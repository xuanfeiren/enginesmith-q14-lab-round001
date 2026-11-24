// Qwen3-14B inference engine for Trainium2, tensor parallel 2 (two logical NeuronCores, LNC=2 each).
// Speaks the EngineSmith protocol (JSON lines on stdin/stdout). Every NEFF is embedded (neffs.S); the host loads
// the checkpoint (safetensors, values unchanged, re-laid out per rank), schedules NEFFs on libnrt (one thread per
// rank: a collective NEFF's nrta_execute_schedule blocks until the peer has scheduled it) and handles the protocol.
// The model is computed by the NEFFs only.
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "json.h"
#include "model.h"

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  char b_[512]; snprintf(b_, sizeof b_, "%s:%d %s -> %d (%s)", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); \
  throw std::runtime_error(b_);} } while (0)

struct NeffBlob { const char* name; const char* start; const char* end; };
extern const NeffBlob NEFFS[];
extern const int NEFF_COUNT;

static const NeffBlob* find_neff(const std::string& n) {
  for (int i = 0; i < NEFF_COUNT; i++) if (n == NEFFS[i].name) return &NEFFS[i];
  throw std::runtime_error("NEFF not embedded: " + n);
}

// ------------------------------------------------------------------------------------------------ safetensors
struct STensor { const uint16_t* data; std::vector<int64_t> shape; std::string dtype; };
struct SafeTensors {
  std::map<std::string, STensor> t;
  void open_dir(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) throw std::runtime_error("cannot open model dir " + dir);
    std::vector<std::string> files;
    while (auto* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors") files.push_back(dir + "/" + n);
    }
    closedir(d);
    for (auto& f : files) {
      int fd = ::open(f.c_str(), O_RDONLY);
      if (fd < 0) throw std::runtime_error("cannot open " + f);
      struct stat st; fstat(fd, &st);
      void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      if (m == MAP_FAILED) throw std::runtime_error("mmap failed " + f);
      ::close(fd);
      const char* base = (const char*)m;
      uint64_t hl; memcpy(&hl, base, 8);
      JParser jp(base + 8, hl);
      JVal h = jp.parse();
      for (auto& kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        STensor s;
        s.dtype = kv.second.get("dtype")->str;
        for (auto& x : kv.second.get("shape")->arr) s.shape.push_back(x.inum);
        int64_t off = kv.second.get("data_offsets")->arr[0].inum;
        s.data = (const uint16_t*)(base + 8 + hl + off);
        t[kv.first] = s;
      }
    }
  }
  const STensor& get(const std::string& n) const {
    auto it = t.find(n);
    if (it == t.end()) throw std::runtime_error("tensor missing: " + n);
    if (it->second.dtype != "BF16") throw std::runtime_error("tensor not bf16: " + n);
    return it->second;
  }
};

static float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

static void parallel_for(int n, int nth, const std::function<void(int)>& f) {
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  std::mutex em; std::string err;
  for (int i = 0; i < nth; i++) th.emplace_back([&] {
    while (true) {
      int k = next++;
      if (k >= n) return;
      try { f(k); } catch (std::exception& e) { std::lock_guard<std::mutex> g(em); err = e.what(); }
    }
  });
  for (auto& x : th) x.join();
  if (!err.empty()) throw std::runtime_error(err);
}

// Weight-stationary tile layout (kernels/qwen3_weights.py ws_layout, generalised): out [128, A*B*128] with
// out[p, (a*B + b)*128 + j] = W_a[j, kts[b]*128 + p], W_a = the 128-row block nsrc[a] (row stride K), zero if null.
static void ws_blocks(uint16_t* out, const std::vector<const uint16_t*>& nsrc, int64_t K, const std::vector<int>& kts) {
  const int64_t A = nsrc.size(), B = kts.size(), rowlen = A * B * 128;
  uint16_t tile[128][128];
  for (int64_t a = 0; a < A; a++)
    for (int64_t b = 0; b < B; b++) {
      int64_t col = (a * B + b) * 128;
      if (!nsrc[a]) { for (int p = 0; p < 128; p++) memset(out + p * rowlen + col, 0, 256); continue; }
      for (int j = 0; j < 128; j++) memcpy(tile[j], nsrc[a] + j * K + (int64_t)kts[b] * 128, 256);
      for (int p = 0; p < 128; p++) {
        uint16_t* o = out + p * rowlen + col;
        for (int j = 0; j < 128; j++) o[j] = tile[j][p];
      }
    }
}

// ------------------------------------------------------------------------------------------------ device state
struct Rank {
  int vnc = 0;
  std::map<std::string, nrt_tensor_t*> t;   // shared inputs by name (one allocation per rank)
  nrt_tensor_t* posA = nullptr; nrt_tensor_t* posB = nullptr;
  std::vector<nrt_model_t*> step;           // one per bucket
  std::vector<nrt_tensor_set_t*> sin[2], sout[2];   // per bucket, per parity (0: posA -> posB, 1: reverse)
  // prefill, per prompt bucket: NEFFs emb / layer / head; input sets [0] emb, [1..L] layer l-1, [L+1] head
  std::vector<nrt_model_t*> pfm[3];
  std::vector<std::vector<nrt_tensor_set_t*>> pfin;
  nrt_tensor_set_t* pfout = nullptr;
  nrt_tensor_t* dummy = nullptr;
  // worker
  std::thread th;
  std::mutex m; std::condition_variable cv;
  std::vector<std::pair<int, int>> job;     // (bucket, parity)
  bool have = false, done = false, quit = false;
  std::string err;
  clk::time_point t_wake, t_sched, t_done;  // diagnostics of the last job
};

static std::vector<Rank> R(W);
static std::vector<int> BUCKETS;   // attention buckets of the embedded step NEFFs
static int NB = 0;
static std::vector<int> PBUCKETS;  // prompt buckets of the embedded prefill NEFFs
static int NPB = 0;
// job item: (b >= 0: step NEFF of attention bucket b, parity) or (-1 - pb: prefill bucket pb, input set index)

static void worker(Rank* k) {
  std::vector<NRT_STATUS> rets;
  while (true) {
    std::vector<std::pair<int, int>> job;
    {
      std::unique_lock<std::mutex> g(k->m);
      k->cv.wait(g, [&] { return k->have || k->quit; });
      if (k->quit) return;
      job.swap(k->job);
      k->have = false;
    }
    std::string err;
    k->t_wake = clk::now();
    try {
      rets.assign(job.size(), NRT_SUCCESS);
      nrta_seq_t seq = 0;
      for (size_t i = 0; i < job.size(); i++) {
        auto [b, par] = job[i];
        nrt_model_t* m; const nrt_tensor_set_t* in; nrt_tensor_set_t* out;
        if (b >= 0) { m = k->step[b]; in = k->sin[par][b]; out = k->sout[par][b]; }
        else { int pb = -1 - b; m = k->pfm[par == 0 ? 0 : par == L + 1 ? 2 : 1][pb]; in = k->pfin[pb][par]; out = k->pfout; }
        NRT_STATUS s;
        while ((s = nrta_execute_schedule(m, in, out, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {}
        CK(s);
        static int dbg_depth = getenv("ENGINE_DBG_DEPTH") ? atoi(getenv("ENGINE_DBG_DEPTH")) : 0;   // experiment
        if (dbg_depth > 0 && i + 1 >= (size_t)dbg_depth) {
          bool d = false; nrta_seq_t w = seq - (dbg_depth - 1);
          while (!d) CK(nrta_is_completed(w, &d));
        }
      }
      k->t_sched = clk::now();
      if (!job.empty()) { bool d = false; while (!d) CK(nrta_is_completed(seq, &d)); }
      k->t_done = clk::now();
      for (auto x : rets) if (x != NRT_SUCCESS) { err = std::string("execution failed: ") + nrt_get_status_as_str(x); break; }
    } catch (std::exception& e) { err = e.what(); }
    std::lock_guard<std::mutex> g(k->m);
    k->err = err; k->done = true; k->cv.notify_all();
  }
}

static std::string job_diag;   // per-rank wake / scheduled / done of the last job, ms after its start
static void run_job(const std::vector<std::pair<int, int>>& job) {
  auto t0 = clk::now();
  for (auto& k : R) { std::lock_guard<std::mutex> g(k.m); k.job = job; k.have = true; k.done = false; k.cv.notify_all(); }
  std::string err;
  for (auto& k : R) {
    std::unique_lock<std::mutex> g(k.m);
    k.cv.wait(g, [&] { return k.done; });
    if (!k.err.empty()) err = k.err;
  }
  if (!err.empty()) throw std::runtime_error(err);
  char b[160]; job_diag.clear();
  auto d = [&](clk::time_point t) { return std::chrono::duration<double, std::milli>(t - t0).count(); };
  for (int r = 0; r < W; r++) { snprintf(b, sizeof b, " r%d %.1f/%.1f/%.1f", r, d(R[r].t_wake), d(R[r].t_sched), d(R[r].t_done)); job_diag += b; }
  snprintf(b, sizeof b, " end %.1f", ms_since(t0)); job_diag += b;
}

static void wr(int r, const std::string& n, const void* p, size_t off, size_t sz) { CK(nrt_tensor_write(R[r].t.at(n), p, off, sz)); }
static void wr_all(const std::string& n, const void* p, size_t off, size_t sz) { for (int r = 0; r < W; r++) wr(r, n, p, off, sz); }

// ------------------------------------------------------------------------------------------------ loading
static void load_weights(const SafeTensors& st) {
  auto t0 = clk::now();
  auto lw = [&](int l, const char* n) { return st.get("model.layers." + std::to_string(l) + "." + n).data; };
  std::vector<int> allk_h(HT); for (int i = 0; i < HT; i++) allk_h[i] = i;
  // tasks: per layer: qkv, o, gu, d (each for 2 ranks x 2 cores); lm head per virtual core; small tensors
  struct Task { int kind, l, v; };
  std::vector<Task> tasks;
  for (int l = 0; l < L; l++) for (int kind = 0; kind < 4; kind++) for (int v = 0; v < C; v++) tasks.push_back({kind, l, v});
  for (int v = 0; v < C; v++) tasks.push_back({4, 0, v});
  std::atomic<size_t> bytes{0};
  parallel_for(tasks.size(), 48, [&](int i) {
    Task tk = tasks[i];
    int r = tk.v / 2, c = tk.v % 2, l = tk.l, v = tk.v;
    std::vector<const uint16_t*> ns; std::vector<int> kts; int64_t K = HID; const char* name = nullptr;
    size_t per = 0;   // elements per (l, c) slice
    if (tk.kind == 0) {          // QKV: q heads v*NQL.., k heads v*NKL.., v heads v*NKL..
      const uint16_t *q = lw(l, "self_attn.q_proj.weight"), *k = lw(l, "self_attn.k_proj.weight"), *vv = lw(l, "self_attn.v_proj.weight");
      for (int a = 0; a < NQL; a++) ns.push_back(q + (int64_t)(v * NQL + a) * 128 * HID);
      for (int a = 0; a < NKL; a++) ns.push_back(k + (int64_t)(v * NKL + a) * 128 * HID);
      for (int a = 0; a < NKL; a++) ns.push_back(vv + (int64_t)(v * NKL + a) * 128 * HID);
      kts = allk_h; name = "wqkv";
    } else if (tk.kind == 1) {   // o_proj: all output tiles, K = this core's heads
      const uint16_t* o = lw(l, "self_attn.o_proj.weight");
      for (int a = 0; a < HT; a++) ns.push_back(o + (int64_t)a * 128 * (NQ * D));
      for (int b = 0; b < NQL; b++) kts.push_back(v * NQL + b);
      K = NQ * D; name = "wo";
    } else if (tk.kind == 2) {   // gate/up: gate tiles v*ITL.., then up tiles v*ITL..
      const uint16_t *g = lw(l, "mlp.gate_proj.weight"), *u = lw(l, "mlp.up_proj.weight");
      for (int a = 0; a < ITL; a++) ns.push_back(g + (int64_t)(v * ITL + a) * 128 * HID);
      for (int a = 0; a < ITL; a++) ns.push_back(u + (int64_t)(v * ITL + a) * 128 * HID);
      kts = allk_h; name = "wgu";
    } else if (tk.kind == 3) {   // down: all output tiles, K = this core's intermediate tiles
      const uint16_t* d = lw(l, "mlp.down_proj.weight");
      for (int a = 0; a < HT; a++) ns.push_back(d + (int64_t)a * 128 * INTER);
      for (int b = 0; b < ITL; b++) kts.push_back(v * ITL + b);
      K = INTER; name = "wd";
    } else {                     // LM head: vocab tiles v*NVL.. (padding tiles zero, masked in the kernel)
      const uint16_t* lm = st.get("lm_head.weight").data;
      for (int a = 0; a < NVL; a++) { int64_t vt = (int64_t)v * NVL + a; ns.push_back(vt < VT ? lm + vt * 128 * HID : nullptr); }
      kts = allk_h; name = "wlm";
    }
    per = (size_t)128 * ns.size() * kts.size() * 128;
    std::vector<uint16_t> buf(per);
    ws_blocks(buf.data(), ns, K, kts);
    size_t off = tk.kind == 4 ? (size_t)c * per : ((size_t)l * 2 + c) * per;
    wr(r, name, buf.data(), off * 2, per * 2);
    bytes += per * 2;
  });
  // embedding, as is
  const STensor& emb = st.get("model.embed_tokens.weight");
  wr_all("embed", emb.data, 0, (size_t)VOCAB * HID * 2);
  // norms (fp32 column layout), RoPE tables, rotation matrix
  auto col = [](const uint16_t* v, int n, float* out) { for (int i = 0; i < n; i++) out[(i % 128) * (n / 128) + i / 128] = bf2f(v[i]); };
  std::vector<float> ln1(L * HID), ln2(L * HID), qn(L * D), kn(L * D), nf(HID);
  for (int l = 0; l < L; l++) {
    col(lw(l, "input_layernorm.weight"), HID, &ln1[l * HID]);
    col(lw(l, "post_attention_layernorm.weight"), HID, &ln2[l * HID]);
    for (int i = 0; i < D; i++) { qn[l * D + i] = bf2f(lw(l, "self_attn.q_norm.weight")[i]); kn[l * D + i] = bf2f(lw(l, "self_attn.k_norm.weight")[i]); }
  }
  col(st.get("model.norm.weight").data, HID, nf.data());
  wr_all("ln1", ln1.data(), 0, ln1.size() * 4); wr_all("ln2", ln2.data(), 0, ln2.size() * 4);
  wr_all("qn", qn.data(), 0, qn.size() * 4); wr_all("kn", kn.data(), 0, kn.size() * 4);
  wr_all("normf", nf.data(), 0, nf.size() * 4);
  std::vector<float> cs(TMAX * 64), sn(TMAX * 64);
  for (int t = 0; t < TMAX; t++) for (int i = 0; i < 64; i++) {
    double inv = 1.0 / pow(ROPE_THETA, (2.0 * i) / D);
    double f = (double)t * inv;
    cs[t * 64 + i] = (float)cos(f); sn[t * 64 + i] = (float)sin(f);
  }
  wr_all("cos", cs.data(), 0, cs.size() * 4); wr_all("sin", sn.data(), 0, sn.size() * 4);
  if (R[0].t.count("cosT")) {   // prefill layout: cosT[d, t] = cos(t * f[d % 64])
    std::vector<float> ct(128 * TMAX), stt(128 * TMAX);
    for (int d = 0; d < 128; d++) for (int t = 0; t < TMAX; t++) { ct[d * TMAX + t] = cs[t * 64 + d % 64]; stt[d * TMAX + t] = sn[t * 64 + d % 64]; }
    wr_all("cosT", ct.data(), 0, ct.size() * 4); wr_all("sinT", stt.data(), 0, stt.size() * 4);
  }
  std::vector<uint16_t> rot(128 * 128, 0);
  for (int m = 0; m < 64; m++) { rot[(m + 64) * 128 + m] = 0xBF80; rot[m * 128 + m + 64] = 0x3F80; }   // -1, +1 in bf16
  wr_all("rot", rot.data(), 0, rot.size() * 2);
  for (int r = 0; r < W; r++) {
    std::vector<float> rkb(128, (float)(r * 2 * NVL * 128));
    wr(r, "rkb", rkb.data(), 0, 512);
  }
  fprintf(stderr, "weights: %.1f GB re-laid out and uploaded in %.1f s\n", bytes / 1e9, ms_since(t0) / 1e3);
}

static void zero_tensor(int r, const std::string& n) {
  nrt_tensor_t* t = R[r].t.at(n);
  size_t sz = nrt_tensor_get_size(t);
  std::vector<char> z(std::min(sz, (size_t)64 << 20), 0);
  for (size_t o = 0; o < sz; o += z.size()) CK(nrt_tensor_write(t, z.data(), o, std::min(z.size(), sz - o)));
}

static std::vector<std::pair<nrt_tensor_t*, nrt_tensor_t*>> dbg_fresh;   // experiment: (slice, standalone copy)
static bool is_layer_tensor(const std::string& n) {
  static const char* names[] = {"ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "kcache", "vcache"};
  for (auto x : names) if (n == x) return true;
  return false;
}

// prefill NEFFs pf<S>_{emb,layer,head}: per-layer inputs bound to slices of the stacked tensors, ids to the head of
// the prompt tensor, scratch s_* to slices of one buffer per name sized for the largest bucket
static void setup_prefill() {
  auto t0 = clk::now();
  for (int i = 0; i < NEFF_COUNT; i++) {
    std::string n = NEFFS[i].name;
    if (n.rfind("pf", 0) == 0 && n.size() > 6 && n.substr(n.size() - 6) == "_layer") PBUCKETS.push_back(atoi(n.c_str() + 2));
  }
  std::sort(PBUCKETS.begin(), PBUCKETS.end());
  NPB = PBUCKETS.size();
  const char* st[3] = {"emb", "layer", "head"};
  for (int pb = 0; pb < NPB; pb++)
    for (int s = 0; s < 3; s++) {
      const NeffBlob* nb = find_neff("pf" + std::to_string(PBUCKETS[pb]) + "_" + st[s]);
      for (int r = 0; r < W; r++) {
        nrt_model_t* m;
        CK(nrt_load_collectives(nb->start, nb->end - nb->start, R[r].vnc, -1, r, W, &m));
        R[r].pfm[s].push_back(m);
      }
    }
  for (int r = 0; r < W; r++) {
    auto& k = R[r];
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, 64, "dummy", &k.dummy));
    CK(nrt_allocate_tensor_set(&k.pfout));
    CK(nrt_add_tensor_to_tensor_set(k.pfout, "output_0", k.dummy));
    // scratch sizes: the largest over all buckets
    std::map<std::string, size_t> scratch;
    for (int pb = 0; pb < NPB; pb++) for (int s = 0; s < 3; s++) {
      nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(k.pfm[s][pb], &info));
      for (uint64_t i = 0; i < info->tensor_count; i++) {
        std::string n = info->tensor_array[i].name;
        if (n.rfind("s_", 0) == 0) scratch[n] = std::max(scratch[n], (size_t)info->tensor_array[i].size);
      }
      nrt_free_model_tensor_info(info);
    }
    for (auto& [n, sz] : scratch) { nrt_tensor_t* t; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, sz, n.c_str(), &t)); k.t[n] = t; }
    std::map<std::pair<std::string, size_t>, nrt_tensor_t*> slices;   // (name, offset) -> slice
    auto slice = [&](const std::string& n, size_t off, size_t sz) {
      auto key = std::make_pair(n, off * 1000003 + sz);
      if (!slices.count(key)) {
        nrt_tensor_t* src = k.t.at(n);
        if (off + sz > nrt_tensor_get_size(src)) throw std::runtime_error("slice out of range: " + n);
        nrt_tensor_t* t; CK(nrt_tensor_allocate_slice(src, off, sz, n.c_str(), &t));
        slices[key] = t;
      }
      return slices[key];
    };
    k.pfin.resize(NPB);
    for (int pb = 0; pb < NPB; pb++) {
      for (int idx = 0; idx < L + 2; idx++) {
        int s = idx == 0 ? 0 : idx == L + 1 ? 2 : 1;
        int l = s == 1 ? idx - 1 : 0;
        nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(k.pfm[s][pb], &info));
        nrt_tensor_set_t* in; CK(nrt_allocate_tensor_set(&in));
        for (uint64_t i = 0; i < info->tensor_count; i++) {
          auto& ti = info->tensor_array[i];
          std::string n = ti.name;
          if (ti.usage != NRT_TENSOR_USAGE_INPUT) continue;
          nrt_tensor_t* t;
          if (n == "ids") t = slice("prompt", 0, ti.size);
          else if (is_layer_tensor(n)) {
            size_t full = nrt_tensor_get_size(k.t.at(n));
            if (full != ti.size * L) throw std::runtime_error("layer slice size mismatch " + n);
            t = slice(n, (size_t)l * ti.size, ti.size);
            const char* fr = getenv("ENGINE_DBG_PF_FRESH");
            if (fr && idx == 1 && (std::string(fr) == "1" || (":" + std::string(fr) + ":").find(":" + n + ":") != std::string::npos)) {   // timing experiment: standalone tensors (garbage values)
              nrt_tensor_t* src = t;
              CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, ti.size, ti.name, &t));
              dbg_fresh.push_back({src, t});
              nrt_tensor_device_allocation_info_t ai; nrt_tensor_get_device_allocation_info(t, &ai);
              nrt_tensor_device_allocation_info_t bi; nrt_tensor_get_device_allocation_info(k.t.at(n), &bi);
              fprintf(stderr, "dbg fresh %s r%d: fresh pa %#lx hbm %d size %zu; stacked pa %#lx hbm %d size %zu\n", n.c_str(), r,
                      (unsigned long)ai.physical_address, ai.hbm_index, ai.size, (unsigned long)bi.physical_address, bi.hbm_index, bi.size);
            }
          } else if (n.rfind("s_", 0) == 0) t = slice(n, 0, ti.size);
          else {
            if (!k.t.count(n)) { CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, ti.size, ti.name, &t)); k.t[n] = t; }
            t = k.t[n];
            if (nrt_tensor_get_size(t) != ti.size) throw std::runtime_error("prefill tensor size mismatch " + n);
          }
          CK(nrt_add_tensor_to_tensor_set(in, ti.name, t));
        }
        nrt_free_model_tensor_info(info);
        k.pfin[pb].push_back(in);
      }
    }
  }
  fprintf(stderr, "prefill: %d buckets loaded in %.1f s\n", NPB, ms_since(t0) / 1e3);
}

static void setup_device(const std::string& model_dir) {
  auto t0 = clk::now();
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  for (int r = 0; r < W; r++) R[r].vnc = r;
  // load the step NEFFs on every rank
  for (int i = 0; i < NEFF_COUNT; i++) {
    std::string n = NEFFS[i].name;
    if (n.rfind("step_b", 0) == 0) BUCKETS.push_back(atoi(n.c_str() + 6));
  }
  std::sort(BUCKETS.begin(), BUCKETS.end());
  NB = BUCKETS.size();
  if (NB == 0 || BUCKETS.back() > TMAX) throw std::runtime_error("bad step NEFF set");
  for (int b = 0; b < NB; b++) {
    const NeffBlob* nb = find_neff("step_b" + std::to_string(BUCKETS[b]));
    for (int r = 0; r < W; r++) {
      nrt_model_t* m;
      CK(nrt_load_collectives(nb->start, nb->end - nb->start, R[r].vnc, -1, r, W, &m));
      R[r].step.push_back(m);
    }
  }
  fprintf(stderr, "nrt_load: %d NEFFs x %d ranks in %.1f s\n", NB, W, ms_since(t0) / 1e3);
  // tensors: allocate the inputs of the first NEFF once per rank, share them with every NEFF
  for (int r = 0; r < W; r++) {
    auto& k = R[r];
    for (int b = 0; b < NB; b++) {
      nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(k.step[b], &info));
      for (int s = 0; s < 2; s++) { nrt_tensor_set_t *a, *o; CK(nrt_allocate_tensor_set(&a)); CK(nrt_allocate_tensor_set(&o)); k.sin[s].push_back(a); k.sout[s].push_back(o); }
      for (uint64_t i = 0; i < info->tensor_count; i++) {
        auto& ti = info->tensor_array[i];
        std::string n = ti.name;
        if (ti.usage == NRT_TENSOR_USAGE_INPUT) {
          if (n == "pos") {
            if (!k.posA) { CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, ti.size, "posA", &k.posA));
                           CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, ti.size, "posB", &k.posB)); }
            CK(nrt_add_tensor_to_tensor_set(k.sin[0][b], "pos", k.posA));
            CK(nrt_add_tensor_to_tensor_set(k.sin[1][b], "pos", k.posB));
            continue;
          }
          if (!k.t.count(n)) {
            nrt_tensor_t* t; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, ti.size, ti.name, &t));
            k.t[n] = t;
          } else if (nrt_tensor_get_size(k.t[n]) != ti.size) throw std::runtime_error("tensor size mismatch " + n);
          CK(nrt_add_tensor_to_tensor_set(k.sin[0][b], ti.name, k.t[n]));
          CK(nrt_add_tensor_to_tensor_set(k.sin[1][b], ti.name, k.t[n]));
        } else {
          if (n != "output_0") throw std::runtime_error("unexpected output " + n);
          CK(nrt_add_tensor_to_tensor_set(k.sout[0][b], ti.name, k.posB));
          CK(nrt_add_tensor_to_tensor_set(k.sout[1][b], ti.name, k.posA));
        }
      }
      nrt_free_model_tensor_info(info);
    }
  }
  setup_prefill();
  for (int r = 0; r < W; r++) { zero_tensor(r, "kcache"); zero_tensor(r, "vcache"); zero_tensor(r, "toks"); zero_tensor(r, "prompt"); }
  SafeTensors st; st.open_dir(model_dir);
  load_weights(st);
  if (getenv("ENGINE_DBG_PF_FRESHCOPY")) for (auto& [a, b] : dbg_fresh) CK(nrt_tensor_copy(a, 0, b, 0, nrt_tensor_get_size(b)));
  for (auto& k : R) k.th = std::thread(worker, &k);
}

// ------------------------------------------------------------------------------------------------ generation
static int cur_pos = 0;      // positions consumed
static int last_tok = -1;    // last emitted token (not consumed yet)
static bool have_state = false;

static int bucket_for(int pos) {
  for (int b = 0; b < NB; b++) if (pos <= BUCKETS[b]) return b;
  throw std::runtime_error("position beyond the largest bucket");
}

// one queued chain: [prefill of bucket pb (if pb >= 0)] + decode steps at positions [p0, p1); returns
// toks[r0 .. r0 + nr)
static std::vector<int> run_chain(int pb, int p0, int p1, int r0, int nr) {
  int32_t pv = p0;
  for (int r = 0; r < W; r++) CK(nrt_tensor_write(R[r].posA, &pv, 0, 4));
  std::vector<std::pair<int, int>> job;
  static int dbg_layers = getenv("ENGINE_DBG_PF_LAYERS") ? atoi(getenv("ENGINE_DBG_PF_LAYERS")) : L;   // timing only
  static bool dbg_same = getenv("ENGINE_DBG_PF_SAMESET") != nullptr;   // timing only: every layer on layer 0's set
  if (pb >= 0) for (int i = 0; i < L + 2; i++) if (i == 0 || i == L + 1 || i <= dbg_layers) job.push_back({-1 - pb, dbg_same && i > 0 && i <= L ? 1 : i});
  for (int p = p0; p < p1; p++) job.push_back({bucket_for(p), (p - p0) % 2});
  run_job(job);
  std::vector<int32_t> out(nr);
  if (nr) CK(nrt_tensor_read(R[0].t.at("toks"), out.data(), (size_t)r0 * 4, out.size() * 4));
  return std::vector<int>(out.begin(), out.end());
}

// decode steps at positions [p0, p1): returns toks[p0+1 .. p1]
static std::vector<int> run_steps(int p0, int p1) {
  if (p1 <= p0) return {};
  return run_chain(-1, p0, p1, p0 + 1, p1 - p0);
}

static bool use_prefill() { static bool off = getenv("ENGINE_NO_PREFILL") != nullptr; return NPB > 0 && !off; }
static int pbucket_for(int n) {   // smallest prompt bucket >= n, else the largest
  for (int b = 0; b < NPB; b++) if (n <= PBUCKETS[b]) return b;
  return NPB - 1;
}

static int maxpos() { return std::min(MAXPOS, BUCKETS.back() + 1); }   // last step at pos <= largest bucket

static void set_plen(int n) { int32_t v = n; wr_all("plen", &v, 0, 4); }

static std::vector<int> generate(const std::vector<int64_t>& prompt, int max_tokens, bool reset) {
  if (max_tokens < 1) throw std::runtime_error("max_tokens must be at least 1");
  std::vector<int> out;
  if (reset) {
    if (prompt.empty()) throw std::runtime_error("a reset request needs a prompt");
    for (auto x : prompt) if (x < 0 || x >= VOCAB) throw std::runtime_error("token id out of range");
    std::vector<int32_t> p(prompt.begin(), prompt.end());
    bool quirk = p.size() == 1;   // the reference pads a one-token prompt with token 0 and predicts after [t, 0]
    if (quirk) p.push_back(0);
    int plen = p.size();
    int consumed = quirk ? 1 : plen;
    if (consumed + max_tokens - 1 > maxpos() || plen > maxpos()) throw std::runtime_error("request longer than the compiled length");
    wr_all("prompt", p.data(), 0, p.size() * 4);
    set_plen(plen);
    have_state = false;
    int m1 = quirk ? 1 : max_tokens;   // tokens produced by the first chain
    if (!use_prefill()) {
      out = run_steps(0, plen + m1 - 1);
      out.erase(out.begin(), out.begin() + (plen - 1));
    } else {
      int pb = pbucket_for(plen), S = PBUCKETS[pb];
      if (plen <= S) {
        out = run_chain(pb, plen, plen + m1 - 1, plen, m1);
      } else {   // longer than the largest bucket: prefill its first S tokens, the rest through decode steps
        set_plen(S);
        run_chain(pb, 0, 0, 0, 0);
        set_plen(plen);
        out = run_chain(-1, S, plen + m1 - 1, plen, m1);
      }
    }
    if (!quirk) {
      cur_pos = plen + max_tokens - 1;
    } else {
      int g = out[0];
      int32_t gv = g;
      wr_all("toks", &gv, 4, 4);   // toks[1] = g: the next input at position 1
      set_plen(1);
      auto rest = run_steps(1, max_tokens);
      out.insert(out.end(), rest.begin(), rest.end());
      cur_pos = max_tokens;
    }
  } else {
    if (!prompt.empty()) throw std::runtime_error("a continuation has an empty prompt");
    if (!have_state) throw std::runtime_error("no state to continue");
    if (cur_pos + max_tokens > maxpos()) throw std::runtime_error("request longer than the compiled length");
    out = run_steps(cur_pos, cur_pos + max_tokens);
    cur_pos += max_tokens;
  }
  last_tok = out.back();
  have_state = true;
  return out;
}

static void warmup() {
  auto t0 = clk::now();
  int32_t p[2] = {0, 0};
  wr_all("prompt", p, 0, 8);
  set_plen(2);
  for (int b = 0; b < NB; b++) {
    int pos = b == 0 ? 0 : BUCKETS[b - 1] + 1;
    int32_t pv = pos;
    for (int r = 0; r < W; r++) CK(nrt_tensor_write(R[r].posA, &pv, 0, 4));
    std::vector<std::pair<int, int>> job = {{b, 0}, {b, 1}, {b, 0}};
    run_job(job);
  }
  for (int pb = 0; pb < NPB; pb++) run_chain(pb, 0, 0, 0, 0);
  zero_tensor(0, "toks"); zero_tensor(1, "toks");
  fprintf(stderr, "warmup %.1f s\n", ms_since(t0) / 1e3);
}

// ------------------------------------------------------------------------------------------------ main
static void setup_env() {
  // the first two visible logical cores are the engine's (the harness lists four; the last two are the judge's)
  const char* v = getenv("NEURON_RT_VISIBLE_CORES");
  int first = 0;
  if (v && *v) first = atoi(v);
  else if (getenv("ENGINESMITH_NEURON_CORE")) first = atoi(getenv("ENGINESMITH_NEURON_CORE"));
  char buf[64];
  snprintf(buf, sizeof buf, "%d-%d", first, first + W - 1);
  setenv("NEURON_RT_VISIBLE_CORES", buf, 1);
  if (!getenv("NEURON_RT_ROOT_COMM_ID")) {
    snprintf(buf, sizeof buf, "localhost:%d", 41000 + (first % 128) * 50 + (int)(getpid() % 50));
    setenv("NEURON_RT_ROOT_COMM_ID", buf, 1);
  }
  if (!getenv("ENGINE_DBG_KEEP_BARRIER")) setenv("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1", 1);
  fprintf(stderr, "cores %s, comm %s\n", getenv("NEURON_RT_VISIBLE_CORES"), getenv("NEURON_RT_ROOT_COMM_ID"));
}

int main(int argc, char** argv) {
  std::string model_dir;
  for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--model" && i + 1 < argc) model_dir = argv[++i];
  if (model_dir.empty()) { fprintf(stderr, "usage: engine --model MODEL_DIR\n"); return 2; }
  std::ios::sync_with_stdio(false);
  // replies go to the original stdout; anything the runtime prints to fd 1 goes to stderr instead
  int reply_fd = dup(1);
  dup2(2, 1);
  setup_env();
  bool ready = false;
  std::string line;
  auto t_reply = clk::now();
  std::string log;
  double last_log_ms = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    auto t_recv = clk::now();
    double gap = ms_since(t_reply);
    int64_t id = 0;
    std::string reply;
    try {
      JParser jp(line.data(), line.size());
      JVal req = jp.parse();
      if (auto* x = req.get("id")) id = x->inum;
      std::string op = req.get("op") ? req.get("op")->str : "";
      if (op == "hello") {
        if (!ready) { setup_device(model_dir); warmup(); ready = true; }
        reply = "{\"id\": " + std::to_string(id) + ", \"ok\": true, \"protocol\": 1}";
      } else if (op == "generate") {
        if (!ready) { setup_device(model_dir); warmup(); ready = true; }
        std::vector<int64_t> prompt;
        if (auto* p = req.get("prompt")) for (auto& x : p->arr) prompt.push_back(x.inum);
        int mt = req.get("max_tokens") ? (int)req.get("max_tokens")->inum : 1;
        bool reset = req.get("reset") ? req.get("reset")->b : true;
        auto t0 = clk::now();
        double parse = std::chrono::duration<double, std::milli>(t0 - t_recv).count();
        auto toks = generate(prompt, mt, reset);
        char b[512];
        snprintf(b, sizeof b, "generate n=%zu max=%d reset=%d: %.1f ms (parse %.2f ms, idle before %.0f ms, last log %.2f ms; last job%s)",
                 prompt.size(), mt, (int)reset, ms_since(t0), parse, gap, last_log_ms, job_diag.c_str());
        log = b;
        reply = "{\"id\": " + std::to_string(id) + ", \"ok\": true, \"tokens\": [";
        for (size_t i = 0; i < toks.size(); i++) { if (i) reply += ", "; reply += std::to_string(toks[i]); }
        reply += "]}";
      } else {
        throw std::runtime_error("unknown op " + op);
      }
    } catch (std::exception& e) {
      fprintf(stderr, "error: %s\n", e.what());
      reply = "{\"id\": " + std::to_string(id) + ", \"ok\": false, \"error\": \"" + json_escape(e.what()) + "\"}";
    }
    // the reply first (one write), the log line after it: a slow stderr file must never delay a reply
    reply += '\n';
    auto tw = clk::now();
    for (size_t o = 0; o < reply.size();) {
      ssize_t n = write(reply_fd, reply.data() + o, reply.size() - o);
      if (n < 0) { if (errno == EINTR) continue; return 1; }
      o += n;
    }
    t_reply = clk::now();
    if (!log.empty()) {
      auto tl = clk::now();
      fprintf(stderr, "%s, reply write %.2f ms\n", log.c_str(), std::chrono::duration<double, std::milli>(t_reply - tw).count());
      last_log_ms = ms_since(tl);
      log.clear();
    }
  }
  for (auto& k : R) { { std::lock_guard<std::mutex> g(k.m); k.quit = true; k.cv.notify_all(); } if (k.th.joinable()) k.th.join(); }
  return 0;
}
