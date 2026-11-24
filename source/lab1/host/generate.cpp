// generate: prefill NEFF + decode-step NEFF on one logical core, sharing every tensor of the same name
// (weights, KV cache, token history), then n_decode decode steps chained on the device. One host read at the end.
//   generate <prefill.neff | emb.neff,layer.neff,head.neff> <decode.neff> <iodir> <n_decode> [reps=1]
// Layered prefill (3 NEFFs): emb once, the 1-layer NEFF once per layer l with every tensor whose pool size is
// L x its NEFF size bound to slice l (nrt_tensor_allocate_slice: weights, KV cache), then head once.
// Inputs from <iodir>/<name>.bin (else zeros). Wiring: the prefill's output_0 (= plen) is the decode's `pos`
// buffer X; decode set A reads X and writes its output_0 into Y, set B reads Y and writes X. Everything (prefill +
// all decode steps) is queued at once with nrta_execute_schedule; the compute queue runs in order.
// Prints prefill time (queued alone), decode us/step and the total; dumps toks to <iodir>/out_toks.bin.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static bool slurp(const std::string& p, std::vector<char>& out) {
  std::ifstream f(p, std::ios::binary); if (!f) return false;
  out.assign((std::istreambuf_iterator<char>(f)), {}); return true;
}
struct Pool {   // one device tensor per name
  std::string dir; std::map<std::string, nrt_tensor_t*> t; std::map<std::string, size_t> sz;
  nrt_tensor_t* get(const std::string& name, size_t size, bool load) {
    auto it = t.find(name);
    if (it != t.end()) { if (sz[name] != size) { fprintf(stderr, "%s: size %zu vs %zu\n", name.c_str(), sz[name], size); exit(1); } return it->second; }
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, size, name.c_str(), &x));
    std::vector<char> buf;
    if (!(load && slurp(dir + "/" + name + ".bin", buf) && buf.size() == size)) buf.assign(size, 0);
    CK(nrt_tensor_write(x, buf.data(), 0, size));
    t[name] = x; sz[name] = size; return x;
  }
};
static nrt_model_t* load(const char* p) {
  std::vector<char> neff; if (!slurp(p, neff)) { fprintf(stderr, "cannot read %s\n", p); exit(1); }
  nrt_model_t* m; auto t0 = clk::now(); CK(nrt_load(neff.data(), neff.size(), 0, 1, &m));
  printf("nrt_load %s: %.1f ms (%zu bytes)\n", p, us_since(t0) / 1e3, neff.size()); return m;
}
static void wait_all(nrta_seq_t seq, std::vector<NRT_STATUS>& rets) {
  bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
  for (auto r : rets) if (r != NRT_SUCCESS) { fprintf(stderr, "exec ret %d (%s)\n", (int)r, nrt_get_status_as_str(r)); exit(1); }
}
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: generate prefill.neff decode.neff iodir n_decode [reps]\n"); return 2; }
  int n = atoi(argv[4]); int reps = argc > 5 ? atoi(argv[5]) : 1;
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  std::string pf = argv[1];
  std::vector<std::string> pfs; { size_t a = 0, b; while ((b = pf.find(',', a)) != std::string::npos) { pfs.push_back(pf.substr(a, b - a)); a = b + 1; } pfs.push_back(pf.substr(a)); }
  bool layered = pfs.size() == 3;
  nrt_model_t* md = load(argv[2]);
  Pool pool; pool.dir = argv[3];
  auto t0 = clk::now();
  nrt_tensor_set_t *inA, *outA, *inB, *outB;
  CK(nrt_allocate_tensor_set(&inA)); CK(nrt_allocate_tensor_set(&outA));
  CK(nrt_allocate_tensor_set(&inB)); CK(nrt_allocate_tensor_set(&outB));
  nrt_tensor_info_array_t* info;
  // decode first: its tensors (full stacked weights / caches) define the pool sizes
  CK(nrt_get_model_tensor_info(md, &info));
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& ti = info->tensor_array[i]; bool in = ti.usage == NRT_TENSOR_USAGE_INPUT; std::string nm = ti.name;
    if (nm == "pos") {
      CK(nrt_add_tensor_to_tensor_set(inA, ti.name, pool.get("posX", ti.size, false)));
      CK(nrt_add_tensor_to_tensor_set(inB, ti.name, pool.get("posY", ti.size, false)));
    } else if (nm == "output_0") {
      CK(nrt_add_tensor_to_tensor_set(outA, ti.name, pool.get("posY", ti.size, false)));
      CK(nrt_add_tensor_to_tensor_set(outB, ti.name, pool.get("posX", ti.size, false)));
    } else {
      auto x = pool.get(nm, ti.size, in);
      CK(nrt_add_tensor_to_tensor_set(in ? inA : outA, ti.name, x));
      CK(nrt_add_tensor_to_tensor_set(in ? inB : outB, ti.name, x));
    }
  }
  // prefill executions: (model, in set, out set)
  struct Ex { nrt_model_t* m; nrt_tensor_set_t* in; nrt_tensor_set_t* out; };
  std::vector<Ex> pre;
  auto bind = [&](nrt_model_t* m, int layer, int L) {
    Ex e{m, nullptr, nullptr}; CK(nrt_allocate_tensor_set(&e.in)); CK(nrt_allocate_tensor_set(&e.out));
    nrt_tensor_info_array_t* inf; CK(nrt_get_model_tensor_info(m, &inf));
    for (uint64_t i = 0; i < inf->tensor_count; i++) {
      auto& ti = inf->tensor_array[i]; bool in = ti.usage == NRT_TENSOR_USAGE_INPUT;
      std::string nm = ti.name; if (nm == "output_0") nm = "posX";
      nrt_tensor_t* x;
      if (layer >= 0 && pool.t.count(nm) && pool.sz[nm] == (size_t)L * ti.size) {
        CK(nrt_tensor_allocate_slice(pool.t[nm], (size_t)layer * ti.size, ti.size, ti.name, &x));
      } else if (layer >= 0 && !pool.t.count(nm) && (nm == "wo_p" || nm == "wd_p")) {
        // per-layer prefill-only weights: allocate the stacked tensor once, from <name>.bin of L x size
        pool.get(nm, (size_t)L * ti.size, true);
        CK(nrt_tensor_allocate_slice(pool.t[nm], (size_t)layer * ti.size, ti.size, ti.name, &x));
      } else x = pool.get(nm, ti.size, in);
      CK(nrt_add_tensor_to_tensor_set(in ? e.in : e.out, ti.name, x));
    }
    pre.push_back(e);
  };
  if (!layered) bind(load(pfs[0].c_str()), -1, 0);
  else {
    nrt_model_t* me = load(pfs[0].c_str()); nrt_model_t* ml = load(pfs[1].c_str()); nrt_model_t* mh = load(pfs[2].c_str());
    int L = 0;
    nrt_tensor_info_array_t* inf; CK(nrt_get_model_tensor_info(ml, &inf));
    for (uint64_t i = 0; i < inf->tensor_count; i++) if (!strcmp(inf->tensor_array[i].name, "wqkv")) L = pool.sz["wqkv"] / inf->tensor_array[i].size;
    printf("layered prefill: %d layers\n", L);
    bind(me, -1, 0);
    for (int l = 0; l < L; l++) bind(ml, l, L);
    bind(mh, -1, 0);
  }
  size_t tot = 0; for (auto& kv : pool.sz) tot += kv.second;
  printf("allocated + uploaded %zu tensors, %.2f GB, in %.1f ms\n", pool.t.size(), tot / 1e9, us_since(t0) / 1e3);
  for (int r = 0; r < reps; r++) {
    std::vector<NRT_STATUS> rets(n + 1, NRT_SUCCESS); nrta_seq_t seq = 0; NRT_STATUS s;
    // prefill alone (to time it), then prefill + decode chain all queued at once
    std::vector<NRT_STATUS> rp(pre.size());
    t0 = clk::now();
    for (size_t j = 0; j < pre.size(); j++) { while ((s = nrta_execute_schedule(pre[j].m, pre[j].in, pre[j].out, 0, &rp[j], &seq)) == NRT_QUEUE_FULL) {} CK(s); }
    wait_all(seq, rp); double tp = us_since(t0);
    t0 = clk::now();
    for (size_t j = 0; j < pre.size(); j++) { while ((s = nrta_execute_schedule(pre[j].m, pre[j].in, pre[j].out, 0, &rp[j], &seq)) == NRT_QUEUE_FULL) {} CK(s); }
    for (int i = 0; i < n; i++) {
      while ((s = nrta_execute_schedule(md, i % 2 ? inB : inA, i % 2 ? outB : outA, 0, &rets[i + 1], &seq)) == NRT_QUEUE_FULL) {}
      CK(s);
    }
    wait_all(seq, rets); wait_all(seq, rp); double tt = us_since(t0);
    printf("GEN rep %d: prefill alone (%zu executions) %.2f ms; prefill + %d decode steps %.2f ms -> decode %.1f us/step (assuming the same prefill time)\n",
           r, pre.size(), tp / 1e3, n, tt / 1e3, (tt - tp) / n);
  }
  auto tt = pool.t["toks"]; size_t ts = pool.sz["toks"];
  std::vector<char> buf(ts); CK(nrt_tensor_read(tt, buf.data(), 0, ts));
  std::ofstream f(pool.dir + "/out_toks.bin", std::ios::binary); f.write(buf.data(), ts);
  nrt_close();
  return 0;
}
