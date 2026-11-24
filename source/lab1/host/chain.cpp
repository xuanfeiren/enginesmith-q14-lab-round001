// chain: run a decode-step NEFF n times back to back on the device with no host work between steps.
//   chain <model.neff> <iodir> <n_steps> <in:out>[,<in:out>...] [reps=1] [vnc_count=1]
// Every tensor is allocated once; inputs come from <iodir>/<name>.bin (else zeros). For every swap pair
// in:out (e.g. pos:output_0) a second device buffer is allocated and two tensor sets are built: set A reads
// buffer X and writes buffer Y, set B reads Y and writes X. All n_steps executions are queued at once with
// nrta_execute_schedule, alternating A/B, then the host waits once. Tensors updated in place (KV cache,
// token history) advance step by step. After the first run every tensor is dumped to <iodir>/out_<name>.bin
// (the swap partner as out_<name>.B.bin). reps>1: the initial state is rewritten (inputs only, small ones
// < 64 MB plus the caches) and the run repeated for timing; prints us/step for each repetition.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
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
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: chain model.neff iodir n_steps in:out[,in:out] [reps] [vnc_count]\n"); return 2; }
  std::string dir = argv[2];
  int n = atoi(argv[3]);
  std::map<std::string, std::string> sw;  // in -> out and out -> in
  { std::stringstream ss(argv[4]); std::string p;
    while (std::getline(ss, p, ',')) { auto c = p.find(':'); sw[p.substr(0, c)] = p.substr(c + 1); sw[p.substr(c + 1)] = p.substr(0, c); } }
  int reps = argc > 5 ? atoi(argv[5]) : 1;
  int vncs = argc > 6 ? atoi(argv[6]) : 1;
  std::vector<char> neff;
  if (!slurp(argv[1], neff)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  nrt_model_t* m; auto t0 = clk::now();
  CK(nrt_load(neff.data(), neff.size(), 0, vncs, &m));
  printf("nrt_load %.1f ms (%zu bytes)\n", us_since(t0) / 1e3, neff.size());
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m, &info));
  nrt_tensor_set_t *inA, *outA, *inB, *outB;
  CK(nrt_allocate_tensor_set(&inA)); CK(nrt_allocate_tensor_set(&outA));
  CK(nrt_allocate_tensor_set(&inB)); CK(nrt_allocate_tensor_set(&outB));
  struct T { std::string name; size_t size; bool in; nrt_tensor_t* t; std::vector<char> init; };
  std::vector<T> ts;
  std::map<std::string, nrt_tensor_t*> byname;
  t0 = clk::now();
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& ti = info->tensor_array[i];
    T t{ti.name, ti.size, ti.usage == NRT_TENSOR_USAGE_INPUT, nullptr, {}};
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, t.size, ti.name, &t.t));
    std::vector<char> buf;
    if (t.in && slurp(dir + "/" + t.name + ".bin", buf)) {
      if (buf.size() != t.size) { fprintf(stderr, "%s: file %zu bytes, tensor %zu\n", t.name.c_str(), buf.size(), t.size); return 1; }
    } else buf.assign(t.size, 0);
    CK(nrt_tensor_write(t.t, buf.data(), 0, t.size));
    bool keep = t.in && (t.size < (64u << 20) || t.name.find("cache") != std::string::npos);
    if (keep) t.init = buf;
    byname[t.name] = t.t;
    ts.push_back(std::move(t));
  }
  printf("upload %.1f ms\n", us_since(t0) / 1e3);
  for (auto& t : ts) {
    printf("  %-4s %-10s %zu bytes%s\n", t.in ? "in" : "out", t.name.c_str(), t.size, sw.count(t.name) ? " (swap)" : "");
    if (!sw.count(t.name)) {
      CK(nrt_add_tensor_to_tensor_set(t.in ? inA : outA, t.name.c_str(), t.t));
      CK(nrt_add_tensor_to_tensor_set(t.in ? inB : outB, t.name.c_str(), t.t));
    } else {
      // set A: own buffer; set B: partner's buffer under this name
      CK(nrt_add_tensor_to_tensor_set(t.in ? inA : outA, t.name.c_str(), t.t));
      CK(nrt_add_tensor_to_tensor_set(t.in ? inB : outB, t.name.c_str(), byname[sw[t.name]]));
    }
  }
  for (int r = 0; r < reps; r++) {
    if (r > 0) for (auto& t : ts) if (!t.init.empty()) CK(nrt_tensor_write(t.t, t.init.data(), 0, t.size));
    std::vector<NRT_STATUS> rets(n); nrta_seq_t seq = 0;
    t0 = clk::now();
    for (int i = 0; i < n; i++) {
      NRT_STATUS s;
      while ((s = nrta_execute_schedule(m, i % 2 ? inB : inA, i % 2 ? outB : outA, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {}
      CK(s);
    }
    bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
    double us = us_since(t0);
    for (auto x : rets) if (x != NRT_SUCCESS) { fprintf(stderr, "exec ret %d (%s)\n", (int)x, nrt_get_status_as_str(x)); exit(1); }
    printf("CHAIN rep %d: %d steps in %.1f us = %.2f us/step\n", r, n, us, us / n);
    if (r == 0) for (auto& t : ts) {
      if (t.name[0] == 'w' || t.name == "embed") continue;   // weights: not dumped
      std::vector<char> buf(t.size); CK(nrt_tensor_read(t.t, buf.data(), 0, t.size));
      std::ofstream f(dir + "/out_" + t.name + ".bin", std::ios::binary); f.write(buf.data(), t.size);
    }
  }
  nrt_close();
  return 0;
}
