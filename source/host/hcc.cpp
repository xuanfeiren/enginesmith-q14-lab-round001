// hcc: host-driven (out-of-NEFF) collectives of libnrt: nrt_cc_global_comm_init + nrta_cc_prepare/nrta_cc_schedule
// (marked "work in progress" in nrt.h / nrt_async.h). One process, one thread per rank (rank r on visible vnc r).
//   NEURON_RT_ROOT_COMM_ID=localhost:<port> NEURON_RT_VISIBLE_CORES=56-57 hcc <W> [stream=0|1]
// All-reduce (fp32 add) of 4 KB .. 16 MB; checks the sum; median latency of prepare+schedule+wait.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int W = atoi(argv[1]);
  bool stream = argc > 2 && atoi(argv[2]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  if (stream) CK(nrt_cc_create_stream(0));
  std::vector<uint32_t> rg(W); for (int i = 0; i < W; i++) rg[i] = i;
  std::vector<std::thread> th;
  size_t sizes[] = {4096, 65536, 1 << 20, 16 << 20};
  std::vector<std::vector<double>> res(W);
  for (int r = 0; r < W; r++) th.emplace_back([&, r] {
    CK(nrt_cc_global_comm_init(r, r, W));
    nrt_cc_comm_t comm{rg.data(), (uint32_t)r, (uint32_t)W, (uint32_t)r, (uint32_t)W, (uint32_t)r};
    for (size_t sz : sizes) {
      nrt_tensor_t *ti, *to;
      CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, r, sz, "i", &ti));
      CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, r, sz, "o", &to));
      std::vector<float> h(sz / 4); for (size_t i = 0; i < h.size(); i++) h[i] = (float)((i % 97) * (r + 1));
      CK(nrt_tensor_write(ti, h.data(), 0, sz));
      nrt_tensor_list_t il{&ti, 1}, ol{&to, 1};
      std::vector<double> l;
      int it = sz >= (1 << 20) ? 20 : 200;
      for (int k = 0; k < it; k++) {
        auto t0 = clk::now();
        nrt_cc_context_t* ctx;
        CK(nrta_cc_prepare(&comm, &il, &ol, NRT_DTYPE_FLOAT32, NRT_OP_ADD, NRT_CC_ALLREDUCE, &ctx));
        NRT_STATUS ret; nrta_seq_t seq;
        CK(nrta_cc_schedule(&ctx, 0, &ret, &seq));
        bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
        CK(ret);
        l.push_back(us_since(t0));
      }
      std::vector<float> o(sz / 4); CK(nrt_tensor_read(to, o.data(), 0, sz));
      double err = 0; float f = W * (W + 1) / 2.0f;
      for (size_t i = 0; i < o.size(); i++) err = std::max(err, (double)fabsf(o[i] - (float)(i % 97) * f));
      std::sort(l.begin(), l.end());
      if (r == 0) printf("W=%d size %8zu host-driven all_reduce: median %.1f us (min %.1f), algbw %.2f GB/s, max err %g\n",
                         W, sz, l[l.size() / 2], l[0], sz / l[l.size() / 2] / 1e3, err);
      nrt_tensor_free(&ti); nrt_tensor_free(&to);
    }
  });
  for (auto& t : th) t.join();
  nrt_close();
  return 0;
}
