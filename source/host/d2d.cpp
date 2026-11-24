// d2d: communication between logical NeuronCores WITHOUT collectives inside a NEFF, natively on libnrt.
//   NEURON_RT_VISIBLE_CORES=56,57 d2d <copy.neff> <vnc_a> <vnc_b>
// Measures, for 4 KB / 64 KB / 1 MB / 16 MB:
//  1. nrt_tensor_copy from a tensor on vnc a to one on vnc b (documented: same core only)
//  2. host bounce: nrt_tensor_read (vnc a -> host buffer) + nrt_tensor_write (host -> vnc b)
//  3. a NEFF loaded on vnc a (copy kernel: output_0 = x) whose OUTPUT tensor is allocated on vnc b, and one whose
//     INPUT tensor is allocated on vnc b: does a kernel on one logical core read / write another core's HBM?
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 4) { fprintf(stderr, "usage: d2d copy.neff vnc_a vnc_b\n"); return 2; }
  int a = atoi(argv[2]), b = atoi(argv[3]);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  size_t sizes[] = {4096, 65536, 1 << 20, 16 << 20};
  for (size_t sz : sizes) {
    nrt_tensor_t *ta, *tb;
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, a, sz, "ta", &ta));
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, b, sz, "tb", &tb));
    std::vector<char> h(sz), h2(sz);
    for (size_t i = 0; i < sz; i++) h[i] = (char)(i * 7 + 3);
    CK(nrt_tensor_write(ta, h.data(), 0, sz));
    NRT_STATUS s = nrt_tensor_copy(ta, 0, tb, 0, sz);
    printf("size %8zu  nrt_tensor_copy vnc %d -> %d: %s\n", sz, a, b, s == NRT_SUCCESS ? "OK" : nrt_get_status_as_str(s));
    int it = sz >= (1 << 20) ? 20 : 200;
    std::vector<double> l;
    for (int i = 0; i < it; i++) {
      auto t0 = clk::now();
      CK(nrt_tensor_read(ta, h2.data(), 0, sz));
      CK(nrt_tensor_write(tb, h2.data(), 0, sz));
      l.push_back(us_since(t0));
    }
    std::vector<char> h3(sz); CK(nrt_tensor_read(tb, h3.data(), 0, sz));
    printf("size %8zu  host bounce read+write: median %.1f us (%.2f GB/s), data %s\n", sz, med(l), sz / med(l) / 1e3,
           memcmp(h.data(), h3.data(), sz) ? "WRONG" : "ok");
    nrt_tensor_free(&ta); nrt_tensor_free(&tb);
  }
  // 3. cross-core tensors in a NEFF's tensor set
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<char> neff((std::istreambuf_iterator<char>(f)), {});
  nrt_model_t* m; CK(nrt_load(neff.data(), neff.size(), a, -1, &m));
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m, &info));
  size_t sz = info->tensor_array[0].size;
  for (int mode = 0; mode < 2; mode++) {
    nrt_tensor_set_t *ins, *outs; CK(nrt_allocate_tensor_set(&ins)); CK(nrt_allocate_tensor_set(&outs));
    nrt_tensor_t *tx, *ty;
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, mode == 1 ? b : a, sz, "x", &tx));
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, mode == 0 ? b : a, sz, "y", &ty));
    std::vector<char> h(sz), h3(sz, 0);
    for (size_t i = 0; i < sz; i++) h[i] = (char)(i * 13 + 1);
    CK(nrt_tensor_write(tx, h.data(), 0, sz)); CK(nrt_tensor_write(ty, h3.data(), 0, sz));
    for (uint64_t i = 0; i < info->tensor_count; i++) {
      auto& ti = info->tensor_array[i];
      CK(nrt_add_tensor_to_tensor_set(ti.usage == NRT_TENSOR_USAGE_INPUT ? ins : outs, ti.name,
                                      ti.usage == NRT_TENSOR_USAGE_INPUT ? tx : ty));
    }
    NRT_STATUS s = nrt_execute(m, ins, outs);
    std::string res = s == NRT_SUCCESS ? "exec OK" : std::string("exec ") + nrt_get_status_as_str(s);
    if (s == NRT_SUCCESS) {
      CK(nrt_tensor_read(ty, h3.data(), 0, sz));
      res += memcmp(h.data(), h3.data(), sz) ? ", output WRONG" : ", output correct";
      std::vector<double> l;
      for (int i = 0; i < 200; i++) { auto t0 = clk::now(); CK(nrt_execute(m, ins, outs)); l.push_back(us_since(t0)); }
      char buf[128]; snprintf(buf, sizeof buf, ", nrt_execute median %.1f us", med(l)); res += buf;
    }
    printf("NEFF on vnc %d, %s tensor on vnc %d (%zu bytes): %s\n", a, mode == 0 ? "OUTPUT" : "INPUT", b, sz, res.c_str());
  }
  nrt_close();
  return 0;
}
