# NKI lab findings (Trainium2, trn2.48xlarge)

Toolchain: Neuron runtime 2.34.10, neuronx-cc 2.27.5334, nki 0.6.0, logical NeuronCore config 2.
All runs on logical cores 56-59 (one experiment per core, `NEURON_RT_VISIBLE_CORES` set for every process), nothing else on those cores.
Every number below is measured unless it says *estimate*. "queued" = `nrta_execute_schedule` with hundreds
in flight, wall time / count, from `host/nrun` (C++ on libnrt); "device" = `neuron-explorer` `total_time`
(whole NEFF incl. its fixed pre/postamble) and `total_exec_time` (kernel body) of the 3rd execution.

## Summary: what an engine should do (all measured on logical cores 56-59, LNC=2 unless noted)

| | ours | reference (NxD Inference) |
|---|---|---|
| Qwen3-0.6B decode, one NEFF per step (28 layers + LM head + argmax), b512 / b4608 bucket | **2.10 / 2.92 ms/step** | 6.26 ms (256 pos), 5.10 ms (4,096) |
| Qwen3-0.6B prefill 256 / 1,024 / 4,096 tokens (whole model, first token included) | **7.3 / 19.8 / 76.3 ms** | 11 / 36 / 154 ms |
| Qwen3-8B decode, one 36-layer NEFF, b512 / b4608 | **23.1 / 25.1 ms/step** (fp8 weights: 13.8, changes the model) | - |
| Qwen3-8B prefill 256 / 1,024 / 4,096 (layered NEFFs) | **37.8 / 129.0 / 552.7 ms** | - |

1. One NEFF per decode step; position, token, KV write and argmax on the device; two tensor sets swapping
   `pos`/`output_0`; queue every step with `nrta_execute_schedule`, read the tokens once (`host/chain.cpp`,
   `host/generate.cpp`). Queued time = device time.
2. Column layout + weight-stationary GEMV; LNC=2 with per-core weight halves and `nisa.sendrecv` (gpsimd_dma)
   partial-sum exchange; static attention buckets chosen by the host.
3. Trigger the weight stream from an engine that does no compute: `dge_mode=hwdge, engine=sync`
   (2.23 -> 2.10 ms; prefill 88.8 -> 76.3 ms). At most ~8k accumulating matmuls per PSUM tile.
4. Prefill feature-major with nkilib `attention_cte` inlined; for long prompts / big models one layer NEFF executed
   per layer on `nrt_tensor_allocate_slice` slices; cross-core scratch as kernel inputs, not internal shared_hbm.
5. Verify on real token sequences against HF fp32 (teacher-forced over hundreds of positions + greedy), not on
   random caches; compare against the reference's own bf16 error.

Status: all four problems solved with measured numbers (sections 1-4). Build / test everything: `./build.sh`
(`./build.sh quick` for a 3-minute smoke test). Tooling:
- `kernels/nkibuild.py`: `build()` traces + compiles an NKI kernel to `neff/<name>.neff` without a device;
  `nrun()` runs a NEFF through `host/nrun` (C++/libnrt) and returns the outputs and updated inputs.
- `host/nrun.cpp`: generic runner: inputs from files, n executions with the same tensor sets (in-place
  updates advance), dumps every tensor, optional queued / sync benchmark.
- `tools/timeline.py`: per-engine / per-DMA-queue timeline from `neuron-explorer view --output-format json`.
- `tools/bench_pos.py`: queued us/exec of an attention NEFF at chosen positions.
- `host/chain.cpp`: a decode-step NEFF executed n times, all queued, two tensor sets that swap `pos`/`output_0`.
- `host/generate.cpp`: prefill NEFF (or emb/layer/head NEFFs bound to per-layer slices) + decode NEFF sharing
  all tensors by name, everything queued at once, one read of the token history at the end.

Pitfall for every test: a Python process that ran a kernel through nki's own runner (`kernel(*arrays)`,
nrtpy) keeps its NeuronCore until it exits; a C++ runner started from that process on the same core fails
with `Logical Neuron Core(s) not available ... (cores busy, ret=-16)`. Compile only in Python; run in C++.

---------------------------------------------------------------------------------------------------------

## 1. Decode attention, KV cache resident in HBM, position read on the device  — SOLVED

Kernel: `kernels/attn_decode.py` (static buckets `attn_b512/1024/2048/4608`, dynamic `attn_dyn*`).
Test: `python tests/test_attn.py <bucket|kernel> [positions...]` (numpy fp32 reference in
`kernels/ref_qwen3.py`).

What the kernel does (Qwen3-0.6B: 16 q heads, 8 kv heads, head_dim 128, TMAX 4,608, LNC=1 NEFF):
q/k RMSNorm per head (eps 1e-6) + RoPE (theta 1e6, rotate-half, fp32 cos/sin tables `[TMAX, 64]` row
picked at `pos`), writes the new K column and V row into the cache **at `pos` read from a device tensor**,
attention over `[0, pos)` from HBM plus the current token from SBUF, GQA softmax, output `[1, 2048]` bf16.

### The mechanism (works)

```python
p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.dma_copy(dst=p_sb, src=pos)                 # pos: [1,1] int32 kernel input in HBM
preg = nisa.register_alloc(); nisa.register_load(preg, p_sb)
# vcache [TMAX, 1024]: row pos
nisa.dma_copy(dst=vcache.ap(pattern=[[1024, 1], [1, 1024]], offset=0, scalar_offset=preg, indirect_dim=0), src=vrow)
# kcache [8, 128, TMAX] (K transposed): column pos of every (head, d) row
nisa.dma_copy(dst=kcache.ap(pattern=[[TMAX, 128], [128 * TMAX, 8]], offset=0, scalar_offset=preg, indirect_dim=2), src=kT_new)  # SBUF [128, 8]
# RoPE row at pos, broadcast to 24 partitions (partition step 0 on the HBM side)
nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[0, 24], [1, 64]], offset=0, scalar_offset=preg, indirect_dim=0))
```
`scalar_offset` is in units of the `indirect_dim` stride of the tensor `.ap()` is called on. The cache
tensors are ordinary kernel inputs; the writes change the caller's buffers in place (verified: the written
row/column equals the reference **bitwise**, every other element of both caches is unchanged).
Position increment on the device: `tensor_scalar(p1, p_sb, add, 1)` then `dma_copy(dst=pos, src=p1)` (in
place, `attn_*_inc` / probes/p2) or to a second buffer (ping-pong two tensor sets: race-free under LNC=2,
see pitfalls). 5 chained executions (sync and queued) advanced pos 37 -> 42 and wrote rows 37..41
(probes/p2_dynwrite_step.py, LNC 1 and 2).

Correctness (tests/test_attn.py, random qkv x2, norm weights 1 +- 0.3, cache garbage beyond pos = 50 / 1000
to prove masking): all buckets PASS at pos 0, 1, 127, 128, 129, 300, bucket-1; max abs error vs fp32
reference 0.3-0.4% of max|out| (within bf16 rounding of the output and of the bf16 probabilities); new K
column bitwise equal to the reference's bf16 k, V row exact.

### Design choices that matter

- **K stored transposed** `[kv_head, d, T]`: a 128-position tile is `[128 d partitions, 128 t]`, the
  stationary operand of the QK matmul with q^T `[128 d, 2 heads]` moving; scores land as `[128 t, 2]`.
  V natural `[T, kv_head*d]` is the stationary of `O^T[d, h] = sum_t V[t, d] P[t, h]` with P `[128 t, 2]`
  moving: **no transposes of P** and the output lands as `O^T [128 d, 16 heads]`, exactly the stationary
  layout an o_proj GEMV wants. Price: the new K is a scatter of 1,024 two-byte elements per layer.
- **Scores `[128 t, tile, 16 heads]`**: softmax reductions run over (tiles x 16) elements per partition
  (37 x 16 at 4,608) instead of 4,608; the per-head max goes partition-wise via TensorE transpose + reduce
  + a ones-matmul broadcast (`ones[16,128]^T @ diag(m)`), the denominators via `ones[128,128]^T @ rowsums`
  (every row of PSUM gets the column sums).
- **Current token from SBUF, cache only `[0, pos)`**: the kernel never reads what it writes in the same
  execution, so no HBM read-after-write ordering is needed; the read of the tile containing `pos` may race
  with the write of `pos`, which is masked anyway. Requires finite cache contents (zero-init once).
- Masking with an additive -30000 (not -inf) before the max; exp(scale*(s-m)) in fp32 on ScalarE, probs
  rounded to bf16 for the PV matmul, normalisation in fp32 after PV. rsqrt for the norm on GpSimd
  (`tensor_scalar(op0=nl.rsqrt, engine=gpsimd)`, more precise than the ScalarE LUT); `nl.power` is not a
  valid `tensor_scalar` op (`ISA validation failed: TensorScalarArithOp ... 'tensor_scalar_valid_ops'`).
- **Issue the cache loads first** in program order (they do not depend on q/k/v): b512 queued
  52.6 -> 34.1 us, b4608 87.8 -> 73.9 us. Each engine's instruction stream is in order; a DMA placed
  after the norm/RoPE chain is triggered only when that engine reaches it.

### Numbers (LNC=1 NEFF, one physical core, core 56)

| variant | positions read | device total / exec (us) | queued us/exec | HBM read |
|---|---|---|---|---|
| static `attn_b512` | 512 | 33.3 / 21.3 | 34.1 | 2.1 MB |
| static `attn_b1024` | 1,024 | 57.0 / 34.3 | - | 4.2 MB |
| static `attn_b2048` | 2,048 | 64.5 / 42.6 | - | 8.4 MB |
| static `attn_b4608` | 4,608 | 76.7 / 64.2 | 73.9 | 18.9 MB |
| static `attn_b4608_hw` (HWDGE, 2 queues) | 4,608 | 74.3 / 59.9 | 69.4 | 18.9 MB |
| static `attn_b4608_rr` (HWDGE+SWDGE, 3 queues) | 4,608 | - | 83.7 | 18.9 MB |
| dynamic `attn_dyn4` (fori_loop, 512/iter) | ceil(pos/512)*512 | - | 48.1 (pos 0), 69.9 (256), 93.1 (1,024), 226 (4,095) | as needed |
| dynamic `attn_dyn4_hw` | same | - | 48.4, 67.6, 86.5, 200 | as needed |
| dynamic `attn_dyn2_hw` (256/iter) | same | - | 48.7, 63.8, 110, 298 | as needed |

Before the reordering (loads after the norm chain): b512 52.6 us queued / 52.1 device total, 30.4 exec.

**Recommendation: static buckets.** One NEFF per bucket (the host knows the step count, so choosing the
NEFF needs no device read and no host write); at 4,608 positions the static kernel streams 18.9 MB in
60-64 us of execution (~300 GB/s on one physical core).

### Dynamic loops (`nl.fori_loop`): works, but slower — measured negative

`nl.fori_loop(0, reg, body)` with a register bound loaded from a device tensor runs on the device, and the
loop register can be a `scalar_offset`. What works inside the body and what does not (probes/p3_fori.py,
probes/p5_fori_variants.py):

| inside a fori_loop body | result |
|---|---|
| DMA with a dynamic offset on the **HBM** side, into a fixed SBUF tile | works |
| `tensor_copy` to / from an SBUF tensor at a dynamic offset (`.ap(scalar_offset=i)`) | works |
| SBUF accumulation across iterations (`tensor_tensor(acc, acc, psum, add)`) | works |
| DMA into an SBUF tensor at a dynamic offset | compiles; `nrt_load` fails: `TDRV:calculate_max_reg_offset OOB access detected in NEFF ... Access size: 33292544, Variable size: 262144` |
| matmul operand at a dynamic SBUF offset | compile error `[NCC_IBIR040] Matmult stationary input tile size must be <= 128x128 but it was = 128x1024` (the whole dim is assumed) |
| PSUM tile accumulated across iterations (declared outside the loop) | compile error `[NCC_IBIR092] Live-in/Live-out MemoryLocation ... not allocated to MemoryType: DRAM` |
| Python dict literal in kernel code (`{1: 0}[CH]`) | parser error `expected string` |

The dynamic kernel (loops over 512-position chunks: QK loop then PV loop, fixed tiles inside) is correct
at every tested position but costs about 10 us per chunk per loop: DMA and compute of one iteration do
not overlap the next. At 4,095 positions it is 3x slower than the static 4,608 bucket.

### Pitfalls

- Under LNC=2 both programs execute the in-place `pos += 1`; it gave the right answer in 5-step tests but
  is a read-modify-write race between the two cores. Write pos+1 to a different tensor (ping-pong sets).
- Default DMAs here are software DGE on GpSimd (`DMA_DIRECT2D` ~0.65 us of GpSimd time each);
  `dge_mode=hwdge` with `engine=sync/scalar` moves them to hardware DGE (small gain at 4,608).
- The kernel-level fixed cost: ~10 us before the first DMA of a small NEFF (semaphore barriers, activation
  table load 1.3 us) and ~10 us after the last one: total_time - total_exec_time = 12-23 us.

---------------------------------------------------------------------------------------------------------

## 2. A whole decode step in one NEFF  — SOLVED (all 28 layers + LM head + device argmax, 2.10 ms/step)

Kernels: `kernels/step_lnc2.py` (LNC=2, the fast one), `kernels/step_decode.py` (LNC=1), both built from the
layer code of `kernels/layer_decode.py`. C++ driver: `host/chain.cpp`. Test:
`LNC=2 python tests/test_step.py step2_b512 40 28` (real Qwen3-0.6B weights, real prompt, compared with HF
transformers on the CPU in fp32 and bf16; `PROMPT_FILE=... PROMPT_TOKENS=4000` for long prompts).

### What one execution does (no host work between steps)

`pos` (device [1,1] int32) -> input token = `prompt[pos]` if `pos < plen` else `toks[pos]` (selected with
fp32 arithmetic on two dynamically indexed loads) -> embedding row gathered with the token in a register
(`embed.ap(pattern=[[1,128],[128,8]], scalar_offset=treg, indirect_dim=0)` lands it directly in column
layout) -> 28 x {RMSNorm, fused QKV, q/k norm, RoPE, K/V write at `pos`, attention over the static bucket,
o_proj, residual, RMSNorm, gate/up, SiLU*up, down, residual} -> final RMSNorm -> LM head (151,936 x 1,024) ->
argmax with ties to the lowest id -> `toks[pos+1] = argmax` (in-place dynamic write) and `output_0 = pos+1`.
The host builds two tensor sets that swap `pos` and `output_0` (set A reads X writes Y, set B the reverse),
queues all steps of a request with `nrta_execute_schedule` alternating A/B, waits once, reads `toks` once. A
request (prompt fed token by token, then generation) is one chain of identical executions: the host writes
the prompt once and reads the tokens once.

### Numbers (measured; queued = `host/chain`, wall time / steps, 3 repetitions of 51 steps; device =
`neuron-explorer` total_time of the 3rd execution; logical core 56)

| NEFF | compile | NEFF size | queued ms/step (end to end from C++) | device total | HBM read | MBU (profiler) |
|---|---|---|---|---|---|---|
| LNC=1 `step_b512` (28 layers + LM head) | 27.5 s | 1.29 MB | 3.46 | 3.466 ms | 1.251 GB | 50% |
| LNC=2 `step2_b512` | 25.9-33.3 s | 1.48 MB | 2.23 | 2.42 ms (profiling on) | 1.252 GB | 72% |
| **LNC=2 `h06b_b512_s4`** (weight DMAs on hwdge/sync, `kernels/step_g.py`) | 26.2 s | 1.48 MB | **2.100** | 2.204 ms | 1.252 GB | 79% |
| **LNC=2 `h06b_b1024_s4`** | 28.2 s | 1.56 MB | **2.231** | - | - | - |
| **LNC=2 `h06b_b2048_s4`** | 30.8 s | 1.70 MB | **2.390** | - | - | - |
| **LNC=2 `h06b_b4608_s4`** (4,000-token prompt teacher-forced: 3972/3999 vs fp32, 32/32 generated) | 44.7 s | 2.04 MB | **2.921** | - | - | - |
| LNC=2 `step2_b1024` | 28.8 s | 1.54 MB | 2.37 | - | - | - |
| LNC=2 `step2_b2048` | 31.3 s | 1.69 MB | 2.62 | - | - | - |
| LNC=2 `step2_b4608` | 41.5 s | 2.01 MB | 3.15 | - | ~1.78 GB (estimate: +0.53 GB of cache) | - |
| reference (NxD Inference) | | | 6.26 at 256 positions, 5.10 at 4,096 (whole request) | 6.42 ms | 1.50 GB | 38% |

(all four LNC=2 rows: 11-token prompt + 40 generated tokens, identical to HF fp32 and bf16; `build/bucket_bench.txt`.)
Static buckets read the whole bucket of cache every step, so their cost does not depend on the position:
each extra 512 positions of bucket costs about 59 MB (28 layers x K+V) and measured 0.13 ms (b512 -> b4608: +0.91 ms for +4,096 positions, ~520 GB/s marginal). An engine should use buckets every 512 positions (or 1,024) and pick the NEFF from the host-known step count. At 2.23 ms the b512 step
moves 1.25 GB at 561 GB/s, at the measured streaming ceiling of an LNC=2 core (p7: 535 GB/s for a pure
134 MB HBM->SBUF stream). The LNC=1 NEFF is DMA-bound on one physical core (360 GB/s).

Correctness (`tests/test_step.py`, HF transformers 5.15 / torch 2.11 on the CPU as reference):
- 11-token prompt + 40 generated, b512, LNC 1 and 2: teacher-forced argmax 11/11 and all 40 generated tokens
  identical to HF fp32 **and** to HF bf16.
- 983-token prompt + 32 generated, b1024: teacher-forced 978/983 vs fp32 (the 5 mismatches are near-ties:
  fp32 logit gap 0.014 median, 0.047 max), 964/983 vs HF bf16; generated 32/32 identical to both.
- 4,000-token prompt + 32 generated, b4608 (4,031 chained steps): teacher-forced 3972/3999 vs fp32,
  3904/3999 vs HF bf16, generated 32/32 identical to both. The kernel is closer to the fp32 model than HF's
  own bf16 run (residual stream kept in fp32 in SBUF, matmuls accumulate in fp32 PSUM).

### LNC=2: how the two physical cores split a layer

Both cores run the same program (`nl.program_id(0)` = 0/1 is usable as an index: `w[l, pid]`, `1 - pid`,
`pid * const` as a `tensor_scalar` operand all trace). Weights are re-laid out per core on the host so
each core streams one contiguous half:
- QKV split by heads (q heads 8c..8c+7, kv heads 4c..4c+3): attention runs locally on 8 q / 4 kv heads,
  reading and writing only that core's half of the cache (`kcache [L, 2, 4, 128, T]`, `vcache [L, 2, T, 512]`).
- o_proj split along K (its heads) and down split along K (its 12 intermediate tiles; gate/up split along N
  over the same tiles): each produces a partial `[128, 8]` fp32 hidden vector, summed with one
  `nisa.sendrecv(src, dst, send_to_rank=1-pid, recv_from_rank=1-pid, pipe_id=0,
  dma_engine=nisa.dma_engine.gpsimd_dma)` + add. Two exchanges per layer.
- LM head split along the vocab (594 tiles each, vocab padded to 1,188 x 128 with zero rows masked to -3e38),
  local argmax, one exchange of (max, lowest-id score), combine.
- Embedding, norms and residual adds run redundantly on both cores; both write the same bytes to `toks` and
  `output_0` (no read-modify-write, so no race).

Exchange cost (probes/p8_sendrecv.py, 64 dependent exchange+add of [128,8] fp32 minus 1, LNC=2): 0.85 us
each with `dma_engine.gpsimd_dma`, 2.0 us with the default `dma_engine.dma`. 57 exchanges per step ~ 50 us.

### Argmax with ties to the lowest id (exact)
Logits in column layout `lg[p, c]` (id = c*128 + p, fp32 SBUF). Per partition: `m_p = max_c`; score =
`(lg == m_p) * (2^20 - id)`, `max` -> the lowest id reaching m_p (ids < 2^20 are exact in fp32). Across
partitions: PE-transpose both columns to rows on partition 0, `M = max`, then `max((row == M) * score)`.
Equality is only ever taken between values that went through the same path, so it is exact.

### GEMV layout used everywhere ("column layout", weight-stationary)
A vector of K lives as `[128, K/128]` (v[kt*128 + p] at [p, kt]). Weight tile (nt, kt) of W^T `[128 k, 128 n]`
is the stationary operand, x[:, kt] moving: `y[:, nt]` lands in PSUM already in column layout, so no
transposes anywhere between layers. Host layout `ws_layout(W)`: `out[p, (nt*KT+kt)*128 + j] = W[nt*128+j,
kt*128+p]` (per partition, the tiles of consecutive nt are contiguous: 4 n-tiles x 8 k-tiles = 8 KB per
partition per DMA, 1 MB per DMA). TensorE rate for this shape: one 128x128 tile per ~30 ns
(probes/p6_pe_rate.py: 1,024 tiles in 30.4 us of TensorE time); x-stationary with 512-wide moving weights
was 2x slower (60.2 us TensorE for the same 2 MB). This contradicts the backend counterexample "weights as
the stationary operand: slower" for the [1, K] x-moving formulation used there; with x as a single moving
*column* the weight-stationary form is the faster one.

### Pitfalls
- NKI re-reads the kernel's Python source (by file and line) while tracing: editing a kernel file (or a helper
  module it imports) while a build of it runs in the background breaks that build with nonsense errors such as
  `_consts() missing value for required arguments 'h', 'w_col', ...` (the line numbers now point into another
  function) and could silently trace other code. Do not edit sources while their builds run.
- ScalarE reloads its activation table (`ACT_TABLE_LOAD`, 1.28 us each) whenever the activation function changes;
  exp (attention) and SiLU (MLP) in every layer cost ~2.6 us per layer (~70 us per 0.6B step, not optimised).
- The 28-layer stack tested with **random** KV caches failed (rel err 0.14, cos 0.989), and so did the
  bf16-rounding numpy reference against the fp32 one (cos 0.85): random caches make attention chaotic over
  28 layers. Test whole-model kernels on real token sequences against a real reference.
- `nrt_load` of the 2 MB step NEFF: 186-255 ms; upload of the 1.4 GB of weights from host: 7-8 s
  (~190 MB/s through `nrt_tensor_write` of whole files: one call per tensor, not optimised).

---------------------------------------------------------------------------------------------------------

## 3. Prefill: whole-model prefill NEFFs, flash attention + TensorE GEMMs — SOLVED (2-2.4x faster than the reference)

Kernel: `kernels/prefill_lnc2.py` (LNC=2). Tests: `tests/test_prefill.py <S> [plen]` (first token vs HF fp32,
K/V cache of every layer vs HF `past_key_values`, against HF bf16's own error on the same metric) and
`tests/test_generate.py <S> <decode bucket> <plen> <n>` (prefill + decode end to end from C++, `host/generate`,
tokens vs HF fp32 greedy; `LAYERED=1` for the per-layer NEFF variant).

### Measured (logical core 57, LNC=2, queued from C++ `host/generate`, whole model incl. embedding + LM head + argmax)

| prompt bucket S | ours | reference (NxD Inference) | model FLOP (GEMM + causal attn) | TFLOP/s | build |
|---|---|---|---|---|---|
| 256 | **7.51 ms** (one NEFF: 7.30 ms) | 11 ms | 233 GFLOP | 31 | layer NEFF 2.8 s; one-NEFF 28 layers 21 s (3.0 MB) |
| 1,024 | **19.8 ms** (one NEFF, before the phase-A fixes: 21.8 ms) | 36 ms | 1,022 GFLOP | 52 | layer NEFF 6 s; one-NEFF 162 s, 8.2 MB |
| 4,096 | **76.3 ms** | 154 ms | 5,532 GFLOP | 73 | layer NEFF 37 s, 1.8 MB; one-NEFF not built (Python tracing > 10 min) |

FLOP count: 2 x S x 28 x 15.73 M weights + 28 x 4,096 x S^2 (16 heads, d 128, causal half of QK and PV).
Correctness: first token = HF fp32 argmax at S = 256 / 1,024; K/V of all 28 layers vs HF fp32 within 0.5%
(S=256) and at most 14.8% at single positions of layer 13 for a 1,021-token prompt, where HF's own bf16 run is
40% off on the same metric; prefill + 31 decode steps reproduce HF fp32 greedy (S=4,096 / 4,093-token prompt:
32/32 tokens; before the phase-A changes it diverged at token 8 on a near-tie, fp32 logit gap 0.005).

### Structure (what to copy)
- **Feature-major activations everywhere** (`h^T [128, 8, S]`, the decode "column layout" with a token axis):
  GEMM = weight tile `[128 k, 128 n]` stationary, activation chunk `[128 k, 512 tokens]` moving, PSUM
  `[128 n, 512]`. No activation transposes between GEMMs; q^T / k^T come out exactly in nkilib's `tp_q=False`
  / `tp_k=False` layouts and in the decode K^T cache layout; V needs one PE transpose per 128x128 block
  (matmul against the identity, exact).
- Per layer, 5 phases over 512-token chunks with the core's weights for the phase resident in SBUF
  (QKV 32 KB, o_proj 16 KB, gate/up 48 KB, down 24 KB per partition), `nisa.core_barrier` between phases:
  A norm1 + QKV + q/k norm + RoPE + K/V cache writes; B attention; C o_proj (N-split) + residual;
  D norm2 + gate/up + SiLU*up; E down (N-split) + residual. The prefill uses N-split copies of o_proj/down
  (`wo_p`, `wd_p`, +56 / +168 MB) and shares every other tensor (weights, caches, token history) with the
  decode NEFF.
- **Attention = nkilib `core/attention/attention_cte.py` `_attention_cte`** (Apache-2.0, in the neuronx-cc
  2.27 wheel, public NKI API only) inlined in our kernel: `q=qT [16, 128, S], k=kT [8, 128, S], v [8, S, 128],
  tp_q=False, tp_k=False, tp_out=True, out_o_hbm=oT`; it splits the 16 heads over the two cores itself (8 each,
  matching our QKV split). Standalone at LNC=2: 73 us (S=256, 3.7 TFLOP/s), 250 us (1,024, 17 TFLOP/s),
  1,384 us (4,096, 49.6 TFLOP/s), max error 0.008 vs fp32 numpy (`prefill/p_nkilib_attn.py`).
- **Layered prefill for long prompts**: 3 NEFFs (`pf_emb`, `pf_layer`, `pf_head`), the layer NEFF executed 28
  times with every per-layer tensor bound to a slice of the stacked tensor (`nrt_tensor_allocate_slice(src,
  l * size, size, ...)`), all 30 executions queued. Costs 0.1-0.2 ms over one NEFF at S=256 / 1,024 and
  makes the S=4,096 build 37 s instead of an unrolled 28-layer trace that did not finish in 10 minutes.
- GEMM rate (`prefill/p_gemm.py`, S=4,096, LNC=2, queued incl. ~20 us fixed cost): 130.5 TFLOP/s (1024 x 4096),
  129.7 (1024 x 6144), 103.5 (3072 x 1024), 95.8 (2048 x 1024).

### What mattered (measured on the S=4,096 layer NEFF, neuron-explorer, 100 us buckets)
Profile before: 3.11 ms per layer, phase A 0.95 ms with VectorE ~83% busy (8 fp32 elementwise passes per head
per chunk), attention 1.2 ms, C+D+E 0.8 ms.
1. Fewer VectorE passes (squares on ScalarE straight from PSUM via `activation(op=nl.square)`, sums of squares
   as bf16 x ones matmuls into fp32 PSUM, norm scale x weight fused in one `scalar_tensor_tensor`): **no gain**
   (88.8 ms), the bottleneck moved to GpSimd (~95% busy).
2. GpSimd was busy issuing the ~48 small DMAs per chunk: default `dma_copy` is a software DGE driven by GpSimd.
   Moving the activation DMAs to hardware DGE (`dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync`):
   **88.8 -> 76.3 ms** at S=4,096, 22.3 -> 19.8 ms at S=1,024.

### Pitfalls
- **Internal `nl.shared_hbm` scratch tensors were not shared between the two cores** in the prefill kernel:
  each core read back only its own writes (zeros elsewhere), deterministic, with `core_barrier` in place; the
  same pattern in small probes (`probes/p10_barrier.py`) did work. Passing every cross-core scratch tensor as a
  kernel input (caller-allocated device buffers) fixed it. Do the same for any LNC=2 exchange through HBM.
- Wrapping a `@nki.jit` library kernel in another `@nki.jit` through a Python conditional traced to an empty
  kernel (`NCC_INKI003 Insufficient number of outputs param ... got 0 outputs`); call the undecorated
  `_attention_cte` instead, or pass kernel kwargs through `build(..., kw=...)` (`kernel[2]` + `_bind_args`).
- nki 0.6 `nl.affine_range` / `nl.sequential_range` are aliases of `range` (fully unrolled); only
  `nl.dynamic_range` / `fori_loop` are device loops. Python unrolling of 28 layers x 8 chunks x nkilib's
  attention makes a 190 MB MLIR at S=1,024 (162 s build) -> use the layered prefill above.

---------------------------------------------------------------------------------------------------------

## 4. Qwen3-8B shapes — SOLVED: decode in one 36-layer NEFF (23.1 ms/step bf16, 13.8 ms fp8), layered prefill (S=256: 37.8 ms)

Kernel: `kernels/step_g.py` (the LNC=2 decode step with the model dimensions as parameters; `g06b_*` reproduces
the 0.6B kernel bit for bit in behaviour, 2.232 ms/step, 40/40 tokens). Tests: `tests/bench_8b.py <kernel> <L>`
(compile time, NEFF size, queued ms/step with zero weights), `tests/test_step_g.py g8b_b512_s1 24 36` (real
Qwen3-8B weights vs HF transformers fp32 on the CPU).

| NEFF (LNC=2, b512 attention bucket) | compile | NEFF | nrt_load | queued ms/step | streamed per step | GB/s |
|---|---|---|---|---|---|---|
| 1 layer + LM head | 31 s | 1.49 MB | 0.2 s | 2.524 | 1.63 GB | 647 |
| 4 layers + LM head | 53.5 s | 2.48 MB | 0.4 s | 4.357 | 2.80 GB | 642 |
| 36 layers + LM head + argmax | 325 s | 13.8 MB | 2.3-2.5 s | 23.86 | 15.21 GB | 637 |
| **same, weight DMAs on hwdge/sync (`h8b_b512_s1`)** | 319 s | 13.8 MB | 2.4 s | **23.11** (real weights: 197/199 teacher-forced, 16/16 generated; after the 8B prefill 24/24) | 15.21 GB | **658** |
| 36 layers, b4608 attention bucket (`g8b_b4608_s1`) | 328 s | 14.6 MB | 2.9 s | 25.11 | 15.8 GB | - |

- Per extra layer: 0.611 ms for 386 MB of weights (632 GB/s). Real weights (36 layers, 17.1 GB of inputs
  incl. caches, upload 55 s through files): 23.86 ms/step, prompt teacher-forced 11/11 and 24/24 generated
  tokens identical to HF fp32 Qwen3-8B ("Berlin. The capital of Italy is Rome. ... Lisbon.").
- Bandwidth floor at the best measured GEMV rate (661-674 GB/s, below): 15.2 GB / 0.67 TB/s = 22.7 ms; the
  step is within 5% of it. HBM use: 16.4 GB weights + 0.7 GB caches (T=4,608) + 0.6 GB LM-head halves in
  24 GiB.
- The step-NEFF question: the compiler accepts all 36 layers in one NEFF (instruction count ~ 212k matmuls
  per core); compile time grows ~8 s per layer. A per-layer NEFF (as in the layered prefill) is the fallback
  if a larger model does not compile; it would cost ~36 x 20 us.

### GEMV bandwidth at 8B shapes: the PSUM accumulation-depth cliff
`probes/p11_gemv_layout.py` (LNC=2, weight-stationary GEMV, queued, 1-column moving x):

| output tiles per core x K tiles, PSUM tile width | bytes | GB/s |
|---|---|---|
| 96 x 32 (8B gate/up per core), 512-wide PSUM tile | 201 MB | 589 |
| 192 x 32, 512 | 403 MB | 600 |
| **512 x 32, 512** (one PSUM tile receives 16,384 accumulating matmuls) | 1,074 MB | **176** |
| 592 x 32, 512 (8B LM head per core) | 1,242 MB | 292 |
| 592 x 32, **256** or 128 | 1,242 MB | **661** |
| 2368 x 8 (K = 1,024), 512 | 1,242 MB | 674 |

Rule: keep at most ~8k accumulating matmuls per PSUM tile (512 x 32 = 16k was 3.7x slower; 256 x 32 = 8k and
512 x 8 = 4k are fine). In the first 8B step build this made the LM head take 4.25 ms instead of 1.9
(1 layer + head: 4.89 -> 2.52 ms). The HBM layout of the weight did not matter (row layout with a 4.8 MB
partition stride vs fully contiguous 1 MB blocks: 292 vs 292 GB/s).

### fp8 weights (weight-only, per-output-channel scale): 1.7x faster, but it changes the model
`probes/p12_fp8_gemv.py`, `kernels/step_g.py` `q06b_*` / `q8b_*`, `Q8=1 python tests/test_step_g.py ...`:
weights `float8_e4m3` (legacy Trn2 e4m3, max 240) as the stationary operand with a bf16 moving x (allowed mixed),
fp32 PSUM, one fp32 scale per output channel and core (`s = absmax / 240`) multiplied after each GEMV.

| | bf16 | fp8 |
|---|---|---|
| 8B LM-head-shaped GEMV, 1.24 GB bf16 (LNC=2) | 1,879 us (661 GB/s) | 996 us (623 GB/s of fp8 bytes); arithmetic vs dequantised ref 6e-7; vs bf16 weights 2.65% rms |
| Qwen3-0.6B decode step b512 | 2.228 ms | **1.353 ms** |
| 0.6B teacher-forced argmax vs HF fp32 (479-token prompt) | 478/479 | 447/479 (mismatch fp32 logit gap median 0.12, max 0.66) |
| Qwen3-8B decode step b512, 36 layers (one NEFF: 329.5 s compile, 14.2 MB) | 23.86 ms | **13.84 ms** |
| 8B teacher-forced argmax vs HF fp32 (199 positions) | 197/199 (near-ties, gap <= 0.012) | 184/199 (gap median 0.197, max 0.307) |
| first 16-24 generated tokens vs HF fp32 | identical | identical |

So per-channel e4m3 costs ~6.5 points of top-1 agreement for both model sizes: a model change, not rounding.
Use it only where the correctness gates allow; measure on long teacher-forced text (short greedy checks hide it).

### 8B prefill: layered prefill works (`kernels/prefill_g.py`)
`python tests/test_prefill_g.py 8b <S> <decode NEFF> <plen> <n>`: emb NEFF + 36 x one-layer NEFF (per-layer tensor
slices) + head NEFF, then the 36-layer decode NEFF, all queued from C++ (`host/generate`); greedy tokens vs HF fp32
Qwen3-8B on the CPU. Design: token blocks of 512 (256 for gate/up + down) with each weight slab streamed once per
block (one core's QKV weights for one 8B layer, 196 KB per partition, do not fit in SBUF); o_proj and down K-split
with the decode weight layouts, partial h per core through a caller-provided `s_p` scratch + `core_barrier`, so
no N-split weight copies (+4.8 GB for 8B, which would not fit next to 16.4 GB of weights + caches in 24 GiB).

| prompt bucket | prefill (all executions queued) | build of the layer NEFF | check |
|---|---|---|---|
| S = 256 (12-token prompt) | **37.8 ms** (38 executions) | 9.2 s, 0.61 MB (head NEFF 28.5 s) | prefill + 23 decode steps: 24/24 tokens = HF fp32 |
| S = 1,024 (1,021-token prompt) | **129.0 ms** (132.4 before slab DMAs on hwdge/sync); 14.5 TFLOP -> 112 TFLOP/s | 31 s, 1.9 MB | prefill + 15 decode steps (b4608 decode NEFF): 16/16 = HF fp32 |
| S = 4,096 (4,093-token prompt) | **552.7 ms** (563.8 before); 61.8 TFLOP -> 112 TFLOP/s | 159 s, 8.5 MB | 16/16 = HF fp32 |
| (0.6B, same kernel: S = 256 / 1,024 / 4,096) | 7.35 / 21.6 / 83.0 ms (87.4 before) | 4.2 / 7.7 / 39 s | 32/32, 16/16, 16/16 tokens = HF fp32 |

The estimate made from the component rates before building it (below: ~125 ms at S = 1,024, ~520 ms at 4,096)
was within 6%.

### 8B prefill components
- GEMMs (`MODEL8B=1 python prefill/p_gemm.py 1024`, LNC=2, queued, S = 1,024): QKV 4096 -> 6144 136.6 TFLOP/s,
  gate/up (one half) 4096 -> 12288 145.6, o_proj 4096 -> 4096 129.4. down 12288 -> 4096 does not compile in this
  probe layout (activations resident): `[NCC_IBIR229] State buffer allocation failed` (x^T [128, 96, 1024] bf16 =
  192 KB per partition).
- Attention (nkilib `attention_cte`, 32 q / 8 kv heads, `HQ=32 python prefill/p_nkilib_attn.py`): S = 1,024
  479 us (17.9 TFLOP/s), S = 4,096 2,732 us (50.3 TFLOP/s), max error 0.008-0.009.
- Why the 0.6B prefill kernel does not carry over: one core's QKV weights for one 8B layer are 196 KB per
  partition, more than SBUF, so phases must stream weight slabs per token chunk (per core and layer at
  S = 4,096 with 1,024-token chunks: ~0.8 GB of weight reads, ~2.6 ms of DMA against ~13 ms of TensorE work).
- *Estimate* made before the kernel existed (measured since: 129.0 / 552.7 ms): S = 1,024: 14.2 TFLOP of GEMMs at ~135
  TFLOP/s = 105 ms + 36 x 0.48 ms attention = ~125 ms; S = 4,096: 56.9 TFLOP / 135 = 421 ms + 36 x 2.73 ms =
  ~520 ms.

### Also measured
- Decode weight DMAs on hardware DGE (`step2_b512_hw`, sync/scalar queues alternating): 2.378 vs 2.235 ms/step
  with the default software DGE -> keep big weight streams on the default DMA path.
- Where the 0.6B step loses against the GEMV ceiling (neuron-explorer, LNC=2, 100 us buckets): the 28 layers
  stream at 500-540 GB/s, the LM head at 610-665 GB/s; TensorE is not saturated in either. The ~15% per-layer
  loss is serial latency at each layer (norms, exchanges, attention between GEMVs) with a limited weight
  prefetch depth; a layer's weights per core (123 KB per partition for 0.6B) do not fit twice in SBUF, so plain
  double buffering of whole layers is not an option. Not optimised further (2.235 ms vs ~1.9 ms floor).
  Tried (all LNC=2, 40/40 tokens still identical): every weight DMA of a layer issued at the top of the layer into
  per-GEMV resident SBUF regions (`p06b_b512_s4`): 2.279 ms (slower); the K/V cache writes (which depend on the
  layer's k/v) moved off the weight queue onto a hardware-DGE queue (`c06b_b512_s4`): 2.230 ms (no change). The
  dips in the DMA rate (5 us buckets: ~20 us per ~65 us layer at 200-400 GB/s) coincide with the serial
  norm/attention/exchange segment; the per-layer time is about the length of the dependency chain (QKV -> q/k
  norm + RoPE -> attention -> o_proj -> exchange -> norm -> gate/up -> down -> exchange), i.e. the 0.6B step is
  latency-bound per layer, not bandwidth-bound. Shortening that chain is the remaining lever.
  Attention with transposed scores `[heads, t]` (`t06b_b512_s4`: the per-head max is one free-axis reduce, exp +
  row sums one ScalarE `activation(..., reduce_op=nl.add, reduce_res=den)`, ~10 fewer dependent hops): 2.246 ms
  (no gain) -> attention hops were not the critical path either.
- **The real cause: weight-DMA triggers stuck behind GpSimd compute.** Instruction-level timeline of one core
  (`sg01`) over one layer: the PE runs in 1-2.7 us bursts (one 1 MB slab each) with 1-3 us gaps; the core's DMA
  rate drops to ~200 GB/s during attention and to 0 for ~2.5 us. Default `dma_copy` is software DGE: its
  descriptors are generated by `DMA_DIRECT2D` instructions in GpSimd's in-order stream, interleaved with GpSimd
  compute (rsqrt, iota, tensor_scalar) that waits on upstream results; every later weight DMA waits behind them.
  **All weight / cache DMAs on hardware DGE triggered from the Sync engine only** (`_dma(..., hw=2)`,
  `dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync`; the Sync engine runs no compute):
  `h06b_b512_s4` **2.230 -> 2.100 ms/step** (40/40 tokens identical). The earlier hardware-DGE try alternated
  the Sync and Scalar (ACT) engines; ACT also runs compute in order, and that variant was slower (2.378 ms).
