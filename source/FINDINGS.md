# NKI lab 2 findings: tensor parallelism across logical NeuronCores (Trainium2, trn2.48xlarge)

Toolchain: Neuron runtime 2.34.10, aws-neuronx-collectives 2.34, neuronx-cc 2.27.5334, nki 0.6.0, logical NeuronCore
config 2 (LNC=2). Lab cores: logical NeuronCores 56-63 only = devices 14 (cores 56-59) and 15 (60-63), which are
direct NeuronLink neighbours (`neuron-ls`). Every process sets `NEURON_RT_VISIBLE_CORES`. Every number is measured
unless it says *estimate*; "queued" = `nrta_execute_schedule` with many executions in flight, wall time / count, from
the C++ drivers in `host/`.

## Summary

| Qwen3 decode step (b512 attention bucket, bf16 weights, one NEFF per core per step, queued, steady state) | ms/step | vs 1 core | check vs HF fp32 (CPU) |
|---|---|---|---|
| 8B, 1 logical core (lab 1, `h8b_b512_s1`) | 23.1 | 1.0x | lab 1 |
| **8B, TP=2** (cores 56-57) | **12.46** | 1.85x | 11/11 teacher-forced + 24/24 generated; 470-token prompt 467/469 (near-ties) + 16/16 |
| **8B, TP=4** (cores 60-63) | **7.10** | 3.25x | 11/11 + 24/24; 470-token prompt 468/469 + 16/16 |
| **8B, TP=8** (two devices, 56-63) | **4.70** | 4.9x | 11/11 + 24/24 |
| **14B, TP=2 / TP=4** (29.5 GB of weights: does not fit one core) | **22.41 / 12.17** | - | 11/11 + 24/24 |
| **32B, TP=4** (62 GB) | **26.06** | - | 11/11 + 24/24 |
| **32B, TP=8** (two devices, cores 56-63) | **15.37** | - | 11/11 + 24/24, same tokens on all 8 ranks; 470-token prompt 468/469 (near-tie) + 16/16 |

Bandwidth floors (estimate: bytes streamed per core / 658 GB/s, lab 1's best single-core GEMV rate): 8B 11.6 / 5.8 /
2.9 ms (TP=2/4/8), 14B 21.3 / 10.7, 32B 24.3 / 12.8. The steps are 5-22% above them (8B TP=8: 62%). The gap is the
collectives (0.5-1.3 ms per step, 7-10 us per all-reduce) plus the per-layer dependency chain lab 1 found on one
core, which dominates once a core's shard is small.

What to do (details and numbers in the sections):
1. Communicate **inside the NEFF** with `nki.collectives` (`all_reduce`, HBM buffers are fastest). Load one SPMD
   NEFF on every rank with `nrt_load_collectives(neff, vnc, -1, rank, W)`, with `NEURON_RT_ROOT_COMM_ID=localhost:<port>`
   set. A decode-sized all-reduce takes 10-15 us measured inside a NEFF (13.4 us for 8 ranks over two devices).
   Set `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` (documented; SPMD only). The runtime's host-driven collectives
   (`nrta_cc_*`) work but cost ~0.6 ms per call. A kernel cannot touch another logical core's tensors, and
   `nrt_tensor_copy` across cores is refused.
2. **Schedule each rank's executions from its own host thread.** `nrta_execute_schedule` of a collective NEFF blocks
   until the peers schedule theirs, so one thread scheduling all ranks deadlocks on the first call.
3. TP split = lab 1's two-physical-core split generalised to 2W "virtual cores" (heads, intermediate tiles, vocab).
   The two physical cores of a rank add their fp32 partials with `nisa.sendrecv`, then all-reduce over the ranks
   from an internal HBM buffer. Two all-reduces per layer plus two small ones for the argmax.

Build / test: `./build.sh` (see the end of this file). Drivers: `host/ccrun` (any collective NEFF, W ranks),
`host/tpchain` (TP decode chain), `host/d2d`, `host/hcc` (non-NEFF options). Kernels: `kernels/step_tp.py`, build
helper `kernels/nb.py` (documented NKI route only).

---------------------------------------------------------------------------------------------------------

## 1. Communication between logical NeuronCores from a native program

### Options and what happened

| option | works? | how | cost |
|---|---|---|---|
| `nki.collectives.all_reduce` / `all_gather` / `collective_permute` inside an NKI kernel, NEFF compiled once, loaded on W ranks | **yes**: 2 and 4 logical cores of one device; 8 over two devices; 2 over two devices (HBM buffers only); not 2 + 2 over two devices | `probes/p1_cc.py`, `host/ccrun.cpp` | 9.7-15 us per KB-sized all-reduce (13.4 us for 8 ranks over two devices, HBM); 47-125 GB/s algbw at 16 MB |
| `nisa.sendrecv` | only between the two physical cores of one logical core (lab 1); its ranks are `nl.program_id` | lab 1 | 0.85 us |
| host-driven collectives `nrt_cc_global_comm_init` + `nrta_cc_prepare` + `nrta_cc_schedule` (headers say "work in progress") | yes (W=2, one device) | `host/hcc.cpp` | 616 us median for 4 KB, 1.43 ms for 16 MB (11.8 GB/s): far too slow for decode |
| `nrt_tensor_copy` between tensors on two logical cores | **no**: `NRT_INVALID`, log `TDRV:tensor_copy Tensors must be allocated on same HBM (SRC: 3) (DST: 2)`; `nrta_tensor_copy` is documented as same-LNC only | `host/d2d.cpp` | - |
| a NEFF on core A with an input or output tensor allocated on core B | **no**: `NMGR:dlr_check_tensor_set_on_same_vtpb_v2 Tensor output_0 allocated on HBM 2 was passed to a model loaded on lnc 0. That is not supported.` (`NRT_INVALID`) | `host/d2d.cpp` | - |
| host bounce (`nrt_tensor_read` to host, `nrt_tensor_write` to the other core) | yes | `host/d2d.cpp` (cores 58, 59) | 8.8 us for 4 KB, 24 us for 64 KB, 171 us for 1 MB, 2.4 ms for 16 MB (~7 GB/s); every step needs a host round trip |
| `nrt_async_sendrecv_*` (`nrt_async_sendrecv.h`, connects by peer IP + lnc) | not tried: documented as "the experimental asynchronous EFA send/recv path" (`NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED`), i.e. network (EFA), not the intra-instance path | | |

### In-NEFF collectives: the mechanism

```python
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
@nki.jit
def k(x, rg):                                   # rg = ReplicaGroup([[0, 1, ..., W-1]]), a non-tensor kernel argument
    a = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)       # or nl.shared_hbm with name="..." (nkilib: required)
    b = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x)                  # collectives may not read / write kernel I/O tensors directly
    ncc.all_reduce(srcs=[a], dsts=[b], op=nl.add, replica_group=rg)
    ...
outs = k[2]([x_rank0, x_rank1], ReplicaGroup([[0, 1]]))     # documented route: lists of arrays, one per rank
```
- **Build**: NKI's documented route handles multi-rank kernels. If a tensor argument is a list, its length is the
  world size. nki compiles **one** NEFF and runs it once on ranks 0..W-1 (= the first W visible cores), loading
  each with `cc_enabled`. So the build process needs `NEURON_RT_VISIBLE_CORES` with W free cores and
  `NEURON_RT_ROOT_COMM_ID=localhost:<port>`. Without it: `NRT:nrt_load_collectives NEURON_RT_ROOT_COMM_ID is not set;
  cannot load NEFF with collective operations!` (`kernels/nb.py` sets both and copies `kernel.neff`.)
- **Run from C++** (`host/ccrun.cpp`, `host/tpchain.cpp`): `nrt_init`, then for every rank r
  `nrt_load_collectives(neff, size, vnc = r, -1, ctx_device_id = r, ctx_device_count = W, &model[r])` (sequential
  loads in one thread work: 24 ms for a small NEFF on 2 ranks, 2.2-3.9 s for the 4-13 MB TP decode NEFFs).
  Tensors are allocated per rank with `nrt_tensor_allocate(DEVICE, vnc = r, ...)`, then the NEFF is executed on every
  rank. Ranks are the visible vncs in order (rank r on vnc r); 8 ranks on 56-63 span the two devices.
- **Executing**: `nrt_execute` (synchronous) from one thread per rank works. `nrta_execute_schedule` **from a single
  thread for all ranks hangs** on the very first call: with `NEURON_RT_LOG_LEVEL=DEBUG` the last line is
  `nrta_execute_schedule API:IN: (model->h_nn=10001, lnc=56, queue=0)` and the call never returns. The call blocks
  until the other ranks have scheduled the same execution. **One scheduling thread per rank** (each queues all its
  executions, then polls `nrta_is_completed`) works and keeps many executions in flight.
- **First executions are slow**: the first chain after loading had 140-200 ms more wall time than the following
  identical chains (8B TP=4 2-layer test: 2.94 ms/step over 71 steps in rep 0, then 1.00-1.02 ms/step). Run a
  warm-up chain after loading, before timing or serving.
- **LNC=2 semantics (probe `probes/p2_lnc_sem.py`)**: under LNC=2 both physical cores run the program. An **SBUF**
  all_reduce reduces each physical core's SBUF buffer with the same physical core of the other ranks:
  `out_r[pid] = sum_r' x_r'[pid]` exactly, for W = 2 and 4. An HBM (shared_hbm) collective works on the logical
  core's one HBM buffer (nkilib `transformer_tkg` puts a `core_barrier` before it).
- **Minimum SBUF collective size**: the free size must be at least max(2, W). A `[1, 1]` tile (W=2), or `[128, 1]`,
  compiles but asserts in `nrt_load`: `tdrv/instr_collectives.c:571: get_enc_tensor_from_collectives_tensor:
  Assertion 'coll_tensor->num_elem[0] > 0 && coll_tensor->num_elem[1] > 0' failed` (`[128,1]`:
  `get_tensor_mr_size: Assertion 'tensor->num_elem[0] != 0 ...'`). With W=4, `[1, 2]` and `[128, 2]` fail the same
  way, while `[1, 4]`, `[1, 8]` and `[128, 4]` work. The data is apparently split into W chunks along the free axis.

### Measured latency and bandwidth (in-NEFF collectives, fp32)

Method (`probes/sweep_cc.py`, `NEURON_RT_DISABLE_EXECUTION_BARRIER=1`): a NEFF with n dependent collectives
(a->b->a ...), n = 1 and n = 33. Each is checked exactly against numpy, after the build run and again from C++
(`host/ccrun`). Executions are queued on all ranks, one scheduling thread per rank, 200 executions (20 at 16 MB)
x 5 repetitions, minimum. per-op = (t33 - t1) / 32. t1 = the whole execution of a one-collective NEFF, which
includes the ~20 us fixed cost of any NEFF execution (lab 1) and two DMAs. algbw = message bytes / per-op.
all_gather: the 16 KB / 1 MB is each rank's contribution, and the chain adds a 1-column tensor_copy per op.

| op, buffers | ranks (cores) | 64 B ([1,16]) / 4 KB ([128,8]) | 16 KB ([128,32]: one decode partial) | 64 KB | 1 MB | 16 MB | t1 |
|---|---|---|---|---|---|---|---|
| all_reduce SBUF | 2 (56-57) | 19.7 / 13.3 us | **11.6 us** | 10.7 us | 37.3 us (28 GB/s) | - | 78-105 us |
| all_reduce SBUF | 4 (56-59) | 35.7 / 14.9 us | **14.9 us** | 14.6 us | 37.4 us (28 GB/s) | - | 85-107 us |
| all_reduce HBM | 2 (56-57) | - / 9.7 us | 9.8 us | 10.2 us | 21.4 us (49 GB/s) | 352.8 us (47.6 GB/s) | 78-90 us; 606 at 16 MB |
| all_reduce HBM | 4 (56-59) | - / 10.9 us | 11.3 us | 13.2 us | 22.7 us (46 GB/s) | 134.6 us (125 GB/s) | 82-95 us; 388 at 16 MB |
| collective_permute HBM (ring i -> i+1) | 2 (56-57) | - / 16.1 us | 16.1 us | - | 25.7 us (41 GB/s) | 171.7 us (98 GB/s) | 81-92 us |
| collective_permute HBM | 4 (56-59) | - / 19.5 us | 19.5 us | - | 29.4 us (36 GB/s) | 175.7 us (95 GB/s) | 86-98 us |
| all_gather SBUF (free axis) | 2 (56-57) | - | 9.0 us | - | 33.9 us | - | 77-103 us |
| all_gather SBUF | 4 (56-59) | - | 11.7 us | - | **fails** (below) | - | 82 us |
| all_reduce SBUF, [128,40] (20 KB: the 14B / 32B partial) | 4 (56-59) / 8 (56-63) | - | 19.2 / 22.5 us | - | - | - | 90 / 97 us |
| all_reduce HBM, [128,40] | 4 (56-59) / 8 (56-63) | - | 11.3 / 14.2 us | - | - | - | 83 / 89 us |
| all_reduce SBUF, **two devices** | 8 (56-63) | - | 22.3 us | - | 125.5 us (8.4 GB/s) | - | 96-201 us |
| all_reduce HBM, **two devices** | 8 (56-63) | - | **13.4 us** | - | 39.8 us (26 GB/s) | 270.5 us (62 GB/s) | 87-116 us; 542 at 16 MB |
| all_reduce HBM, **two devices** | 2 (59-60: core 59 on device 14, 60 on 15) | - | 30.9 us | - | 43.4 us (24 GB/s) | 263.8 us (63.6 GB/s) | 100-113 us; 507 at 16 MB |
| all_reduce SBUF, two devices | 2 (59-60) | **fails** (below) | | | | | |
| any, two devices, 2 + 2 ranks | 4 (58-61) | **fails** (below) | | | | | |

- Decode-sized collectives cost **~10 us (2 ranks) to ~15 us (4 ranks) within a device, 13.4 us (HBM) for 8 ranks
  over two devices**, nearly flat from 4 KB to 64 KB: latency, not bandwidth. A 1-partition tile is slower than a 128-partition tile of the same or larger size
  (64 B [1,16]: 19.7 / 35.7 us; 4 KB [128,8]: 13.3 / 14.9 us).
- **HBM buffers are faster than SBUF for all-reduce**, increasingly so with more ranks and for odd shapes: 16 KB
  9.8 vs 11.6 us (2 ranks), 11.3 vs 14.9 (4), 13.4 vs 22.3 (8 over two devices); the [128, 40] partial of the
  5120-wide models 11.3 vs 19.2 us (4 ranks). 1 MB: 49 vs 28 GB/s. 4 ranks reach 125 GB/s algbw at 16 MB (2
  ranks 47.6, 8 ranks over two devices 62). Section 2 uses this: HBM all-reduces in the decode step.
- **Topology rules found** (all measured failures, error text below): (a) `NEURON_RT_VISIBLE_CORES` must be one
  contiguous range. `58,62` gives `NRT:nrt_allocate_neuron_cores Logical Neuron cores are not contiguous` at
  `nrt_init`. (b) Groups spanning two devices must use whole devices, or 1 + 1 cores with HBM buffers: 8 ranks on
  56-63 work, and 2 ranks on 59-60 work with HBM buffers. 2 + 2 ranks on 58-61 fail at the first execution:
  `ENC:enc_init_comm [rank 2] failed (2) to init a collective algorithm. reason: no_hier no_mesh
  replica-group:Replica-group 0: [0,1,2,3]` then `TDRV:kbl_exec_build_and_load_cc_resources_internal failed to load
  operations`. SBUF buffers on 59-60 fail with an assertion in `enc.cc:4101 enc_post_operation`. (c) The 2-rank
  cross-device group is slow for small messages (30.9 us at 16 KB vs 13.4 us for 8 ranks over the same devices).
- `collective_permute` is slower than `all_reduce` at every size here (16.1 vs 9.8 us at 16 KB). A
  pairwise-exchange all-reduce built from permutes would not beat the library all-reduce.
- all_gather SBUF 1 MB per rank at 4 ranks: compiles, then every rank fails at its first execution:
  `ENC:enc_post_function [nec_dev 57] Failed to post function 0` / `TDRV:kbl_exec_build_and_load_cc_resources_internal
  failed to load operations` / `NRT_INVALID`. 2 ranks and 16 KB at 4 ranks work.
- **The collectives execution barrier.** Without `NEURON_RT_DISABLE_EXECUTION_BARRIER=1`, executions of a small
  4-rank collective NEFF were bimodal, 82-900 us each. Median over 15 repetitions of 200 queued executions:
  875 us for both the 1- and the 33-all-reduce NEFF, minimum 119 us. 2 ranks were not affected (~80 us).
  Polling with sleeps (`CC_POLL_US`) did not change it. With the variable set, the same NEFFs run in a stable
  81.5 us (n=1) and 433.6 us (n=33), with min / median / max within 2%. The runtime documentation
  (`nrt-env-var-reference`) describes it as "Drop the collectives barrier between executions. Valid only when the
  workload is genuinely SPMD and the caller runs its own barrier". Our chains are SPMD (every rank runs the same
  sequence of the same NEFF, and the in-NEFF collectives synchronize). The 36-layer decode steps did not change
  with it (7.51 vs 7.53 ms at TP=4, 12.99 vs 13.00 at TP=2): long executions hide the barrier.
- Host-driven collectives (`host/hcc`, 2 ranks, cores 58-59): 4 KB 616 us, 64 KB 638 us, 1 MB 1.08 ms, 16 MB
  1.43 ms (median of prepare + schedule + wait; exact sums). Usable for rare large transfers, not per layer.

---------------------------------------------------------------------------------------------------------

## 2. Tensor-parallel decode step: Qwen3-8B over 2 and 4 logical NeuronCores — SOLVED

Kernel `kernels/step_tp.py` (`step_tp`, one SPMD NEFF per TP degree). Host split + test `tests/test_tp.py`. Driver
`host/tpchain.cpp`. Command: `python tests/test_tp.py <W> [G=24] [L=36]` (env `MODEL`, `CORES`, `REPS`, `NOCC=1`
for the no-communication decomposition, `NOBUILD=1`).

### Design
- 2W "virtual cores" g = 2 x rank + pid. Lab 1 split one logical core over its two physical cores; the same split
  is now applied over the 2W virtual cores. Per virtual core: NQL = 32/2W q heads, NKL = 8/2W kv heads (8B W=2:
  8 / 2; W=4: 4 / 1), ITL = 96/2W intermediate tiles (24 / 12), NVL = ceil(1187/2W) vocab tiles (297 / 149; padded
  ids masked to -3e38). QKV and gate/up are N-split, o_proj and down are K-split (they produce partial
  [128, HT] fp32 hidden vectors), and attention plus the KV cache are local to each virtual core. Weights are
  re-laid out per rank at load time (values unchanged).
- Per layer: o_proj partial -> `ncc.all_reduce` (SBUF, fp32 add) over the W ranks, by each physical core for its
  own partial -> `nisa.sendrecv` + add between the two physical cores (lab 1). The same after down. That is 2
  all-reduces + 2 sendrecvs per layer.
- Argmax: each core takes (max, BIGC - lowest id) over its vocab slice. The two physical cores combine with a
  sendrecv. Over the ranks, two `all_reduce(max)` of a [1, 8] tile: M = max gm, then max((gm == M) * score). It is
  exact (ties go to the lowest id), and every rank writes the same token to its own `toks`.
- Embedding, norms, residual, token selection: redundant on every core. Each rank has its own pos/toks and swaps
  them between two tensor sets (lab 1 chain). Executions are queued per rank from one thread per rank.
- Kernel arguments: the TP constants (`W`, `rg` = ReplicaGroup, head counts) are non-tensor kwargs. The vocab id
  base of a rank comes in a per-rank input tensor `rkb` [128, 1] fp32, so all ranks run the same NEFF.

### Numbers (36 layers + LM head + argmax, b512 bucket, weight DMAs on hwdge/sync, queued, REPS=3, steady state)

| | per rank HBM | compile (incl. 1 run) | NEFF | nrt_load (all ranks) | ms/step | streamed per rank per step (estimate) | bandwidth floor at 658 GB/s (estimate) |
|---|---|---|---|---|---|---|---|
| lab 1, 1 core | 17.1 GB | 319 s | 13.8 MB | 2.4 s | 23.1 | 15.2 GB | 23.1 |
| TP=2 (56-57), first version | 9.16 GB | 170 s | 7.2 MB | 2.3 s | 13.00 | 7.6 GB | 11.6 |
| TP=2, `PF=4`, [128,8] argmax tiles | 9.16 GB | 174 s | 7.2 MB | | 12.90 | 7.6 GB | 11.6 |
| **TP=2, final (+ `HBMCC=1`: per-layer all-reduces on HBM buffers)** | 9.16 GB | 173 s | 7.3 MB | | **12.46** | 7.6 GB | 11.6 |
| TP=4 (60-63), first version | 5.20 GB | 103 s | 4.0 MB | 2.2 s | 7.53 | 3.8 GB | 5.8 |
| TP=4, `PF=4`, [128,8] argmax tiles | 5.20 GB | 101 s | 4.0 MB | | 7.44 | 3.8 GB | 5.8 |
| **TP=4, final (+ `HBMCC=1`)** | 5.20 GB | 98 s | 4.0 MB | | **7.10** | 3.8 GB | 5.8 |
| TP=8 (56-63, two devices; kv heads replicated: 16 virtual cores, 8 kv heads), final | 3.39 GB | 63 s | 2.5 MB | 2.1 s | **4.70** | 1.9 GB | 2.9 |
| TP=2, final, **b4608** attention bucket | | 179 s | 7.8 MB | | **13.48** (+1.02 vs b512) | | |
| TP=4, final, **b4608** | | 104 s | 4.3 MB | | **8.17** (+1.07 vs b512) | | |

Decomposition (`NOCC=1`: rank 0's shard of the W-way split, the same kernel built with W=1 and run alone on one
core, with no communication): TP=2 shard 11.71 ms, TP=4 shard 6.57 ms, TP=8 shard 4.12 ms (so TP=8 collectives
cost 0.58 ms per step). Collectives therefore cost **1.29 ms (W=2)
and 0.96 ms (W=4) per step**, 17 / 13 us for each of the 74 all-reduces on the critical path. Without them the
shards stream at 650 GB/s (W=2), 580 GB/s (W=4) and 460 GB/s (W=8). The smaller the shard, the more the step is
bound by lab 1's per-layer dependency chain (TP=8: 114 us per layer for ~53 MB per core). That chain, not
communication, is the main lever left for high TP degrees. b4608: lab 1's single core paid +2.0 ms for the bigger
bucket (23.1 -> 25.1). TP=2 / TP=4 pay +1.0 / +1.1 ms although each core reads 1/W of the cache, so long-bucket
attention is latency-bound per layer (36 more PE tiles per layer in the serial chain), not bandwidth-bound.

What was tried on top (TP=4 unless noted, all still 11/11 + 24/24 vs HF fp32):
- `NEURON_RT_DISABLE_EXECUTION_BARRIER=1`: 7.51 vs 7.53 ms (TP=2: 12.99 vs 13.00). No effect on long NEFFs.
- Device contention: 4 no-communication shards in 4 processes on cores 60-63 at the same time ran 6.57-6.59
  ms/step each, the same as one alone (6.57). There is no HBM or power interference between the logical cores of
  one device, so the ~1 ms really is all-reduce latency.
- `PF=4`: the first 4 weight slabs (4 x 1 MB per physical core) of the next GEMV (gate/up, or the next layer's QKV)
  are issued before each all-reduce, so they stream during it: 7.48 ms (-0.05). Small: the weight queue was not
  idle during the collective for long. For 32B TP=4: 27.50 vs 27.42 ms (no gain).
- Argmax all-reduces on [128, 8] tiles (value in row 0, the op's identity elsewhere) instead of [1, 8]: 7.44 ms
  (-0.04). Section 1: a 1-partition collective is 2x slower.
- **`HBMCC=1`, the best variant**: the two physical cores first add their partials with `nisa.sendrecv`
  (0.85 us). Both cores then DMA the identical sum to an internal `nl.shared_hbm` buffer (named; collectives may
  not use kernel I/O tensors). One HBM `all_reduce` follows (an HBM collective runs once per logical core), then
  both cores DMA the result back to SBUF (`_xchg_add_tp_hbm`). TP=2 12.90 -> **12.46 ms**, TP=4 7.44 -> **7.10 ms**,
  32B TP=4 27.50 -> 26.06, 32B TP=8 17.05 -> 15.37, all still 11/11 + 24/24. The per-step collective cost is now
  12.46 - 11.71 = 0.75 ms (W=2) and 7.10 - 6.57 = 0.53 ms (W=4), i.e. 10 / 7 us per all-reduce: HBM all-reduces
  are faster (section 1), and the extra DMAs partly overlap the weight stream.
- Not tried: bf16 partials (latency-bound, and they change the numerics), and a collective built from
  `collective_permute` (slower than all_reduce in isolation).

Correctness: real Qwen3-8B weights, prompt "The capital of France is Paris. The capital of Germany is" fed token by
token then 24 greedy tokens, compared with HF transformers fp32 on the CPU. Teacher-forced 11/11, generated 24/24
identical (" Berlin. The capital of Italy is Rome. ... Lisbon. The") for W=2 and W=4, with identical token
histories on all ranks. 2-layer versions were also checked (W=2, W=4: 11/11 + 8/8 vs the truncated HF model).
Long prompt (TP=2, `PROMPT_FILE=~/nki-lab/build/longprompt.txt PROMPT_TOKENS=470`, 485 chained steps): teacher-forced
**467/469** vs HF fp32 (the 2 mismatches are near-ties, fp32 logit gap median 0.030, max 0.048), 16/16 generated.
Lab 1's single-core 8B step got 197/199 on its test.

---------------------------------------------------------------------------------------------------------

## 3. A model that needs it: Qwen3-14B and Qwen3-32B in bf16 — SOLVED (compile, load, run, correct)

Same kernel and test (`MODEL=~/models/Qwen3-14B` or `Qwen3-32B`, `python tests/test_tp.py <W> 24`).
Downloaded with `hf download Qwen/Qwen3-14B` / `Qwen/Qwen3-32B` into ~/models (28 / 62 GB on disk).
Bandwidth floor = bytes streamed per rank per step (layer shards + LM-head shard + KV bucket; estimate from the
shapes) / 658 GB/s, the best GEMV rate measured on one core (lab 1).

| model | TP (cores) | per rank HBM | compile (incl. 1 run) | NEFF | nrt_load | ms/step, first version | **ms/step, final** (`PF=4 HBMCC=1`) | shard alone (`NOCC`) | floor (estimate) | check vs HF fp32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 14B (40 layers, hidden 5120, 40 q / 8 kv heads, 29.5 GB: does not fit one core) | 2 (56-57 / 60-61) | 15.93 GB | 298-314 s | 13.1 MB | 3.9-4.1 s | 22.8 | **22.41** | - | 21.3 | 11/11, 24/24 |
| 14B | 4 (60-63) | 8.75 GB | 161-163 s | 6.9 MB | 3.6-3.7 s | 13.03 | **12.17** | - | 10.7 | 11/11, 24/24 |
| 32B (64 layers, 64 q / 8 kv heads, intermediate 25600, 62 GB) | 4 (56-59 / 60-63) | 17.86 GB | 357-372 s | 15.5 MB | 9.2 s | 27.42 | **26.06** | 24.72 | 24.3 | 11/11, 24/24 |
| 32B | **8 (56-63, two devices)** | 10.28 GB | 205-209 s | 8.6 MB | 9.1 s | 17.05 | **15.37** | 14.34 | 12.8 | 11/11, 24/24, same tokens on all 8 ranks |

Collectives per step (full minus shard): 32B TP=4 2.70 ms first version (130 all-reduces, 21 us each) -> 1.34 ms
with HBM all-reduces; TP=8 2.71 -> 1.03 ms. The first version's 21 us per all-reduce for the 5120-wide models is
explained by section 1: an SBUF all-reduce of the [128, 40] partial costs 19.2 us at 4 ranks (16 KB [128, 32]: 14.9),
an HBM one 11.3 us.

Long prompt, 32B TP=8 (`PROMPT_FILE=~/nki-lab/build/longprompt.txt PROMPT_TOKENS=470`, 485 chained steps at
15.48 ms/step): teacher-forced **468/469** vs HF fp32 Qwen3-32B (the mismatch is a near-tie, fp32 logit gap 0.005),
16/16 generated.

All runs: prompt "The capital of France is Paris. The capital of Germany is" + 24 greedy tokens, identical to HF
transformers fp32 on the CPU (" Berlin. The capital of Italy is Rome. ... Lisbon. The").

What broke or needed changing on the way:
- **Collective tile minimum size** (section 1): the [1, 1] / [1, 2] argmax all-reduce asserted in `nrt_load`
  at W=2 / W=4.
- **Intermediate size not divisible by the virtual-core count**: 32B has 200 intermediate tiles and 16 virtual
  cores at TP=8. The host zero-pads gate/up rows and down columns to 208 tiles (13 per virtual core). This is a
  layout change only (SiLU(0) x 0 = 0 adds nothing) and costs 4% more MLP bytes.
- **Fewer kv heads than virtual cores** (8 kv heads, 16 virtual cores at TP=8): each kv head is replicated on the
  2 virtual cores whose q heads use it, and each keeps its own cache copy. The kernel needs no change (NKL = 1).
- **Host-side weight split time**: 32B took 689 s single-threaded (numpy re-layout); 68-71 s with the 64 layers
  re-laid out in 16 threads.
- Not run: 32B on TP=2. It would need ~31 GB of weights per rank (estimate from the shapes) in a 24 GiB core.
- The first version lost 2.7 ms per step to collectives on 32B: an SBUF all-reduce of a [128, 40] tile is slow
  (19.2 us at 4 ranks, 22.5 at 8). HBM all-reduces fixed most of it (above).
- TP=8 across two devices works with the same NEFF code and driver; ranks 0-3 are on device 14 and 4-7 on device 15.

---------------------------------------------------------------------------------------------------------

## Files, commands, reproduction

All in ~/nki-lab2 (`source env.sh` first: venv on PATH, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, PYTHONPATH).

| file | what |
|---|---|
| `build.sh` | `./build.sh host` (C++ drivers), `quick` (+ collective probes + 2-layer 8B TP=2 / TP=4 vs HF, 2.7 min measured), default (+ 36-layer 8B TP=2 / TP=4), `big` (14B TP=2 / 4, 32B TP=4 / 8), `comm` (the section 1 sweep) |
| `kernels/nb.py` | `build(name, "module:func", cores, *args)`: documented NKI route in a fresh process (`NKI_ARTIFACTS_DIR`, `NEURON_RT_VISIBLE_CORES`, `NEURON_RT_ROOT_COMM_ID`), copies `kernel.neff` to `neff/`; `ccrun(...)`: runs a collective NEFF through `host/ccrun` |
| `kernels/step_tp.py` | the TP decode step (`step_tp`; kwargs `W`, `rg`, `PF`, `HBMCC`, head / tile counts) |
| `kernels/tp_build.py` | build target: zero arrays of the per-rank shapes, `step_tp[2]` with lists (one per rank) |
| `kernels/layer_decode.py`, `step_decode.py`, `qwen3_weights.py`, `ref_qwen3.py` | copied unchanged from lab 1 (GEMV, RMSNorm, token selection, safetensors reader, layouts, RoPE tables) |
| `tests/test_tp.py` | `python tests/test_tp.py <W> [G] [L]`: per-rank weight split (threads), build, `host/tpchain`, HF fp32 comparison; env `MODEL CORES TB PF HBMCC NOCC NOBUILD NOREF REPS PROMPT_FILE PROMPT_TOKENS` |
| `probes/p1_cc.py`, `probes/sweep_cc.py` | collective chains (all_reduce SBUF / HBM, collective_permute, all_gather) and the latency / bandwidth sweep (`OUT=`, `N=`, `CC_REPS=`) |
| `probes/p2_lnc_sem.py` | what an SBUF collective means under LNC=2 |
| `probes/p0_copy.py` | copy kernel for `host/d2d` |
| `host/ccrun.cpp` | any collective NEFF on W ranks from one process (`CC_MT=1`: one scheduling thread per rank; `CC_SYNC=1`: `nrt_execute` per thread; `CC_REPS`, `CC_POLL_US`) |
| `host/tpchain.cpp` | the TP decode chain: W ranks, two tensor sets per rank swapping `pos` / `output_0`, one scheduling thread per rank, reps |
| `host/d2d.cpp`, `host/hcc.cpp` | non-collective options: cross-core copy / tensors, host bounce; host-driven `nrta_cc_*` all-reduce |
| `skill-drafts/tensor-parallel.md` | the backend skill draft |
| `build/sweep_cc3.txt`, `build/sweep_cc_x*.txt`, `build/logs/*.log` | raw sweep results and run logs |

Environment for every multi-rank run: `NEURON_RT_VISIBLE_CORES=<contiguous range in 56-63>`,
`NEURON_RT_ROOT_COMM_ID=localhost:<port>` (`LAB2_COMM_ID`; each concurrent job its own port),
`NEURON_RT_DISABLE_EXECUTION_BARRIER=1`.

Pitfalls of the lab itself: two of my jobs collided on cores twice (a sweep moving onto 56-63 while a build wanted
56-57: `nrt_init` failed with `Failed to initialize NRT runtime`). Give every concurrent job a disjoint core range
for its whole lifetime. Do not edit `kernels/step_tp.py` while one of its builds traces it (lab 1 pitfall). One
build was restarted for this reason.
