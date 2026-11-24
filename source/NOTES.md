# Qwen3-14B engine for Trainium2 (tensor parallel 2): notes

## What it is (round 001)

One native executable (`target/engine`, links libnrt only) that speaks the EngineSmith protocol and runs Qwen3-14B
bf16 tensor parallel over the first two visible logical NeuronCores (LNC=2 each: 4 "virtual cores" = 2 ranks x 2
physical cores). All NEFFs are embedded (`.incbin`, `target/gen/neffs.S`), built at build time through NKI's
documented route (`kernels/build_neffs.py`: `NKI_ARTIFACTS_DIR` + `@nki.jit` called with per-rank numpy lists).

| piece | file | what |
|---|---|---|
| decode step | `kernels/step_tp.py` (lab 2, unchanged) | one SPMD NEFF per step: embedding, 40 layers, LM head, argmax (ties to lowest id), `toks[pos+1]`, `pos+1`; in-NEFF `nki.collectives.all_reduce` (HBM buffers) of the o_proj / down partials; `PF=6 HBMCC=1`, `SL=2` (weight slabs of 2 output tiles). Attention buckets 512 / 1152 / 2048 / 4224 / 4608 (`step_b*`), chosen per step by the host (`pos <= TB`) |
| prefill | `kernels/prefill_tp.py` (from lab 1 `prefill_g.py`) | layered: `pf<S>_emb`, `pf<S>_layer` (executed 40 times on per-layer tensor slices), `pf<S>_head` (final norm + LM head + TP argmax, writes `toks[plen]`). Prompt buckets S = 256 / 512 / 1024 / 2048 / 4096. TP change: the o_proj / down partials are summed over the two physical cores (scratch `s_p` + barrier) and then SBUF-all-reduced over the ranks in 256-token sub-chunks as bf16 (`_combine_tp`, `PF_CC` in targets.py) |
| host | `src/engine.cpp`, `src/json.h`, `src/model.h` | safetensors mmap + per-rank re-layout (values unchanged, 48 threads, 2.7 s for 28 GB), NEFF load (`nrt_load_collectives` per rank), shared device tensors bound by name, per-layer slices (`nrt_tensor_allocate_slice`), one scheduling thread per rank, whole request queued as one chain (prefill NEFFs + decode steps), tokens read once per request |
| targets / shapes | `kernels/targets.py` | the NEFF list and per-rank shapes (the contract with `src/model.h`) |

Request flow: the prompt is written to `prompt` (the prefill's `ids` is a slice of it), `plen` and `pos` written;
then one queued chain: prefill of the smallest bucket >= plen (42 executions) + the decode steps; `toks` read once.
A prompt longer than 4,096 tokens: the first 4,096 by prefill (with `plen` temporarily 4,096), the rest by decode
steps. **Reference quirk** (NxD Inference): a one-token prompt `[t]` is answered as the prediction after `[t, 0]`;
the engine prefills `[t, 0]`, then rewrites `toks[1] = g`, `plen = 1` and continues at position 1, as the
reference does. Padding positions of a prefill bucket write K/V garbage beyond plen; decode steps overwrite each
position before it is read and mask everything at or after pos.

Runtime environment set by the engine itself: `NEURON_RT_VISIBLE_CORES` = the first two of the given list (the
harness lists four; the last two are the judge's), `NEURON_RT_ROOT_COMM_ID=localhost:<port from core and pid>`,
`NEURON_RT_DISABLE_EXECUTION_BARRIER=1`. libnrt prints to fd 1: the engine dups the original stdout for replies
and points fd 1 at stderr.

Build: `bash build.sh` -> `kernels/build_neffs.py target/neff` (cache `target/neffcache/<name>-<key>.neff`, key =
the kernel sources the target depends on + the target's shapes + neuronx-cc / nki versions; two parallel 2-core
jobs on `BUILD_CORES`, default the run's cores ENGINESMITH_NEURON_CORE..+3) -> g++. From an empty cache: step NEFFs
~320-340 s each (5), prefill layer NEFFs 24 / 35 / 41 / 75 / 144 s (S = 256 .. 4096), emb / head 11-49 s; a clean
clone (`git clone` + `bash build.sh`, empty cache, cores 56-59) built in 18 min 6 s. Cached: 3 s. Never put a third job on cores outside
the run's four: at formal-build time 60-63 may belong to someone else.

## Measured (round 001, cores 56-57, engine alone, `tests/compare.py` / `tests/smoke.py`, wall time per request)

| | 256 | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|
| prefill bucket, whole request generate(P, 1) | 46 ms | 87 ms | 165 ms | 328 ms | 654 ms |
| decode, ms/token (generate((), 16..64)) | 22.4 (b512) | | 22.8 (b1536-2048) | | 23.7 (b4608) |

Correctness: prefill path vs token-by-token decode-step path (`ENGINE_NO_PREFILL=1`), random prompts of 2, 3, 17,
100, 129, 200, 255, 256, 300, 513, 1,025, 2,049, 4,097 tokens + 16-24 generated tokens: identical streams.
Real-text prompt "The capital of France is Paris. The capital of Germany is": " Berlin. The capital of Italy is
Rome. ... Lisbon. The" (as HF fp32 in lab 2).

Profile of one prefill layer at S = 4,096 built with W=1 (rank 0's shard, no collectives; `neuron-explorer`):
13.0 ms, TensorE active 86%, DMA active 97%, HBM read 5.7 GB (weights re-read per token block), throttling limit
0.5 part of the time. The TP layer costs ~16 ms (654 ms / 40): ~3 ms per layer is the cross-rank all-reduce of the
two [128, 5120, 4096] fp32 partials (SBUF, 128-token messages).

## dev-eval (public prompts)

| run | p256 prefill / decode | p1024 | p4096 | score |
|---|---|---|---|---|
| dev-eval 002 (2 rounds), fp32 prefill all-reduce | 2.943 / 1.282 | 2.230 / 1.334 | 1.518 / 1.285 | **1.6725** (prefill 2.15, decode 1.30), 0 mismatches |
| dev-eval 003, bf16 prefill all-reduce (outliers, below) | 0.600 / 1.280 | 1.450 / 1.351 | 1.659 / 1.298 | 1.2161, 0 mismatches in timed cases; controls p2047 / p2048 / p2049 1 mismatch each (margins 0.125-0.25) |
| dev-eval 004, bf16 all-reduce + reply written before the stderr log | 3.100 / 1.287 | 2.410 / 1.350 | 1.655 / 1.296 | **1.7409** (prefill 2.31, decode 1.31), 0 mismatches |

Reference (NxD Inference, TP 2) per request: prefill 137 / 368 / 993 ms, decode 64 tokens 1,835 / 1,939 / 1,937 ms
(28.7 / 30.3 / 30.3 ms per token). dev-eval 001 (1 round) had candidate prefill outliers (p256 522 ms, p4096 995 ms
by the harness while the engine's own timer said 46 / 653 ms); not reproduced by dev-eval 002 or by the harness's
own client (`enginesmith.protocol.Engine`) run directly. dev-eval 003 had them again (p256 197 / 276 ms for 44 ms
of engine time, p1024 301 / 220 for 152): the engine wrote its stderr log line *before* the reply, and on this shared
machine (other runs building, load ~35) a small write to the scratch-dir file can block; the time fell between the
engine's timer and the reply. Now the reply goes out first with one write(2), the log after it (it records the
previous log's own duration). dev-eval 004: no outliers.

## Prefill all-reduce experiments (S = 4,096, whole request, cores 60-61, engine alone)

| variant | prefill ms |
|---|---|
| fp32, 128-token SBUF messages ([128, 20 x 128] fp32 = 1.3 MB), combine after all blocks (round 001 first version) | 654 |
| no all-reduce at all (`PF_NOAR=1`, wrong tokens: timing only) | 558 |
| fp32, combine each block right after its partial (`OVL=1`) | 658 |
| fp32, software-pipelined: launch block b's all-reduce, consume it after block b+1 (`OVL=2`) | 671 |
| fp32, MLP token blocks of 512 instead of 256 (`TEB=512`: half the MLP weight re-reads) | 657 |
| fp32, 256-token messages (2.6 MB) | fails at run time: `enc_post_operation: Assertion alg == ENC_ALG_MESH` |
| **bf16 messages, 128 tokens** | 619 |
| **bf16 messages, 256 tokens (1.3 MB) -- adopted (`PF_CC`)** | **599** (256: 44, 1,024: 153) |

The cross-rank all-reduce does not overlap with the GEMMs in any arrangement tried; bytes and call count are what
cost. bf16 partials: the per-rank sum of the two physical cores' fp32 partials is rounded to bf16, summed over the
ranks in bf16 and added to the fp32 residual (NxD Inference all-reduces bf16 activations too). Not bitwise equal to
the fp32 version: on 9 random prompts vs the decode-step path, 8 identical, one diverged at generated token 6.

## Decode experiments (64 tokens after a 256-token prompt, bucket 512, cores 60-61; whole request)

| step NEFF variant | ms per 64 tokens |
|---|---|
| SL=1, PF=4, weight DMAs on hwdge / sync (round 001 first version) | 1427.7 |
| hw=1 (weight DMAs alternating sync / scalar hwdge queues) | 1438.4 (slower) |
| **SL=2 (2 output tiles per slab: 20 KB contiguous per partition), PF=4 -- adopted** | **1420.3** (tokens identical on 10 prompts x 65 tokens) |
| SL=4, PF=2 | 1465.1 (slower) |
| SL=2, PF=8 | does not compile (QKV has 7 slabs) |
| SL=2, PF=2 | 1420.0 |
| **SL=2, PF=6 -- adopted** | **1416.5** (vs 1420.2 for PF=4 in the same session; tokens identical on 6 prompts x 65 tokens) |

Buckets: the timed decode of p1024 runs positions 1024..1088, of p4096 4096..4160; buckets 1152 and 4224 (instead
of 2048 and 4608) took decode 64 tokens from 1453 -> 1424 ms (p1024) and 1513 -> 1478 ms (p4096). 3072 dropped
(only control prompts reach it; they use 4224). Device profile of the b512 step (`neuron-explorer capture` with
`--collectives-worker-count=2 -r 2 -m multi.txt`, all 21 inputs given as .npy, `pos` = 300): 23.2 ms (sync
execution), 14.04 GB read from HBM (606 GB/s), MBU 0.85, DMA active 99.6%, collectives 0.93 ms in 82 ops (mostly
hidden behind the weight stream). Decode is DMA-throughput bound.

dev-eval 005 (2 rounds, SL=2 + buckets 1152 / 4224, b3072 still present): p256 3.115 / 1.289, p1024 2.419 / 1.376,
p4096 1.658 / 1.321, **score 1.7555** (prefill 2.32, decode 1.33), 0 mismatches in the timed cases.

dev-eval 006 (**5 rounds**, the formal shape; SL=2 PF=4, buckets 512 / 1152 / 2048 / 4224 / 4608): p256 3.095 /
1.272, p1024 2.408 / 1.351, p4096 1.656 / 1.297, **score 1.7376** (prefill 2.311, decode 1.306), 0 mismatching steps
in all 15 timed streams. Candidate times are stable to 0.3% over rounds; the reference's decode varies 1,810-1,985 ms.

dev-eval 007 (2 rounds, final round-001 engine: SL=2 PF=6): p256 3.111 / 1.296, p1024 2.415 / 1.366, p4096
1.657 / 1.306, **score 1.7504** (prefill 2.318, decode 1.322), 0 mismatching steps. Hello ~37-47 s.

## Prefill profile (S = 4,096, one layer NEFF)

- `neuron-explorer` profiles collective NEFFs on 2 ranks (`--collectives-worker-count=2 -r 2 -i 0`); random inputs
  are fine for the layer NEFF. TP layer (bf16 all-reduce): 13.45 ms device time, 32 cc ops 1.65 ms, TensorE active
  83%, DMA active 97%, HBM read 5.7 GB, throttled to a 50% utilization limit for 25% of the time (same with zero
  inputs: 13.44 ms). With `TEB=512` (MLP blocks of 512 tokens): 13.21 ms, 3.6 GB read, but end to end unchanged
  (600 vs 599 ms), so it was not adopted.
- In the engine each layer costs 14.8-15.0 ms whether 1 or 40 layers run (ENGINE_DBG_PF_LAYERS), whether executions
  are queued 1, 2, 3 or all at a time (ENGINE_DBG_DEPTH), whether every layer uses one tensor set or its own
  (ENGINE_DBG_PF_SAMESET), with or without NEURON_RT_DISABLE_EXECUTION_BARRIER. The ~1.5 ms per layer between the
  profiler's total_time and the engine is **the weight values**: binding the layer's weights to freshly allocated
  (uninitialised, presumably zero) tensors took S=4,096 from 599 to 537 ms (`ENGINE_DBG_PF_FRESH=1`, all layers on
  one set); copying layer 0's real weights into those fresh tensors (`ENGINE_DBG_PF_FRESHCOPY=1`) brought it back to
  599 ms. Same HBM index for both (`nrt_tensor_get_device_allocation_info`). So real data draws more power and the
  device throttles harder than with the profiler's inputs: the prefill is power-limited (wgu alone: 550 ms).
- W=1 shard (no collectives), TEB=512: 12.53 ms; without attention (PF_NOATT=1, timing only) 11.06 ms: nkilib
  attention is 1.47 ms per layer (43 GFLOP per core: ~29 TFLOP/s); the GEMMs run at ~86% of TensorE bf16 peak.
  Prefill is compute- (and power-) bound; what is left is attention efficiency and the ~41 ms of all-reduce.

## Debug knobs (environment of the engine process; timing experiments only, never set by the harness)

`ENGINE_NO_PREFILL=1` (prompt through decode steps, the fp32 path), `ENGINE_DBG_PF_LAYERS=n` (run only n prefill
layers), `ENGINE_DBG_PF_SAMESET=1`, `ENGINE_DBG_PF_FRESH=1|name:name`, `ENGINE_DBG_PF_FRESHCOPY=1`,
`ENGINE_DBG_DEPTH=n` (at most n executions in flight), `ENGINE_DBG_KEEP_BARRIER=1`. Kernel env at build time:
`PF_NOAR=1`, `PF_NOATT=1` (wrong results, timing only; they are not part of the cache key: never build the engine
with them set). Test tools: `tests/seq.py` (the harness's per-round sequence), `tests/compare.py` (two engines token
for token), `tests/smoke.py`; `CORES=60-63` runs them on the experiment cores.

## Ideas not done yet (estimates)

- prefill: the unexplained 1.5 ms per layer (engine vs profiler, ~60 ms at 4,096); attention (1.47 ms per layer).
- decode: DMA-bound at ~606 GB/s per logical core (profile); the measured best GEMV rate was 658-674 GB/s.
- hello takes ~47 s (5 + 15 NEFF loads, 26 s of it nrt_load of the step NEFFs): not scored.
