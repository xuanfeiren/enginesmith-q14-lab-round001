# Seed: kernel lab 2 (tensor parallelism)

The work of kernel lab 2, not yet an engine: FINDINGS.md and kernels/step_tp.py hold a measured, tested tensor-parallel
decode step for Qwen3-8B, 14B and 32B (Qwen3-14B over 2 logical NeuronCores: 22.4 ms/step, identical to HF fp32 on
real tokens), with the in-NEFF all-reduce (nki.collectives), the SPMD load (nrt_load_collectives) and C++ drivers on
libnrt (host/tpchain). Prefill for 14B is not built here: kernel lab 1 (copied under lab1/: the layered prefill in lab1/kernels/prefill_g.py, in the
skill\x27s backends/nki-cpp/prefill.md) and nkilib\x27s attention_cte are the starting points.

To turn it into the engine: a native executable that speaks the EngineSmith protocol (build.sh -> target/engine) and
runs tensor parallel 2 over the first two visible NeuronCores (NEURON_RT_VISIBLE_CORES lists four; the last two are
the reference judge\x27s), every NEFF embedded, built only through NKI\x27s documented route (NKI_ARTIFACTS_DIR +
@nki.jit; kernels/nb.py). Never commit target/ or built NEFFs. Experiments use NeuronCores 60-63.
