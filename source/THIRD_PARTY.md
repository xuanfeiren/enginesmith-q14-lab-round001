# Third-party code

| project | files used | how | license |
|---|---|---|---|
| AWS NKI Library (`nkilib`), github.com/aws-neuron/nki-library, the copy shipped in the neuronx-cc 2.27.5334 wheel (`site-packages/nkilib/`) | `core/attention/attention_cte.py` (`_attention_cte`, prefill attention) and its helpers | imported at build time by `kernels/prefill_tp.py` and traced into the prefill layer NEFFs; not copied into this repository | Apache-2.0 (Copyright Amazon.com, Inc. or its affiliates) |

All other kernels (`kernels/*.py`) and the host code (`src/`) are this project's own (the EngineSmith kernel labs 1
and 2), using only the public NKI API (`nki.language`, `nki.isa`, `nki.collectives`, `@nki.jit`) and libnrt.
