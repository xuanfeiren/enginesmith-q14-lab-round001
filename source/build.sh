#!/bin/bash
# Build the engine: NKI kernels -> NEFFs (documented NKI route, cached under target/neffcache), embedded into one
# native executable target/engine that links only libnrt (+ libc/libstdc++).
# Compiling a NEFF runs it once, so the build needs free NeuronCores: BUILD_CORES (default: the run's cores,
# ENGINESMITH_NEURON_CORE .. +3, as two 2-core jobs).
set -e
cd "$(dirname "$0")"
V=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
export PATH=$V/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
NRT=/opt/aws/neuron
mkdir -p target/neff target/gen
python kernels/build_neffs.py target/neff
NEFFS=$(python -c "import sys; sys.path.insert(0, 'kernels'); import targets; print(' '.join(t['name'] for t in targets.TARGETS))")
{
  echo '.section .rodata'
  for n in $NEFFS; do
    echo ".global neff_${n}_start"; echo ".global neff_${n}_end"; echo ".balign 64"
    echo "neff_${n}_start:"; echo ".incbin \"target/neff/${n}.neff\""; echo "neff_${n}_end:"
  done
  echo '.section .note.GNU-stack,"",@progbits'
} > target/gen/neffs.S
{
  echo 'struct NeffBlob { const char* name; const char* start; const char* end; };'
  for n in $NEFFS; do echo "extern \"C\" const char neff_${n}_start[], neff_${n}_end[];"; done
  echo 'extern const NeffBlob NEFFS[] = {'
  for n in $NEFFS; do echo "  {\"${n}\", neff_${n}_start, neff_${n}_end},"; done
  echo '};'
  echo "extern const int NEFF_COUNT = sizeof(NEFFS) / sizeof(NEFFS[0]);"
} > target/gen/neff_table.cpp
g++ -O2 -std=c++17 -I$NRT/include -Isrc src/engine.cpp target/gen/neff_table.cpp target/gen/neffs.S -o target/engine \
  -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
echo "built target/engine ($(du -h target/engine | cut -f1))"
