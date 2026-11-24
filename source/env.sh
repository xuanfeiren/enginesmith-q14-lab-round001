# source me. Lab 2 cores: 56-63 only.
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
export PATH=$V/bin:$PATH
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-56-59}
export PYTHONPATH=$LAB_DIR/kernels:$LAB_DIR/probes:$PYTHONPATH
