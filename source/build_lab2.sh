#!/bin/bash
# Build + self-check lab 2 (tensor parallelism over logical NeuronCores). Lab cores 56-63 only.
#   ./build.sh host     C++ drivers only
#   ./build.sh quick    drivers + collective probes + 2-layer Qwen3-8B TP=2 / TP=4 decode vs HF (~10 min)
#   ./build.sh          quick + full 36-layer Qwen3-8B TP=2 and TP=4 (~30 min, ~/models/Qwen3-8B)
#   ./build.sh big      Qwen3-14B TP=2 / TP=4, Qwen3-32B TP=4 / TP=8 (~1.5 h, ~/models/Qwen3-14B, Qwen3-32B)
#   ./build.sh comm     the communication sweep of FINDINGS section 1 (~1 h, cores 56-63)
set -e
cd "$(dirname "$0")"
source ./env.sh
NRT=/opt/aws/neuron
mkdir -p neff build/logs
for t in ccrun tpchain d2d hcc; do
  g++ -O2 -std=c++17 -I$NRT/include host/$t.cpp -o host/$t -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
done
echo "host tools built: host/ccrun host/tpchain host/d2d host/hcc"
[ "$1" = "host" ] && exit 0
export NEURON_RT_DISABLE_EXECUTION_BARRIER=1      # SPMD chains: documented latency win (FINDINGS section 1)
export LAB2_COMM_ID=${LAB2_COMM_ID:-localhost:54001}
export PF=${PF:-4} HBMCC=${HBMCC:-1}              # the final decode variant (FINDINGS section 2)
run() { echo "== $*"; "$@" 2>&1 | grep -v "Warning\|Loading weights" | grep -E "PASS|FAIL|wrote /|CHAIN rep [12]|identical|agree|err|rror|BENCH|per_op" ; }
if [ "$1" = "comm" ]; then
  for k in sbuf hbm perm ag; do
    s=64,16384,65536,1048576; [ $k = hbm ] && s=4096,16384,65536,1048576,16777216
    [ $k = perm ] && s=4096,16384,1048576,16777216; [ $k = ag ] && s=16384,1048576
    run python probes/sweep_cc.py $k "56-57;56-59;56-63" $s
  done
  run python probes/sweep_cc.py hbm "59-60" 16384,1048576,16777216     # 2 ranks over two devices (HBM only)
  exit 0
fi
if [ "$1" != "big" ]; then
  # problem 1: in-NEFF all_reduce (W=2), checked in the build run and from C++; LNC=2 semantics of SBUF collectives
  run python -c "import sys; sys.path.insert(0,'kernels'); import nb; print('p1 all_reduce W=2', nb.build('ar_hbm_w2', 'p1_cc:run', '56-57', 'hbm', 2, 128, 32, 1)[0])"
  run python -c "import sys; sys.path.insert(0,'kernels'); import nb; print('p2 semantics W=4 err', nb.build('p2_sem_w4', 'p2_lnc_sem:run', '56-59', 4)[0])"
  # problem 2: tensor-parallel decode, 2 layers (fast) then full
  REPS=3 run python tests/test_tp.py 2 8 2
  CORES=56-59 REPS=3 run python tests/test_tp.py 4 8 2
  [ "$1" = "quick" ] && exit 0
  REPS=3 run python tests/test_tp.py 2 24
  CORES=56-59 REPS=3 run python tests/test_tp.py 4 24
  exit 0
fi
# problem 3
MODEL=~/models/Qwen3-14B CORES=56-57 REPS=3 run python tests/test_tp.py 2 24
MODEL=~/models/Qwen3-14B CORES=56-59 REPS=3 run python tests/test_tp.py 4 24
MODEL=~/models/Qwen3-32B CORES=56-59 REPS=3 run python tests/test_tp.py 4 24
MODEL=~/models/Qwen3-32B CORES=56-63 REPS=3 run python tests/test_tp.py 8 24
