"""Build target for nb.build: compile step_tp for given per-rank input shapes (documented route: the kernel is called
with lists of numpy arrays, one per rank; zero-filled, since only shapes matter for the NEFF)."""
import numpy as np
import ml_dtypes
from nki.collectives import ReplicaGroup
import step_tp

DT = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}
ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "normf",
         "wlm", "rot", "cos", "sin", "kcache", "vcache", "rkb"]


def build_step(spec, W, TB, SL, hw, NQL, NKL, ITL, kernel="step_tp", PF=0, HBMCC=0):
    k = getattr(step_tp, kernel)
    mk = lambda: [np.zeros(spec[n][0], DT[spec[n][1]]) for n in ORDER]
    kw = dict(TB=TB, SL=SL, hw=hw, NQL=NQL, NKL=NKL, ITL=ITL, W=W, PF=PF, HBMCC=HBMCC)
    if W == 1:
        k[2](*mk(), rg=None, **kw)
    else:
        per = [mk() for _ in range(W)]
        k[2](*[[per[r][i] for r in range(W)] for i in range(len(ORDER))], rg=ReplicaGroup([list(range(W))]), **kw)
    return "ok"
