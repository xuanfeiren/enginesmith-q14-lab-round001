"""Probe 2: what does an SBUF all_reduce mean under LNC=2? Each physical core (program_id p) loads its own slice
x[p] into SBUF and all-reduces it over the W ranks; out[p] receives the result. Expected (hypothesis): per-physical-
core reduction, out_r[p] = sum_r' x_r'[p]. Also the HBM variant: both cores write different halves of one
shared_hbm tensor, core_barrier, all_reduce from HBM."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as ncc
from nki.collectives import ReplicaGroup


@nki.jit
def sem_sbuf(x, rg):
    _, P, F = x.shape
    pid = nl.program_id(0)
    a = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf, name="cc_a")
    b = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf, name="cc_b")
    out = nl.ndarray((2, P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=a, src=x[pid])
    ncc.all_reduce(srcs=[a], dsts=[b], op=nl.add, replica_group=rg)
    nisa.dma_copy(dst=out[pid], src=b)
    return out


def run(W):
    rng = np.random.default_rng(0)
    xs = [rng.integers(-8, 8, size=(2, 128, 32)).astype(np.float32) for _ in range(W)]
    rg = ReplicaGroup([list(range(W))])
    outs = sem_sbuf[2](xs, rg)
    s = sum(xs)
    return {"per_core_sum_err": [float(np.abs(np.asarray(o) - s).max()) for o in outs],
            "out0_core0_vs_sum_over_cores": [float(np.abs(np.asarray(o)[0] - s.sum(0)).max()) for o in outs]}
