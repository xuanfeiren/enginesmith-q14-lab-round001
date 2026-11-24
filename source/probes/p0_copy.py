"""tiny copy kernel (output_0 = x) for host/d2d."""
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def copy_k(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=out, src=t)
    return out


def run(F):
    x = np.arange(128 * F, dtype=np.float32).reshape(128, F)
    o = copy_k[2](x)
    return float(np.abs(np.asarray(o) - x).max())
