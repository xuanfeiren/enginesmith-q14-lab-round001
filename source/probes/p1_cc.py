"""Probe 1: nki.collectives inside a NEFF. all_reduce (add) of a [P, F] fp32 tensor across W ranks,
repeated n times in a dependent chain (src -> dst -> src ...), HBM or SBUF buffers.
Build + check through the documented route (lists of numpy arrays, one per rank)."""
import os
import numpy as np
import nki
import nki.isa as nisa
import nki.language as nl
import nki.collectives as ncc
from nki.collectives import ReplicaGroup


@nki.jit
def ar_hbm(x, rg, n):
    P, F = x.shape
    a = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_a")
    b = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_b")
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=a, src=x)
    for i in range(n):
        if i % 2 == 0:
            ncc.all_reduce(srcs=[a], dsts=[b], op=nl.add, replica_group=rg)
        else:
            ncc.all_reduce(srcs=[b], dsts=[a], op=nl.add, replica_group=rg)
    if n % 2 == 1:
        nisa.dma_copy(dst=out, src=b)
    else:
        nisa.dma_copy(dst=out, src=a)
    return out


@nki.jit
def ar_sbuf(x, rg, n):
    P, F = x.shape
    a = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf, name="cc_a")
    b = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf, name="cc_b")
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=a, src=x)
    for i in range(n):
        if i % 2 == 0:
            ncc.all_reduce(srcs=[a], dsts=[b], op=nl.add, replica_group=rg)
        else:
            ncc.all_reduce(srcs=[b], dsts=[a], op=nl.add, replica_group=rg)
    if n % 2 == 1:
        nisa.dma_copy(dst=out, src=b)
    else:
        nisa.dma_copy(dst=out, src=a)
    return out


def run(kind, W, P, F, n, lnc=2):
    rng = np.random.default_rng(0)
    xs = [rng.integers(-8, 8, size=(P, F)).astype(np.float32) for _ in range(W)]
    rg = ReplicaGroup([list(range(W))])
    k = {"hbm": ar_hbm, "sbuf": ar_sbuf}[kind]
    outs = k[lnc](xs, rg, n) if lnc > 1 else k(xs, rg, n)
    s = sum(xs)
    ref = s * (W ** (n - 1))
    errs = [float(np.abs(np.asarray(o) - ref).max()) for o in outs]
    return {"max_err_per_rank": errs, "ref_absmax": float(np.abs(ref).max())}


@nki.jit
def perm_hbm(x, pairs, n):
    """n dependent ring permutes (rank i -> i+1) of an HBM tensor."""
    P, F = x.shape
    a = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_a")
    b = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm, name="cc_b")
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=a, src=x)
    for i in range(n):
        if i % 2 == 0:
            ncc.collective_permute(srcs=[a], dsts=[b], source_target_pairs=pairs)
        else:
            ncc.collective_permute(srcs=[b], dsts=[a], source_target_pairs=pairs)
    if n % 2 == 1:
        nisa.dma_copy(dst=out, src=b)
    else:
        nisa.dma_copy(dst=out, src=a)
    return out


@nki.jit
def ag_sbuf(x, rg, W, n):
    """n all_gathers (SBUF, along the free axis): [P, F] -> [P, W*F]; each next source is this rank's... (the
    first F columns of the previous result, i.e. rank 0's block: a data dependency so the ops serialize)."""
    P, F = x.shape
    a = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    g = nl.ndarray((P, W * F), dtype=x.dtype, buffer=nl.sbuf)
    out = nl.ndarray((P, W * F), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=a, src=x)
    for i in range(n):
        ncc.all_gather(srcs=[a], dsts=[g], replica_group=rg, collective_dim=1)
        if i < n - 1:
            nisa.tensor_copy(dst=a, src=g[:, 0:F])
    nisa.dma_copy(dst=out, src=g)
    return out


def run2(kind, W, P, F, n, lnc=2):
    rng = np.random.default_rng(0)
    xs = [rng.integers(-8, 8, size=(P, F)).astype(np.float32) for _ in range(W)]
    if kind == "perm":
        outs = perm_hbm[lnc](xs, tuple((i, (i + 1) % W) for i in range(W)), n)   # a list would be taken as per-rank inputs
        refs = [xs[(r - n) % W] for r in range(W)]
    else:
        outs = ag_sbuf[lnc](xs, ReplicaGroup([list(range(W))]), W, n)
        g0 = np.concatenate([xs[0]] + [xs[r] for r in range(1, W)], 1) if n == 1 else np.concatenate([xs[0]] * W, 1)
        refs = [g0] * W
    return {"max_err_per_rank": [float(np.abs(np.asarray(o) - rf).max()) for o, rf in zip(outs, refs)]}
