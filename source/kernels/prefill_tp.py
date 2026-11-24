"""Tensor-parallel layered prefill for Qwen3-14B (W ranks x 2 physical cores = 2W virtual cores), adapted from lab 1's
prefill_g.py (Qwen3-8B, one logical core). Three kinds of NEFF for each prompt bucket S: STAGE 1 embedding, STAGE 2
one layer (executed once per layer on per-layer tensor slices), STAGE 3 final norm + LM head + argmax on token plen-1
(the vocabulary split over the 2W virtual cores, step_tp._lm_argmax_tp).

Per virtual core (rank r, physical core pid): q heads (2r+pid)*NQL.., kv heads (2r+pid)*NKL.., intermediate tiles
(2r+pid)*ITL.. -- the decode step's split and weight layouts (step_tp.py), so prefill and decode share every tensor.
Attention is local: nkilib attention_cte over the rank's 2*NQL q heads, split over the two cores as our QKV split.
o_proj and down are K-split: each core writes a partial h for all hidden tiles to s_p[pid]; after a barrier core pid
adds s_p[0] + s_p[1] for its half of the hidden tiles, all-reduces that over the ranks (SBUF, per physical core:
the same physical core of every rank holds the same half) and adds it to h.
Scratch inputs (caller-allocated, shared by the two cores): s_h [128, HT, S] fp32, s_q [2NQL, 128, S],
s_k [2NKL, 128, S], s_v [2NKL, S, 128], s_o [2NQL, 128, S] bf16, s_p [2, 128, HT, S] fp32.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _consts, _rms_col, D, EPS
import nki.collectives as ncc
from step_tp import _lm_argmax_tp
from nkilib.core.attention.attention_cte import _attention_cte

SCALE = 0.08838834764831845
import os as _os
SLAB_HW = int(_os.environ.get("PG_SLAB_HW", "1"))


def _hd(dst, src):
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _norm_blk(h, t0, n, w_sb, onesb, HT, xo, xo0):
    """RMSNorm of tokens [t0, t0+n) of h [128, HT, S] fp32 (HBM) into xo[:, :, xo0:xo0+n] bf16 (SBUF)."""
    hc = nl.ndarray((128, HT, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(hc, h[:, :, nl.ds(t0, n)])
    sq = nl.ndarray((128, HT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=hc)
    ss = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.psum)
    for kt in range(HT):
        nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq[:, kt, :])
    rstd = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / (HT * 128), op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    for kt in range(HT):
        nisa.scalar_tensor_tensor(dst=xo[:, kt, xo0:xo0 + n], data=hc[:, kt, :], op0=nl.multiply,
                                  operand0=w_sb[:, kt:kt + 1], op1=nl.multiply, operand1=rstd)


def _slab(W, nt, KT):
    """weight slab: on a hardware DGE queue triggered from the Sync engine (no compute there), see FINDINGS 4"""
    t = nl.ndarray((128, KT * 128), dtype=W.dtype, buffer=nl.sbuf)
    if SLAB_HW:
        nisa.dma_copy(dst=t, src=W[:, nl.ds(nt * KT * 128, KT * 128)], dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=t, src=W[:, nl.ds(nt * KT * 128, KT * 128)])
    return t


def _combine(h, s_p, pid, t0, n, HT):
    """h[:, own half, t0:t0+n] += s_p[0] + s_p[1] (own half of the hidden tiles)."""
    HH = HT // 2
    a = nl.ndarray((128, HH, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(t0, n)])
    b = nl.ndarray((128, HH, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(t0, n)])
    c = nl.ndarray((128, HH, n), dtype=nl.float32, buffer=nl.sbuf)
    _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(t0, n)])
    nisa.tensor_tensor(dst=a, data1=a, data2=b, op=nl.add)
    nisa.tensor_tensor(dst=a, data1=a, data2=c, op=nl.add)
    _hd(h[:, nl.ds(pid * HH, HH), nl.ds(t0, n)], a)


NOAR = int(_os.environ.get("PF_NOAR", "0"))
NOATT = int(_os.environ.get("PF_NOATT", "0"))   # timing experiment only: skip the cross-rank all-reduce


def _combine_tp(h, s_p, pid, t0, n, HT, rg, W, CCN=128, CCBF=0):
    """h[:, own half, t0:t0+n] += sum over ranks of (s_p[0] + s_p[1]) (own half of the hidden tiles), in sub-chunks of
    CCN tokens (an SBUF all-reduce message of [128, HT/2 * CCN]; fp32, or bf16 with CCBF=1)."""
    HH = HT // 2
    NC = CCN
    if n < NC:
        NC = n
    cdt = nl.float32
    if CCBF:
        cdt = nl.bfloat16
    for sc in range(n // NC):
        u0 = t0 + sc * NC
        b = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        c = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        sm = nl.ndarray((128, HH * NC), dtype=cdt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sm.reshape((128, HH, NC)), data1=b, data2=c, op=nl.add)
        red = sm
        if W > 1 and not NOAR:
            red = nl.ndarray((128, HH * NC), dtype=cdt, buffer=nl.sbuf)
            ncc.all_reduce(srcs=[sm], dsts=[red], op=nl.add, replica_group=rg)
        a = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        nisa.tensor_tensor(dst=a, data1=a, data2=red.reshape((128, HH, NC)), op=nl.add)
        _hd(h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)], a)


def _combine_launch(s_p, pid, t0, n, HT, rg, W):
    """first half of _combine_tp: (s_p[0] + s_p[1]) own half -> all-reduce over the ranks; returns the pending
    (u0, NC, reduced tile) list. The consumer (_combine_finish) is issued later in program order, so the in-order
    engines keep working on the next block while the all-reduce is in flight."""
    HH = HT // 2
    NC = 128
    if n < NC:
        NC = n
    pend = []
    for sc in range(n // NC):
        u0 = t0 + sc * NC
        b = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(b, s_p[0, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        c = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(c, s_p[1, :, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        sm = nl.ndarray((128, HH * NC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sm.reshape((128, HH, NC)), data1=b, data2=c, op=nl.add)
        red = sm
        if W > 1 and not NOAR:
            red = nl.ndarray((128, HH * NC), dtype=nl.float32, buffer=nl.sbuf)
            ncc.all_reduce(srcs=[sm], dsts=[red], op=nl.add, replica_group=rg)
        pend.append((u0, NC, red))
    return pend


def _combine_finish(h, pid, HT, pend):
    HH = HT // 2
    for i in range(len(pend)):
        u0 = pend[i][0]
        NC = pend[i][1]
        red = pend[i][2]
        a = nl.ndarray((128, HH, NC), dtype=nl.float32, buffer=nl.sbuf)
        _hd(a, h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)])
        nisa.tensor_tensor(dst=a, data1=a, data2=red.reshape((128, HH, NC)), op=nl.add)
        _hd(h[:, nl.ds(pid * HH, HH), nl.ds(u0, NC)], a)


def _pg(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cosT, sinT, kcache, vcache,
        rkb, s_h, s_q, s_k, s_v, s_o, s_p, STAGE, NQL, NKL, ITL, W, rg, OVL=0, TEB=256, CCN=128, CCBF=0):
    S = ids.shape[0]
    HT = embed.shape[1] // 128
    HID = HT * 128
    NQ = 2 * NQL
    pid = nl.program_id(0)
    TB = 512
    if S < 512:
        TB = S
    TE = TEB                  # block for gate/up + down (act block [128, ITL, TE] bf16)
    if S < TE:
        TE = S
    NSUB = 256                # norm sub-chunk
    if TB < 256:
        NSUB = TB
    ones, ident, idf, ones16 = _consts()
    onesb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onesb, value=1.0)
    pl_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pl_sb, src=plen)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    h = s_h

    if STAGE == 1:
        NB = S // 128
        for bi in range(NB // 2):
            b = 2 * bi + pid
            idx = nl.ndarray((128, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=idx, src=ids[nl.ds(b * 128, 128), :])
            er = nl.ndarray((128, HID), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=er, src=embed.ap(pattern=[[HID, 128], [1, HID]], offset=0, vector_offset=idx, indirect_dim=0))
            et = nl.ndarray((128, HT, 128), dtype=nl.float32, buffer=nl.sbuf)
            for kt in range(HT):
                tp = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=tp, stationary=er[:, kt * 128:(kt + 1) * 128], moving=ident)
                nisa.tensor_copy(dst=et[:, kt, :], src=tp)
            _hd(h[:, :, nl.ds(b * 128, 128)], et)
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out

    if STAGE == 2:
        l = 0
        rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rot_sb, src=rot)
        w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w1, src=ln1[l])
        w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w2, src=ln2[l])
        qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qnc, src=qn[l])
        knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=knc, src=kn[l])
        NQKV = NQL + 2 * NKL
        # ===== A: norm1 + QKV + q/k norm + RoPE + cache writes
        for blk in range(S // TB):
            t0 = blk * TB
            x1 = nl.ndarray((128, HT, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
            for sb in range(TB // NSUB):
                _norm_blk(h, t0 + sb * NSUB, NSUB, w1, onesb, HT, x1, sb * NSUB)
            cs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(cs, cosT[:, nl.ds(t0, TB)])
            sn = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
            _hd(sn, sinT[:, nl.ds(t0, TB)])
            for nt in range(NQKV):
                wt = _slab(wqkv[l, pid], nt, HT)
                ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                for kt in range(HT):
                    nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=x1[:, kt, :])
                if nt < NQL + NKL:
                    sq = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=sq, op=nl.square, data=ps)
                    ss = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=ss, stationary=onesb, moving=sq)
                    rs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=rs, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
                    nisa.tensor_scalar(dst=rs, data=rs, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
                    yb = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    if nt < NQL:
                        nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=qnc, op1=nl.multiply, operand1=rs)
                    else:
                        nisa.scalar_tensor_tensor(dst=yb, data=ps, op0=nl.multiply, operand0=knc, op1=nl.multiply, operand1=rs)
                    rp = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=rp, stationary=rot_sb, moving=yb)
                    xc = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=xc, data1=yb, data2=cs, op=nl.multiply)
                    xs = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=xs, data1=rp, data2=sn, op=nl.multiply)
                    ro = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=ro, data1=xc, data2=xs, op=nl.add, engine=nisa.engine.gpsimd)
                    if nt < NQL:
                        _hd(s_q[pid * NQL + nt, :, nl.ds(t0, TB)], ro)
                    else:
                        j = nt - NQL
                        _hd(s_k[pid * NKL + j, :, nl.ds(t0, TB)], ro)
                        _hd(kcache[l, pid, j, :, nl.ds(t0, TB)], ro)
                else:
                    j = nt - NQL - NKL
                    vb = nl.ndarray((128, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=vb, op=nl.copy, data=ps)
                    vr = nl.ndarray((128, TB // 128, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for tb in range(TB // 128):
                        tp = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=tp, stationary=vb[:, tb * 128:(tb + 1) * 128], moving=ident)
                        nisa.tensor_copy(dst=vr[:, tb, :], src=tp)
                    for tb in range(TB // 128):
                        _hd(s_v[pid * NKL + j, nl.ds(t0 + tb * 128, 128), :], vr[:, tb, :])
                        _hd(vcache[l, pid, nl.ds(t0 + tb * 128, 128), j * D:(j + 1) * D], vr[:, tb, :])
        nisa.core_barrier(data=s_q, cores=(0, 1))
        nisa.core_barrier(data=s_k, cores=(0, 1))
        nisa.core_barrier(data=s_v, cores=(0, 1))
        # ===== B: attention (nkilib), heads split over the cores as our QKV split
        if NOATT == 0:   # NOATT: timing experiment only
            _attention_cte(q=s_q, k=s_k, v=s_v, scale=SCALE, causal_mask=True, tp_q=False, tp_k=False, tp_out=True,
                           out_o_hbm=s_o)
        nisa.core_barrier(data=s_o, cores=(0, 1))
        # ===== C: o_proj K-split (own heads) -> partial h for all hidden tiles -> s_p[pid]
        pend = None
        for blk in range(S // TB):
            t0 = blk * TB
            oc = nl.ndarray((128, NQL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in range(NQL):
                _hd(oc[:, hh, :], s_o[pid * NQL + hh, :, nl.ds(t0, TB)])
            for nt in range(HT):
                wt = _slab(wo[l, pid], nt, NQL)
                ps = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.psum)
                for kt in range(NQL):
                    nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=oc[:, kt, :])
                po = nl.ndarray((128, TB), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=po, op=nl.copy, data=ps)
                _hd(s_p[pid, :, nt, nl.ds(t0, TB)], po)
            if OVL == 1:   # combine this block now: its all-reduce can overlap the next block's GEMMs
                nisa.core_barrier(data=s_p, cores=(0, 1))
                _combine_tp(h, s_p, pid, t0, TB, HT, rg, W)
            if OVL == 2:   # software-pipelined: launch this block's all-reduce, consume the previous block's
                nisa.core_barrier(data=s_p, cores=(0, 1))
                nxt = _combine_launch(s_p, pid, t0, TB, HT, rg, W)
                if pend is not None:
                    _combine_finish(h, pid, HT, pend)
                pend = nxt
        if OVL == 2:
            _combine_finish(h, pid, HT, pend)
            pend = None
        if not OVL:
            nisa.core_barrier(data=s_p, cores=(0, 1))
            for blk in range(S // TB):
                _combine_tp(h, s_p, pid, blk * TB, TB, HT, rg, W, CCN, CCBF)
        nisa.core_barrier(data=h, cores=(0, 1))
        # ===== D+E: norm2 + gate/up (own intermediate tiles) + SiLU*up + down K-split -> partial -> combine
        for blk in range(S // TE):
            t0 = blk * TE
            x2 = nl.ndarray((128, HT, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
            for sb in range(TE // NSUB):
                _norm_blk(h, t0 + sb * NSUB, NSUB, w2, onesb, HT, x2, sb * NSUB)
            act = nl.ndarray((128, ITL, TE), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i in range(ITL):
                wg = _slab(wgu[l, pid], i, HT)
                pg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                for kt in range(HT):
                    nisa.nc_matmul(dst=pg, stationary=wg[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                wu = _slab(wgu[l, pid], ITL + i, HT)
                pu = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                for kt in range(HT):
                    nisa.nc_matmul(dst=pu, stationary=wu[:, kt * 128:(kt + 1) * 128], moving=x2[:, kt, :])
                sg = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sg, op=nl.silu, data=pg)
                nisa.tensor_tensor(dst=act[:, i, :], data1=sg, data2=pu, op=nl.multiply)
            for nt in range(HT):
                wt = _slab(wd[l, pid], nt, ITL)
                ps = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.psum)
                for kt in range(ITL):
                    nisa.nc_matmul(dst=ps, stationary=wt[:, kt * 128:(kt + 1) * 128], moving=act[:, kt, :])
                po = nl.ndarray((128, TE), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=po, op=nl.copy, data=ps)
                _hd(s_p[pid, :, nt, nl.ds(t0, TE)], po)
            if OVL == 1:
                nisa.core_barrier(data=s_p, cores=(0, 1))
                _combine_tp(h, s_p, pid, t0, TE, HT, rg, W)
            if OVL == 2:
                nisa.core_barrier(data=s_p, cores=(0, 1))
                nxt = _combine_launch(s_p, pid, t0, TE, HT, rg, W)
                if pend is not None:
                    _combine_finish(h, pid, HT, pend)
                pend = nxt
        if OVL == 2:
            _combine_finish(h, pid, HT, pend)
            pend = None
        if not OVL:
            nisa.core_barrier(data=s_p, cores=(0, 1))
            for blk in range(S // TB):
                _combine_tp(h, s_p, pid, blk * TB, TB, HT, rg, W, CCN, CCBF)
        nisa.core_barrier(data=h, cores=(0, 1))
        nisa.dma_copy(dst=pos_out, src=pl_sb)
        return pos_out

    # STAGE 3: LM head on token plen - 1
    lm1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lm1, data=pl_sb, op0=nl.add, operand0=-1)
    lreg = nisa.register_alloc()
    nisa.register_load(lreg, lm1)
    hl = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=hl, src=h.ap(pattern=[[HT * S, 128], [S, HT]], offset=0, scalar_offset=lreg, indirect_dim=2))
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(hl, wf, HT, HID, ones, nl.bfloat16)
    NVL = wlm.shape[2] // (HT * 128)
    tokf = _lm_argmax_tp(xf, wlm[pid], pid, rkb, 1, 2, HT, NVL, rg, W)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    preg = nisa.register_alloc()
    nisa.register_load(preg, pl_sb)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0), src=tok_i)
    nisa.dma_copy(dst=pos_out, src=pl_sb)
    return pos_out



@nki.jit
def prefill_tp(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cosT, sinT, kcache, vcache,
               rkb, s_h, s_q, s_k, s_v, s_o, s_p, STAGE, NQL, NKL, ITL, W, rg, OVL=0, TEB=256, CCN=128, CCBF=0):
    return _pg(ids, plen, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cosT, sinT, kcache, vcache,
               rkb, s_h, s_q, s_k, s_v, s_o, s_p, STAGE, NQL, NKL, ITL, W, rg, OVL, TEB, CCN, CCBF)
