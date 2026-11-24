"""Generic (model-size parameterised) version of step_lnc2.py; one code for Qwen3-0.6B and Qwen3-8B.
Original description follows.

Whole Qwen3-0.6B decode step in one LNC=2 NEFF: both physical cores of the logical core stream half of every
weight and exchange [128, 8] fp32 partial sums through nisa.sendrecv (SBUF -> SBUF, gpsimd_dma, ~0.85 us).

Split per layer (core c = nl.program_id(0)):
  QKV     N-split by heads: q heads 8c..8c+7, kv heads 4c..4c+3 (16 of the 32 output tiles)
  attn    local: the core's 8 q heads against its 4 kv heads; reads / writes only its half of the cache
  o_proj  K-split (its 8 heads' rows) -> partial h -> exchange + add
  gate/up N-split (intermediate tiles 12c..12c+11 of gate and of up) -> local SiLU*up
  down    K-split (the same 12 intermediate tiles) -> partial -> exchange + add
  LM head N-split (vocab tiles 594c..594c+593, vocab padded to 1188 tiles), local argmax, exchange (max, id).
Everything else (embedding, RMSNorms, residual) is computed redundantly on both cores.

Inputs: like step_decode, but weights pre-split per core (see tests/test_step2.py `split_inputs`):
  wqkv [L, 2, 128, 16*8*128], wo [L, 2, 128, 8*8*128], wgu [L, 2, 128, 24*8*128], wd [L, 2, 128, 8*12*128],
  wlm [2, 128, 594*8*128]; kcache [L, 2, 4, 128, TMAX], vcache [L, 2, TMAX, 512].
Both cores write the same values to toks[pos+1] and to output_0 (= pos+1): identical bytes, no read-modify-write.
"""
import nki
import nki.language as nl
import nki.isa as nisa
from layer_decode import _consts, _gemv, _rms_col, _dma, D, EPS, NEG, SCALE
from step_decode import _token_in, BIGC

VOCAB = 151936
NVL = 594    # vocab tiles per core (2 * 594 * 128 = 152064 >= VOCAB)


def _xchg_add(part, pid):
    """part [128, n] fp32 on each core -> sum over both cores (on both)."""
    n = part.shape[1]
    mine = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mine, src=part)
    other = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=mine, dst=other, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    s = nl.ndarray((128, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=s, data1=mine, data2=other, op=nl.add)
    return s


def _embed_g(embed, t_sb, HT):
    treg = nisa.register_alloc()
    nisa.register_load(treg, t_sb)
    eb = nl.ndarray((128, HT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=eb, src=embed.ap(pattern=[[1, 128], [128, HT]], offset=0, scalar_offset=treg, indirect_dim=0))
    h = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=h, src=eb)
    return h


def _wload(W, n, qi, hw):
    """whole per-core weight block W [128, n] -> SBUF, in 1 MB DMAs (8 KB per partition)"""
    t = nl.ndarray((128, n), dtype=W.dtype, buffer=nl.sbuf)
    step = 4096
    o = 0
    while o < n:
        m = step
        if n - o < step:
            m = n - o
        _dma(t[:, o:o + m], W[:, o:o + m], qi, hw)
        qi += 1
        o += m
    return t, qi


def _gemv_res(xT, wsb, KT, NT, ps):
    for nt in range(NT):
        for k in range(KT):
            nisa.nc_matmul(dst=ps[:, nt:nt + 1], stationary=wsb[:, (nt * KT + k) * 128:(nt * KT + k + 1) * 128],
                           moving=xT[:, k:k + 1])


def _attn_t(qkT, vrow, kall, vall, pb, idf, ident, NQL, NKL, G, NQK, TB, NT):
    """Decode attention with transposed scores S^T [NQL heads, TB+1] (heads on partitions): the per-head max is one
    free-axis reduce and exp + row sums one ScalarE activation; P is PE-transposed back for PV. Fewer dependent
    cross-engine hops than the [t, heads] form."""
    # zero-padded q stationaries: kv head hh only feeds its G q heads' output rows
    qpad = nl.ndarray((128, NKL, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=qpad, value=0.0)
    for hh in range(NKL):
        nisa.tensor_copy(dst=qpad[:, hh, G * hh:G * hh + G], src=qkT[:, G * hh:G * hh + G])
    ST = nl.ndarray((NQL, TB + 1), dtype=nl.float32, buffer=nl.sbuf)
    tr = nl.ndarray((NQL, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=tr, pattern=[[1, TB]], offset=0, channel_multiplier=0)
    mrow = nl.ndarray((NQL, TB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mrow, data=tr, op0=nl.greater_equal, operand0=pb[0:NQL, 0:1], op1=nl.multiply, operand1=NEG)
    CW = 512
    if TB < 512:
        CW = TB
    for c in range(TB // CW):
        ps = nl.ndarray((NQL, CW), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NKL):
            nisa.nc_matmul(dst=ps, stationary=qpad[:, hh, :], moving=kall[:, hh, c * CW:(c + 1) * CW])
        nisa.tensor_tensor(dst=ST[:, c * CW:(c + 1) * CW], data1=ps, data2=mrow[:, c * CW:(c + 1) * CW], op=nl.add)
    psc = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        nisa.nc_matmul(dst=psc, stationary=qpad[:, hh, :], moving=qkT[:, NQL + hh:NQL + hh + 1])
    nisa.tensor_copy(dst=ST[:, TB:TB + 1], src=psc)
    m = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=ST, axis=1)
    negm = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=negm, data=m, op0=nl.multiply, operand0=-SCALE)
    E = nl.ndarray((NQL, TB + 1), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=E, op=nl.exp, data=ST, scale=SCALE, bias=negm, reduce_op=nl.add, reduce_res=den,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rden = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    Pn = nl.ndarray((NQL, TB + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Pn, data=E, op0=nl.multiply, operand0=rden)
    PT = nl.ndarray((128, NT, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(NT):
        tp = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tp, stationary=Pn[:, j * 128:(j + 1) * 128], moving=ident[0:NQL, 0:NQL])
        nisa.tensor_copy(dst=PT[:, j, :], src=tp)
    tpc = nl.ndarray((1, NQL), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tpc, stationary=Pn[:, TB:TB + 1], moving=ident[0:NQL, 0:NQL])
    PTc = nl.ndarray((1, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=PTc, src=tpc)
    ops_ = nl.ndarray((D, NQL), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        for j in range(NT):
            nisa.nc_matmul(dst=ops_[:, G * hh:G * hh + G], stationary=vall[:, j, hh * D:(hh + 1) * D],
                           moving=PT[:, j, G * hh:G * hh + G])
        nisa.nc_matmul(dst=ops_[:, G * hh:G * hh + G], stationary=vrow[0:1, hh * D:(hh + 1) * D],
                       moving=PTc[0:1, G * hh:G * hh + G])
    oT = nl.ndarray((D, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=oT, src=ops_)
    return oT


def _layer2(h, l, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
            preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, HT, NQL, NKL, ITL, QS=None, PRE=0, CWQ=0, ATT2=0):
    """QS = (sqkv, so, sgu, sd): fp8 weights with per-output-channel fp32 scales (column layout per core),
    applied to the PSUM result of each GEMV (before the cross-core exchange for o_proj / down)."""
    HID = HT * 128
    G = NQL // NKL
    TMAX = vcache.shape[2]
    NT = TB // 128
    NS = NT + 1
    qi = 0
    kall = nl.ndarray((D, NKL, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
    vall = nl.ndarray((128, NT, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    for hh in range(NKL):
        _dma(kall[:, hh, :], kcache[l, pid, hh, :, 0:TB], qi, hw)
        qi += 1
    for j in range(NT):
        _dma(vall[:, j, :], vcache[l, pid, j * 128:(j + 1) * 128, :], qi, hw)
        qi += 1
    if PRE:
        NQKV_ = NQL + 2 * NKL
        wq_sb, qi = _wload(wqkv[l, pid], NQKV_ * HT * 128, qi, hw)
        wo_sb, qi = _wload(wo[l, pid], HT * NQL * 128, qi, hw)
        wg_sb, qi = _wload(wgu[l, pid], 2 * ITL * HT * 128, qi, hw)
        wd_sb, qi = _wload(wd[l, pid], HT * ITL * 128, qi, hw)
    w1 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1, src=ln1[l])
    w2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2, src=ln2[l])
    qnc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qnc, src=qn[l])
    knc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=knc, src=kn[l])

    x1 = _rms_col(h, w1, HT, HID, ones, nl.bfloat16)
    NQKV = NQL + 2 * NKL
    qkv_ps = nl.ndarray((128, NQKV), dtype=nl.float32, buffer=nl.psum)
    if PRE:
        _gemv_res(x1, wq_sb, HT, NQKV, qkv_ps)
    else:
        qi = _gemv(x1, wqkv[l, pid], HT, 0, NQKV, qkv_ps, 0, SL, hw, qi)
    qkv = nl.ndarray((128, NQKV), dtype=nl.float32, buffer=nl.sbuf)
    if QS is None:
        nisa.tensor_copy(dst=qkv, src=qkv_ps)
    else:
        s1 = nl.ndarray((128, NQKV), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s1, src=QS[0][l, pid])
        nisa.tensor_tensor(dst=qkv, data1=qkv_ps, data2=s1, op=nl.multiply)
    NQK = NQL + NKL
    sq = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=qkv[:, 0:NQK], data2=qkv[:, 0:NQK], op=nl.multiply)
    ss = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ss, stationary=ones, moving=sq)
    rstd = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xn, data1=qkv[:, 0:NQK], data2=rstd, op=nl.multiply)
    qkn = nl.ndarray((128, NQK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qkn[:, 0:NQL], data=xn[:, 0:NQL], op0=nl.multiply, operand0=qnc)
    nisa.tensor_scalar(dst=qkn[:, NQL:NQK], data=xn[:, NQL:NQK], op0=nl.multiply, operand0=knc)
    rps = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rps, stationary=rot_sb, moving=qkn)
    xc = nl.ndarray((128, NQK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xc, data=qkn, op0=nl.multiply, operand0=cs_col)
    qkT = nl.ndarray((128, NQK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=qkT, data=rps, op0=nl.multiply, operand0=sn_col, op1=nl.add, operand1=xc)
    vT = nl.ndarray((128, NKL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vT, src=qkv[:, NQK:NQKV])
    vr_ps = nl.ndarray((1, NKL * D), dtype=nl.float32, buffer=nl.psum)
    for hh in range(NKL):
        nisa.nc_matmul(dst=vr_ps[0:1, hh * D:(hh + 1) * D], stationary=vT[:, hh:hh + 1], moving=ident)
    vrow = nl.ndarray((1, NKL * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vrow, src=vr_ps)
    if CWQ:
        # compute-dependent writes on their own (hardware DGE) queue: in the weight stream's in-order queue they
        # block every later weight DMA until this layer's k/v exist
        nisa.dma_copy(dst=kcache[l, pid].ap(pattern=[[TMAX, D], [D * TMAX, NKL]], offset=0, scalar_offset=preg, indirect_dim=2),
                      src=qkT[:, NQL:NQK], dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        nisa.dma_copy(dst=vcache[l, pid].ap(pattern=[[NKL * D, 1], [1, NKL * D]], offset=0, scalar_offset=preg, indirect_dim=0),
                      src=vrow, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=kcache[l, pid].ap(pattern=[[TMAX, D], [D * TMAX, NKL]], offset=0, scalar_offset=preg, indirect_dim=2),
                      src=qkT[:, NQL:NQK])
        nisa.dma_copy(dst=vcache[l, pid].ap(pattern=[[NKL * D, 1], [1, NKL * D]], offset=0, scalar_offset=preg, indirect_dim=0),
                      src=vrow)

    if ATT2:
        oT = _attn_t(qkT, vrow, kall, vall, pb, idf, ident, NQL, NKL, G, NQK, TB, NT)
    else:
        io = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=io, pattern=[[128, NT]], offset=0, channel_multiplier=1)
        madd = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=madd, data=io, op0=nl.greater_equal, operand0=pb, op1=nl.multiply, operand1=NEG)
        S = nl.ndarray((128, NS, NQL), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(NT):
            ps = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
            for hh in range(NKL):
                nisa.nc_matmul(dst=ps[:, G * hh:G * hh + G], stationary=kall[:, hh, j * 128:(j + 1) * 128],
                               moving=qkT[:, G * hh:G * hh + G])
            nisa.tensor_scalar(dst=S[:, j, :], data=ps, op0=nl.add, operand0=madd[:, j:j + 1])
        nisa.memset(dst=S[:, NT, :], value=NEG)
        psc = nl.ndarray((1, NQL), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NKL):
            nisa.nc_matmul(dst=psc[0:1, G * hh:G * hh + G], stationary=qkT[:, NQL + hh:NQL + hh + 1], moving=qkT[:, G * hh:G * hh + G])
        nisa.tensor_copy(dst=S[0:1, NT, :], src=psc)
        pm = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=pm, op=nl.maximum, data=S.ap(pattern=[[NS * NQL, 128], [1, NQL], [NQL, NS]], offset=0), axis=2)
        pmT = nl.ndarray((NQL, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=pmT, data=pm)
        m = nl.ndarray((NQL, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=pmT, axis=1)
        dg = nl.ndarray((NQL, NQL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dg, data=idf[0:NQL, 0:NQL], op0=nl.multiply, operand0=m)
        mb = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=mb, stationary=ones16[0:NQL, :], moving=dg)
        mbs = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mbs, src=mb)
        Sm = nl.ndarray((128, NS, NQL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=Sm, data1=S, data2=mbs.ap(pattern=[[NQL, 128], [0, NS], [1, NQL]], offset=0), op=nl.subtract)
        E = nl.ndarray((128, NS, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=E, op=nl.exp, data=Sm, scale=SCALE)
        es = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=es, op=nl.add, data=E.ap(pattern=[[NS * NQL, 128], [1, NQL], [NQL, NS]], offset=0), axis=2)
        den = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=den, stationary=ones, moving=es)
        rden = nl.ndarray((128, NQL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rden, data=den)
        ops_ = nl.ndarray((D, NQL), dtype=nl.float32, buffer=nl.psum)
        for hh in range(NKL):
            for j in range(NT):
                nisa.nc_matmul(dst=ops_[:, G * hh:G * hh + G], stationary=vall[:, j, hh * D:(hh + 1) * D],
                               moving=E[:, j, G * hh:G * hh + G])
            nisa.nc_matmul(dst=ops_[:, G * hh:G * hh + G], stationary=vrow[0:1, hh * D:(hh + 1) * D],
                           moving=E[0:1, NT, G * hh:G * hh + G])
        oT = nl.ndarray((D, NQL), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=oT, data1=ops_, data2=rden, op=nl.multiply)

    y_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    if PRE:
        _gemv_res(oT, wo_sb, NQL, HT, y_ps)
    else:
        qi = _gemv(oT, wo[l, pid], NQL, 0, HT, y_ps, 0, SL, hw, qi)
    if QS is None:
        y = _xchg_add(y_ps, pid)
    else:
        s2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s2, src=QS[1][l, pid])
        ys = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ys, data1=y_ps, data2=s2, op=nl.multiply)
        y = _xchg_add(ys, pid)
    h2 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h2, data1=h, data2=y, op=nl.add)

    x2 = _rms_col(h2, w2, HT, HID, ones, nl.bfloat16)
    gu_ps = nl.ndarray((128, 2 * ITL), dtype=nl.float32, buffer=nl.psum)
    if PRE:
        _gemv_res(x2, wg_sb, HT, 2 * ITL, gu_ps)
    else:
        qi = _gemv(x2, wgu[l, pid], HT, 0, 2 * ITL, gu_ps, 0, SL, hw, qi)
    gu = gu_ps
    if QS is not None:
        s3 = nl.ndarray((128, 2 * ITL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s3, src=QS[2][l, pid])
        gu = nl.ndarray((128, 2 * ITL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gu, data1=gu_ps, data2=s3, op=nl.multiply)
    sg = nl.ndarray((128, ITL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=gu[:, 0:ITL])
    act = nl.ndarray((128, ITL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=act, data1=sg, data2=gu[:, ITL:2 * ITL], op=nl.multiply)
    d_ps = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.psum)
    if PRE:
        _gemv_res(act, wd_sb, ITL, HT, d_ps)
    else:
        qi = _gemv(act, wd[l, pid], ITL, 0, HT, d_ps, 0, SL, hw, qi)
    if QS is None:
        d = _xchg_add(d_ps, pid)
    else:
        s4 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s4, src=QS[3][l, pid])
        dsc = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dsc, data1=d_ps, data2=s4, op=nl.multiply)
        d = _xchg_add(dsc, pid)
    h3 = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h3, data1=h2, data2=d, op=nl.add)
    return h3


def _lm_argmax2(xf, wlm_c, pid, SL, hw, HT, slm_c=None):
    # at most 8,192 accumulating matmuls per PSUM tile: 512 x 32 (8B) made the GEMV 3.7x slower (probes/p11)
    CH = 8192 // HT
    if CH > 512:
        CH = 512
    lg = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    qi = 0
    c0 = 0
    while c0 < NVL:
        n = CH
        if NVL - c0 < CH:
            n = NVL - c0
        ps = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.psum)
        qi = _gemv(xf, wlm_c, HT, c0, c0 + n, ps, 0, SL, hw, qi)
        if slm_c is None:
            nisa.tensor_copy(dst=lg[:, c0:c0 + n], src=ps[:, 0:n])
        else:
            s5 = nl.ndarray((128, CH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s5[:, 0:n], src=slm_c[:, c0:c0 + n])
            nisa.tensor_tensor(dst=lg[:, c0:c0 + n], data1=ps[:, 0:n], data2=s5[:, 0:n], op=nl.multiply)
        c0 += n
    # global ids of this core's logits: (pid*NVL + col)*128 + p; padded ids (>= VOCAB) masked out
    gid = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=gid, pattern=[[128, NVL]], offset=0, channel_multiplier=1)
    pf_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pf_i, pattern=[[0, 1]], offset=0, channel_multiplier=0)
    base = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=base, data=pf_i, op0=nl.add, operand0=pid * (NVL * 128))
    nisa.tensor_scalar(dst=gid, data=gid, op0=nl.add, operand0=base)
    msk = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=msk, data=gid, op0=nl.greater_equal, operand0=float(VOCAB), op1=nl.multiply, operand1=-3.0e38)
    nisa.tensor_tensor(dst=lg, data1=lg, data2=msk, op=nl.add)
    pm = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm[:, 0:1], op=nl.maximum, data=lg, axis=1)
    nisa.tensor_scalar(dst=gid, data=gid, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    eq = nl.ndarray((128, NVL), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eq, data=lg, op0=nl.equal, operand0=pm[:, 0:1])
    nisa.tensor_tensor(dst=eq, data1=eq, data2=gid, op=nl.multiply)
    nisa.tensor_reduce(dst=pm[:, 1:2], op=nl.maximum, data=eq, axis=1)
    rows = nl.ndarray((1, 2, 128), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(2):
        tp = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=pm[:, c:c + 1])
        nisa.tensor_copy(dst=rows[:, c, :], src=tp)
    loc = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)      # (max, BIGC - lowest id) of this core
    nisa.tensor_reduce(dst=loc[:, 0:1], op=nl.maximum, data=rows[:, 0, :], axis=1)
    e2 = nl.ndarray((1, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e2, data=rows[:, 0, :], op0=nl.equal, operand0=loc[:, 0:1])
    nisa.tensor_tensor(dst=e2, data1=e2, data2=rows[:, 1, :], op=nl.multiply)
    nisa.tensor_reduce(dst=loc[:, 1:2], op=nl.maximum, data=e2, axis=1)
    oth = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(src=loc, dst=oth, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    both = nl.ndarray((1, 2, 2), dtype=nl.float32, buffer=nl.sbuf)   # [., core-ish, (max, score)]
    nisa.tensor_copy(dst=both[:, 0, :], src=loc)
    nisa.tensor_copy(dst=both[:, 1, :], src=oth)
    gm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gm, data1=loc[:, 0:1], data2=oth[:, 0:1], op=nl.maximum)
    e3 = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e3, data=both[:, :, 0], op0=nl.equal, operand0=gm)
    nisa.tensor_tensor(dst=e3, data1=e3, data2=both[:, :, 1], op=nl.multiply)
    bs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=bs, op=nl.maximum, data=e3, axis=1)
    tokf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tokf, data=bs, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=BIGC)
    return tokf


def _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin,
           kcache, vcache, TB, SL, hw, NQL, NKL, ITL, QS=None, slm=None, PRE=0, CWQ=0, ATT2=0):
    L = wqkv.shape[0]
    HT = embed.shape[1] // 128
    HID = HT * 128
    pid = nl.program_id(0)
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)
    cs_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    sn_col = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    for half in range(2):
        nisa.dma_copy(dst=cs_col[half * 64:(half + 1) * 64, :],
                      src=cos.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
        nisa.dma_copy(dst=sn_col[half * 64:(half + 1) * 64, :],
                      src=sin.ap(pattern=[[1, 64], [1, 1]], offset=0, scalar_offset=preg, indirect_dim=0))
    ones, ident, idf, ones16 = _consts()
    rot_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_sb, src=rot)
    t_sb = _token_in(pos, plen, prompt, toks, p_sb)
    h = _embed_g(embed, t_sb, HT)
    for l in range(L):
        h = _layer2(h, l, pid, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, rot_sb, kcache, vcache,
                    preg, pb, cs_col, sn_col, ones, ident, idf, ones16, TB, SL, hw, HT, NQL, NKL, ITL, QS, PRE, CWQ, ATT2)
    wf = nl.ndarray((128, HT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wf, src=normf)
    xf = _rms_col(h, wf, HT, HID, ones, nl.bfloat16)
    slm_c = None
    if slm is not None:
        slm_c = slm[pid]
    tokf = _lm_argmax2(xf, wlm[pid], pid, SL, hw, HT, slm_c)
    tok_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_i, src=tokf)
    nisa.dma_copy(dst=toks.ap(pattern=[[1, 1], [1, 1]], offset=1, scalar_offset=preg, indirect_dim=0), src=tok_i)
    p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
    pos_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=pos_out, src=p1)
    return pos_out


# q heads / kv heads / intermediate tiles per core: 06b (8, 4, 12), 8b (16, 4, 48); s = n-tiles per weight DMA


@nki.jit
def g06b_b512_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 1, 0, 8, 4, 12)


@nki.jit
def g06b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 8, 4, 12)


@nki.jit
def g06b_b1024_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 1024, 1, 0, 8, 4, 12)


@nki.jit
def g06b_b1024_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 1024, 4, 0, 8, 4, 12)


@nki.jit
def g06b_b2048_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 2048, 1, 0, 8, 4, 12)


@nki.jit
def g06b_b2048_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 2048, 4, 0, 8, 4, 12)


@nki.jit
def g06b_b4608_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 1, 0, 8, 4, 12)


@nki.jit
def g06b_b4608_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 4, 0, 8, 4, 12)


@nki.jit
def g8b_b512_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 1, 0, 16, 4, 48)


@nki.jit
def g8b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 16, 4, 48)


@nki.jit
def g8b_b1024_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 1024, 1, 0, 16, 4, 48)


@nki.jit
def g8b_b1024_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 1024, 4, 0, 16, 4, 48)


@nki.jit
def g8b_b2048_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 2048, 1, 0, 16, 4, 48)


@nki.jit
def g8b_b2048_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 2048, 4, 0, 16, 4, 48)


@nki.jit
def g8b_b4608_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 1, 0, 16, 4, 48)


@nki.jit
def g8b_b4608_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 4, 0, 16, 4, 48)

# fp8 weight variants: weights float8_e4m3 + per-output-channel fp32 scales sqkv [L,2,128,NQKV], so [L,2,128,HT],
# sgu [L,2,128,2*ITL], sd [L,2,128,HT], slm [2,128,594]; SL=2 keeps 1 MB per weight DMA at K=4096 (fp8)


@nki.jit
def q06b_b512_s2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 2, 0, 8, 4, 12, (sqkv, so, sgu, sd), slm)


@nki.jit
def q06b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 8, 4, 12, (sqkv, so, sgu, sd), slm)


@nki.jit
def q06b_b512_s8(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 8, 0, 8, 4, 12, (sqkv, so, sgu, sd), slm)


@nki.jit
def q06b_b4608_s2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 2, 0, 8, 4, 12, (sqkv, so, sgu, sd), slm)


@nki.jit
def q06b_b4608_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 4, 0, 8, 4, 12, (sqkv, so, sgu, sd), slm)


@nki.jit
def q06b_b4608_s8(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 8, 0, 8, 4, 12, (sqkv, so, sgu, sd), slm)


@nki.jit
def q8b_b512_s2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 2, 0, 16, 4, 48, (sqkv, so, sgu, sd), slm)


@nki.jit
def q8b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 16, 4, 48, (sqkv, so, sgu, sd), slm)


@nki.jit
def q8b_b512_s8(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 8, 0, 16, 4, 48, (sqkv, so, sgu, sd), slm)


@nki.jit
def q8b_b4608_s2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 2, 0, 16, 4, 48, (sqkv, so, sgu, sd), slm)


@nki.jit
def q8b_b4608_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 4, 0, 16, 4, 48, (sqkv, so, sgu, sd), slm)


@nki.jit
def q8b_b4608_s8(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, sqkv, so, sgu, sd, slm):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 8, 0, 16, 4, 48, (sqkv, so, sgu, sd), slm)


@nki.jit
def p06b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    """whole-layer weight prefetch into resident SBUF regions"""
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 8, 4, 12, None, None, 1)


@nki.jit
def c06b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    """KV-cache writes on a separate hardware-DGE queue"""
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 8, 4, 12, None, None, 0, 1)


@nki.jit
def t06b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    """transposed-score attention (_attn_t)"""
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 0, 8, 4, 12, None, None, 0, 0, 1)


@nki.jit
def t8b_b512_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 1, 0, 16, 4, 48, None, None, 0, 0, 1)


@nki.jit
def t06b_b4608_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    """transposed-score attention (_attn_t)"""
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 4, 0, 8, 4, 12, None, None, 0, 0, 1)


@nki.jit
def t8b_b4608_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 1, 0, 16, 4, 48, None, None, 0, 0, 1)


@nki.jit
def h06b_b512_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    """weight / cache DMAs on hardware DGE from the sync engine only"""
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 4, 2, 8, 4, 12)


@nki.jit
def h8b_b512_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 512, 1, 2, 16, 4, 48)


@nki.jit
def h06b_b4608_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    """weight / cache DMAs on hardware DGE from the sync engine only"""
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 4, 2, 8, 4, 12)


@nki.jit
def h8b_b4608_s1(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 4608, 1, 2, 16, 4, 48)


@nki.jit
def h06b_b1024_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 1024, 4, 2, 8, 4, 12)


@nki.jit
def h06b_b2048_s4(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache):
    return _step2(pos, plen, prompt, toks, embed, ln1, ln2, qn, kn, wqkv, wo, wgu, wd, normf, wlm, rot, cos, sin, kcache, vcache, 2048, 4, 2, 8, 4, 12)
