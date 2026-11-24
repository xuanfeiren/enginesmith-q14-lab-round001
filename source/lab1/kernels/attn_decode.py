"""Decode attention for Qwen3-0.6B shapes, KV cache resident in HBM, position read on the device.

Layouts (all fixed at compile time):
  qkv    [1, 4096] bf16   output of the fused QKV projection: q heads 0..15 | k heads 0..7 | v heads 0..7 (x128)
  qn_w, kn_w [1, 128] bf16  q_norm / k_norm weights (RMSNorm over head_dim, eps 1e-6)
  cos, sin [TMAX, 64] fp32  RoPE tables, theta 1e6, rotate-half convention (built once at load time)
  kcache [8, 128, TMAX] bf16  K transposed: kcache[h, d, t]  (so a tile is [128 d partitions, t] for TensorE)
  vcache [TMAX, 1024]   bf16  V natural: vcache[t, h*128 + d]
  pos    [1, 1] int32         number of tokens already in the cache = position of the new token
Output: attn [1, 2048] bf16 (head-major, = o_proj input), and the cache updated in place at column/row `pos`.

Attention covers cache positions [0, pos) read from HBM plus the current token taken from SBUF, so the
kernel never reads back what it writes (no RAW hazard through HBM). Positions >= pos in the tiles read are
masked to -30000 before the max; the cache must hold finite values (zero-initialise it once).

Scores are laid out [128 t (partitions), tile j, 16 heads]: softmax reductions run over a few hundred
elements per partition instead of thousands; cross-partition max / sum use TensorE (transpose, ones-matmul).

Two variants of the cache sweep:
  static  (attn_b<TB>):   reads the bucket [0, TB) with unrolled loops; the host picks a NEFF per bucket.
  dynamic (attn_dyn<CH>): nl.fori_loop over ceil(pos / (128*CH)) chunks, so one NEFF reads only what is needed.
"""
import nki
import nki.language as nl
import nki.isa as nisa

D = 128
NQ = 16
NKV = 8
NH = NQ + NKV
EPS = 1e-6
NEG = -30000.0
SCALE = 0.08838834764831845  # 1/sqrt(128)


def _dma(dst, src, i, hw):
    """hw=0: compiler default (software DGE on GpSimd); hw=1: HWDGE alternating sync/scalar queues;
    hw=2: round-robin over sync, scalar (HWDGE) and gpsimd (SWDGE)."""
    if hw == 0:
        nisa.dma_copy(dst=dst, src=src)
    elif hw == 1:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync if i % 2 == 0 else nisa.engine.scalar)
    else:
        if i % 3 == 2:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.swdge)
        else:
            nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.sync if i % 3 == 0 else nisa.engine.scalar)


def _prep(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    """pos -> register + fp32 broadcast; q/k RMSNorm + RoPE; returns (p_sb, preg, pb, qkT [128, 24] bf16,
    vrow [1, 1024] bf16) and writes the new K column / V row into the cache at pos."""
    TMAX = vcache.shape[0]
    p_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=p_sb, src=pos)
    preg = nisa.register_alloc()
    nisa.register_load(preg, p_sb)
    pb_i = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pb_i, src=pos.ap(pattern=[[0, 128], [1, 1]], offset=0))
    pb = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pb, src=pb_i)

    # q / k rows [24, 128], RMSNorm per head
    x = nl.ndarray((NH, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=qkv.ap(pattern=[[D, NH], [1, D]], offset=0))
    w = nl.ndarray((NH, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w[0:NQ, :], src=qn_w.ap(pattern=[[0, NQ], [1, D]], offset=0))
    nisa.dma_copy(dst=w[NQ:NH, :], src=kn_w.ap(pattern=[[0, NKV], [1, D]], offset=0))
    sq = nl.ndarray((NH, D), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((NH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=sq, op=nl.square, data=x, reduce_op=nl.add, reduce_res=ss)
    # rstd = rsqrt(ss/128 + eps) on GpSimd (higher precision than the ScalarE LUT)
    rstd = nl.ndarray((NH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=ss, op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=EPS)
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0, engine=nisa.engine.gpsimd)
    xn = nl.ndarray((NH, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xn, data=x, op0=nl.multiply, operand0=rstd)
    nisa.tensor_tensor(dst=xn, data1=xn, data2=w, op=nl.multiply)

    # RoPE (rotate half) at position pos
    cs = nl.ndarray((NH, 64), dtype=nl.float32, buffer=nl.sbuf)
    sn = nl.ndarray((NH, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs, src=cos.ap(pattern=[[0, NH], [1, 64]], offset=0, scalar_offset=preg, indirect_dim=0))
    nisa.dma_copy(dst=sn, src=sin.ap(pattern=[[0, NH], [1, 64]], offset=0, scalar_offset=preg, indirect_dim=0))
    t1 = nl.ndarray((NH, 64), dtype=nl.float32, buffer=nl.sbuf)
    t2 = nl.ndarray((NH, 64), dtype=nl.float32, buffer=nl.sbuf)
    rb = nl.ndarray((NH, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t1, data1=xn[:, 0:64], data2=cs, op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=xn[:, 64:128], data2=sn, op=nl.multiply)
    nisa.tensor_tensor(dst=rb[:, 0:64], data1=t1, data2=t2, op=nl.subtract)
    t3 = nl.ndarray((NH, 64), dtype=nl.float32, buffer=nl.sbuf)
    t4 = nl.ndarray((NH, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t3, data1=xn[:, 64:128], data2=cs, op=nl.multiply)
    nisa.tensor_tensor(dst=t4, data1=xn[:, 0:64], data2=sn, op=nl.multiply)
    nisa.tensor_tensor(dst=rb[:, 64:128], data1=t3, data2=t4, op=nl.add)

    # transpose to [128 d, 24]: q^T columns 0..15, k^T columns 16..23
    tp = nl.ndarray((D, NH), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tp, data=rb)
    qkT = nl.ndarray((D, NH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qkT, src=tp)
    vrow = nl.ndarray((1, NKV * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=vrow, src=qkv[0:1, (NQ + NKV) * D:(NQ + 2 * NKV) * D])

    # write the new K column / V row at pos (in place)
    nisa.dma_copy(dst=kcache.ap(pattern=[[TMAX, D], [D * TMAX, NKV]], offset=0, scalar_offset=preg, indirect_dim=2),
                  src=qkT[:, NQ:NH])
    nisa.dma_copy(dst=vcache.ap(pattern=[[NKV * D, 1], [1, NKV * D]], offset=0, scalar_offset=preg, indirect_dim=0),
                  src=vrow)
    return p_sb, preg, pb, qkT, vrow


def _current_scores(qkT, S2d, col):
    """Score of the current token for all 16 heads -> S2d[0:1, col:col+16] (rest of that slot set by caller)."""
    psc = nl.ndarray((1, NQ), dtype=nl.float32, buffer=nl.psum)
    for h in range(NKV):
        nisa.nc_matmul(dst=psc[0:1, 2 * h:2 * h + 2], stationary=qkT[:, NQ + h:NQ + h + 1], moving=qkT[:, 2 * h:2 * h + 2])
    nisa.tensor_copy(dst=S2d[0:1, col:col + NQ], src=psc)


def _softmax(S, NS):
    """S [128, NS, 16] fp32 (masked) -> (E [128, NS, 16] bf16 unnormalised probs, rden [128, 16] = 1/sum)."""
    pm = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pm, op=nl.maximum, data=S.ap(pattern=[[NS * NQ, 128], [1, NQ], [NQ, NS]], offset=0), axis=2)
    pmT = nl.ndarray((NQ, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pmT, data=pm)
    m = nl.ndarray((NQ, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=pmT, axis=1)
    ident = nl.ndarray((NQ, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=ident, pattern=[[-1, NQ]], offset=0, channel_multiplier=1)       # p - c
    nisa.tensor_scalar(dst=ident, data=ident, op0=nl.equal, operand0=0.0)           # 1 on the diagonal
    dg = nl.ndarray((NQ, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dg, data=ident, op0=nl.multiply, operand0=m)
    ones16 = nl.ndarray((NQ, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones16, value=1.0)
    mb = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=mb, stationary=ones16, moving=dg)                            # every row = per-head max
    mbs = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mbs, src=mb)
    Sm = nl.ndarray((128, NS, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Sm, data1=S, data2=mbs.ap(pattern=[[NQ, 128], [0, NS], [1, NQ]], offset=0), op=nl.subtract)
    E = nl.ndarray((128, NS, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=E, op=nl.exp, data=Sm, scale=SCALE)
    es = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=es, op=nl.add, data=E.ap(pattern=[[NS * NQ, 128], [1, NQ], [NQ, NS]], offset=0), axis=2)
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    den = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=den, stationary=ones, moving=es)                             # every row = per-head sum
    rden = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden, data=den)
    return E, rden


def _finish(ops_, rden, p_sb, pos, inc_pos):
    o = nl.ndarray((D, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o, data1=ops_, data2=rden, op=nl.multiply)
    oT_ps = nl.ndarray((NQ, D), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=oT_ps, data=o)
    oT = nl.ndarray((NQ, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=oT, src=oT_ps)
    out = nl.ndarray((1, NQ * D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.ap(pattern=[[D, NQ], [1, D]], offset=0), src=oT)
    if inc_pos:
        p1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=p1, data=p_sb, op0=nl.add, operand0=1)
        nisa.dma_copy(dst=pos, src=p1)
    return out


def _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, TB, inc_pos, hw):
    NT = TB // 128
    NS = NT + 1  # + slot for the current token

    # load the cache bucket [0, TB) first: it does not depend on this step's q/k/v
    # (position pos itself is masked, so racing with this step's write of column pos is harmless)
    kall = nl.ndarray((D, NKV, TB), dtype=nl.bfloat16, buffer=nl.sbuf)
    vall = nl.ndarray((128, NT, NKV * D), dtype=nl.bfloat16, buffer=nl.sbuf)
    qi = 0
    for h in range(NKV):
        _dma(kall[:, h, :], kcache[h, :, 0:TB], qi, hw)
        qi += 1
    for j in range(NT):
        _dma(vall[:, j, :], vcache[j * 128:(j + 1) * 128, :], qi, hw)
        qi += 1

    p_sb, preg, pb, qkT, vrow = _prep(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos)

    # mask: madd[p, j] = 0 if j*128+p < pos else NEG
    io = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[128, NT]], offset=0, channel_multiplier=1)
    madd = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=madd, data=io, op0=nl.greater_equal, operand0=pb, op1=nl.multiply, operand1=NEG)

    S = nl.ndarray((128, NS, NQ), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(NT):
        ps = nl.ndarray((128, NQ), dtype=nl.float32, buffer=nl.psum)
        for h in range(NKV):
            nisa.nc_matmul(dst=ps[:, 2 * h:2 * h + 2], stationary=kall[:, h, j * 128:(j + 1) * 128],
                           moving=qkT[:, 2 * h:2 * h + 2])
        nisa.tensor_scalar(dst=S[:, j, :], data=ps, op0=nl.add, operand0=madd[:, j:j + 1])
    nisa.memset(dst=S[:, NT, :], value=NEG)
    _current_scores(qkT, S.reshape((128, NS * NQ)), NT * NQ)

    E, rden = _softmax(S, NS)

    ops_ = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.psum)
    for h in range(NKV):
        for j in range(NT):
            nisa.nc_matmul(dst=ops_[:, 2 * h:2 * h + 2], stationary=vall[:, j, h * D:(h + 1) * D],
                           moving=E[:, j, 2 * h:2 * h + 2])
        nisa.nc_matmul(dst=ops_[:, 2 * h:2 * h + 2], stationary=vrow[0:1, h * D:(h + 1) * D],
                       moving=E[0:1, NT, 2 * h:2 * h + 2])
    return _finish(ops_, rden, p_sb, pos, inc_pos)


def _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, CH, inc_pos, hw, parts=3):
    """fori_loop over chunks of CH tiles (CH*128 positions); only ceil(pos/(128*CH)) chunks are read.
    Rules found by probes/p5_fori_variants.py (nki 0.6, ncc 2.27): inside a fori_loop, dynamic offsets are
    fine on HBM DMA operands and on tensor_copy SBUF operands, but NOT on DMA SBUF destinations (nrt_load
    rejects the NEFF), NOT on matmul SBUF operands (compile error), and PSUM cannot be accumulated across
    iterations (compile error) -> load into fixed tiles, copy scores out, accumulate the output in SBUF."""
    TMAX = vcache.shape[0]
    CW = CH * 128
    NCH = TMAX // CW
    NS = (NCH + 1) * CH          # one extra chunk of slots; its first slot holds the current token
    CUR = NCH * CH

    p_sb, preg, pb, qkT, vrow = _prep(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos)

    # number of chunks = (pos + CW - 1) // CW, as a register
    nch_f = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nch_f, data=p_sb, op0=nl.add, operand0=float(CW - 1))
    nch = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    sh = 7
    if CH == 2:
        sh = 8
    if CH == 4:
        sh = 9
    if CH == 8:
        sh = 10
    nisa.tensor_scalar(dst=nch, data=nch_f, op0=nl.right_shift, operand0=sh)
    nreg = nisa.register_alloc()
    nisa.register_load(nreg, nch)

    S = nl.ndarray((128, NCH + 1, CH * NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=S, value=0.0)
    k4 = kcache.reshape((NKV, D, NCH, CW))
    v3 = vcache.reshape((NCH, CW, NKV * D))

    def qk_body(i):
        kt = nl.ndarray((D, NKV, CW), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in range(NKV):
            _dma(kt[:, h, :], k4.ap(pattern=[[NCH * CW, D], [1, CW]], offset=h * D * NCH * CW, scalar_offset=i, indirect_dim=2), h, hw)
        ps = nl.ndarray((128, CH * NQ), dtype=nl.float32, buffer=nl.psum)
        for c in range(CH):
            for h in range(NKV):
                nisa.nc_matmul(dst=ps[:, c * NQ + 2 * h:c * NQ + 2 * h + 2], stationary=kt[:, h, c * 128:(c + 1) * 128],
                               moving=qkT[:, 2 * h:2 * h + 2])
        nisa.tensor_copy(dst=S.ap(pattern=[[(NCH + 1) * CH * NQ, 128], [1, CH * NQ]], offset=0, scalar_offset=i, indirect_dim=1),
                         src=ps)

    if parts & 1:
        nl.fori_loop(0, nreg, qk_body)

    # mask every slot: madd[p, j] = 0 if j*128+p < pos else NEG (current-token slot fixed up below)
    io = nl.ndarray((128, NS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=io, pattern=[[128, NS]], offset=0, channel_multiplier=1)
    madd = nl.ndarray((128, NS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=madd, data=io, op0=nl.greater_equal, operand0=pb, op1=nl.multiply, operand1=NEG)
    S3 = S.reshape((128, NS, NQ))
    nisa.tensor_tensor(dst=S3, data1=S3, data2=madd.ap(pattern=[[NS, 128], [1, NS], [0, NQ]], offset=0), op=nl.add)
    nisa.memset(dst=S3[:, CUR, :], value=NEG)
    _current_scores(qkT, S.reshape((128, NS * NQ)), CUR * NQ)

    E, rden = _softmax(S3, NS)
    E2 = E.reshape((128, NCH + 1, CH * NQ))

    # output accumulator in SBUF, initialised with the current token's contribution
    ps0 = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.psum)
    for h in range(NKV):
        nisa.nc_matmul(dst=ps0[:, 2 * h:2 * h + 2], stationary=vrow[0:1, h * D:(h + 1) * D],
                       moving=E[0:1, CUR, 2 * h:2 * h + 2])
    acc = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=acc, src=ps0)

    def pv_body(i):
        vt = nl.ndarray((128, CH, NKV * D), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(CH):
            _dma(vt[:, c, :], v3.ap(pattern=[[NKV * D, 128], [1, NKV * D]], offset=c * 128 * NKV * D, scalar_offset=i, indirect_dim=0), c + 1, hw)
        et = nl.ndarray((128, CH, NQ), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=et, src=E2.ap(pattern=[[(NCH + 1) * CH * NQ, 128], [NQ, CH], [1, NQ]], offset=0, scalar_offset=i, indirect_dim=1))
        ps = nl.ndarray((D, NQ), dtype=nl.float32, buffer=nl.psum)
        for h in range(NKV):
            for c in range(CH):
                nisa.nc_matmul(dst=ps[:, 2 * h:2 * h + 2], stationary=vt[:, c, h * D:(h + 1) * D], moving=et[:, c, 2 * h:2 * h + 2])
        nisa.tensor_tensor(dst=acc, data1=acc, data2=ps, op=nl.add)

    if parts & 2:
        nl.fori_loop(0, nreg, pv_body)
    return _finish(acc, rden, p_sb, pos, inc_pos)


@nki.jit
def attn_b512(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 512, False, 0)


@nki.jit
def attn_b1024(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 1024, False, 0)


@nki.jit
def attn_b2048(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 2048, False, 0)


@nki.jit
def attn_b4608(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4608, False, 0)


@nki.jit
def attn_b512_hw(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 512, False, 1)


@nki.jit
def attn_b4608_hw(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4608, False, 1)


@nki.jit
def attn_b512_rr(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 512, False, 2)


@nki.jit
def attn_b4608_rr(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_static(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4608, False, 2)


@nki.jit
def attn_dyn4(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4, False, 0)


@nki.jit
def attn_dyn4_hw(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4, False, 1)


@nki.jit
def attn_dyn2_hw(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 2, False, 1)


@nki.jit
def attn_dyn8_hw(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 8, False, 1)


@nki.jit
def attn_dyn4_inc(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4, True, 0)


@nki.jit
def attn_dyn4_p1(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4, False, 0, 1)


@nki.jit
def attn_dyn4_p2(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4, False, 0, 2)


@nki.jit
def attn_dyn4_p0(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos):
    return _attn_dynamic(qkv, qn_w, kn_w, cos, sin, kcache, vcache, pos, 4, False, 0, 0)
