"""The NEFFs of the Qwen3-14B engine (tensor parallel 2) and how each is traced: zero arrays of the per-rank shapes
(only shapes matter for a NEFF). The shapes here are the contract with the host (src/engine.cpp)."""
import numpy as np

# Qwen3-14B
L, HID, HT, NQ, NKV, D, INTER, VOCAB = 40, 5120, 40, 40, 8, 128, 17408, 151936
W = 2                       # tensor-parallel ranks (logical NeuronCores)
C = 2 * W                   # virtual cores (2 physical cores per rank)
NQL, NKL, ITL = NQ // C, NKV // C, INTER // 128 // C      # 10, 2, 34
VT = -(-VOCAB // 128)       # 1187
NVL = -(-VT // C)           # 297
TMAX = 4608                 # positions in the KV cache (longest request 4,096 + 65)

STEP_BUCKETS = [512, 1152, 2048, 4224, 4608]   # timed decode: 256+65 / 1024+65 / 4096+65 positions; 4608 = the reference's length
STEP_SL = 2          # weight slabs of 2 output tiles (decode 64 tokens at b512: 1427.7 -> 1420.3 ms, tokens identical)
STEP_PF = 6          # slabs of the next GEMV issued before each all-reduce (PF=4: 1420.2, 2: 1420.0, 6: 1416.5 ms)

STEP_SPEC = {
    "pos": ((1, 1), "i32"), "plen": ((1, 1), "i32"), "prompt": ((TMAX, 1), "i32"), "toks": ((TMAX + 1, 1), "i32"),
    "embed": ((VOCAB, HID), "bf16"),
    "ln1": ((L, 128, HT), "f32"), "ln2": ((L, 128, HT), "f32"), "qn": ((L, 128, 1), "f32"), "kn": ((L, 128, 1), "f32"),
    "wqkv": ((L, 2, 128, (NQL + 2 * NKL) * HT * 128), "bf16"), "wo": ((L, 2, 128, HT * NQL * 128), "bf16"),
    "wgu": ((L, 2, 128, 2 * ITL * HT * 128), "bf16"), "wd": ((L, 2, 128, HT * ITL * 128), "bf16"),
    "normf": ((128, HT), "f32"), "wlm": ((2, 128, NVL * HT * 128), "bf16"), "rot": ((128, 128), "bf16"),
    "cos": ((TMAX, 64), "f32"), "sin": ((TMAX, 64), "f32"),
    "kcache": ((L, 2, NKL, 128, TMAX), "bf16"), "vcache": ((L, 2, TMAX, NKL * 128), "bf16"), "rkb": ((128, 1), "f32"),
}
STEP_ORDER = ["pos", "plen", "prompt", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "normf",
              "wlm", "rot", "cos", "sin", "kcache", "vcache", "rkb"]

PREFILL_BUCKETS = [256, 512, 1024, 2048, 4096]
PF_OVL = 0          # prefill layer: 1 / 2 = combine each block right after its partial / software-pipelined (no gain)
PF_CC = dict(CCBF=1, CCN=256)   # prefill all-reduce: bf16 messages of 256 tokens (fp32 x 128: 654 -> 599 ms at 4,096)
STAGES = {1: "emb", 2: "layer", 3: "head"}


def prefill_spec(S):
    """per-layer tensors have a leading layer dim of 1: the host binds them to slices of the stacked tensors"""
    sp = dict(STEP_SPEC)
    for n in ["ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "kcache", "vcache"]:
        sp[n] = ((1,) + sp[n][0][1:], sp[n][1])
    sp.update({"ids": ((S, 1), "i32"), "cosT": ((128, TMAX), "f32"), "sinT": ((128, TMAX), "f32"),
               "s_h": ((128, HT, S), "f32"), "s_q": ((2 * NQL, 128, S), "bf16"), "s_k": ((2 * NKL, 128, S), "bf16"),
               "s_v": ((2 * NKL, S, 128), "bf16"), "s_o": ((2 * NQL, 128, S), "bf16"),
               "s_p": ((2, 128, HT, S), "f32")})
    return sp


PREFILL_ORDER = ["ids", "plen", "toks", "embed", "ln1", "ln2", "qn", "kn", "wqkv", "wo", "wgu", "wd", "normf", "wlm",
                 "rot", "cosT", "sinT", "kcache", "vcache", "rkb", "s_h", "s_q", "s_k", "s_v", "s_o", "s_p"]

STEP_DEPS = ["step_tp.py", "layer_decode.py", "step_decode.py"]
TARGETS = [dict(name=f"step_b{tb}", kind="step", TB=tb, SL=STEP_SL, PF=STEP_PF, cost=10 + tb // 512, deps=STEP_DEPS,
                spec=sorted((k, list(v[0]), v[1]) for k, v in STEP_SPEC.items())) for tb in STEP_BUCKETS]
TARGETS += [dict(name=f"pf{S}_{STAGES[st]}", kind="prefill", S=S, STAGE=st, cost=(S // 256 if st == 2 else 1),
                 deps=STEP_DEPS + ["prefill_tp.py"],
                 spec=sorted((k, list(v[0]), v[1]) for k, v in prefill_spec(S).items()))
            for S in PREFILL_BUCKETS for st in (1, 2, 3)]
for t in TARGETS:
    if t["kind"] == "prefill" and t["STAGE"] == 2:
        t["OVL"] = PF_OVL
        t.update(PF_CC)


def _zeros(spec, order):
    import ml_dtypes
    DT = {"bf16": ml_dtypes.bfloat16, "f32": np.float32, "i32": np.int32}
    return [np.zeros(spec[n][0], DT[spec[n][1]]) for n in order]


def run(t):
    from nki.collectives import ReplicaGroup
    if t["kind"] == "step":
        import step_tp
        per = [_zeros(STEP_SPEC, STEP_ORDER) for _ in range(W)]
        args = [[per[r][i] for r in range(W)] for i in range(len(STEP_ORDER))]
        step_tp.step_tp[2](*args, rg=ReplicaGroup([list(range(W))]), TB=t["TB"], SL=t.get("SL", 1), hw=t.get("hw", 2),
                           NQL=NQL, NKL=NKL, ITL=ITL, W=W, PF=t.get("PF", 4), HBMCC=1)
    elif t["kind"] == "prefill":
        import prefill_tp
        sp = prefill_spec(t["S"])
        per = [_zeros(sp, PREFILL_ORDER) for _ in range(W)]
        for r in range(W):
            per[r][PREFILL_ORDER.index("plen")][:] = 1   # the head reads h at plen - 1: keep the build run in bounds
        args = [[per[r][i] for r in range(W)] for i in range(len(PREFILL_ORDER))]
        prefill_tp.prefill_tp[2](*args, STAGE=t["STAGE"], NQL=NQL, NKL=NKL, ITL=ITL, W=W,
                                 rg=ReplicaGroup([list(range(W))]), OVL=t.get("OVL", 0),
                                 TEB=t.get("TEB", 256), CCN=t.get("CCN", 128), CCBF=t.get("CCBF", 0))
    else:
        raise ValueError(t["kind"])
