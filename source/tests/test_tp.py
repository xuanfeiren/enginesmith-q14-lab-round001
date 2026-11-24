"""Tensor-parallel Qwen3 decode (kernels/step_tp.py) on W logical NeuronCores vs HF transformers fp32 on the CPU.
usage: python tests/test_tp.py <W> [G=16] [L=all] ; env MODEL (default ~/models/Qwen3-8B), TB (512), SL (1), HW (2),
CORES (default 56..56+W-1), NOBUILD=1, REPS, PROMPT, NOREF=1 (timing only)."""
import sys, os, json, time, shutil, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import nb
from tp_build import ORDER
from ref_qwen3 import rope_tables
from qwen3_weights import SafeTensors, ws_layout, col_layout, rope_rot_matrix, bf16

TMAX = 4608


def split_tp(st, cfg, L, ids, W):
    """Per-rank input dicts. C = 2W virtual cores; virtual core g = 2*rank + pid gets heads / tiles block g."""
    H = cfg["hidden_size"]; HT = H // 128; NQ = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
    IT = cfg["intermediate_size"] // 128
    C = 2 * W
    # kv heads: split if C divides NKV, else replicated (C a multiple of NKV: C/NKV virtual cores share a kv head,
    # each keeps its own copy of that head's cache)
    # intermediate tiles: if C does not divide IT, gate/up rows and down columns are zero-padded to ni*C tiles
    # (a layout change: SiLU(0) * 0 = 0 contributes nothing; weight values are unchanged)
    assert NQ % C == 0 and (NKV % C == 0 or C % NKV == 0), (NQ, NKV, C)
    nq, nk, ni = NQ // C, max(1, NKV // C), -(-IT // C)
    g = lambda l, n: st.get(f"model.layers.{l}.{n}")
    VT = -(-cfg["vocab_size"] // 128)
    NVL = -(-VT // C)
    lmn = "lm_head.weight" if "lm_head.weight" in st.index else "model.embed_tokens.weight"
    lm = ws_layout(st.get(lmn)).reshape(128, VT, HT, 128)
    lm = np.concatenate([lm, np.zeros((128, NVL * C - VT, HT, 128), bf16)], 1)
    cos, sin = rope_tables(TMAX)
    pr = np.zeros((TMAX, 1), np.int32); pr[:len(ids), 0] = ids
    common = dict(
        pos=np.zeros((1, 1), np.int32), plen=np.array([[len(ids)]], np.int32), prompt=pr,
        toks=np.zeros((TMAX + 1, 1), np.int32), embed=np.ascontiguousarray(st.get("model.embed_tokens.weight")),
        ln1=np.stack([col_layout(g(l, "input_layernorm.weight")).astype(np.float32) for l in range(L)]),
        ln2=np.stack([col_layout(g(l, "post_attention_layernorm.weight")).astype(np.float32) for l in range(L)]),
        qn=np.stack([g(l, "self_attn.q_norm.weight").reshape(128, 1).astype(np.float32) for l in range(L)]),
        kn=np.stack([g(l, "self_attn.k_norm.weight").reshape(128, 1).astype(np.float32) for l in range(L)]),
        normf=col_layout(st.get("model.norm.weight")).astype(np.float32),
        rot=rope_rot_matrix().astype(bf16), cos=cos, sin=sin)
    ranks = [dict(common) for _ in range(W)]
    for r in range(W):
        ranks[r]["wqkv"] = np.empty((L, 2, 128, (nq + 2 * nk) * HT * 128), bf16)
        ranks[r]["wo"] = np.empty((L, 2, 128, HT * nq * 128), bf16)
        ranks[r]["wgu"] = np.empty((L, 2, 128, 2 * ni * HT * 128), bf16)
        ranks[r]["wd"] = np.empty((L, 2, 128, HT * ni * 128), bf16)
        ranks[r]["wlm"] = np.ascontiguousarray(np.stack([lm[:, NVL * (2 * r + c):NVL * (2 * r + c + 1)] for c in range(2)])).reshape(2, 128, -1)
        ranks[r]["kcache"] = np.zeros((L, 2, nk, 128, TMAX), bf16)
        ranks[r]["vcache"] = np.zeros((L, 2, TMAX, nk * 128), bf16)
        ranks[r]["rkb"] = np.full((128, 1), r * 2 * NVL * 128, np.float32)
    def one_layer(l):
        qkv = ws_layout(np.concatenate([g(l, "self_attn.q_proj.weight"), g(l, "self_attn.k_proj.weight"),
                                        g(l, "self_attn.v_proj.weight")])).reshape(128, NQ + 2 * NKV, HT * 128)
        wo = ws_layout(g(l, "self_attn.o_proj.weight")).reshape(128, HT, NQ, 128)
        gu = ws_layout(np.concatenate([g(l, "mlp.gate_proj.weight"), g(l, "mlp.up_proj.weight")])).reshape(128, 2 * IT, HT * 128)
        wd = ws_layout(g(l, "mlp.down_proj.weight")).reshape(128, HT, IT, 128)
        if ni * C != IT:
            z = np.zeros((128, ni * C - IT, HT * 128), bf16)
            gu = np.concatenate([gu[:, :IT], z, gu[:, IT:], z], 1)
            wd = np.concatenate([wd, np.zeros((128, HT, ni * C - IT, 128), bf16)], 2)
            IT_ = ni * C
        else:
            IT_ = IT
        for r in range(W):
            for c in range(2):
                v = 2 * r + c
                q0, k0, i0 = v * nq, (v * NKV) // C if NKV < C else v * nk, v * ni
                ranks[r]["wqkv"][l, c] = np.concatenate([qkv[:, q0:q0 + nq], qkv[:, NQ + k0:NQ + k0 + nk],
                                                         qkv[:, NQ + NKV + k0:NQ + NKV + k0 + nk]], 1).reshape(128, -1)
                ranks[r]["wo"][l, c] = wo[:, :, q0:q0 + nq].reshape(128, -1)
                ranks[r]["wgu"][l, c] = np.concatenate([gu[:, i0:i0 + ni], gu[:, IT_ + i0:IT_ + i0 + ni]], 1).reshape(128, -1)
                ranks[r]["wd"][l, c] = wd[:, :, i0:i0 + ni].reshape(128, -1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:      # numpy copies release the GIL: re-layout layers in parallel
        list(ex.map(one_layer, range(L)))
    return ranks, (nq, nk, ni)


DTN = {np.dtype(bf16): "bf16", np.dtype(np.float32): "f32", np.dtype(np.int32): "i32"}


def run_chain(neff, ranks, n_steps, reps, cores, W, keep=None):
    import subprocess, tempfile
    d = keep or tempfile.mkdtemp(prefix="tp_", dir=os.path.join(nb.LAB, "build"))
    t0 = time.time()
    for r, ins in enumerate(ranks):
        os.makedirs(os.path.join(d, f"r{r}"), exist_ok=True)
        for k, v in ins.items():
            np.ascontiguousarray(v).tofile(os.path.join(d, f"r{r}", k + ".bin"))
    print(f"wrote inputs in {time.time() - t0:.1f} s", flush=True)
    cs = nb.parse_cores(cores)
    env = {**os.environ, "NEURON_RT_VISIBLE_CORES": cores,
           "NEURON_RT_ROOT_COMM_ID": os.environ.get("LAB2_COMM_ID", f"localhost:{51000 + os.getpid() % 2000}")}
    p = subprocess.run([os.path.join(nb.LAB, "host", "tpchain"), neff, d, str(W), str(n_steps), "pos:output_0",
                        str(reps)], capture_output=True, text=True, env=env, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError(f"tpchain failed: {p.stdout[-3000:]}\n{p.stderr[-5000:]}")
    toks = [np.fromfile(os.path.join(d, f"r{r}", "out_toks.bin"), np.int32) for r in range(W)]
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return toks, p.stdout


if __name__ == "__main__":
    W = int(sys.argv[1])
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3-8B"))
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    L = int(sys.argv[3]) if len(sys.argv) > 3 else cfg["num_hidden_layers"]
    TB, SL, HW = int(os.environ.get("TB", 512)), int(os.environ.get("SL", 1)), int(os.environ.get("HW", 2))
    PF = int(os.environ.get("PF", 0))     # weight slabs prefetched across each all-reduce
    HBMCC = int(os.environ.get("HBMCC", 0))   # 1: per-layer all-reduces on internal HBM buffers
    cores = os.environ.get("CORES", f"56-{56 + W - 1}" if W > 1 else "56")
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = open(os.environ["PROMPT_FILE"]).read() if os.environ.get("PROMPT_FILE") else \
        os.environ.get("PROMPT", "The capital of France is Paris. The capital of Germany is")
    ids = tok(text)["input_ids"][:int(os.environ.get("PROMPT_TOKENS", "100000"))]
    P = len(ids)
    t0 = time.time()
    ranks, (nq, nk, ni) = split_tp(SafeTensors(MODEL), cfg, L, ids, W)
    print(f"split for W={W} (per virtual core: {nq} q heads, {nk} kv heads, {ni} intermediate tiles) in "
          f"{time.time() - t0:.1f} s; per rank {sum(v.nbytes for v in ranks[0].values()) / 1e9:.2f} GB", flush=True)
    mname = os.path.basename(MODEL.rstrip("/")).replace("Qwen3-", "q").lower()
    name = os.environ.get("NAME", f"tp{W}_{mname}_L{L}_b{TB}_s{SL}_h{HW}" + (f"_pf{PF}" if PF else "") + ("_hbmcc" if HBMCC else ""))
    WB = W
    if os.environ.get("NOCC"):
        # timing decomposition: rank 0's shard of the W-way split, built with W=1 (no collectives) and run alone
        # on one core: the step time without any cross-core communication (tokens are meaningless)
        WB, ranks, name, cores = 1, ranks[:1], name + "_nocc", cores.split("-")[0].split(",")[0]
        os.environ["NOREF"] = "1"
    if not os.environ.get("NOBUILD"):
        spec = {k: (ranks[0][k].shape, DTN[ranks[0][k].dtype]) for k in ORDER}
        _, dt = nb.build(name, "tp_build:build_step", cores, spec, WB, TB, SL, HW, nq, nk, ni, "step_tp", PF, HBMCC)
    neff = os.path.join(nb.LAB, "neff", name + ".neff")
    toks, so = run_chain(neff, ranks, P + G - 1, int(os.environ.get("REPS", "1")), cores, WB)
    for l in so.splitlines():
        if "CHAIN" in l or "nrt_load" in l or "upload" in l: print(l)
    del ranks
    same_ranks = all((t == toks[0]).all() for t in toks)
    print("token history identical on all ranks:", same_ranks)
    got = toks[0][1:P + G].tolist()
    if os.environ.get("NOREF"):
        sys.exit(0)
    import torch
    m = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if L < cfg["num_hidden_layers"]:
        m.model.layers = m.model.layers[:L]; m.config.num_hidden_layers = L
    with torch.no_grad():
        full = m.generate(torch.tensor([ids]), max_new_tokens=G, do_sample=False, num_beams=1)[0].tolist()
        lg = m(torch.tensor([full])).logits[0].float().numpy()
    ref_pred = lg.argmax(-1)[:P + G - 1].tolist()
    tf = sum(int(a == b) for a, b in zip(got[:P - 1], ref_pred[:P - 1]))
    gen_k, gen_r = got[P - 1:], full[P:P + G]
    same = next((i for i in range(G) if gen_k[i] != gen_r[i]), G)
    print(f"[fp32] prompt teacher-forced argmax agree {tf}/{P - 1}; generated: first {same}/{G} tokens identical")
    mis = [i for i in range(P - 1) if got[i] != ref_pred[i]]
    if mis:
        gaps = [float(lg[i][ref_pred[i]] - lg[i][got[i]]) for i in mis]
        print(f"   teacher-forced mismatches: fp32 logit gap median {float(np.median(gaps)):.3f} max {max(gaps):.3f}")
    ok = same_ranks and tf >= min(P - 2, int(0.98 * (P - 1))) and same >= min(G, 8)
    if same < G:
        i = P - 1 + same
        gap = float(lg[i][gen_r[same]] - lg[i][gen_k[same]])
        print(f"   divergence at gen {same}: kernel {gen_k[same]} ref {gen_r[same]}, fp32 logit gap {gap:.4f}")
        ok = ok or (same_ranks and gap < 0.1)
    print("kernel text:", repr(tok.decode(gen_k)))
    print("PASS" if ok else "FAIL")
