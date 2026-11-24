"""Start an engine like the harness does (clean env, scratch cwd), send hello + a few generate requests, print tokens
and times. usage: python tests/smoke.py ENGINE [CORES=56-59] ; env PROMPT, N (max_tokens), LENS (extra random
prompt lengths, comma list)"""
import json, os, subprocess, sys, tempfile, time, random
eng = os.path.abspath(sys.argv[1]); cores = sys.argv[2] if len(sys.argv) > 2 else "56-59"
MODEL = "/home/ubuntu/models/Qwen3-14B"
scr = tempfile.mkdtemp(prefix="q14smoke_")
env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": scr, "TMPDIR": scr, "HF_HUB_OFFLINE": "1",
       "NEURON_RT_VISIBLE_CORES": cores}
p = subprocess.Popen([eng, "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=open(os.path.join(scr, "stderr.log"), "w"), cwd=scr, env=env, text=True)
def call(**req):
    t0 = time.time(); p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()
    line = p.stdout.readline(); dt = time.time() - t0
    if not line:
        print(open(os.path.join(scr, "stderr.log")).read()[-3000:]); sys.exit(1)
    return json.loads(line), dt
try:
    r, dt = call(id=1, op="hello"); print("hello", r, f"{dt:.1f} s", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(os.environ.get("PROMPT", "The capital of France is Paris. The capital of Germany is"))["input_ids"]
    n = int(os.environ.get("N", "24"))
    r, dt = call(id=2, op="generate", prompt=ids, max_tokens=1, reset=True); print("prefill", r, f"{dt*1e3:.0f} ms")
    r2, dt = call(id=3, op="generate", prompt=[], max_tokens=n - 1, reset=False); print("decode", f"{dt*1e3:.0f} ms", f"{dt*1e3/(n-1):.2f} ms/tok")
    allt = r["tokens"] + r2["tokens"]
    print(repr(tok.decode(allt)))
    r3, dt = call(id=4, op="generate", prompt=ids, max_tokens=n, reset=True)
    print("one request same:", r3["tokens"] == allt, f"{dt*1e3:.0f} ms")
    r4, _ = call(id=5, op="generate", prompt=[ids[0]], max_tokens=4, reset=True); print("1-token", r4)
    for L in [int(x) for x in os.environ.get("LENS", "").split(",") if x]:
        pr = [random.randrange(151936) for _ in range(L)]
        r, dt = call(id=6, op="generate", prompt=pr, max_tokens=1, reset=True)
        r2, dt2 = call(id=7, op="generate", prompt=[], max_tokens=64, reset=False)
        print(f"len {L}: prefill {dt*1e3:.0f} ms, decode 64 {dt2*1e3:.0f} ms ({dt2*1e3/64:.2f} ms/tok)", r["tokens"], r2["tokens"][:8], flush=True)
finally:
    p.stdin.close(); p.wait(timeout=120)
    print("stderr tail:\n" + open(os.path.join(scr, "stderr.log")).read()[-1500:])
