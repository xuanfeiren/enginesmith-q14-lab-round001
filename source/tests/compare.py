"""Compare two engine configurations token for token on random prompts (generate(P, 1) + generate((), N)).
usage: python tests/compare.py ENGINE_A[:ENV=V,..] ENGINE_B[:ENV=V,..] LEN[,LEN..] [N=32] [SEED=0]
Runs A then B (one at a time, cores 56-59 as the harness gives them). Prints per-prompt first divergence."""
import json, os, subprocess, sys, tempfile, time, random

MODEL = "/home/ubuntu/models/Qwen3-14B"


class Eng:
    def __init__(self, spec, cores=os.environ.get("CORES", "56-59")):
        path, _, envs = spec.partition(":")
        scr = tempfile.mkdtemp(prefix="q14cmp_")
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": scr, "TMPDIR": scr, "NEURON_RT_VISIBLE_CORES": cores}
        for kv in filter(None, envs.split(",")):
            k, _, v = kv.partition("="); env[k] = v
        self.err = os.path.join(scr, "stderr.log")
        self.p = subprocess.Popen([os.path.abspath(path), "--model", MODEL], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(self.err, "w"), cwd=scr, env=env, text=True)
        self.i = 0

    def call(self, **req):
        self.i += 1
        req["id"] = self.i
        t0 = time.time(); self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(open(self.err).read()[-3000:])
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(str(r))
        return r, time.time() - t0

    def close(self):
        self.p.stdin.close(); self.p.wait(timeout=120)


def run(spec, prompts, N):
    e = Eng(spec)
    e.call(op="hello")
    out = []
    for pr in prompts:
        a, t1 = e.call(op="generate", prompt=pr, max_tokens=1, reset=True)
        b, t2 = e.call(op="generate", prompt=[], max_tokens=N, reset=False)
        out.append((a["tokens"] + b["tokens"], t1, t2))
    e.close()
    return out


if __name__ == "__main__":
    A, B = sys.argv[1], sys.argv[2]
    lens = [int(x) for x in sys.argv[3].split(",")]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    rng = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    prompts = [[rng.randrange(151936) for _ in range(L)] for L in lens]
    ra = run(A, prompts, N)
    rb = run(B, prompts, N)
    same = 0
    for L, (ta, a1, a2), (tb, b1, b2) in zip(lens, ra, rb):
        d = next((i for i in range(len(ta)) if ta[i] != tb[i]), None)
        same += d is None
        print(f"len {L}: {'identical' if d is None else f'diverge at {d}'}  A {a1*1e3:.0f}+{a2*1e3:.0f} ms  B {b1*1e3:.0f}+{b2*1e3:.0f} ms", flush=True)
    print(f"{same}/{len(lens)} identical")
