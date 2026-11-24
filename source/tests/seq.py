"""The harness's per-round sequence on one fresh engine process: hello, warmup generate(2048, 2), then per length
generate(P, 1) + generate((), 64). Prints wall times. usage: python tests/seq.py ENGINE[:ENV=V] [LENS=256,1024,4096]"""
import sys, random, time, os
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare import Eng
e = Eng(sys.argv[1])
lens = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "256,1024,4096").split(",")]
rng = random.Random(5)
r, t = e.call(op="hello"); print(f"hello {t:.1f} s")
r, t = e.call(op="generate", prompt=[rng.randrange(151936) for _ in range(int(os.environ.get("WARM", "2048")))], max_tokens=2, reset=True); print(f"warmup {t*1e3:.1f} ms")
for L in lens:
    r, t1 = e.call(op="generate", prompt=[rng.randrange(151936) for _ in range(L)], max_tokens=1, reset=True)
    r, t2 = e.call(op="generate", prompt=[], max_tokens=64, reset=False)
    print(f"{L}: prefill {t1*1e3:.1f} ms decode {t2*1e3:.1f} ms", flush=True)
e.close()
print(open(e.err).read()[-800:])
