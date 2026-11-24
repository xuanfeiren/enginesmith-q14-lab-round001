"""Sweep: in-NEFF all_reduce latency/bandwidth. For each (kind, cores, bytes) build n=1 and n=N chains, check
correctness in the build run (nrtpy) and in C++ (host/ccrun, one scheduling thread per rank), time queued execs.
per-op = (t_N - t_1) / (N - 1). Results appended to build/sweep_cc.txt."""
import os, sys, json
import numpy as np
import nb

N = int(os.environ.get('N', 33))
def one(kind, cores, nbytes):
    W = len(nb.parse_cores(cores))
    P = 128 if nbytes >= 128 * 4 * 8 else 1
    F = nbytes // (P * 4)
    res = {}
    for n in (1, N):
        name = f"ar_{kind}_c{cores.replace(',', '_')}_{nbytes}_n{n}"
        try:
            fn = "p1_cc:run2" if kind in ("perm", "ag") else "p1_cc:run"
            chk, dt = nb.build(name, fn, cores, kind, W, P, F, n)
        except Exception as e:
            return {"error": str(e).splitlines()[-1][:300] if str(e) else repr(e), "msg": str(e)[-1500:]}
        rng = np.random.default_rng(1)
        xs = [rng.integers(-8, 8, size=(P, F)).astype(np.float32) for _ in range(W)]
        os.environ["CC_MT"] = "1"
        os.environ["CC_REPS"] = os.environ.get("CC_REPS", "5")
        OF = F * W if kind == "ag" else F
        outs, so = nb.ccrun(name, [{"x": x} for x in xs], {"output_0": ((P, OF), np.float32)}, n_exec=1,
                            bench=200 if nbytes <= (1 << 20) else 20, cores=cores)
        if kind == "perm":
            refs = [xs[(r - n) % W] for r in range(W)]
        elif kind == "ag":
            refs = [np.concatenate(xs, 1) if n == 1 else np.concatenate([xs[0]] * W, 1)] * W
        else:
            refs = [sum(xs) * (W ** (n - 1))] * W
        err = max(float(np.abs(o["output_0"] - rf).max()) for o, rf in zip(outs, refs))
        us = float([l for l in so.splitlines() if l.startswith("BENCH")][0].split()[2])   # min over CC_REPS
        res[n] = {"build_check": chk["max_err_per_rank"], "cpp_err": err, "us": us}
    per = (res[N]["us"] - res[1]["us"]) / (N - 1)
    return {"t1": res[1]["us"], f"t{N}": res[N]["us"], "per_op_us": per, "GBps_algbw": nbytes / per / 1e3,
            "errs": [res[1]["cpp_err"], res[N]["cpp_err"]], "build_errs": [res[1]["build_check"], res[N]["build_check"]]}

if __name__ == "__main__":
    kind = sys.argv[1]
    for cores in sys.argv[2].split(";"):
        for nbytes in [int(x) for x in sys.argv[3].split(",")]:
            r = one(kind, cores, nbytes)
            line = json.dumps({"kind": kind, "cores": cores, "bytes": nbytes, **r})
            print(line, flush=True)
            open(os.path.join(nb.LAB, "build", os.environ.get("OUT", "sweep_cc.txt")), "a").write(line + "\n")
