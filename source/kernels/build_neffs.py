"""Build every NEFF the engine embeds, through NKI's documented route only (NKI_ARTIFACTS_DIR set before nki is
imported, the @nki.jit kernel called with numpy arrays: lists of arrays, one per rank, for the collective kernels).

usage: python kernels/build_neffs.py OUT_DIR [name ...]
  OUT_DIR/<name>.neff is written for every target; a cache under target/neffcache keyed by the kernel sources, the
  target's spec and the toolchain is reused across builds.
env: BUILD_CORES  comma list of core ranges, one per parallel job (each job needs W contiguous cores), default
                  from ENGINESMITH_NEURON_CORE: "C-(C+1),(C+2)-(C+3)"
"""
import hashlib, json, os, pickle, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import targets


def key_of(t):
    """the kernel sources the target depends on, the target itself (incl. its shapes) and the toolchain"""
    h = hashlib.sha256()
    for fn in sorted(t["deps"]):
        h.update(fn.encode()); h.update(open(os.path.join(HERE, fn), "rb").read())
    h.update(json.dumps(t, sort_keys=True).encode())
    try:
        import importlib.metadata as md
        h.update(md.version("neuronx-cc").encode()); h.update(md.version("nki").encode())
    except Exception:
        pass
    return h.hexdigest()[:20]


def build_one(t, cores, port, cache):
    k = key_of(t)
    cached = os.path.join(cache, f"{t['name']}-{k}.neff")
    if os.path.exists(cached):
        return t["name"], cached, 0.0, True
    art = os.path.join(cache, "work", t["name"])
    shutil.rmtree(art, ignore_errors=True)
    os.makedirs(art)
    env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
           "NEURON_RT_VISIBLE_CORES": cores, "NEURON_RT_ROOT_COMM_ID": f"localhost:{port}",
           "PYTHONPATH": HERE + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import pickle, sys, targets; targets.run(pickle.loads(bytes.fromhex(sys.argv[1])))")
    t0 = time.time()
    with open(os.path.join(art, "build.log"), "w") as log:
        p = subprocess.run([sys.executable, "-c", code, pickle.dumps(t).hex()], env=env, cwd=art,
                           stdout=log, stderr=subprocess.STDOUT, timeout=3500)
    dt = time.time() - t0
    if p.returncode != 0 or not os.path.exists(os.path.join(art, "kernel.neff")):
        tail = open(os.path.join(art, "build.log"), errors="replace").read()[-6000:]
        raise RuntimeError(f"build of {t['name']} failed ({dt:.0f} s) on cores {cores}:\n{tail}")
    shutil.copy(os.path.join(art, "kernel.neff"), cached + ".tmp")
    os.replace(cached + ".tmp", cached)
    shutil.rmtree(art, ignore_errors=True)
    return t["name"], cached, dt, False


def main():
    # kernel switches read from the environment are not part of the cache key: refuse to build with them set
    bad = [v for v in ("PF_NOAR", "PF_NOATT", "PG_SLAB_HW") if os.environ.get(v)]
    if bad:
        sys.exit(f"build_neffs: unset {', '.join(bad)} (timing-only kernel switches) before building the engine")
    out = sys.argv[1]
    want = sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(ROOT, "target", "neffcache")
    os.makedirs(cache, exist_ok=True)
    ts = [t for t in targets.TARGETS if not want or t["name"] in want]
    base = int(os.environ.get("ENGINESMITH_NEURON_CORE", "56"))
    pools = os.environ.get("BUILD_CORES", f"{base}-{base + 1},{base + 2}-{base + 3}").split(",")
    # jobs: sorted longest first; each worker owns one core range for its whole lifetime
    import queue, threading
    q = queue.Queue()
    for t in sorted(ts, key=lambda t: -t.get("cost", 1)):
        q.put(t)
    errs, done = [], []
    def worker(i, cores):
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            try:
                r = build_one(t, cores, 61000 + (os.getpid() % 500) * 8 + i, cache)
                done.append(r)
                print(f"  {r[0]}: {'cached' if r[3] else f'built in {r[2]:.0f} s'} ({cores})", flush=True)
            except Exception as e:
                errs.append(str(e))
                print(str(e), flush=True)
    th = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(pools)]
    for x in th: x.start()
    for x in th: x.join()
    if errs:
        sys.exit(1)
    for name, path, _, _ in done:
        shutil.copy(path, os.path.join(out, name + ".neff"))


if __name__ == "__main__":
    main()
