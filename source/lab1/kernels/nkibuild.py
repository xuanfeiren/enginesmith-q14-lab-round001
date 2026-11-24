"""Build-time helper: trace + compile an NKI kernel to a standalone NEFF; optionally run it once
on the device (via nki's nrtpy standalone path) for a self-check. Adapted from the backend example."""
import os, shutil, time
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
import numpy as np
import ml_dtypes  # noqa
from nki.compiler.frontend import resolve_frontend_cls
from nki.framework.compiled import StandaloneKernel

bf16 = ml_dtypes.bfloat16
LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(kern, name, *args, run=None, lnc=2, outdir=None, kw=None):
    """Returns (outputs or None, compile_seconds). run=None -> env COMPILE_ONLY decides (default run)."""
    if run is None:
        run = os.environ.get("NKI_RUN_PY", "0") == "1"  # default: compile only, test via host/nrun
    art = os.path.join(LAB, "build", name)
    shutil.rmtree(art, ignore_errors=True)
    os.makedirs(art)
    k = kern[lnc] if lnc > 1 else kern
    sk = k._to_subclass(StandaloneKernel, _frontend_cls=resolve_frontend_cls(), artifacts_dir=art)
    t0 = time.time()
    if not run:
        from dataclasses import replace
        from nki.framework.compiled import compile_kernel_to_nir
        from nki.compiler.driver import _compile_bir_to_neff
        opts = replace(sk._compile_opts(), artifacts_dir=art, output_path=os.path.join(art, "kernel.neff"))
        inputs = dict(sk._bind_args(args, kw or {}))
        nir = compile_kernel_to_nir(sk, inputs=inputs, compile_opts=opts,
                                    frontend=sk._frontend_cls(enable_backend_opt=False), enable_cache=False)
        _compile_bir_to_neff(nir, opts, {k: v for k, v in inputs.items() if isinstance(v, np.ndarray)})
        out = None
    else:
        out = sk(*args, **(kw or {}))
    dt = time.time() - t0
    outdir = outdir or os.path.join(LAB, "neff")
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"{name}.neff")
    shutil.copy(os.path.join(art, "kernel.neff"), dst)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e6:.2f} MB) in {dt:.1f} s", flush=True)
    return out, dt


def nrun(neff, inputs, outputs, n_exec=1, mode="sync", bench=0, core=None, quiet=True):
    """Run a NEFF through the C++ libnrt runner (host/nrun). inputs: {name: array};
    outputs: {name: (shape, dtype)} for outputs AND for inputs whose updated value you want back.
    Returns ({name: array}, stdout)."""
    import subprocess, tempfile
    d = tempfile.mkdtemp(prefix="nrun_", dir=os.path.join(LAB, "build"))
    for k, v in inputs.items():
        np.ascontiguousarray(v).tofile(os.path.join(d, k + ".bin"))
    env = dict(os.environ)
    if core is not None:
        env["NEURON_RT_VISIBLE_CORES"] = str(core)
    assert env.get("NEURON_RT_VISIBLE_CORES", "") and int(env["NEURON_RT_VISIBLE_CORES"].split("-")[0]) >= 56
    p = subprocess.run([os.path.join(LAB, "host", "nrun"), neff, d, str(n_exec), mode, str(bench)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"nrun failed: {p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    if not quiet:
        print(p.stdout)
    res = {}
    for k, (shape, dt) in outputs.items():
        res[k] = np.fromfile(os.path.join(d, "out_" + k + ".bin"), dtype=dt).reshape(shape)
    shutil.rmtree(d, ignore_errors=True)
    return res, p.stdout


def chain(neff, inputs, outputs, n_steps, swaps, reps=1, vncs=1, keep_dir=None):
    """Run host/chain: n_steps queued executions alternating two tensor sets that swap the pairs in `swaps`
    ("pos:output_0"). outputs: {name: (shape, dtype)} of tensors to read back. Returns (dict, stdout)."""
    import subprocess, tempfile
    d = keep_dir or tempfile.mkdtemp(prefix="chain_", dir=os.path.join(LAB, "build"))
    os.makedirs(d, exist_ok=True)
    for k, v in inputs.items():
        np.ascontiguousarray(v).tofile(os.path.join(d, k + ".bin"))
    env = dict(os.environ)
    assert env.get("NEURON_RT_VISIBLE_CORES", "") and int(env["NEURON_RT_VISIBLE_CORES"].split("-")[0]) >= 56
    p = subprocess.run([os.path.join(LAB, "host", "chain"), neff, d, str(n_steps), swaps, str(reps), str(vncs)],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"chain failed: {p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    res = {k: np.fromfile(os.path.join(d, "out_" + k + ".bin"), dtype=dt).reshape(shape)
           for k, (shape, dt) in outputs.items()}
    if not keep_dir:
        shutil.rmtree(d, ignore_errors=True)
    return res, p.stdout
