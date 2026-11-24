"""Queued us/exec of an attention NEFF at given positions (host/nrun bench mode, other inputs zero).
usage: bench_pos.py neff pos1 [pos2 ...]"""
import sys, os, subprocess, numpy as np
neff = sys.argv[1]
for p in map(int, sys.argv[2:]):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"build/bench_{os.getpid()}_{p}"); os.makedirs(d, exist_ok=True)
    np.array([[p]], np.int32).tofile(f"{d}/pos.bin")
    o = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host", "nrun"), neff, d, "1", "sync", "400"], capture_output=True, text=True).stdout
    q = [l for l in o.splitlines() if "BENCH queued" in l][0].split()[2]
    print(f"{os.path.basename(neff)} pos={p}: {q} us/exec queued", flush=True)
