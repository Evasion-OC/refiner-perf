"""Benchmark the spectral-mix forward paths across n.

Run on a CUDA machine:
    python kernels/bench.py --out kernels/results/sweep.csv

Arms: eager | rewrite | compile_eager | compile_rewrite | triton
Method: 10 warmup iters, then median and IQR over 100 timed iters,
CUDA events around the op, one stream, no grad. Records device and versions.
"""
import argparse, csv, json, statistics, sys, time
import torch
sys.path.insert(0, "src")
from graph_transformer import LanczosSpectralMix
from spectral_mix_opt import mix_eager, mix_rewrite, HAS_TRITON
if HAS_TRITON:
    from spectral_mix_opt import mix_triton

NS = [1_000, 10_000, 50_000, 200_000, 1_000_000]
D, C, M = 16, 128, 8
WARMUP, ITERS = 10, 100


def time_cuda(fn):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))          # ms
    ts.sort()
    q = statistics.quantiles(ts, n=4)
    return ts[len(ts)//2], q[0], q[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kernels/results/sweep.csv")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "benchmark requires CUDA"
    dev, dtype = "cuda", getattr(torch, args.dtype)

    mod = LanczosSpectralMix(d_model=C, num_filters=M).to(dev, dtype).eval()
    ce = torch.compile(mix_eager)
    cr = torch.compile(mix_rewrite)
    arms = {"eager": mix_eager, "rewrite": mix_rewrite,
            "compile_eager": ce, "compile_rewrite": cr}
    if HAS_TRITON:
        arms["triton"] = mix_triton

    rows = []
    for n in NS:
        h = torch.randn(n, C, device=dev, dtype=dtype)
        V, _ = torch.linalg.qr(torch.randn(n, D, device=dev, dtype=torch.float32))
        V = V.to(torch.float64)               # matches the repo's fp64 eigvec convention
        ev = torch.sort(torch.rand(D, device=dev, dtype=dtype)).values
        for name, fn in arms.items():
            try:
                with torch.no_grad():
                    med, lo, hi = time_cuda(lambda: fn(h, ev, V, mod.phi, mod.proj))
                rows.append({"n": n, "arm": name, "ms_median": med, "ms_q1": lo, "ms_q3": hi})
                print(f"n={n:>9,}  {name:<16} {med:8.3f} ms  [{lo:.3f}, {hi:.3f}]")
            except torch.cuda.OutOfMemoryError:
                rows.append({"n": n, "arm": name, "ms_median": float("nan"),
                             "ms_q1": float("nan"), "ms_q3": float("nan")})
                print(f"n={n:>9,}  {name:<16} OOM")
                torch.cuda.empty_cache()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    meta = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "triton": HAS_TRITON, "dtype": args.dtype, "d": D, "c": C, "m": M,
            "warmup": WARMUP, "iters": ITERS,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    with open(args.out.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
