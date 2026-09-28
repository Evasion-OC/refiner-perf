"""Per-layer quantization report for the shipped refiner checkpoints.

For every weight tensor and every scheme in {int8, int4} x {per-tensor, per-channel}:
measured SQNR, the closed-form predicted SQNR (per-tensor only; the model's
uniform-noise assumption), dynamic range max|W|/RMS, and sizes.

Usage:  python quantization/report.py [--ckpt models/spectral_refiner_k16_eps_0_03.pt]
Writes results/report_<name>.csv and prints the summary table.
"""
from __future__ import annotations
import argparse, csv, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from quantize import fake_quantize, sqnr_db, dynamic_range, predicted_sqnr_db

SCHEMES = [(8, False), (8, True), (4, False), (4, True)]


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd


def weight_tensors(state):
    for k, v in state.items():
        if k.endswith("weight") and v.dim() >= 2:      # skip biases and norms
            yield k, v.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/spectral_refiner_k16_eps_0_03.pt")
    args = ap.parse_args()
    state = load_state(args.ckpt)
    rows = []
    for name, W in weight_tensors(state):
        row = {"layer": name, "shape": "x".join(map(str, W.shape)), "numel": W.numel(),
               "dyn_range": round(dynamic_range(W), 2),
               "pred_sqnr8_db": round(predicted_sqnr_db(W, 8), 1),
               "pred_sqnr4_db": round(predicted_sqnr_db(W, 4), 1)}
        for bits, pc in SCHEMES:
            Wq = fake_quantize(W, bits=bits, per_channel=pc)
            row[f"sqnr{bits}{'pc' if pc else 'pt'}_db"] = round(sqnr_db(W, Wq), 1)
        rows.append(row)

    total = sum(r["numel"] for r in rows)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    os.makedirs("quantization/results", exist_ok=True)
    out = f"quantization/results/report_{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    print(f"{name}: {len(rows)} weight tensors, {total:,} weight params")
    print(f"{'layer':46s} {'dynR':>5s} {'i8pt':>6s} {'i8pc':>6s} {'i4pt':>6s} {'i4pc':>6s}  (SQNR dB)")
    for r in sorted(rows, key=lambda r: r["sqnr4pt_db"])[:8]:
        print(f"{r['layer']:46s} {r['dyn_range']:5.1f} {r['sqnr8pt_db']:6.1f} "
              f"{r['sqnr8pc_db']:6.1f} {r['sqnr4pt_db']:6.1f} {r['sqnr4pc_db']:6.1f}")
    print("... (worst 8 by int4 per-tensor SQNR; full table in", out + ")")
    fp32_mb = total * 4 / 1e6
    print(f"weights on disk: fp32 {fp32_mb:.2f} MB -> int8 {fp32_mb/4:.2f} MB -> int4 {fp32_mb/8:.2f} MB")


if __name__ == "__main__":
    main()
