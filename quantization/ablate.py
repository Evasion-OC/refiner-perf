"""Layer-wise quantization sensitivity of the deployed refiner encoder.

Metric that runs anywhere, no dataset needed: relative output drift of the
full encoder on fixed synthetic probe graphs,
    drift = ||f_q(x) - f(x)|| / ||f(x)||,
with one weight tensor quantized at a time (everything else fp32). Whole-model
rows quantize every weight tensor at once. Also checks that whole-model
quantization keeps the encoder's invariances (vertex permutation and
degenerate-eigenspace rotation).

Usage:  python quantization/ablate.py [--ckpt models/spectral_refiner_k16_eps_0_03.pt]
Writes quantization/results/ablation_<name>.csv
"""
from __future__ import annotations
import argparse, copy, csv, os, sys
import torch
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(__file__))
from graph_transformer import SpectralGraphTransformer
from quantize import fake_quantize
from report import load_state

SCHEMES = [(8, False), (8, True), (4, False), (4, True)]
torch.manual_seed(0)


def build_encoder(ckpt_path):
    """The shipped checkpoint is the full PPO policy; extract its encoder.

    Keys are prefixed encoder.* alongside actor_vertex/actor_block/critic heads.
    Drift is defined on the encoder output; the heads still appear in report.py.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    enc = SpectralGraphTransformer(
        in_dim=sd["in_dim"], d_model=sd["d_model"], n_heads=sd["n_heads"],
        n_layers=sd["n_layers"], pe_dim=sd["pe_dim"], n_eigs=sd["n_eigs"],
        use_local=True, global_kind="lanczos").eval()
    enc_state = {k[len("encoder."):]: v for k, v in sd["model_state"].items()
                 if k.startswith("encoder.")}
    enc.load_state_dict(enc_state, strict=True)
    return enc, sd


def probe_graph(n, in_dim, d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    # random sparse symmetric adjacency, ~8 edges per node
    src = torch.randint(0, n, (8 * n,), generator=g)
    dst = torch.randint(0, n, (8 * n,), generator=g)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ai = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    aw = torch.ones(ai.shape[1])
    deg = torch.zeros(n).index_add_(0, ai[1], aw).clamp_min(1)
    # Laplacian eigenpairs, bottom d, fp64 like the repo's ARPACK path
    A = torch.zeros(n, n, dtype=torch.float64)
    A[ai[0], ai[1]] = 1.0
    L = torch.diag(A.sum(1)) - A
    ev, V = torch.linalg.eigh(L)          # fp64 solve, then the pipeline convention: fp32 into the encoder
    return x, ev[:d].float(), V[:, :d].float(), ai, aw, 1.0 / deg


def quantized_copy(enc, bits, per_channel, only_layer=None):
    q = copy.deepcopy(enc)
    with torch.no_grad():
        for name, p in q.named_parameters():
            if name.endswith("weight") and p.dim() >= 2 and (only_layer in (None, name)):
                p.copy_(fake_quantize(p, bits=bits, per_channel=per_channel))
    return q


def drift(enc_q, enc, probes):
    ds = []
    with torch.no_grad():
        for x, ev, V, ai, aw, di in probes:
            a = enc(x, ev, V, ai, aw, di)
            b = enc_q(x, ev, V, ai, aw, di)
            ds.append(float((b - a).norm() / a.norm().clamp_min(1e-30)))
    return sum(ds) / len(ds)


def invariance_preserved(enc_q, probes):
    """Vertex permutation and degenerate-block rotation on the quantized model."""
    x, ev, V, ai, aw, di = probes[0]
    n = x.shape[0]
    with torch.no_grad():
        base = enc_q(x, ev, V, ai, aw, di)
        # permutation
        perm = torch.randperm(n)
        inv = torch.empty_like(perm); inv[perm] = torch.arange(n)
        ai_p = inv[ai]
        out_p = enc_q(x[perm], ev, V[perm], ai_p, aw, di[perm])
        d_perm = float((out_p - base[perm]).norm() / base.norm())
        # rotate a (near-)degenerate pair of eigvecs: force exact degeneracy first
        ev2 = ev.clone(); ev2[1] = ev2[2]
        base2 = enc_q(x, ev2, V, ai, aw, di)
        th = torch.tensor(0.7, dtype=V.dtype)
        R = torch.eye(V.shape[1], dtype=V.dtype)
        R[1, 1] = R[2, 2] = torch.cos(th); R[1, 2] = -torch.sin(th); R[2, 1] = torch.sin(th)
        out_r = enc_q(x, ev2, V @ R, ai, aw, di)
        d_rot = float((out_r - base2).norm() / base2.norm())
    return d_perm, d_rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/spectral_refiner_k16_eps_0_03.pt")
    ap.add_argument("--probes", type=int, default=3)
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()
    enc, sd = build_encoder(args.ckpt)
    probes = [probe_graph(args.n, sd["in_dim"], sd["n_eigs"], seed=100 + i)
              for i in range(args.probes)]

    rows = []
    for bits, pc in SCHEMES:
        q = quantized_copy(enc, bits, pc)
        d_all = drift(q, enc, probes)
        dp, dr = invariance_preserved(q, probes)
        rows.append({"layer": "ALL", "bits": bits, "per_channel": pc,
                     "drift": round(d_all, 6), "inv_perm": f"{dp:.2e}", "inv_rot": f"{dr:.2e}"})
        print(f"ALL  int{bits} {'pc' if pc else 'pt'}: drift {d_all:.4f}  "
              f"perm-inv {dp:.1e}  rot-inv {dr:.1e}")

    names = [n for n, p in enc.named_parameters() if n.endswith("weight") and p.dim() >= 2]
    for name in names:
        for bits, pc in [(4, False)]:                       # ranking scheme: harshest simple one
            q = quantized_copy(enc, bits, pc, only_layer=name)
            rows.append({"layer": name, "bits": bits, "per_channel": pc,
                         "drift": round(drift(q, enc, probes), 6), "inv_perm": "", "inv_rot": ""})

    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    os.makedirs("quantization/results", exist_ok=True)
    out = f"quantization/results/ablation_{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    per_layer = [r for r in rows if r["layer"] != "ALL"]
    per_layer.sort(key=lambda r: -r["drift"])
    print(f"\nmost sensitive layers under int4 per-tensor (single-layer ablation):")
    for r in per_layer[:6]:
        print(f"  {r['layer']:50s} drift {r['drift']:.4f}")
    print("full table:", out)


if __name__ == "__main__":
    main()
