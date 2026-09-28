"""
GPS-hybrid graph transformer encoder.

Each layer combines a global attention branch with a local message-passing branch
(Rampasek et al. 2022, "Recipe for a General, Powerful, Scalable Graph Transformer"), because
partition refinement is fundamentally a local operation (boundary-vertex moves). A global-only
transformer lacks the local inductive bias the task needs. Structure also enters the
global branch through the spectral PE (StableSpectralPE).

  layer(h) = h + Attn(h)            # global
                + MPNN(h, adj)      # local (skipped if no adjacency is passed)
                + FFN(h)

Invariances preserved: the local MPNN uses adjacency + node features (both eigenbasis-free)
and is permutation-equivariant, so the whole encoder stays permutation-equivariant and, with
the stable PE, sign/basis-invariant (checked numerically below, when each repeated eigenspace
is included in full).
The local branch is adj-gated: pass no adjacency -> global-only (backward compatible).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_pe import StableSpectralPE, SpecformerPE, _LegacySignNet


class LocalMPNN(nn.Module):
    """Local mean-aggregation message passing for the GPS hybrid (permutation-equivariant)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_self = nn.Linear(d_model, d_model)
        self.W_neigh = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h, adj_indices, adj_weights, deg_inv):
        src, dst = adj_indices[0], adj_indices[1]
        # cast adjacency to h's dtype so index_add_ matches under autocast (h is bf16, adj_* fp32)
        msg = h[src] * adj_weights.unsqueeze(1).to(h.dtype)
        aggr = torch.zeros_like(h).index_add_(0, dst, msg) * deg_inv.unsqueeze(1).to(h.dtype)
        return F.gelu(self.W_self(h) + self.W_neigh(aggr))


class LanczosSpectralMix(nn.Module):
    """Multi-scale learnable spectral mixing on the low-rank eigenbasis (LanczosNet, Liao et al. 2019).

    Replaces the O(n^2) dense-attention global branch with O(m*n*d*d_model) spectral
    convolution. For m learnable filters, the global mixing of features h is
        Z_l = V diag(g_l(lambda)) V^T h            # smooth spectral conv of the features
    where (lambda, V) are the bottom-d Laplacian eigenpairs already computed for the PE (an exact
    low-rank Ritz basis; compute_eigenvectors uses ARPACK = implicitly-restarted Lanczos). g_l is a
    per-eigenvalue MLP, so one filter can realise any function of lambda. This subsumes the
    multi-scale powers R^t that LanczosNet propagates. Cost is linear in n (d = n_eigs, m small), so
    the global branch is no longer the quadratic bottleneck. Sign- and basis-invariant when each
    repeated eigenspace lies wholly inside the d selected eigenpairs: the branch acts through
    V diag(.) V^T, and g depends only on the eigenvalues, so equal eigenvalues get equal weights."""

    def __init__(self, d_model, num_filters=8, phi_hidden=64):
        super().__init__()
        self.m = num_filters
        self.phi = nn.Sequential(
            nn.Linear(1, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, num_filters))
        self.proj = nn.Linear(num_filters * d_model, d_model)

    def forward(self, h, eigvals, eigvecs):
        """h: (n, d_model); eigvals: (d,); eigvecs V: (n, d)  ->  (n, d_model)."""
        g = self.phi(eigvals.unsqueeze(-1))                              # (d, m) filter weights
        Vt_h = eigvecs.transpose(0, 1).to(h.dtype) @ h                   # (d, d_model) = V^T h
        scaled = Vt_h.unsqueeze(-1) * g.unsqueeze(1).to(h.dtype)         # (d, d_model, m) diag(g_l) V^T h
        # A single (n,d) @ (d, d_model*m) matmul, identical to einsum("nd,dcm->ncm") + reshape.
        # It never materialises the (n,d,d_model,m) intermediate, so this step is
        # O(n * d * d_model * m), linear in n.
        Z = eigvecs.to(h.dtype) @ scaled.reshape(scaled.shape[0], -1)    # (n, d_model*m) = V (.)
        return self.proj(Z)                                              # (n, d_model)


class GraphTransformerLayer(nn.Module):
    """GPS layer: pre-norm global branch (attn | lanczos) + optional local MPNN branch + FFN."""

    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.0, use_local=True,
                 global_kind="attn", num_filters=8):
        super().__init__()
        self.use_local = use_local
        self.global_kind = global_kind
        if global_kind == "attn":
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        elif global_kind == "lanczos":
            self.glob = LanczosSpectralMix(d_model, num_filters=num_filters)
        else:
            raise ValueError(f"unknown global_kind {global_kind!r}")
        self.norm_attn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(), nn.Linear(ffn_mult * d_model, d_model))
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        if use_local:
            self.local = LocalMPNN(d_model)
            self.norm_local = nn.LayerNorm(d_model)

    def forward(self, h, eigvals=None, eigvecs=None, adj_indices=None, adj_weights=None,
                deg_inv=None, key_padding_mask=None):
        x = self.norm_attn(h)                                            # h: (1, n, d)
        if self.global_kind == "attn":
            a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        else:                                                            # lanczos spectral mix
            a = self.glob(x.squeeze(0), eigvals, eigvecs).unsqueeze(0)
        h = h + self.drop(a)
        if self.use_local and adj_indices is not None:                  # local branch (adj-gated)
            hl = self.local(self.norm_local(h.squeeze(0)), adj_indices, adj_weights, deg_inv)
            h = h + self.drop(hl.unsqueeze(0))
        h = h + self.drop(self.ffn(self.norm_ffn(h)))
        return h


class SpectralGraphTransformer(nn.Module):
    """Node features + spectral PE -> GPS-hybrid transformer -> node embeddings."""

    def __init__(self, in_dim, d_model=128, n_heads=8, n_layers=4, pe_dim=32, num_filters=8,
                 pe_mode="diag", pe_kind="stable", n_eigs=16, ffn_mult=4, dropout=0.0, use_local=True,
                 global_kind="attn"):
        super().__init__()
        assert pe_kind in ("stable", "specformer", "signnet", "none"), pe_kind
        assert global_kind in ("attn", "lanczos"), global_kind
        self.pe_kind, self.n_eigs, self.use_local = pe_kind, n_eigs, use_local
        self.global_kind = global_kind
        if pe_kind == "stable":
            self.pe = StableSpectralPE(num_filters=num_filters, pe_dim=pe_dim, mode=pe_mode); pe_out = pe_dim
        elif pe_kind == "specformer":
            self.pe = SpecformerPE(pe_dim=pe_dim, num_filters=num_filters); pe_out = pe_dim
        elif pe_kind == "signnet":
            out = max(1, pe_dim // n_eigs); self.pe = _LegacySignNet(hidden=64, out=out); pe_out = out * n_eigs
        else:
            self.pe = None; pe_out = 0
        self.in_proj = nn.Linear(in_dim + pe_out, d_model)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model, n_heads, ffn_mult, dropout, use_local,
                                   global_kind=global_kind, num_filters=num_filters)
             for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x, eigvals, eigvecs, adj_indices=None, adj_weights=None, deg_inv=None):
        """x:(n,in_dim) features; eigvals:(d,); eigvecs:(n,d); optional adjacency -> (n,d_model)."""
        if self.pe_kind == "none":
            h = x
        elif self.pe_kind in ("stable", "specformer"):
            h = torch.cat([x, self.pe(eigvals, eigvecs)], dim=-1)
        else:
            V = eigvecs[:, :self.n_eigs]
            if V.shape[1] < self.n_eigs:
                V = torch.cat([V, V.new_zeros(V.shape[0], self.n_eigs - V.shape[1])], dim=-1)
            h = torch.cat([x, self.pe(V)], dim=-1)
        h = self.in_proj(h).unsqueeze(0)
        for layer in self.layers:                                       # eigvals/eigvecs used by lanczos branch
            h = layer(h, eigvals, eigvecs, adj_indices, adj_weights, deg_inv)
        return self.norm(h).squeeze(0)

    def graph_embed(self, x, eigvals, eigvecs, adj_indices=None, adj_weights=None, deg_inv=None):
        return self.forward(x, eigvals, eigvecs, adj_indices, adj_weights, deg_inv).mean(dim=0)


if __name__ == "__main__":
    import numpy as np, scipy.sparse as sp
    torch.manual_seed(0)
    n, d, in_dim = 14, 6, 5
    eigvals = torch.tensor([0.0, 1.0, 1.0, 1.0, 2.5, 3.7])      # 3-fold degenerate block
    V, _ = torch.linalg.qr(torch.randn(n, d))
    X = torch.randn(n, in_dim)

    rng = np.random.RandomState(0); row, col = [], []
    for i in range(n):
        j = (i + 1) % n; row += [i, j]; col += [j, i]
    for _ in range(n):
        a, b = rng.randint(0, n), rng.randint(0, n)
        if a != b: row += [a, b]; col += [b, a]
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n)); A.data[:] = 1.0; A.sum_duplicates(); A.data[:] = 1.0

    def gt(A):
        coo = A.tocoo()
        idx = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
        w = torch.tensor(coo.data, dtype=torch.float32)
        deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
        return idx, w, torch.tensor(1.0 / deg, dtype=torch.float32)

    adj = gt(A)
    enc = SpectralGraphTransformer(in_dim=in_dim, d_model=64, n_heads=8, n_layers=3, use_local=True).eval()
    md = lambda a, b: (a - b).abs().max().item()

    with torch.no_grad():
        glob = enc(X, eigvals, V)                                   # global-only (no adj)
        gps = enc(X, eigvals, V, *adj)                              # GPS (with local branch)
        # GPS basis + sign invariance (adj fixed; re-basis / sign-flip eigvecs)
        Qb, _ = torch.linalg.qr(torch.randn(3, 3)); Vb = V.clone(); Vb[:, 1:4] = V[:, 1:4] @ Qb
        d_basis = md(gps, enc(X, eigvals, Vb, *adj))
        d_sign = md(gps, enc(X, eigvals, V * torch.tensor([1., -1., 1., -1., 1., 1.]), *adj))
        # GPS permutation equivariance: permute nodes and relabel adjacency
        perm = torch.randperm(n); pn = perm.numpy()
        Aperm = A[pn][:, pn]
        d_perm = md(gps[perm], enc(X[perm], eigvals, V[perm], *gt(Aperm)))
        local_effect = md(glob, gps)                                # local branch must change output

    print(f"GPS output shape     : {tuple(gps.shape)}")
    print(f"local-branch effect  : {local_effect:.3e}   (>0 => local MPNN is active)")
    print(f"GPS perm-equivariance: {d_perm:.2e}   (expect ~1e-5)")
    print(f"GPS basis-invariance : {d_basis:.2e}   (expect ~1e-5)")
    print(f"GPS sign-invariance  : {d_sign:.2e}   (expect ~1e-5)")
    ok = d_perm < 1e-3 and d_basis < 1e-3 and d_sign < 1e-3 and local_effect > 1e-3
    print("PASS" if ok else "FAIL",
          "(GPS hybrid, local+global: permutation-equivariant and sign/basis-invariant)")

    # Lanczos spectral-mix backbone (global branch swapped) keeps the same invariances.
    encL = SpectralGraphTransformer(in_dim=in_dim, d_model=64, n_heads=8, n_layers=3,
                                    use_local=True, global_kind="lanczos").eval()
    has_lanczos = all(getattr(l, "global_kind", None) == "lanczos" for l in encL.layers)
    sign = torch.tensor([1., -1., 1., -1., 1., 1.])
    with torch.no_grad():
        baseL = encL(X, eigvals, V, *adj)
        QbL, _ = torch.linalg.qr(torch.randn(3, 3)); VbL = V.clone(); VbL[:, 1:4] = V[:, 1:4] @ QbL
        dL_basis = md(baseL, encL(X, eigvals, VbL, *adj))
        dL_sign = md(baseL, encL(X, eigvals, V * sign, *adj))
        permL = torch.randperm(n); Ap = A[permL.numpy()][:, permL.numpy()]
        dL_perm = md(baseL[permL], encL(X[permL], eigvals, V[permL], *gt(Ap)))
    print(f"Lanczos backbone     : global_kind=lanczos on all layers={has_lanczos}")
    print(f"Lanczos perm-equiv   : {dL_perm:.2e}   (expect ~1e-5)")
    print(f"Lanczos basis-inv    : {dL_basis:.2e}   (expect ~1e-5)")
    print(f"Lanczos sign-inv     : {dL_sign:.2e}   (expect ~1e-5)")
    okL = has_lanczos and dL_perm < 1e-3 and dL_basis < 1e-3 and dL_sign < 1e-3
    print("PASS" if okL else "FAIL",
          "(Lanczos backbone: permutation-equivariant and sign/basis-invariant)")
