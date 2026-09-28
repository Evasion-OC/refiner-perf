"""
Stable spectral positional encoding: sign-invariant, and basis-invariant when each repeated
eigenspace is included in full.

Replaces the element-wise SignNet (phi(v) + phi(-v) per eigenvector), which is
sign-invariant but not basis-invariant: under an orthogonal re-basis of a degenerate eigenspace its
output changes, so on CFI / spectrally-blind families it realises no invariant
separation above 1-WL.

Construction (SPE-style; Huang et al. 2024 "Stable & Expressive PEs"; Lim et al. 2023
BasisNet; multi-filter spectral conv a la LanczosNet, Liao et al. 2019). Given Laplacian
eigenpairs (lambda_l, v_l) stacked as eigvals (d,) and eigvecs V (n, d):

    Phi = phi(lambda)                      # (d, m)  per-eigenvalue filter weights
    W_l = V diag(Phi[:, l]) V^T            # (n, n)  smooth spectral filter of L
    PE  = rho( per-node readout of {W_l} ) # (n, pe_dim)

Two modes:
  - "diag" (default, O(n d m)): per-node feature is diag(W_l)_i = sum_d v_{i,d}^2 Phi[d,l]
    (a learnable heat-kernel-signature). Scalable; never materialises an n x n matrix.
  - "full" (O(m n^2)): materialises W_l and reads [diag, max|.|, l2-norm] per row,
    capturing off-diagonal structure. For small graphs / expressiveness probes.

Invariances (checked numerically in __main__):
  sign       v_l -> -v_l               : invariant (outer-product / squared dependence)
  basis      orthogonal re-basis of a  : invariant when that eigenspace lies wholly inside
             degenerate eigenspace        the d selected eigenpairs. Phi gives equal
                                          eigenvalues equal weights, so a complete eigenspace
                                          enters only through its eigenprojector, which is
                                          basis-free. The code keeps exactly d eigenpairs, so
                                          the cutoff can split an eigenspace.
  perm       node permutation          : equivariant (V row-permutes with the nodes)
  stability  lambda -> lambda + eps    : Lipschitz (smooth filter phi)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class StableSpectralPE(nn.Module):
    def __init__(
        self,
        num_filters: int = 8,
        pe_dim: int = 32,
        phi_hidden: int = 64,
        rho_hidden: int = 64,
        mode: str = "diag",
    ):
        super().__init__()
        assert mode in ("diag", "full"), mode
        self.mode = mode
        self.m = num_filters
        # per-eigenvalue filter MLP: scalar lambda -> m filter weights
        self.phi = nn.Sequential(
            nn.Linear(1, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden), nn.ReLU(),
            nn.Linear(phi_hidden, num_filters),
        )
        feat_per_filter = 1 if mode == "diag" else 3
        self.rho = nn.Sequential(
            nn.Linear(num_filters * feat_per_filter, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, pe_dim),
        )

    def forward(self, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        """eigvals: (d,)  eigvecs: (n, d)  ->  PE: (n, pe_dim)"""
        Phi = self.phi(eigvals.unsqueeze(-1))                 # (d, m)
        if self.mode == "diag":
            # diag(V diag(Phi_l) V^T)_i = sum_d v_{i,d}^2 * Phi[d, l]
            S = (eigvecs ** 2) @ Phi                          # (n, m)
        else:
            # W[l] = (V * Phi[:, l]) @ V^T   -> (m, n, n)
            VW = eigvecs.unsqueeze(0) * Phi.transpose(0, 1).unsqueeze(1)   # (m, n, d)
            W = VW @ eigvecs.transpose(0, 1).unsqueeze(0)                  # (m, n, n)
            diag = torch.diagonal(W, dim1=1, dim2=2).transpose(0, 1)       # (n, m)
            amax = W.abs().amax(dim=2).transpose(0, 1)                     # (n, m)
            l2 = W.norm(dim=2).transpose(0, 1)                             # (n, m)
            S = torch.cat([diag, amax, l2], dim=1)                         # (n, 3m)
        return self.rho(S)                                                 # (n, pe_dim)


class SpecformerPE(nn.Module):
    """Specformer-style set-to-set spectral PE (Bo, Shi, Wang, Liao, ICLR 2023, arXiv:2303.01028).

    Generalises StableSpectralPE's per-eigenvalue *independent* scalar filter to a filter
    conditioned on the whole spectrum. A Transformer over the eigenvalue set (each eigenvalue attends
    to all others) produces spectrum-aware filter weights g (d, m); the per-node readout reuses the
    same basis-invariant diagonal form S = (V^2) @ g, so it drops into the existing PE contract
    (eigvals (d,), eigvecs (n, d)) -> (n, pe_dim).

    Invariances (checked numerically in __main__):
      sign   v_l -> -v_l            : invariant ((V^2) readout)
      basis  re-basis degenerate    : invariant when that eigenspace lies wholly inside the d
             eigenspace                selected eigenpairs. The eigenvalue encoding rho(lambda)
                                       and the self-attention carry no token-order positional
                                       encoding, so equal eigenvalues map to equal g, and
                                       (V^2) @ g uses only the eigenprojector diagonal
                                       (basis-free)
      perm   node permutation       : equivariant (V row-permutes with the nodes)
      stab   lambda -> lambda + eps : Lipschitz (smooth sinusoidal rho + smooth transformer)
    """

    def __init__(self, pe_dim: int = 32, num_filters: int = 8, n_freq: int = 16, d_attn: int = 32,
                 n_heads: int = 4, n_layers: int = 2, rho_hidden: int = 64):
        super().__init__()
        self.m = num_filters
        # fixed sinusoidal eigenvalue encoding: a function of lambda only (equal lambda -> equal code)
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, 6.0, n_freq)))
        self.lift = nn.Linear(2 * n_freq + 1, d_attn)
        enc = nn.TransformerEncoderLayer(d_attn, n_heads, dim_feedforward=2 * d_attn,
                                         dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.to_filter = nn.Linear(d_attn, num_filters)
        self.rho = nn.Sequential(
            nn.Linear(num_filters, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, pe_dim),
        )

    def _encode_eig(self, eigvals: torch.Tensor) -> torch.Tensor:        # (d,) -> (d, 2*n_freq+1)
        ang = eigvals.unsqueeze(-1) * self.freqs.unsqueeze(0)            # (d, n_freq)
        return torch.cat([eigvals.unsqueeze(-1), torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, eigvals: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        """eigvals: (d,)  eigvecs: (n, d)  ->  PE: (n, pe_dim)"""
        e = self.lift(self._encode_eig(eigvals))                         # (d, d_attn)
        e = self.tf(e.unsqueeze(0)).squeeze(0)                           # (d, d_attn) set-to-set
        g = self.to_filter(e)                                            # (d, m) spectrum-conditioned
        S = (eigvecs ** 2) @ g                                           # (n, m) basis-invariant readout
        return self.rho(S)                                              # (n, pe_dim)


# Legacy element-wise SignNet, inlined for the contrast test.
class _LegacySignNet(nn.Module):
    def __init__(self, hidden: int = 32, out: int = 8):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, V: torch.Tensor) -> torch.Tensor:  # V: (n, d)
        return torch.cat(
            [self.phi(V[:, i:i + 1]) + self.phi(-V[:, i:i + 1]) for i in range(V.shape[1])],
            dim=-1,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 12, 6
    # eigenvalues with a 3-fold degenerate block (value 1.0) at indices {1,2,3}
    eigvals = torch.tensor([0.0, 1.0, 1.0, 1.0, 2.5, 3.7])
    V, _ = torch.linalg.qr(torch.randn(n, d))      # orthonormal eigenvector columns

    def maxdiff(a, b):
        return (a - b).abs().max().item()

    sign_mask = torch.tensor([1., 1., -1., 1., -1., 1.])
    Qb, _ = torch.linalg.qr(torch.randn(3, 3))     # re-basis of the degenerate block

    print("module                sign-flip   basis-rotate   node-perm   eig-perturb(1e-3)")
    for mode in ("diag", "full"):
        pe = StableSpectralPE(num_filters=8, pe_dim=16, mode=mode).eval()
        with torch.no_grad():
            base = pe(eigvals, V)
            d_sign = maxdiff(base, pe(eigvals, V * sign_mask))
            Vb = V.clone(); Vb[:, 1:4] = V[:, 1:4] @ Qb
            d_basis = maxdiff(base, pe(eigvals, Vb))
            perm = torch.randperm(n)
            d_perm = maxdiff(base[perm], pe(eigvals, V[perm]))
            d_stab = maxdiff(base, pe(eigvals + 1e-3 * torch.randn(d), V))
        print(f"StableSpectralPE[{mode:4s}]  {d_sign:.2e}    {d_basis:.2e}     {d_perm:.2e}    {d_stab:.2e}")

    # SpecformerPE (set-to-set spectral filter) has the same invariances as StableSpectralPE
    spe = SpecformerPE(pe_dim=16, num_filters=8).eval()
    with torch.no_grad():
        base = spe(eigvals, V)
        sp_sign = maxdiff(base, spe(eigvals, V * sign_mask))
        Vb = V.clone(); Vb[:, 1:4] = V[:, 1:4] @ Qb
        sp_basis = maxdiff(base, spe(eigvals, Vb))
        perm = torch.randperm(n)
        sp_perm = maxdiff(base[perm], spe(eigvals, V[perm]))
        sp_stab = maxdiff(base, spe(eigvals + 1e-3 * torch.randn(d), V))
    print(f"SpecformerPE           {sp_sign:.2e}    {sp_basis:.2e}     {sp_perm:.2e}    {sp_stab:.2e}")
    spec_ok = sp_sign < 1e-3 and sp_basis < 1e-3 and sp_perm < 1e-3 and sp_stab < 1e-1

    sn = _LegacySignNet().eval()
    with torch.no_grad():
        b = sn(V)
        Vb = V.clone(); Vb[:, 1:4] = V[:, 1:4] @ Qb
        sn_sign = maxdiff(b, sn(V * sign_mask))
        sn_basis = maxdiff(b, sn(Vb))
    print(f"SignNet (legacy)       {sn_sign:.2e}    {sn_basis:.2e}     (basis != 0: not basis-invariant)")
    print("\nExpected: basis-rotate near 0 for StableSpectralPE (basis-invariant), "
          "well above 0 for legacy SignNet.")
    print("PASS" if spec_ok else "FAIL",
          "(SpecformerPE: sign and basis invariance, permutation equivariance, stability)")
