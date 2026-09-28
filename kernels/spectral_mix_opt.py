"""Optimised forward paths for LanczosSpectralMix (src/graph_transformer.py).

Three implementations of the same map, verified against each other:

  eager    the module's current forward, restated here as the reference.
  rewrite  algebraic rewrite by matmul associativity. proj(V @ S) with
              proj = Linear(c*m -> c) equals V @ (S @ W^T) + b, and S @ W^T
              is (d, c): tiny and independent of n. The (n, c*m) intermediate
              Z is never materialised. Exact in real arithmetic; differs from
              eager only at floating-point rounding.
  triton   fused tall-skinny GEMM out = V @ M + b for the n-scale step,
              with the bias folded into the epilogue. GPU only; guarded import.

Shapes, repo defaults: V (n, d=16), h (n, c=128), m=8 filters.
Multiplies in the last two eager steps: n*d*(c*m) + n*(c*m)*c = 147,456 n.
After the rewrite: d*(c*m)*c (n-independent) + n*d*c = 2,048 n  (~72x fewer).
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # macOS / no GPU
    HAS_TRITON = False


def mix_eager(h, eigvals, eigvecs, phi, proj):
    """Reference: the module's forward, restated."""
    g = phi(eigvals.unsqueeze(-1))                                   # (d, m)
    Vt_h = eigvecs.transpose(0, 1).to(h.dtype) @ h                   # (d, c)
    scaled = Vt_h.unsqueeze(-1) * g.unsqueeze(1).to(h.dtype)         # (d, c, m)
    Z = eigvecs.to(h.dtype) @ scaled.reshape(scaled.shape[0], -1)    # (n, c*m)
    return proj(Z)                                                   # (n, c)


def mix_rewrite(h, eigvals, eigvecs, phi, proj):
    """Associativity rewrite: identical map, no (n, c*m) intermediate."""
    g = phi(eigvals.unsqueeze(-1))                                   # (d, m)
    V = eigvecs.to(h.dtype)
    Vt_h = V.transpose(0, 1) @ h                                     # (d, c)
    S = (Vt_h.unsqueeze(-1) * g.unsqueeze(1).to(h.dtype)).reshape(Vt_h.shape[0], -1)  # (d, c*m)
    M = S @ proj.weight.to(h.dtype).transpose(0, 1)                  # (d, c)   n-independent
    out = V @ M                                                      # (n, c)
    if proj.bias is not None:
        out = out + proj.bias.to(h.dtype)
    return out


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": bn, "BLOCK_C": bc}, num_warps=w)
            for bn in (64, 128, 256) for bc in (32, 64, 128) for w in (2, 4, 8)
        ],
        key=["n", "c", "d"],
    )
    @triton.jit
    def _vm_bias_kernel(V, M, B, Out, n, c, d,
                        sVn, sVd, sMd, sMc, sOn, sOc,
                        BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
        """Out[n, c] = V[n, d] @ M[d, c] + B[c].  d is tiny (16): one K-tile, no loop."""
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        rd = tl.arange(0, BLOCK_D)
        v = tl.load(V + rn[:, None] * sVn + rd[None, :] * sVd,
                    mask=(rn[:, None] < n) & (rd[None, :] < d), other=0.0)
        m = tl.load(M + rd[:, None] * sMd + rc[None, :] * sMc,
                    mask=(rd[:, None] < d) & (rc[None, :] < c), other=0.0)
        acc = tl.dot(v, m)
        b = tl.load(B + rc, mask=rc < c, other=0.0)
        acc = acc + b[None, :]
        tl.store(Out + rn[:, None] * sOn + rc[None, :] * sOc, acc,
                 mask=(rn[:, None] < n) & (rc[None, :] < c))

    def mix_triton(h, eigvals, eigvecs, phi, proj):
        """rewrite path with the n-scale GEMM + bias fused into one Triton kernel."""
        g = phi(eigvals.unsqueeze(-1))
        V = eigvecs.to(h.dtype).contiguous()
        Vt_h = V.transpose(0, 1) @ h
        S = (Vt_h.unsqueeze(-1) * g.unsqueeze(1).to(h.dtype)).reshape(Vt_h.shape[0], -1)
        M = (S @ proj.weight.to(h.dtype).transpose(0, 1)).contiguous()   # (d, c)
        n, d = V.shape
        c = M.shape[1]
        out = torch.empty((n, c), device=h.device, dtype=h.dtype)
        bias = proj.bias.to(h.dtype).contiguous() if proj.bias is not None \
            else torch.zeros(c, device=h.device, dtype=h.dtype)
        BLOCK_D = max(16, triton.next_power_of_2(d))
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), triton.cdiv(c, meta["BLOCK_C"]))
        _vm_bias_kernel[grid](V, M, bias, out, n, c, d,
                              V.stride(0), V.stride(1), M.stride(0), M.stride(1),
                              out.stride(0), out.stride(1), BLOCK_D=BLOCK_D)
        return out
else:
    def mix_triton(*a, **k):
        raise RuntimeError("triton not available on this machine; run bench.py on a CUDA box")
