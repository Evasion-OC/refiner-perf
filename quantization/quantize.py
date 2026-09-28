"""Hand-written weight-only symmetric fake quantization.

Written by hand rather than via torch.ao because the point is exact control
of the scheme and a readable error model. Fake quantization (quantize then
dequantize in fp32) measures the accuracy impact of a scheme exactly; it
makes no latency claim and needs no int kernels.

Error model, symmetric b-bit, per tensor:
    s = max|W| / (2^(b-1) - 1),  What = s * clamp(round(W/s))
    E[||E||_F^2] ~= numel * s^2 / 12   (rounding error ~ U[-s/2, s/2])
so SQNR ~= 12 (2^(b-1)-1)^2 / numel * ||W||_F^2 / max|W|^2 :
the error is governed by the dynamic range (max|W| against RMS), not by the spectrum.
Per-channel gives each output channel its own s, so one outlier channel
stops poisoning the rest.
"""
from __future__ import annotations
import torch


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1          # 127 for int8, 7 for int4, 3 for int3


def fake_quantize(W: torch.Tensor, bits: int = 8, per_channel: bool = False) -> torch.Tensor:
    """Symmetric fake quantization. per_channel scales along dim 0 (out-channels)."""
    Q = qmax(bits)
    if per_channel and W.dim() >= 2:
        amax = W.abs().flatten(1).amax(dim=1).clamp_min(1e-12)      # (out,)
        s = (amax / Q).view(-1, *([1] * (W.dim() - 1)))
    else:
        s = (W.abs().max().clamp_min(1e-12) / Q)
    return (W / s).round().clamp(-Q, Q) * s


def sqnr_db(W: torch.Tensor, What: torch.Tensor) -> float:
    err = (W - What).pow(2).sum()
    if err == 0:
        return float("inf")
    return float(10.0 * torch.log10(W.pow(2).sum() / err))


def dynamic_range(W: torch.Tensor) -> float:
    """max|W| / RMS(W): the outlier statistic that governs per-tensor SQNR."""
    return float(W.abs().max() / W.pow(2).mean().sqrt().clamp_min(1e-30))


def predicted_sqnr_db(W: torch.Tensor, bits: int) -> float:
    """The closed-form prediction from the uniform-noise model, per tensor."""
    Q = qmax(bits)
    return float(10.0 * torch.log10(
        torch.tensor(12.0 * Q * Q) * W.pow(2).mean() / W.abs().max().pow(2)))


def _sanity_against_torch_ao():
    """One-tensor check against torch's own fake-quant reference."""
    import torch.ao.quantization as tq  # noqa: F401  (presence check)
    torch.manual_seed(0)
    W = torch.randn(64, 128)
    s = W.abs().max() / qmax(8)
    ref = torch.fake_quantize_per_tensor_affine(W, float(s), 0, -qmax(8), qmax(8))
    ours = fake_quantize(W, bits=8, per_channel=False)
    diff = (ours - ref).abs()
    n_mismatch = int((diff > 1e-9).sum())
    max_steps = float(diff.max() / s)
    # PyTorch multiplies by 1/s where this code divides by s, so an element can land one step
    # apart. More than one step, or more than 0.1% of the elements, would be a bug.
    assert max_steps <= 1.01, f"element off by {max_steps:.2f} steps"
    assert n_mismatch <= max(1, W.numel() // 1000), f"{n_mismatch} mismatches"
    return n_mismatch, max_steps


if __name__ == "__main__":
    nm, ms = _sanity_against_torch_ao()
    print(f"check against torch: {nm} element(s) one step apart, max {ms:.2f} steps  OK")
