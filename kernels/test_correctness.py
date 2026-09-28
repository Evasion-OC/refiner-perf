"""Correctness tests for the optimised spectral-mix paths.

CPU-runnable (rewrite vs eager); the triton test self-skips without a GPU.
Tolerances (relative): the rewrite agrees with the reference to 1e-5, the reference with the module
to 1e-7, the basis-invariance test to 1e-5, and the Triton kernel with the rewrite to 1e-3 because
its product runs in TF32.
"""
import math, sys, torch
sys.path.insert(0, "src")
from graph_transformer import LanczosSpectralMix
from spectral_mix_opt import mix_eager, mix_rewrite, HAS_TRITON

torch.manual_seed(0)


def make_case(n=4096, d=16, c=128, m=8, dtype=torch.float32, degenerate=False):
    mod = LanczosSpectralMix(d_model=c, num_filters=m).to(dtype).eval()
    h = torch.randn(n, c, dtype=dtype)
    A = torch.randn(n, d, dtype=torch.float64)
    V, _ = torch.linalg.qr(A)                       # orthonormal columns, fp64 like the repo's eig path
    ev = torch.sort(torch.rand(d, dtype=dtype)).values
    if degenerate:
        ev[3:7] = ev[3]                             # a 4-fold degenerate block
    return mod, h, ev, V


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def test_rewrite_matches_eager():
    for n in (257, 4096):
        for degenerate in (False, True):
            mod, h, ev, V = make_case(n=n, degenerate=degenerate)
            with torch.no_grad():
                r = rel(mix_rewrite(h, ev, V, mod.phi, mod.proj),
                        mix_eager(h, ev, V, mod.phi, mod.proj))
            assert r < 1e-5, f"rewrite vs eager rel err {r:.2e} (n={n}, deg={degenerate})"
    print("rewrite == eager: OK")


def test_module_forward_matches_reference():
    mod, h, ev, V = make_case()
    with torch.no_grad():
        r = rel(mod(h, ev, V), mix_eager(h, ev, V, mod.phi, mod.proj))
    assert r == 0.0 or r < 1e-7, f"reference drifted from module: {r:.2e}"
    print("reference == module.forward: OK")


def test_degenerate_basis_invariance():
    """Rotating the basis inside a degenerate eigenblock must not change the output."""
    mod, h, ev, V = make_case(degenerate=True)
    blk = slice(3, 7)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    V2 = V.clone(); V2[:, blk] = V[:, blk] @ Q
    with torch.no_grad():
        for f in (mix_eager, mix_rewrite):
            r = rel(f(h, ev, V2, mod.phi, mod.proj), f(h, ev, V, mod.phi, mod.proj))
            assert r < 1e-5, f"{f.__name__} not basis-invariant: {r:.2e}"
    print("degenerate-basis invariance (both paths): OK")


def test_triton_matches_rewrite():
    if not (HAS_TRITON and torch.cuda.is_available()):
        print("triton path: SKIPPED (no CUDA)")
        return
    from spectral_mix_opt import mix_triton
    mod, h, ev, V = make_case(n=100_000)
    dev = "cuda"
    mod, h, ev, V = mod.to(dev), h.to(dev), ev.to(dev), V.to(dev)
    with torch.no_grad():
        r = rel(mix_triton(h, ev, V, mod.phi, mod.proj),
                mix_rewrite(h, ev, V, mod.phi, mod.proj))
    # On Ampere, tl.dot uses TF32 for fp32 inputs by default (10-bit mantissa)
    # while PyTorch's matmul does not, so the two paths round differently and
    # the tolerance here is 1e-3. Without TF32 the difference is below 1e-5.
    assert r < 1e-3, f"triton vs rewrite rel err {r:.2e}"
    print("triton == rewrite: OK")


if __name__ == "__main__":
    test_module_forward_matches_reference()
    test_rewrite_matches_eager()
    test_degenerate_basis_invariance()
    test_triton_matches_rewrite()
    print("all correctness tests passed")
