"""One profiler capture of the original and the rewrite at n=50k, 20 forward passes each."""
import sys, torch
sys.path.insert(0, "src"); sys.path.insert(0, "kernels")
from graph_transformer import LanczosSpectralMix
from spectral_mix_opt import mix_eager, mix_rewrite
from torch.profiler import profile, ProfilerActivity

assert torch.cuda.is_available()
mod = LanczosSpectralMix(d_model=128, num_filters=8).cuda().eval()
h = torch.randn(50_000, 128, device="cuda")
ev = torch.sort(torch.rand(16, device="cuda")).values
V, _ = torch.linalg.qr(torch.randn(50_000, 16, device="cuda"))
V = V.double()
for fn, name in ((mix_eager, "eager"), (mix_rewrite, "rewrite")):
    with torch.no_grad():
        for _ in range(5):
            fn(h, ev, V, mod.phi, mod.proj)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(20):
                fn(h, ev, V, mod.phi, mod.proj)
            torch.cuda.synchronize()
    print("===", name)
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=8))
