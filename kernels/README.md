# Speeding up the spectral mixer

`LanczosSpectralMix` (`src/graph_transformer.py`) is the model's global branch: a learned spectral
convolution on a precomputed low-rank eigenbasis of the graph Laplacian. The eigenpairs are computed
once per graph with ARPACK, and this module then runs in each layer of every forward pass.

## The rewrite

The original forward pass ends with

    Z = V @ S        # (n, d) @ (d, c*m) -> (n, c*m)
    out = proj(Z)    # Linear(c*m -> c)

With the repository's defaults (d = 16, c = 128, m = 8), c*m is 1024. Since matrix multiplication is
associative,

    proj(V @ S) = V @ (S @ W_proj^T) + b

and `S @ W_proj^T` is only (16, 128), whatever the value of n. For these two steps the number of
multiplies drops from 147,456 n to 2,048 n + 2.1 million, about 72 times fewer, and the (n, 1024)
intermediate is never built. At n = 50,000 that intermediate has 51.2 million elements, 205 MB in fp32
and 102 MB in bf16. The rewrite matches the original to within 1e-5 (relative, fp32) in
`test_correctness.py`, including on degenerate eigenspaces, where both versions keep the model's
invariance to the choice of basis.

## The Triton kernel

After the rewrite, the work that grows with n is a tall, narrow matrix product (d = 16) plus a bias.
`_vm_bias_kernel` computes `out = V @ M + b` in a single kernel, with one tile along d, and is
autotuned over block sizes and warp counts.

## Results

NVIDIA A100-PCIE-40GB, torch 2.13.0+cu130. Each figure is the median of 100 timed runs after 10
warm-up runs, in milliseconds, timed with CUDA events. TF32 was off for PyTorch's fp32 matrix
products, which is the PyTorch default. The Triton kernel's `tl.dot` used TF32, Triton's default on
the A100. The correctness tests passed on the GPU in the same job, with the Triton kernel checked
against the rewrite to 1e-3 because of TF32.

fp32:

| n | original | rewrite | compile(original) | compile(rewrite) | Triton |
|---|---|---|---|---|---|
| 1,000     | 0.296  | 0.278 | 0.297  | 0.262 | 0.399 |
| 10,000    | 0.449  | 0.270 | 0.461  | 0.301 | 0.460 |
| 50,000    | 1.296  | 0.308 | 1.313  | 0.474 | 0.337 |
| 200,000   | 4.130  | 0.551 | 4.107  | 1.011 | 0.425 |
| 1,000,000 | 20.217 | 2.053 | 20.148 | 4.346 | 1.445 |

bf16:

| n | original | rewrite | compile(original) | compile(rewrite) | Triton |
|---|---|---|---|---|---|
| 1,000     | 0.284 | 0.303 | 0.300 | 0.307 | 0.450 |
| 10,000    | 0.286 | 0.309 | 0.341 | 0.344 | 0.464 |
| 50,000    | 0.440 | 0.318 | 0.477 | 0.349 | 0.345 |
| 200,000   | 0.982 | 0.406 | 0.939 | 0.309 | 0.358 |
| 1,000,000 | 4.334 | 1.286 | 4.236 | 0.870 | 0.938 |

- `torch.compile` (default mode) applied to the original code is within 5% of the original's time, or slower, at every
  size and in both precisions, so it does not find the rewrite.
- In a profile of 20 forward passes of the original code at n = 50,000, the (n, 1024) product
  (`ampere_sgemm_32x128_tn`) takes 30.3 of the 37.5 ms of GPU time, 81%. The CUDA-event time of one
  forward pass falls 4.2 times at 50,000 nodes and 9.8 times at a million (fp32).
- The Triton kernel is slower than the rewrite up to 50,000 nodes and faster from 200,000. At a
  million nodes in fp32 it is 1.42 times faster than the rewrite and 14 times faster than the
  original, with its product in TF32. In bf16, `torch.compile` applied to the rewrite beats the kernel at the two
  largest sizes (0.870 against 0.938 ms at a million).

## Files

- `spectral_mix_opt.py`: the original computation, the rewrite, and the Triton path (GPU only)
- `test_correctness.py`: CPU tests, plus a GPU test for the kernel
- `bench.py`: sweep over n = 1k, 10k, 50k, 200k and 1M for the five variants; writes CSV and metadata
- `profile_capture.py`: the profile in `results/profile.txt`
