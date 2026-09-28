# refiner-perf

Performance and quantization work on the graph transformer from
[multilevel-partition-refinement](https://github.com/Evasion-OC/multilevel-partition-refinement).
The timings were measured on an NVIDIA A100-PCIE-40GB. The model code in `src/` and the four
checkpoints in `models/` come from that repository.

## Speeding up the spectral mixer (`kernels/`)

The spectral mixer runs in every layer of every forward pass. Because matrix multiplication is
associative, its output projection can be applied before the product with the eigenvectors instead of
after it. That avoids building an (n, 1024) intermediate and cuts the multiplies in that part of the
layer by about 72 times. A fused Triton kernel then handles the remaining tall, narrow product.

| nodes (fp32, ms) | original | rewrite | torch.compile on the original | Triton kernel |
|---|---|---|---|---|
| 50,000 | 1.296 | 0.308 | 1.313 | 0.337 |
| 1,000,000 | 20.217 | 2.053 | 20.148 | 1.445 |

The rewrite is 4.2 times faster than the original at 50,000 nodes and 9.8 times faster at a million.
`torch.compile` applied to the original code does not find it. Profiling shows the product the rewrite
removes took 81% of the original GPU time. The Triton kernel is slower than the rewrite up to 50,000
nodes and faster from 200,000 on, by 1.42 times at a million. Its matrix product ran in TF32, Triton's
default on the A100, while the PyTorch versions ran in strict fp32, so that comparison is not like for
like. `kernels/README.md` has the full fp32 and bf16 results.

These timings are for the mixer alone, at the class defaults (16 eigenvectors, 128 channels). The
trained checkpoints are smaller and refine graphs of at most 4,000 nodes, so this is not a speed-up of
the partitioner.

## Quantization (`quantization/`)

Simulated int8 and int4 weight quantization of the four checkpoints, per tensor and per channel,
written by hand. The int8 per-tensor case was checked against `torch.fake_quantize_per_tensor_affine`.
The closed-form error model predicts the measured per-tensor SQNR to a median of 0.1 dB. At int8 the
encoder's output changes by at most 1.3%. At int4 it changes by 4.5% to 19% with per-tensor scales and
by 3.2% to 11.5% with per-channel scales. The model's invariances survive quantization to within
1e-6. `quantization/README.md` has the details.

## Running it

`submit_a100.sbatch` runs the correctness tests, both benchmark sweeps and the profile on one A100
under SLURM, then the quantization ablations of three of the four checkpoints (k16 was run
separately). The ablations run on the CPU. Its partition and environment lines are for the
cluster it ran on. Without a GPU, `python kernels/test_correctness.py` runs the CPU tests (the GPU test
skips itself) and `python quantization/quantize.py` runs the quantizer check.
