# Weight quantization of the refiner

Symmetric, weight-only, post-training quantization of the four shipped checkpoints (k = 4, 8, 16 and
32). The scheme is written by hand in `quantize.py` and simulated in fp32: each weight is quantized and
then dequantized.

## Error model

For b-bit symmetric quantization with scale s = max|W| / (2^(b-1) - 1), the rounding error is roughly
uniform on [-s/2, s/2], so

    SQNR ~= 12 (2^(b-1) - 1)^2 / numel * ||W||_F^2 / max|W|^2

This predicts about 6 dB lost per bit removed, and an error set by each tensor's dynamic range,
max|W| / RMS(W).

## Results

Per weight tensor (`report.py`), in all four checkpoints:

- The formula predicts the measured per-tensor SQNR to a median of 0.1 dB.
- The median SQNR is about 42 dB at int8 and 16.5 dB at int4, 6.3 dB per bit.
- SQNR ranks almost exactly by dynamic range (Spearman correlation -0.99 in every checkpoint).
- Per-channel scales add a median of 2.45 to 3.1 dB per tensor.
- The weight matrices take 0.71 to 0.72 MB in fp32. Packed, int8 would need a quarter of that and
  int4 an eighth, plus the scales.

Change in the encoder's output on fixed synthetic probe graphs with every weight quantized (`ablate.py`):

| checkpoint | int8 per tensor | int8 per channel | int4 per tensor | int4 per channel |
|---|---|---|---|---|
| k4  | 0.80% | 0.50% | 16.0% | 9.9% |
| k8  | 1.27% | 0.52% | 19.1% | 11.5% |
| k16 | 0.52% | 0.34% | 8.7% | 5.8% |
| k32 | 0.28% | 0.16% | 4.5% | 3.2% |

- Quantizing one tensor at a time (int4, per tensor), `in_proj` is the most sensitive in every
  checkpoint, changing the output by 3.0% to 13.4% on its own. Most of the next most sensitive tensors
  are in the first layer.
- The k32 checkpoint is much less sensitive than k4 and k8. These results don't show why.
- The quantized models keep the model's invariances. Relabelling the nodes, or rotating the basis
  inside a degenerate eigenspace, changes their output by less than 1e-6.

## Limitations

- The output change on probe graphs stands in for accuracy. The partitioning quality of the quantized
  models isn't measured here.
- Compared with `torch.fake_quantize_per_tensor_affine` on the 64 x 128 test tensor, one element
  differs by one quantization step, because PyTorch multiplies by 1/s where `quantize.py` divides by
  s. The check in `quantize.py` allows for this.
- Only weights are quantized, not activations.

## Files

- `quantize.py`: the quantizer, SQNR and dynamic-range statistics, and the check against PyTorch
- `report.py`: per-tensor SQNR table for a checkpoint (CSV in `results/`)
- `ablate.py`: output change with the whole model or one tensor quantized, and the invariance checks
