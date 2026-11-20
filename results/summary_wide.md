# Summary of the wide-data comparison (AV2, 400 scenes / 250 logs, seed 43,44,45,46,47)

## Trajectory (m, mean [log-level bootstrap 95% interval])

| Comparison | Value |
|---|---|
| BF16 vs BF16, different seed only (mean over seed pairs) | 2.302 [2.158, 2.446] |
| BF16, same CoT, different diffusion noise only (teacher forcing) | 3.604 [3.303, 3.927] |
| fp8body vs BF16, same seed | 0.041 [0.030, 0.055] (1.8 [1.3, 2.4] % of the BF16 seed spread) |
| fp8all vs BF16, same seed | 0.045 [0.036, 0.055] (2.0 [1.6, 2.4] % of the BF16 seed spread) |

| Error vs. recorded drive | ADE (seed mean) | Difference (variant − BF16) | minADE (5 seeds) difference |
|---|---|---|---|
| BF16 | 3.248 [2.978, 3.525] | — | — |
| fp8body | 3.246 [2.976, 3.523] | -0.003 [-0.012, 0.005] | 0.007 [0.001, 0.014] |
| fp8all | 3.253 [2.981, 3.529] | 0.004 [-0.004, 0.013] | 0.001 [-0.008, 0.010] |

### Reference: navigation text always "Continue straight" (BF16)

- ADE 4.127 [3.752, 4.520], difference from GT navigation text 0.878 [0.642, 1.140], minADE 2.412 [2.112, 2.717], CoT same as with GT navigation text 71.2 [66.4, 75.5] %
- ADE difference by motion: left +1.61 (n=63), other +0.16 (n=80), right +2.65 (n=81), stop +0.00 (n=19), straight +0.15 (n=157)

## Reasoning

| | CoT match at same seed | Length difference (tokens) | NOMINAL share difference | Median divergence position |
|---|---|---|---|---|
| BF16 vs BF16, different seed | 23.3 [20.0, 26.6] % | (BF16 length 18.6 [17.9, 19.2]) | (BF16 20.6 [17.3, 24.1] %) | |
| fp8body vs BF16 | 91.2 [90.0, 92.5] % | -0.18 [-0.26, -0.09] | 1.2 [0.7, 1.7] pt | 10.0 |
| fp8all vs BF16 | 87.2 [85.7, 88.7] % | 0.01 [-0.07, 0.09] | 0.2 [-0.1, 0.5] pt | 8.0 |

## Teacher forcing (CoT fixed to BF16 seed 43)

| Variant | KL (T=1) | KL ÷ BF16 rounding jitter (eager) | Top-1 match | Swaps | Trajectory difference ÷ diffusion noise spread |
|---|---|---|---|---|---|
| eager | 0.000079 [0.000074, 0.000084] | — | 99.79 [99.69, 99.88] % | 18/7530 | 0.21 [0.19, 0.23] % |
| fp8body | 0.000261 [0.000249, 0.000274] | 3.28 [3.07, 3.50] | 99.56 [99.39, 99.71] % | 32/7530 | 0.51 [0.47, 0.57] % |
| fp8all | 0.000426 [0.000402, 0.000452] | 5.37 [5.06, 5.68] | 99.42 [99.25, 99.59] % | 49/7530 | 0.52 [0.47, 0.57] % |

## Per scene category (FP8 same-seed trajectory difference ÷ BF16 seed spread)

### motion

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp8all ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| left | 63 (63) | 2.283 | 1.2% [1, 2] / 90% / +0.003 / 2.4e-04 | 1.8% [1, 3] / 85% / +0.007 / 4.1e-04 |
| other | 80 (76) | 2.099 | 1.7% [1, 2] / 90% / +0.003 / 2.6e-04 | 2.5% [2, 3] / 84% / +0.009 / 4.8e-04 |
| right | 81 (75) | 2.355 | 1.2% [1, 2] / 90% / +0.008 / 2.4e-04 | 2.1% [1, 3] / 84% / +0.016 / 4.7e-04 |
| stop | 19 (16) | 1.681 | 1.3% [1, 2] / 92% / +0.008 / 2.6e-04 | 2.9% [1, 6] / 83% / +0.033 / 5.8e-04 |
| straight | 157 (105) | 2.462 | 2.4% [1, 4] / 92% / -0.015 / 2.8e-04 | 1.7% [1, 2] / 91% / -0.008 / 3.7e-04 |

### city

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp8all ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| ATX | 12 (8) | 2.607 | 0.7% [0, 1] / 88% / -0.002 / 3.1e-04 | 3.3% [1, 8] / 83% / +0.045 / 5.8e-04 |
| DTW | 55 (34) | 2.710 | 1.5% [1, 2] / 92% / -0.015 / 2.5e-04 | 2.5% [2, 4] / 86% / +0.004 / 4.2e-04 |
| MIA | 130 (81) | 2.376 | 1.7% [1, 3] / 92% / +0.007 / 2.4e-04 | 2.0% [1, 3] / 88% / -0.000 / 3.9e-04 |
| PAO | 15 (8) | 2.232 | 4.6% [1, 11] / 89% / -0.084 / 3.1e-04 | 2.6% [1, 5] / 85% / +0.035 / 4.4e-04 |
| PIT | 142 (91) | 2.122 | 1.8% [1, 3] / 91% / +0.004 / 2.7e-04 | 1.5% [1, 2] / 86% / +0.000 / 4.4e-04 |
| WDC | 46 (28) | 2.107 | 2.1% [1, 3] / 90% / -0.011 / 2.7e-04 | 1.6% [1, 3] / 90% / +0.010 / 4.5e-04 |

### split

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp8all ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| train | 100 (100) | 2.363 | 1.4% [1, 2] / 90% / +0.007 / 2.4e-04 | 2.1% [1, 3] / 85% / +0.018 / 4.4e-04 |
| val | 300 (150) | 2.282 | 1.9% [1, 3] / 92% / -0.006 / 2.7e-04 | 1.9% [2, 2] / 88% / -0.000 / 4.2e-04 |

### nav_group

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp8all ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| Continue straight | 236 (128) | 2.311 | 2.1% [1, 3] / 92% / -0.009 / 2.7e-04 | 2.0% [1, 3] / 89% / -0.001 / 4.1e-04 |
| Turn left | 73 (64) | 2.265 | 1.5% [1, 2] / 90% / +0.005 / 2.5e-04 | 1.7% [1, 3] / 85% / +0.006 / 4.3e-04 |
| Turn right | 91 (78) | 2.310 | 1.2% [1, 2] / 91% / +0.008 / 2.4e-04 | 2.1% [1, 3] / 84% / +0.017 / 4.7e-04 |

### behavior

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp8all ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| stop | 19 (16) | 1.681 | 1.3% [1, 2] / 92% / +0.008 / 2.6e-04 | 2.9% [1, 6] / 83% / +0.033 / 5.8e-04 |
| straight | 221 (135) | 2.352 | 2.2% [1, 3] / 92% / -0.009 / 2.7e-04 | 1.8% [1, 2] / 90% / -0.001 / 4.0e-04 |
| turn_left | 66 (66) | 2.348 | 1.2% [1, 2] / 89% / +0.002 / 2.5e-04 | 2.0% [1, 3] / 84% / +0.001 / 4.2e-04 |
| turn_right | 94 (87) | 2.279 | 1.2% [1, 2] / 91% / +0.007 / 2.4e-04 | 2.1% [1, 3] / 84% / +0.013 / 4.6e-04 |

## Speed (single-input measurement, including ViT, runs that built a new graph excluded)

| Variant | Runs | ms per token (slope) | Time median | 90% | Max | Under 1 s | Length median / 90% / max | Length that fits in 1 s (regression) |
|---|---|---|---|---|---|---|---|---|
| base | 2000 | 15.48 | 831 | 938 | 1282 | 98.3% | 20 / 27 / 40 | 31 |
| fp8all | 2000 | 9.17 | 699 | 761 | 942 | 100.0% | 20 / 27 / 40 | 53 |
| fp8body | 2000 | 10.30 | 722 | 793 | 925 | 100.0% | 20 / 27 / 40 | 47 |

BF16 CoT length distribution (tokens, bins [0, 5, 10, 20, 30, 40, 50, 60, 80, 120, 257]): [0, 413, 479, 1051, 56, 1, 0, 0, 0, 0]

Runs excluded for building a new graph: 0, max temperature 81.0°C
