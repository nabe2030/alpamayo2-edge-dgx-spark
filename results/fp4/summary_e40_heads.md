# Summary of the wide-data comparison (AV2, 40 scenes / 37 logs, seed 43,44,45)

## Trajectory (m, mean [log-level bootstrap 95% interval])

| Comparison | Value |
|---|---|
| BF16 vs BF16, different seed only (mean over seed pairs) | 2.856 [2.406, 3.362] |
| BF16, same CoT, different diffusion noise only (teacher forcing) | 3.268 [2.532, 4.041] |
| fp4gptq_d10 vs BF16, same seed | 0.045 [0.027, 0.070] (1.6 [0.9, 2.4] % of the BF16 seed spread) |
| fp4gptq_d10_h8 vs BF16, same seed | 0.035 [0.022, 0.052] (1.2 [0.8, 1.9] % of the BF16 seed spread) |
| fp4gptq_d10_h4 vs BF16, same seed | 0.060 [0.030, 0.106] (2.1 [1.0, 3.8] % of the BF16 seed spread) |

| Error vs. recorded drive | ADE (seed mean) | Difference (variant − BF16) | minADE (5 seeds) difference |
|---|---|---|---|
| BF16 | 3.271 [2.685, 3.953] | — | — |
| fp4gptq_d10 | 3.266 [2.680, 3.949] | -0.004 [-0.027, 0.015] | 0.011 [-0.011, 0.050] |
| fp4gptq_d10_h8 | 3.275 [2.691, 3.951] | 0.005 [-0.005, 0.017] | 0.013 [-0.010, 0.051] |
| fp4gptq_d10_h4 | 3.294 [2.712, 3.972] | 0.024 [-0.002, 0.065] | 0.033 [-0.003, 0.097] |

## Reasoning

| | CoT match at same seed | Length difference (tokens) | NOMINAL share difference | Median divergence position |
|---|---|---|---|---|
| BF16 vs BF16, different seed | 12.5 [5.0, 21.6] % | (BF16 length 20.4 [18.8, 22.0]) | (BF16 11.7 [4.4, 20.2] %) | |
| fp4gptq_d10 vs BF16 | 85.0 [77.8, 91.5] % | 0.14 [-0.20, 0.55] | 0.0 [0.0, 0.0] pt | 9.0 |
| fp4gptq_d10_h8 vs BF16 | 86.7 [80.5, 92.5] % | 0.05 [-0.21, 0.32] | 0.0 [0.0, 0.0] pt | 13.5 |
| fp4gptq_d10_h4 vs BF16 | 71.7 [62.0, 81.2] % | -0.44 [-0.84, -0.09] | 0.0 [0.0, 0.0] pt | 12.5 |

## Teacher forcing (CoT fixed to BF16 seed 43)

| Variant | KL (T=1) | KL ÷ BF16 rounding jitter (eager) | Top-1 match | Swaps | Trajectory difference ÷ diffusion noise spread |
|---|---|---|---|---|---|
| eager | 0.000078 [0.000064, 0.000092] | — | 100.00 [100.00, 100.00] % | 0/825 | 0.24 [0.17, 0.33] % |
| fp4gptq_d10 | 0.000415 [0.000331, 0.000507] | 5.33 [4.47, 6.29] | 99.34 [98.80, 99.78] % | 6/825 | 0.61 [0.45, 0.79] % |
| fp4gptq_d10_h8 | 0.000649 [0.000512, 0.000806] | 8.35 [6.68, 10.30] | 99.19 [98.58, 99.75] % | 8/825 | 0.62 [0.46, 0.80] % |
| fp4gptq_d10_h4 | 0.002009 [0.001700, 0.002319] | 25.85 [22.16, 30.10] | 98.16 [97.31, 98.95] % | 16/825 | 0.64 [0.47, 0.84] % |

## Per scene category (FP8 same-seed trajectory difference ÷ BF16 seed spread)

### motion

| Category | Count (logs) | BF16 seed spread m | fp4gptq_d10 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h8 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h4 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|---|
| left | 5 (5) | 1.751 | 4.9% [1, 8] / 87% / +0.029 / 6.1e-04 | 4.4% [1, 8] / 93% / +0.024 / 8.1e-04 | 10.1% [1, 20] / 60% / +0.124 / 2.3e-03 |
| other | 8 (7) | 2.310 | 3.0% [1, 6] / 71% / -0.031 / 5.2e-04 | 0.7% [0, 1] / 79% / +0.009 / 9.6e-04 | 1.4% [1, 3] / 58% / +0.001 / 2.4e-03 |
| right | 10 (10) | 3.321 | 1.0% [1, 1] / 80% / -0.007 / 3.7e-04 | 1.0% [1, 1] / 77% / -0.004 / 6.0e-04 | 2.3% [1, 4] / 60% / +0.030 / 2.0e-03 |
| stop | 2 (2) | 5.016 | 0.9% / 67% / +0.009 / 5.8e-04 | 0.8% / 83% / +0.005 / 8.9e-04 | 0.8% / 83% / +0.014 / 2.5e-03 |
| straight | 15 (15) | 2.918 | 0.9% [1, 1] / 98% / -0.001 / 3.0e-04 | 1.1% [1, 2] / 96% / +0.001 / 4.3e-04 | 0.9% [1, 1] / 89% / -0.000 / 1.7e-03 |

### city

| Category | Count (logs) | BF16 seed spread m | fp4gptq_d10 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h8 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h4 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|---|
| ATX | 3 (3) | 2.726 | 1.3% [1, 3] / 100% / +0.017 / 6.5e-04 | 2.6% [1, 12] / 78% / +0.027 / 9.7e-04 | 2.9% [1, 6] / 56% / +0.063 / 3.6e-03 |
| DTW | 6 (6) | 1.793 | 1.1% [1, 2] / 83% / +0.004 / 4.4e-04 | 0.6% [0, 1] / 89% / +0.002 / 6.5e-04 | 1.2% [1, 2] / 72% / +0.007 / 1.7e-03 |
| MIA | 13 (13) | 3.312 | 0.8% [1, 1] / 90% / -0.003 / 3.6e-04 | 0.8% [1, 1] / 90% / -0.001 / 4.9e-04 | 1.4% [1, 3] / 85% / +0.016 / 1.7e-03 |
| PIT | 13 (11) | 3.050 | 1.6% [1, 3] / 87% / +0.010 / 3.5e-04 | 1.5% [1, 3] / 87% / +0.010 / 6.7e-04 | 3.0% [1, 8] / 64% / +0.046 / 1.9e-03 |
| WDC | 5 (4) | 2.521 | 4.7% [2, 7] / 60% / -0.069 / 5.5e-04 | 1.3% [0, 3] / 80% / -0.006 / 8.1e-04 | 1.5% [0, 3] / 67% / -0.018 / 2.5e-03 |

### split

| Category | Count (logs) | BF16 seed spread m | fp4gptq_d10 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h8 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h4 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|---|
| train | 10 (10) | 2.897 | 1.7% [1, 4] / 87% / +0.018 / 4.1e-04 | 1.7% [1, 4] / 80% / +0.019 / 6.6e-04 | 4.7% [1, 11] / 70% / +0.100 / 2.0e-03 |
| val | 30 (27) | 2.843 | 1.5% [1, 2] / 84% / -0.012 / 4.2e-04 | 1.0% [1, 2] / 89% / -0.000 / 6.5e-04 | 1.2% [1, 2] / 72% / -0.002 / 2.0e-03 |

### nav_group

| Category | Count (logs) | BF16 seed spread m | fp4gptq_d10 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h8 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h4 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|---|
| Continue straight | 24 (22) | 2.919 | 1.4% [1, 2] / 90% / -0.014 / 3.8e-04 | 0.9% [1, 1] / 92% / +0.003 / 6.2e-04 | 1.0% [1, 1] / 82% / +0.003 / 1.9e-03 |
| Turn left | 5 (5) | 1.751 | 4.9% [1, 8] / 87% / +0.029 / 6.1e-04 | 4.4% [1, 8] / 93% / +0.024 / 8.1e-04 | 10.1% [1, 20] / 60% / +0.124 / 2.3e-03 |
| Turn right | 11 (11) | 3.221 | 1.2% [1, 2] / 73% / +0.002 / 3.9e-04 | 1.0% [1, 1] / 73% / -0.001 / 6.4e-04 | 2.3% [1, 4] / 55% / +0.025 / 2.1e-03 |

### behavior

| Category | Count (logs) | BF16 seed spread m | fp4gptq_d10 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h8 ratio / CoT match / ΔADE / KL | fp4gptq_d10_h4 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|---|
| stop | 2 (2) | 5.016 | 0.9% / 67% / +0.009 / 5.8e-04 | 0.8% / 83% / +0.005 / 8.9e-04 | 0.8% / 83% / +0.014 / 2.5e-03 |
| straight | 22 (21) | 2.805 | 1.5% [1, 3] / 88% / -0.012 / 3.9e-04 | 1.0% [1, 2] / 91% / +0.004 / 6.2e-04 | 1.0% [1, 1] / 82% / -0.002 / 1.9e-03 |
| turn_left | 4 (4) | 1.868 | 5.0% [1, 9] / 83% / +0.049 / 6.1e-04 | 4.3% [1, 9] / 92% / +0.044 / 8.0e-04 | 11.0% [1, 22] / 58% / +0.168 / 2.2e-03 |
| turn_right | 12 (12) | 2.919 | 1.1% [1, 2] / 83% / -0.010 / 3.7e-04 | 1.1% [1, 2] / 78% / -0.008 / 6.2e-04 | 2.5% [1, 5] / 56% / +0.024 / 2.0e-03 |

## Speed (single-input measurement, including ViT, runs that built a new graph excluded)

| Variant | Runs | ms per token (slope) | Time median | 90% | Max | Under 1 s | Length median / 90% / max | Length that fits in 1 s (regression) |
|---|---|---|---|---|---|---|---|---|
| base | 120 | 15.43 | 843 | 943 | 1072 | 95.0% | 21 / 27 / 36 | 31 |
| fp4gptq_d10 | 120 | 7.85 | 674 | 730 | 793 | 100.0% | 21 / 28 / 36 | 62 |
| fp4gptq_d10_h4 | 120 | 6.31 | 639 | 679 | 734 | 100.0% | 20 / 27 / 36 | 77 |
| fp4gptq_d10_h8 | 120 | 6.74 | 651 | 694 | 756 | 100.0% | 21 / 27 / 36 | 72 |

BF16 CoT length distribution (tokens, bins [0, 5, 10, 20, 30, 40, 50, 60, 80, 120, 257]): [0, 14, 23, 76, 7, 0, 0, 0, 0, 0]

Runs excluded for building a new graph: 0, max temperature 74.0°C
