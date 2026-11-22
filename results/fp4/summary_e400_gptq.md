# Summary of the wide-data comparison (AV2, 400 scenes / 250 logs, seed 43,44,45,46,47)

## Trajectory (m, mean [log-level bootstrap 95% interval])

| Comparison | Value |
|---|---|
| BF16 vs BF16, different seed only (mean over seed pairs) | 2.304 [2.160, 2.447] |
| BF16, same CoT, different diffusion noise only (teacher forcing) | 3.603 [3.303, 3.925] |
| fp8body vs BF16, same seed | 0.039 [0.029, 0.053] (1.7 [1.3, 2.3] % of the BF16 seed spread) |
| fp4gptq_d10 vs BF16, same seed | 0.072 [0.052, 0.097] (3.1 [2.3, 4.1] % of the BF16 seed spread) |

| Error vs. recorded drive | ADE (seed mean) | Difference (variant − BF16) | minADE (5 seeds) difference |
|---|---|---|---|
| BF16 | 3.250 [2.979, 3.527] | — | — |
| fp8body | 3.245 [2.975, 3.522] | -0.005 [-0.014, 0.003] | 0.005 [-0.004, 0.014] |
| fp4gptq_d10 | 3.239 [2.963, 3.521] | -0.010 [-0.030, 0.010] | 0.028 [-0.008, 0.096] |

## Reasoning

| | CoT match at same seed | Length difference (tokens) | NOMINAL share difference | Median divergence position |
|---|---|---|---|---|
| BF16 vs BF16, different seed | 23.4 [20.1, 26.7] % | (BF16 length 18.6 [17.9, 19.2]) | (BF16 20.6 [17.3, 24.1] %) | |
| fp8body vs BF16 | 91.2 [89.9, 92.5] % | -0.20 [-0.29, -0.11] | 1.2 [0.7, 1.8] pt | 10.0 |
| fp4gptq_d10 vs BF16 | 89.2 [87.7, 90.5] % | 0.01 [-0.09, 0.10] | 0.0 [-0.5, 0.5] pt | 10.0 |

## Teacher forcing (CoT fixed to BF16 seed 43)

| Variant | KL (T=1) | KL ÷ BF16 rounding jitter (eager) | Top-1 match | Swaps | Trajectory difference ÷ diffusion noise spread |
|---|---|---|---|---|---|
| eager | 0.000078 [0.000073, 0.000083] | — | 99.79 [99.66, 99.89] % | 16/7541 | 0.21 [0.19, 0.24] % |
| fp8body | 0.000263 [0.000251, 0.000276] | 3.37 [3.16, 3.60] | 99.45 [99.25, 99.62] % | 39/7541 | 0.51 [0.46, 0.56] % |
| fp4gptq_d10 | 0.000386 [0.000358, 0.000415] | 4.95 [4.61, 5.30] | 99.27 [99.06, 99.47] % | 56/7541 | 0.66 [0.58, 0.74] % |

## Per scene category (FP8 same-seed trajectory difference ÷ BF16 seed spread)

### motion

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp4gptq_d10 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| left | 63 (63) | 2.283 | 1.1% [1, 2] / 92% / +0.005 / 2.4e-04 | 2.2% [2, 3] / 89% / -0.012 / 4.2e-04 |
| other | 80 (76) | 2.100 | 1.7% [1, 2] / 88% / +0.005 / 2.7e-04 | 5.7% [3, 9] / 87% / +0.046 / 4.5e-04 |
| right | 81 (75) | 2.361 | 0.9% [1, 1] / 91% / +0.004 / 2.4e-04 | 1.9% [1, 3] / 86% / -0.012 / 3.6e-04 |
| stop | 19 (16) | 1.682 | 1.2% [1, 2] / 93% / +0.006 / 2.6e-04 | 6.9% [1, 19] / 86% / -0.098 / 6.0e-04 |
| straight | 157 (105) | 2.462 | 2.4% [1, 4] / 92% / -0.019 / 2.8e-04 | 2.7% [2, 4] / 92% / -0.027 / 3.3e-04 |

### city

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp4gptq_d10 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| ATX | 12 (8) | 2.608 | 0.8% [0, 1] / 87% / -0.002 / 3.3e-04 | 2.2% [1, 4] / 92% / -0.001 / 5.6e-04 |
| DTW | 55 (34) | 2.711 | 1.3% [1, 2] / 92% / -0.013 / 2.6e-04 | 4.9% [2, 9] / 84% / +0.026 / 4.3e-04 |
| MIA | 130 (81) | 2.377 | 1.6% [1, 3] / 92% / +0.008 / 2.4e-04 | 2.7% [2, 4] / 90% / +0.001 / 3.6e-04 |
| PAO | 15 (8) | 2.231 | 4.6% [1, 11] / 91% / -0.083 / 3.1e-04 | 4.9% [1, 11] / 92% / -0.084 / 4.3e-04 |
| PIT | 142 (91) | 2.125 | 1.6% [1, 3] / 91% / -0.003 / 2.7e-04 | 2.1% [2, 3] / 90% / -0.009 / 3.7e-04 |
| WDC | 46 (28) | 2.107 | 2.1% [1, 3] / 89% / -0.013 / 2.7e-04 | 5.1% [1, 9] / 90% / -0.069 / 4.0e-04 |

### split

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp4gptq_d10 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| train | 100 (100) | 2.368 | 1.0% [1, 1] / 91% / +0.005 / 2.4e-04 | 2.2% [2, 3] / 88% / -0.015 / 3.9e-04 |
| val | 300 (150) | 2.283 | 2.0% [1, 3] / 91% / -0.008 / 2.7e-04 | 3.5% [2, 5] / 89% / -0.009 / 3.8e-04 |

### nav_group

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp4gptq_d10 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| Continue straight | 236 (128) | 2.311 | 2.2% [1, 3] / 91% / -0.011 / 2.7e-04 | 3.7% [2, 5] / 91% / -0.010 / 3.8e-04 |
| Turn left | 73 (64) | 2.265 | 1.1% [1, 1] / 92% / +0.005 / 2.5e-04 | 2.9% [2, 4] / 88% / -0.012 / 4.4e-04 |
| Turn right | 91 (78) | 2.316 | 0.9% [1, 1] / 91% / +0.004 / 2.5e-04 | 1.9% [1, 3] / 86% / -0.010 / 3.7e-04 |

### behavior

| Category | Count (logs) | BF16 seed spread m | fp8body ratio / CoT match / ΔADE / KL | fp4gptq_d10 ratio / CoT match / ΔADE / KL |
|---|---|---|---|---|
| stop | 19 (16) | 1.682 | 1.2% [1, 2] / 93% / +0.006 / 2.6e-04 | 6.9% [1, 19] / 86% / -0.098 / 6.0e-04 |
| straight | 221 (135) | 2.352 | 2.2% [1, 3] / 91% / -0.012 / 2.8e-04 | 3.2% [2, 4] / 90% / -0.013 / 3.7e-04 |
| turn_left | 66 (66) | 2.349 | 1.2% [1, 2] / 90% / +0.005 / 2.4e-04 | 4.0% [2, 7] / 89% / +0.026 / 4.2e-04 |
| turn_right | 94 (87) | 2.284 | 0.9% [1, 1] / 91% / +0.003 / 2.5e-04 | 1.9% [1, 3] / 87% / -0.011 / 3.6e-04 |

## Speed (single-input measurement, including ViT, runs that built a new graph excluded)

| Variant | Runs | ms per token (slope) | Time median | 90% | Max | Under 1 s | Length median / 90% / max | Length that fits in 1 s (regression) |
|---|---|---|---|---|---|---|---|---|
| base | 2000 | 15.55 | 833 | 940 | 1143 | 97.9% | 20 / 27 / 40 | 31 |
| fp4gptq_d10 | 2000 | 7.81 | 669 | 722 | 892 | 100.0% | 20 / 27 / 40 | 62 |
| fp8body | 2000 | 10.30 | 723 | 793 | 928 | 100.0% | 20 / 27 / 40 | 47 |

BF16 CoT length distribution (tokens, bins [0, 5, 10, 20, 30, 40, 50, 60, 80, 120, 257]): [0, 413, 474, 1056, 56, 1, 0, 0, 0, 0]

Runs excluded for building a new graph: 0, max temperature 78.0°C
