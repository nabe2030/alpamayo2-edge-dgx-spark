# Speed summary (AV2, log 9e9bcfb7, wide_ft, BF16 / FP8 / FP4)

- All times in ms. e2e is wall-clock time including preprocessing (synchronized before and after). Stages are CUDA events synchronized at the boundaries.
- Re-rendering to the virtual cameras is done during input preparation and is not included in e2e (see the "Re-rendering" section below).

## Single input (t0 = 8.5 s, 1 first run + 20 steady-state runs)

| Condition | seed | Tokens | e2e median | Quartiles | Min–max | First run | preprocess | vision | prefill_llm | decode | expert | traj | ms per token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_bf16_s43 | 43 | 18 | **788.2** | 787.0–789.5 | 784.9–793.0 | 11767.3 | 17.7 | 240.6 | 176.4 | 298.4 | 51.3 | 1.3 | 16.58 |
| C_bf16_s45 | 45 | 19 | **802.6** | 801.2–805.4 | 796.1–816.5 | 12197.8 | 17.6 | 240.6 | 174.7 | 314.2 | 51.5 | 1.3 | 16.54 |
| C_fp4_gptq:0.1_s43 | 43 | 18 | **641.4** | 639.1–643.4 | 634.7–653.5 | 14664.3 | 17.6 | 241.3 | 174.5 | 152.5 | 51.1 | 1.3 | 8.47 |
| C_fp4_gptq:0.1_s45 | 45 | 19 | **663.7** | 648.5–668.3 | 645.0–691.1 | 12475.7 | 17.9 | 242.3 | 175.6 | 160.9 | 51.1 | 1.3 | 8.47 |
| C_fp4h8_gptq:0.1_s43 | 43 | 18 | **624.5** | 623.1–627.1 | 621.9–634.0 | 18839.3 | 17.4 | 240.4 | 174.7 | 135.0 | 52.6 | 1.2 | 7.50 |
| C_fp4h8_gptq:0.1_s45 | 45 | 19 | **631.7** | 631.0–632.4 | 622.2–641.2 | 13522.3 | 17.1 | 241.8 | 175.0 | 141.6 | 52.6 | 1.3 | 7.45 |
| C_fp8body_s43 | 43 | 18 | **687.9** | 686.1–690.1 | 684.1–699.6 | 12236.3 | 17.3 | 241.4 | 174.5 | 198.0 | 52.4 | 1.3 | 11.00 |
| C_fp8body_s45 | 45 | 19 | **699.3** | 697.5–701.1 | 695.2–704.4 | 11042.1 | 17.8 | 240.9 | 175.2 | 209.1 | 52.3 | 1.3 | 11.01 |

## Continuous (steps counted = t0 ≥ 4.0 s)

| Condition | Steps | Images through ViT (median) | e2e median | p10 | p90 | Max | Share over 1 s | With reasoning: median / over | NOMINAL: median / over (steps) | Cold first step | Tokens median / p90 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|

### Per-stage medians of continuous steps

| Condition | preprocess | vision | prefill_llm | decode | expert | traj |
|---|---|---|---|---|---|---|

### Fit against token count (continuous steps)

| Condition | e2e ≈ a + b × tokens | Max tokens within 1000 ms | Residual SD | Decode slope (ms per token) |
|---|---|---|---|---|
