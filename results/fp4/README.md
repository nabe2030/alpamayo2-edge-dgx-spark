# Results of the FP4 / TensorRT Edge-LLM article (AV2, one NVIDIA GB10)

All numbers were measured on one DGX Spark (GB10) on 2026-11-21 and 2026-11-22, with the versions listed in
[`docs/article_fp4_en.md`](../../docs/article_fp4_en.md) section 2. No AV2 images are included here.

## Files

| File | Content | Made by |
|---|---|---|
| `summary_speed_fp4.md/json` | Single input (log 9e9bcfb7, t0 = 8.5 s, seeds 43 / 45, 20 runs): BF16 (four optimizations), FP8 body, FP4 body (GPTQ), FP4 body + FP8 output layer | `measure/run_speed_fp4.sh`, `measure/summarize_speed.py` |
| `summary_stream_fp4.md/json` | Continuous inference on 12 val logs, 10 Hz (1,428 steps) and 1 Hz (432 steps) | `measure/run_speed_fp4.sh`, `measure/summarize_multi.py` |
| `summary_e40_methods.md/json` | 40 scenes × 3 seeds: FP8 body and the FP4 rounding methods (article 4.1, figure 5) | `measure/run_quality_fp4.sh`, `measure/summarize_wide.py` |
| `summary_e400_gptq.md/json` | 400 scenes × 5 seeds: FP8 body and FP4 body (GPTQ) against BF16 (article 4.3) | same |
| `summary_e40_heads.md/json` | 40 scenes × 3 seeds: FP4 body (GPTQ) with the output layer in BF16 / FP8 / FP4 (reference) | same |
| `summary_firsttoken_fp4.md` | First reasoning token in the 139 stop/go scenes (article 4.3, figure 6) | `measure/av2_firsttoken.py`, `measure/summarize_firsttoken.py` |
| `greedy20_summary.jsonl` | Greedy reasoning on 20 scenes, PyTorch with quantized weights (fake quantization) vs. PyTorch BF16 | `measure/greedy_fakeq.py`, `measure/cmp_greedy.py` |
| `fp4_gemv_bench.json` | FP4 / FP8 / BF16 decode GEMVs on their own (synthetic weights of the real shapes) | `kernels/fp4_gemv/bench_fp4_gemv.py` |
| `calib_logs_48.csv` | The 48 AV2 train logs used for the GPTQ calibration (none of them is in the 400 evaluation scenes): log ID, t0 (s from the start of the log) and motion kind | `measure/select_calib_scenes.py` |
| `calib_logs_48_heldout.csv` | 48 other train logs, used only to check the calibration (held out); same columns | same, excluding the logs above |

In the tables, `fp4gptq_d10` is the FP4 body with GPTQ (damping 0.1), the recommended configuration; `fp4rtn` is plain
rounding, `fp4mse` the block-scale search, `fp4gptqm_d10` GPTQ + scale search, `_h8` / `_h4` the output layer in FP8 / FP4.
In `greedy20_summary.jsonl`, labels without "decode-only" use the quantized weights in prefill too; "decode-only" uses
them only for the decode steps (prefill in BF16), which is what the FP4 kernel path does.

## Rounding error on the real weights (28 layers × qkv / o / fc1 / fc2, `measure/fp4_qerr.py`)

Output error = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ), mean over the 112 projections. H = Σ x xᵀ of the projection inputs during BF16
reasoning decoding (5,468 tokens on the calibration logs; the held-out H comes from the other 48 logs).

| Method | Weight error | Output error, calibration H | Output error, held-out H |
|---|---|---|---|
| Plain rounding (RTN) | 0.898% | 0.299% | 0.298% |
| Scale search, unweighted | 0.737% | 0.250% | 0.249% |
| Scale search, weighted by diag(H) | 0.980% | 0.153% | 0.154% |
| GPTQ, damping 0.01 | 3.687% | 0.015% | 0.049% |
| **GPTQ, damping 0.1 (adopted)** | 3.050% | 0.022% | **0.046%** |
| GPTQ, damping 0.5 | 2.892% | 0.035% | 0.054% |
| GPTQ + scale search, damping 0.1 | 2.902% | 0.014% | 0.030% |

GPTQ for the 28 layers takes 65 s on GB10 (GPTQ + scale search: 177 s).

## TensorRT Edge-LLM (VLM part only, `tensorrt/`)

TensorRT Edge-LLM 0.12.0, TensorRT 10.16.1.11, ModelOpt 0.45.0. Same input as the single-input measurement (log 9e9bcfb7,
t0 = 8.5 s, prompt 3,261 tokens, 22 images, greedy reasoning of 18 tokens). The W4A16 rows use `tensorrt/pack_qkv_maxcopy.patch`.

### Stages (ms)

| Configuration | ViT | Prefill | Reasoning, per token | VLM part, GPU | VLM part, including CPU work |
|---|---|---|---|---|---|
| PyTorch BF16 (four optimizations) | 241 | 176 | 16.6 | about 716 | — |
| PyTorch FP8 body (custom kernel) | 243 | 178 | 11.0 | about 619 | — |
| PyTorch FP4 body (custom kernel) | 241 | 175 | 8.47 | about 569 | — |
| TensorRT FP16 | 187 | 161 | 17.3 | 650 | 758 |
| TensorRT NVFP4 W4A4 | 188 | 104 | 9.98 | 470 | 578 |
| TensorRT NVFP4 W4A16 + workaround | 210 | 184 | 9.83 | 574 | 685 |

- PyTorch rows: sum of the per-stage CUDA events (`a2e_bench2.py`). TensorRT rows: GPU windows of the engine calls from an
  nsys capture (`tensorrt/nsys_stage_split.py`); the last column is the wall clock of one request from Python
  (`tensorrt/trt_speed.sh`), which adds about 110 ms of CPU work in the runtime (image intake, tokenization, ...).
  Preprocessing and the expert are not included on either side. The two ways of measuring differ, so compare totals as a guide only.
- Matrix products of the body, from the nsys capture: in prefill W4A4 (CUTLASS FP4 × FP4) 32.5 ms vs. W4A16 (Marlin) 124.2 ms;
  per decode token 5.98 ms (W4A4) vs. 5.16 ms (W4A16). Our FP4 kernel reads the body in about 3.6 ms per token.

### Greedy reasoning vs. PyTorch BF16 (20 scenes)

| Configuration | Identical reasoning | Same top-1 at step 0 | KL per step (over the PyTorch top 20), median / mean |
|---|---|---|---|
| TensorRT FP16 | 18 / 20 | 19 / 20 | 3.3e-5 / 2.9e-4 |
| TensorRT NVFP4 W4A4 | 10 / 20 | 17 / 20 | 1.8e-3 / 6.6e-3 |
| TensorRT NVFP4 W4A16, as released | 0 / 20 | 0 / 20 | 2.35 (first step) |
| TensorRT NVFP4 W4A16 + workaround | 13 / 20 | 18 / 20 | 9.4e-4 / 3.7e-3 |
| (reference) PyTorch FP4 GPTQ, decode only | 18 / 20 | 20 / 20 | 8.3e-5 / 6.1e-4 |
| (reference) PyTorch FP4 plain rounding, prefill too | 14 / 20 | 18 / 20 | 1.1e-3 / 3.9e-3 |

The TensorRT side bans only the text end-of-sequence token (`logit_bias` holds at most 1,024 ids, so the 4,000 trajectory
tokens cannot be banned); the PyTorch side applies Alpamayo's masks. The reference rows are in `greedy20_summary.jsonl`.

### The W4A16 bug: KV cache right after prefill, cosine against the FP16 engine

| Engine | Layer 0 K | Layer 0 V | Layer 13 K / V | Layer 27 K / V |
|---|---|---|---|---|
| NVFP4 W4A4 | 0.9996 | 0.9768 | 0.9982 / 0.9860 | 0.9964 / 0.9863 |
| NVFP4 W4A16, as released | −0.0002 | 0.0017 | 0.23 / 0.01 | 0.15 / 0.02 |
| NVFP4 W4A16 + workaround | 0.9998 | 0.9837 | 0.9993 / 0.9951 | 0.9979 / 0.9928 |

TensorRT-only reproducer (`tensorrt/w4a16_bug/issue_4949_repro.py`), maximum absolute error of `[x, 2x]`:

| TensorRT | M = 1 | M = 2 | M = 33 | M = 512 |
|---|---|---|---|---|
| 10.15.1.29 | 0 (correct) | 7.5 | 8.9 | 10.2 |
| 10.16.1.11 | 0 (correct) | 5.6 | 8.4 | 9.2 |
| 11.5.0.63 | 0 (correct) | 8.0 | 7.9 | 9.9 |

The errors for M ≥ 2 vary from run to run with the random input (for example 6.8 / 9.7 / 9.4 in another run on 10.16.1.11);
M = 1 is always exact.
