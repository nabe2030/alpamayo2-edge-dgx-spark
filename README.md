# Running Alpamayo 2 Edge on Argoverse 2 at 1 Hz on a single DGX Spark (GB10)

![Official (tuned) vs. the four optimizations + FP8 body only, real time on one DGX Spark](docs/figs_av2/demo_realtime.gif)

*One Argoverse 2 drive (log 9e9bcfb7, 4–15 s) played in real time. Left: official tuned (about 1.9 s per inference). Right: the four optimizations + FP8 body only (about 0.7 s). Each prediction appears when its inference finishes; times and predictions were measured on this log. Images: Argoverse 2 (© 2021 Argo AI, LLC), CC BY-NC-SA 4.0.*

**Article:** the write-up is [`docs/article_en.md`](docs/article_en.md). A Japanese version of the article is on [Qiita](https://qiita.com/nabe2030/items/f220eb860dda549e966d).

We ran NVIDIA's driving VLA model [Alpamayo 2 Edge](https://huggingface.co/nvidia/Alpamayo2-Edge) on camera
footage from the public [Argoverse 2 Sensor](https://www.argoverse.org/av2.html) dataset (AV2). With four inference
optimizations and a custom FP8 decode kernel, we measured whether continuous 1 Hz inference fits on one NVIDIA GB10,
and compared the outputs (trajectories and reasoning text) of FP8 against BF16. This repository holds the code and the results.

It does not contain the model weights, the upstream code or any AV2 data. Get each from its distributor.

## Contents

| Folder | Contents |
|---|---|
| `kernels/fp8_gemv/` | CUDA kernel for FP8 (E4M3) weight × BF16 vector products (`fp8_gemv.cu`), build script (`build.sh`), and a standalone benchmark on synthetic data that needs neither a dataset nor the model (`bench_fp8_gemv.py`) |
| `accel/` | The four optimizations (`a2e_stagea.py`) and the FP8 weights for CoT decoding (`a2e_fp8.py`). Both patch the loaded model from outside; the upstream code is not edited |
| `av2_input/` | Scripts that build Alpamayo 2 Edge inputs (the same dict as the official loader) from AV2 logs: download, scene selection, re-rendering to virtual cameras, navigation text |
| `measure/` | Speed measurements (single input / continuous 10 Hz and 1 Hz / including re-rendering), BF16 vs FP8 output comparisons, the stopped-scene study, the input-layout comparison (`av2_eval.py`), summaries and plots |
| `results/` | Result tables and plots on AV2 (no AV2 images) |
| `results_av2_images/` | Figures that contain AV2 images (with attribution; CC BY-NC-SA 4.0; see `results_av2_images/LIST.md`) |
| `docs/` | The English article (`article_en.md`) and its figures. `docs/figs_av2/` holds article figures 6, 7 and 11 (they contain AV2 images; CC BY-NC-SA 4.0; source, modifications and license in `docs/figs_av2/README.md`). They are drawn by `measure/article_figs.py` |
| `LICENSES/` | Apache-2.0 (this code and the upstream code) and CC BY-NC-SA 4.0 (AV2) |

## The four optimizations and FP8

The numbers (#1 etc.) are our working names and match the section headings in `accel/a2e_stagea.py`. The article calls them ① to ④ (#1 = ①, #9 = ②, #2 = ③, #3 = ④).

1. **#1 Load the processor once** (`install_processor_once`): the official preprocessing calls `AutoProcessor.from_pretrained` on every call (still true on upstream main `02f55a7` as of 2026-11-20). Read it once instead. This is the same change as the upstream proposal PR #11.
2. **#9 Resize images on the GPU** (`GpuPrep`): resize, patchify and normalize the 22 images on the GPU. Resized uint8 pixels can differ from upstream by at most 1. The prompt, token ids and image grid sizes match upstream.
3. **#2 Reuse ViT results** (`VitFeatureCache`): with continuous input, a frame comes back as history at later timestamps, so reuse its earlier ViT output (keyed by camera, timestamp, grid and a pixel fingerprint).
4. **#3 Compile the ViT too** (CUDA graphs, `VisionAccel`). The official acceleration leaves the ViT eager.
- **FP8 decoding** (`a2e_fp8.py` + `kernels/fp8_gemv`): during CoT decoding (batch 1), read the weights of the language model's linear layers (the 28-layer body) as E4M3 with per-row scales. The output layer can be FP8 too (`--q all`), but that shifts the first reasoning token slightly towards "proceed", so **we recommend FP8 body only (`--q body`)**. Prefill, ViT and the trajectory expert stay in BF16.

These sit on top of the official acceleration (`alpamayo2_edge.accel`, `mode="tuned"`).

## How to run

### 1. Requirements

- Upstream code: https://github.com/NVlabs/alpamayo2-edge (we measured commit `02f55a7`) in `$A2E_ROOT/src/alpamayo2-edge`, with its `.venv` built as in the upstream README (including FlashAttention 2; on GB10 we built flash-attn 2.8.3 for sm_121).
- Model: https://huggingface.co/nvidia/Alpamayo2-Edge in `$A2E_ROOT/models/Alpamayo2-Edge` (check the terms on the model page). We measured revision `f8951d0`; main as of 2026-11-20 (`b92b6aa`) has the same weights and configs and differs only in the model card.
  Example: `hf download nvidia/Alpamayo2-Edge --local-dir $A2E_ROOT/models/Alpamayo2-Edge`
- To build the FP8 kernel: an nvcc that supports sm_121 (CUDA 12.9 or later) and ninja (e.g. `$PY -m pip install ninja`, or a ninja binary on PATH; without it the build stops with `Ninja is required to load C++ extensions`). We measured with CUDA 12.9 (matching PyTorch's cu129). The CUDA 13.0 nvcc of DGX OS 7.6.0 (`/usr/local/cuda`) also built the kernel and passed the standalone benchmark (2026-11-20).
- AV2 data: step 4 below downloads only the frames needed for each scene (about 10 MB per scene).

### 2. Environment

```bash
cp env.example.sh env.sh      # edit A2E_ROOT, A2E_WORK (and CUDA_HOME if needed)
source env.sh                 # sets PY (the upstream venv's python), PYTHONPATH, A2E_REPO, ... and cds to A2E_WORK
```

`env.sh` sets `A2E_ROOT`, `A2E_WORK`, `A2E_MODEL`, `A2E_FP8_SO` (where the kernel .so goes), `A2E_REPO` (this repository),
`PY`, `CUDA_HOME`, `PYTHONPATH` (upstream `src` plus `accel/`, `av2_input/` and `measure/` of this repository) and the compiler cache paths.

### 3. FP8 kernel (no data, no model)

```bash
PY=$PY $A2E_REPO/kernels/fp8_gemv/build.sh $(dirname $A2E_FP8_SO)    # a few minutes; stops if it already exists
$PY $A2E_REPO/kernels/fp8_gemv/bench_fp8_gemv.py $A2E_FP8_SO --out $A2E_WORK/fp8_gemv_bench.json
```

The default architectures are sm_120 / sm_121 (GB10). Change `ARCH_FLAGS` for other GPUs (FP8 conversion needs sm_89 or later).
The benchmark reports correctness (relative error against the same FP8 weights dequantized to FP32; below 0.2% is normal)
and the time of the kernel, BF16 `F.linear`, and a read-only kernel.

### 4. Inference on one AV2 scene (minimal check)

```bash
# Scene list (CSV with columns scene_id, split, log_id, t0_rel_s; t0_rel_s is seconds from the first frame of the log)
mkdir -p $A2E_WORK/data/av2 $A2E_WORK/inputs $A2E_WORK/raw/speed
printf "scene_id,split,log_id,t0_rel_s\nv_9e9bcfb7_8.5,val,9e9bcfb7-601d-3d80-bc12-ef7025174beb,8.5\n" > $A2E_WORK/scenes.csv
# Download only the needed frames from the public S3 bucket (anonymous HTTPS). The third argument is a folder of fully
# downloaded logs (an empty folder is fine)
$PY $A2E_REPO/av2_input/av2_fetch_frames.py $A2E_WORK/scenes.csv $A2E_WORK/data/av2 $A2E_WORK/data/av2/full $A2E_WORK/scenes_t0.csv 1
# Build the input (re-render to virtual cameras, navigation text from the recorded drive; layout wide_ft)
$PY $A2E_REPO/av2_input/av2_make_input.py $A2E_WORK/data/av2/val/9e9bcfb7-601d-3d80-bc12-ef7025174beb 8.5 $A2E_WORK/inputs/9e9bcfb7_8.5.pt
# Single-input timing: official tuned + the four optimizations + FP8 body only. The first run compiles for a few minutes
$PY $A2E_REPO/measure/a2e_bench2.py --model $A2E_MODEL --mode single --input pt:$A2E_WORK/inputs/9e9bcfb7_8.5.pt \
    --accel tuned --feat p,g,c,v --q body --runs 5 --label C_body_s43 --out $A2E_WORK/raw/speed/C_body_s43.jsonl
# Summary (Markdown and JSON)
$PY $A2E_REPO/measure/summarize_speed.py $A2E_WORK/raw/speed $A2E_WORK/raw/speed/summary_speed.md $A2E_WORK/raw/speed/summary_speed.json
```

`--feat` selects the optimizations (p = #1, g = #9, c = #2, v = #3) and `--q` the FP8 scope (`none` / `body` / `head` / `all`).
For the official path use `--accel tuned --feat "" --q none` (or `--accel off`).
On this scene (seed 43, 18 reasoning tokens) we measured 1927 ms for official tuned and 695 ms with the four optimizations + FP8 body only (median of 20 runs).

### 5. Full measurements

- Speed: `measure/run_speed.sh` (single / continuous), `measure/run_stream_multi.sh` (continuous over 12 logs), `measure/run_e2e.sh` (including re-rendering)
- BF16 vs FP8: `measure/run_wide.sh` (`wide_eval.py`), `measure/av2_stopgo.py` (more seeds on stopped scenes)
- Input layouts and navigation text (BF16): `measure/av2_eval.py` (its records `raw/cfg.jsonl` and `raw/samples/` are used for article figure 6)
- Article figures 6, 7 and 11: `measure/article_figs.py` (see `docs/figs_av2/README.md`)
- Summaries: `summarize_speed.py`, `summarize_multi.py`, `summarize_wide.py`, `summarize_stopgo.py`
- `run_*.sh` read `env.sh` in this repository (set `A2E_ENV=/path/to/env.sh` to use another). Put full logs in `$A2E_WORK/data/av2/val/` (`av2_input/av2_fetch.py`).

## How the AV2 inputs are built (important assumptions)

- **Camera re-mapping**: Alpamayo 2 Edge expects NVIDIA's camera rig (120° front wide, 30° front tele, sides, rears). AV2 has seven ring cameras with other headings and fields of view. Every slot is re-rendered from the ring cameras into a 1920×1080 "virtual camera" using the calibration.
  - Front and cross (side-front): 120° equidistant (f-theta)
  - Rear: 70°
  - Tele: the central 30° of the front-center camera
  - Areas no source camera covers stay black. Filling them with gray or edge colors made no difference.
- **The navigation text is derived from the recorded drive** (the ego vehicle's actual future motion), e.g. "Turn left in 10m", "Continue straight". In a real car it comes from route planning, so this leaks part of the answer. For reference we also report runs with "Continue straight" everywhere.
- **Re-rendering time is reported separately from inference.** "Inference only" numbers exclude it; numbers including it are reported separately (`results/`).
- Timestamp jitter: AV2 frame timestamps deviate from the 50 ms grid by a few ns, so "the frame at or before the requested time" uses a 1 ms tolerance.

## Environment and versions

| | |
|---|---|
| Machine | NVIDIA DGX Spark (GB10, compute capability 12.1), DGX OS 7.6.0, driver 580.178.04 |
| PyTorch | 2.8.0+cu129 (CUDA 12.9), torchvision 0.23.0, triton 3.4.0 |
| transformers | 5.17.0 |
| flash-attn | 2.8.3 (built for sm_121) |
| alpamayo2-edge | `02f55a7` (same as main on 2026-11-20) |
| Model | nvidia/Alpamayo2-Edge revision `f8951d095de3589b3face1dec3f4394c8ecedb38` (downloaded 2026-11-18; same weights and configs as main `b92b6aa` on 2026-11-20) |
| FP8 kernel | `kernels/fp8_gemv/fp8_gemv.cu`, nvcc 12.9, `-O3 --use_fast_math`, sm_120/sm_121 |
| Sampling | Same as the official `inference`: temperature 0.6, top_p 0.98, 10 diffusion steps, 1 sample |

## AV2 access and license

- Source: `s3://argoverse/datasets/av2/sensor/` (anonymous access; see https://argoverse.github.io/user-guide/getting_started.html )
- License: **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/ ). Terms of use: https://www.argoverse.org/about.html#terms-of-use
- Attribution used in the figures: `Images: Argoverse 2 (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)`
- Citation: Wilson et al., "Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting", NeurIPS Datasets and Benchmarks 2021
- **Figures containing AV2 images** (`results_av2_images/` and `docs/figs_av2/`) are adaptations of AV2 images and are released under CC BY-NC-SA 4.0 (non-commercial, share-alike), separately from the code license.
- No AV2 images, raw tensors or AV2 files are included.

## License

- **Code: Apache License 2.0** (`LICENSES/Apache-2.0.txt`). Copyright (c) 2026 [nabe2030](https://github.com/nabe2030).
- Our own code files (`.py`, `.sh`, `.cu`) start with `SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030` and `SPDX-License-Identifier: Apache-2.0`.
- Two files contain modified upstream code (each starts with the original copyright notice, nabe2030's copyright notice for the modifications, and a statement of the changes):
  - `accel/a2e_fp8.py`, `DecodeStepQ.forward`: follows the CoT decode step in upstream `alpamayo2_edge/accel/decode.py`; its linear layers call the FP8 kernel.
  - `accel/a2e_stagea.py`, `GpuPrep`: follows upstream `alpamayo2_edge/helper.py` `prepare_model_inputs` and moves image processing to the GPU.
  - Upstream: NVlabs/alpamayo2-edge, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
- Figures in `results_av2_images/` and `docs/figs_av2/`: CC BY-NC-SA 4.0 (above).
- Upstream files, model weights (NVIDIA's license) and AV2 data are not included.

## Results (AV2, one NVIDIA GB10)

Tables are in `results/summary_*.md/json` and `results/compare_3configs.md`; plots in `results/fig_*.png`.

### Single input (log 9e9bcfb7, t0 = 8.5 s, seed 43, 18 tokens, median of 20 runs)

| Setup | Time |
|---|---|
| Official, no acceleration | 2524 ms |
| Official tuned | 1927 ms (847 ms without the 1081 ms preprocessing) |
| + load the processor once | 932 ms |
| + all four optimizations | 790 ms |
| + FP8 body only (recommended) | 695 ms |
| + FP8 body + output layer (reference) | 678 ms |

### Continuous (12 logs, `compare_3configs.md`)

| Setup | Inference only, 10 Hz median / max | Inference only, 1 Hz median / max | With re-rendering, 1 Hz median / max | Over 1 s | Per token |
|---|---|---|---|---|---|
| Four optimizations (BF16) | 641 / 867 ms | 757 / 977 ms | 849 / 1077 ms | 0% inference only, 1.4% with re-rendering | 15.4 ms |
| + FP8 body only (recommended) | 528 / 685 ms | 638 / 780 ms | 736 / 878 ms | 0% | 10.0 ms |
| + FP8 body + output layer (reference) | 512 / 644 ms | 620 / 752 ms | 715 / 852 ms | 0% | 9.2 ms |

- Inference only: 1428 steps at 10 Hz and 432 steps at 1 Hz per setup. With re-rendering: 432 steps at 1 Hz.
- Reasoning length that fits in 1 s (1 Hz fit): 37 tokens (BF16), 58 (FP8 body only), 63 (FP8 body + output layer). The longest reasoning over the 12 logs was 37 tokens.

- Slow steps are explained almost entirely by reasoning length (time ≈ fixed part + per-token time × tokens).
- Re-rendering used the faster renderer with pixel-identical output (`av2_render_fast.py`): about 94 ms at 1 Hz.

### BF16 vs FP8 outputs (400 scenes × 5 seeds, `summary_wide.md`)
- With the same seed, the trajectory difference is 1.8% (FP8 body only) / 2.0% (FP8 body + output layer) of the BF16 seed-to-seed spread.
- The change in error against the recorded drive is within ±0.004 m, with no trend by maneuver or city.

### Scenes where the car should stay stopped (22 scenes × 64 seeds, plus 117 scenes × 16 seeds, `summary_stopgo.md`)
- The share of trajectories that start moving (≥ 2 m in 3 s) does not change between BF16 and FP8 (−0.1 pt, 95% CI [−0.5, +0.4]).
- With the output layer in FP8, the probability that the first reasoning token is a "proceed"-type token rises (+1.45 pt over 22 scenes, same direction in all 22). FP8 on the output layer alone gives the same direction, so the output layer is the main source.
- With FP8 body only it is +0.37 pt (−0.05 pt over the 117 scenes).
