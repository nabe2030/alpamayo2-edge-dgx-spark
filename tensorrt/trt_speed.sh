#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# trt_speed.sh <engine_dir> <label> [runs] [checkpoint_dir]
# Stage timing of one TensorRT Edge-LLM engine on the AV2 single input (log 9e9bcfb7, t0 = 8.5 s), greedy reasoning.
#   full_miss : ViT (cache defeated) + prefill 3,261 tokens + reasoning 18 tokens -> "VLM part", comparable to PyTorch vision + prefill + decode
#   full_hit  : same, ViT outputs cached (identical images)                      -> ViT = full_miss - full_hit
#   one_miss  : ViT + prefill + 1 generated token                                -> decode per token = (full_miss - one_miss) / 17
#   The runtime caches ViT outputs by image content and the cache cannot be turned off from Python, so "miss" runs change
#   one pixel of every image per run (trt_run.py --perturb 1). Outputs are compared on the original images.
#   Input: the reference written by trt_ref_dump.py ($A2E_TRT/ref_9e9bcfb7, see README.md). GPU-only stage times come from
#   an nsys capture split with nsys_stage_split.py. Records: $A2E_TRT/raw/speed_<label>.jsonl (+ .env: temperature, other GPU processes).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${A2E_ROOT:?set A2E_ROOT}" "${A2E_WORK:?set A2E_WORK}"
source "${A2E_ENV:-$HERE/../env.sh}"   # sets PY, A2E_MODEL, A2E_TRT, A2E_EDGELLM_VENV, A2E_QUANT_VENV
E=$1; L=$2; R=${3:-20}; M=${4:-$A2E_TRT/models/a2e-reasoner}
source $A2E_EDGELLM_VENV/bin/activate
mkdir -p $A2E_TRT/raw
OUT=$A2E_TRT/raw/speed_$L.jsonl; rm -f $OUT
run(){ echo "env $(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) procs=[$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ')]" >> $A2E_TRT/raw/speed_$L.env
  python $HERE/trt_run.py --engine-dir $E --model-dir $M --ref $A2E_TRT/ref_9e9bcfb7 --mode greedy --warmup 2 --runs $R "$@" --out $OUT 2>&1 | grep -v 'INFO\]' | tail -1; }
run --perturb 1 --label full_miss
run --perturb 0 --label full_hit
run --perturb 1 --max-gen 1 --label one_miss
