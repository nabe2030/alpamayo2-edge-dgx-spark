#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# build_engines.sh — TensorRT Edge-LLM engines (LLM + visual) with the direct builder `trtedge build` (no ONNX):
#   fp16        : $A2E_TRT/engines/fp16         from models/a2e-reasoner
#   nvfp4_w4a4  : $A2E_TRT/engines/nvfp4_w4a4   from models/a2e-reasoner-nvfp4 (weights and activations in 4 bits)
#   nvfp4_w4a16 : $A2E_TRT/engines/nvfp4_w4a16  same checkpoint, --no-quantize-activations (weights only in 4 bits).
#                 On TensorRT 10.15 / 10.16 / 11.5 this engine gives garbage output unless pack_qkv_maxcopy.patch is
#                 applied to tensorrt_edgellm (see README.md in this folder; NVIDIA/TensorRT#4949).
#   The visual engine is FP16 in all three. Each build takes 10-40 s on GB10.
set -u
: "${A2E_TRT:?source env.sh first}"
source $A2E_EDGELLM_VENV/bin/activate
cd $A2E_TRT
C="--max-input-len 4096 --max-kv-cache-capacity 4096 --max-batch-size 1 --max-image-tokens 4096 --max-image-tokens-per-image 1024"
b(){ local E=$1; shift; [ -d engines/$E ] && { echo "engines/$E exists (not rebuilt)"; return; }
  echo "$E start $(date +%T)"; trtedge build --engine-dir engines/$E --components llm,visual $C "$@"; echo "$E rc=$? $(date +%T)"; }
b fp16 --model-dir models/a2e-reasoner
b nvfp4_w4a4 --model-dir models/a2e-reasoner-nvfp4
b nvfp4_w4a16 --model-dir models/a2e-reasoner-nvfp4 --no-quantize-activations
