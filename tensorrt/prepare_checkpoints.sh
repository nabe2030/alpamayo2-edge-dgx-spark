#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# prepare_checkpoints.sh — checkpoints of the Alpamayo 2 Edge VLM part for TensorRT Edge-LLM (see tensorrt/README.md).
#   1. FP16 path: rename the VLM part to the Cosmos3-Edge (reasoner) layout the Edge-LLM builder reads (expert dropped,
#      tensors copied byte for byte)                                                       -> $A2E_TRT/models/a2e-reasoner
#   2. NVFP4 path: the VLM part in the Transformers layout (only "vlm." dropped)            -> $A2E_TRT/models/a2e-vlm-hf
#      AV2 calibration batches (32 full prompts from the FP4 calibration logs)              -> $A2E_TRT/calib_av2.pt
#      ModelOpt NVFP4 through Edge-LLM's quantization flow (language model 28 layers; output layer, embeddings and ViT
#      stay FP16)                                                                           -> $A2E_TRT/models/a2e-vlm-nvfp4
#      rename to the builder layout                                                         -> $A2E_TRT/models/a2e-reasoner-nvfp4
#   Venvs: the upstream Alpamayo venv ($PY) for 1 and the calibration batches; $A2E_QUANT_VENV (torch + nvidia-modelopt +
#   tensorrt-edgellm[quantize]) for the quantization. Needs $A2E_WORK/raw/calib/calib48_scenes.csv from measure/run_calib_fp4.sh.
set -eu
: "${A2E_TRT:?source env.sh first}"
HERE=$(cd "$(dirname "$0")" && pwd)
M=$A2E_TRT/models; mkdir -p $M
[ -d $M/a2e-reasoner ] || $PY $HERE/make_cosmos3_reasoner_ckpt.py --src $A2E_MODEL --dst $M/a2e-reasoner
[ -d $M/a2e-vlm-hf ] || $PY $HERE/make_cosmos3_reasoner_ckpt.py --src $A2E_MODEL --dst $M/a2e-vlm-hf --hf
[ -f $A2E_TRT/calib_av2.pt ] || $PY $HERE/trt_calib_av2.py --model $A2E_MODEL --scenes $A2E_WORK/raw/calib/calib48_scenes.csv \
  --use-t0 --n 32 --out $A2E_TRT/calib_av2.pt
[ -d $M/a2e-vlm-nvfp4 ] || $A2E_QUANT_VENV/bin/python $HERE/trt_quantize_av2.py --model-dir $M/a2e-vlm-hf \
  --calib $A2E_TRT/calib_av2.pt --out $M/a2e-vlm-nvfp4
[ -d $M/a2e-reasoner-nvfp4 ] || $A2E_QUANT_VENV/bin/python $HERE/hf_to_edge_names.py --src $M/a2e-vlm-nvfp4 --dst $M/a2e-reasoner-nvfp4
ls $M
