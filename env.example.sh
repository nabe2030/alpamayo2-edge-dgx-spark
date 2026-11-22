# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# env.example.sh — copy to env.sh, edit A2E_ROOT and A2E_WORK, and load with `source env.sh`. measure/run_*.sh also loads it.
#   A2E_ROOT: location of upstream code (src/alpamayo2-edge and its .venv), the model (models/Alpamayo2-Edge) and kernel builds (build/)
#   A2E_WORK: location of AV2 data (data/av2), inputs (inputs/), measurement logs (raw/) and compile cache (cache/)
# Copy to env.sh, set A2E_ROOT and A2E_WORK, then `source env.sh`.
export A2E_ROOT=${A2E_ROOT:-/path/to/root}
export A2E_WORK=${A2E_WORK:-/path/to/work}
export A2E_MODEL=${A2E_MODEL:-$A2E_ROOT/models/Alpamayo2-Edge}
export A2E_FP8_SO=${A2E_FP8_SO:-$A2E_ROOT/build/fp8_gemv/v1/fp8_gemv.so}
# FP4 kernel (kernels/fp4_gemv/build.sh), cache of quantized FP4 bodies, and the GPTQ calibration statistics (measure/run_calib_fp4.sh)
export A2E_FP4_SO=${A2E_FP4_SO:-$A2E_ROOT/build/fp4_gemv/v1/fp4_gemv.so}
export A2E_FP4_CACHE=${A2E_FP4_CACHE:-$A2E_WORK/cache/fp4w}
export A2E_CALIB_IN=${A2E_CALIB_IN:-$A2E_WORK/raw/calib/H_calib48.pt}
# TensorRT Edge-LLM (tensorrt/): venv with tensorrt-edgellm, venv used for NVFP4 quantization, and the work folder
export A2E_EDGELLM_VENV=${A2E_EDGELLM_VENV:-$A2E_ROOT/venv-edgellm}
export A2E_QUANT_VENV=${A2E_QUANT_VENV:-$A2E_ROOT/venv-quant}
export A2E_TRT=${A2E_TRT:-$A2E_WORK/trt}
# location of this repository (where env.sh is placed)
export A2E_REPO=${A2E_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}
O=$A2E_ROOT
N=$A2E_WORK
# python of the upstream venv (created following the upstream README)
PY=${PY:-$O/src/alpamayo2-edge/.venv/bin/python}
# CUDA used to build the FP8 kernel (same major version as PyTorch's CUDA; CUDA 12.9 for torch 2.8.0+cu129)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
# ptxas used by triton on sm_121 (CUDA 12.9 or later). Used when the one bundled with PyTorch is too old
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
export TORCHINDUCTOR_CACHE_DIR=$N/cache/inductor TRITON_CACHE_DIR=$N/cache/triton
# make the upstream packages and this repository's accel/, av2_input/ and measure/ importable
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$A2E_REPO/accel:$A2E_REPO/av2_input:$A2E_REPO/measure:$O/src/alpamayo2-edge/src
mkdir -p "$N"
cd "$N"
