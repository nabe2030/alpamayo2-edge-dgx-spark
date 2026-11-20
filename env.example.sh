# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# env.example.sh — env.sh に写して A2E_ROOT と A2E_WORK を書き換え、`source env.sh` で読む。measure/run_*.sh も読む。
#   A2E_ROOT: 上流のコード(src/alpamayo2-edge と、その .venv)、モデル(models/Alpamayo2-Edge)、カーネルのビルド(build/)の置き場所
#   A2E_WORK: AV2 のデータ(data/av2)、入力(inputs/)、測定の記録(raw/)、コンパイルのキャッシュ(cache/)の置き場所
# Copy to env.sh, set A2E_ROOT and A2E_WORK, then `source env.sh`.
export A2E_ROOT=${A2E_ROOT:-/path/to/root}
export A2E_WORK=${A2E_WORK:-/path/to/work}
export A2E_MODEL=${A2E_MODEL:-$A2E_ROOT/models/Alpamayo2-Edge}
export A2E_FP8_SO=${A2E_FP8_SO:-$A2E_ROOT/build/fp8_gemv/v1/fp8_gemv.so}
# このリポジトリの場所(env.sh を置いた所)
export A2E_REPO=${A2E_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}
O=$A2E_ROOT
N=$A2E_WORK
# 上流の venv の python(上流の README の手順で作ったもの)
PY=${PY:-$O/src/alpamayo2-edge/.venv/bin/python}
# FP8 カーネルのビルドに使う CUDA(PyTorch の CUDA と同じメジャー版。torch 2.8.0+cu129 なら CUDA 12.9)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
# sm_121 で triton が使う ptxas(CUDA 12.9 以降)。PyTorch に同梱のものが古いときに使う
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}
export TORCHINDUCTOR_CACHE_DIR=$N/cache/inductor TRITON_CACHE_DIR=$N/cache/triton
# 上流のパッケージと、このリポジトリの accel/・av2_input/・measure/ を読めるようにする
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$A2E_REPO/accel:$A2E_REPO/av2_input:$A2E_REPO/measure:$O/src/alpamayo2-edge/src
mkdir -p "$N"
cd "$N"
