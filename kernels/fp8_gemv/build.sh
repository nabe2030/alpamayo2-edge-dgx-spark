#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# build.sh <出力ディレクトリ> — fp8_gemv.cu を PyTorch の拡張(.so)にビルドする。
#   要るもの: CUDA 12.9 以降の nvcc(CUDA_HOME)、ninja、PyTorch(CUDA 版)の入った Python(PY、既定は python)。
#   既定のアーキテクチャは sm_120 / sm_121(GB10 など)。ARCH_FLAGS で変えられる。
#   出力: <出力ディレクトリ>/fp8_gemv.so と BUILD.md(ビルドの条件)。既にあれば止まる(作ったものは変えない)。
set -eu
OUT=${1:?出力ディレクトリ}
SRC=$(cd "$(dirname "$0")" && pwd)/fp8_gemv.cu
PY=${PY:-python}
: "${CUDA_HOME:?CUDA_HOME を設定する}"
ARCH_FLAGS=${ARCH_FLAGS:-"-gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_120,code=sm_120"}
FLAGS="-O3 --use_fast_math -lineinfo $ARCH_FLAGS"
[ -e "$OUT/fp8_gemv.so" ] && { echo "$OUT/fp8_gemv.so は作成済み(変えない)。別のディレクトリにする"; exit 1; }
mkdir -p "$OUT"
cp "$SRC" "$OUT/"
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0} MAX_JOBS=${MAX_JOBS:-4}
cd "$OUT"
$PY -I - "$PWD" "$FLAGS" <<'PYEOF'
import sys, torch
from torch.utils.cpp_extension import load
out, flags = sys.argv[1], sys.argv[2].split()
load(name="fp8_gemv", sources=[f"{out}/fp8_gemv.cu"], build_directory=out, extra_cuda_cflags=flags,
     extra_cflags=["-O3"], is_python_module=False, verbose=True)
torch.ops.load_library(f"{out}/fp8_gemv.so")
print("ops:", torch.ops.a2e.fp8_gemv, torch.ops.a2e.read_only)
PYEOF
{
  echo "# BUILD — fp8_gemv"
  echo "- 日時: $(date '+%Y-%m-%d %H:%M %Z')"
  echo "- ソース: fp8_gemv.cu sha256 $(sha256sum fp8_gemv.cu | cut -c1-64)"
  echo "- nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1)"
  echo "- gcc: $(gcc --version | head -1)"
  echo "- torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  echo "- nvcc フラグ: $FLAGS(+ torch.utils.cpp_extension の既定フラグ。TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
  # cuobjdump は CUDA の一部の配布物(nvcc だけを展開したものなど)には無い。無ければ飛ばす
  if [ -x "$CUDA_HOME/bin/cuobjdump" ]; then echo '```'; "$CUDA_HOME/bin/cuobjdump" --list-elf fp8_gemv.so 2>&1 | head; echo '```'
  else echo "- (cuobjdump が無いので、.so の中のアーキテクチャの一覧は省いた)"; fi
} > BUILD.md
cat BUILD.md
