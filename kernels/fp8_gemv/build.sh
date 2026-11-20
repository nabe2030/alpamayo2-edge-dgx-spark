#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# build.sh <output directory> — builds fp8_gemv.cu into a PyTorch extension (.so).
#   Requires: nvcc from CUDA 12.9 or later (CUDA_HOME), ninja, and a Python with PyTorch (CUDA build) (PY, default python).
#   Default architectures are sm_120 / sm_121 (GB10, etc.). Change with ARCH_FLAGS.
#   Output: <output directory>/fp8_gemv.so and BUILD.md (build conditions). Stops if it already exists (does not change what was built).
set -eu
OUT=${1:?output directory}
SRC=$(cd "$(dirname "$0")" && pwd)/fp8_gemv.cu
PY=${PY:-python}
: "${CUDA_HOME:?set CUDA_HOME}"
ARCH_FLAGS=${ARCH_FLAGS:-"-gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_120,code=sm_120"}
FLAGS="-O3 --use_fast_math -lineinfo $ARCH_FLAGS"
[ -e "$OUT/fp8_gemv.so" ] && { echo "$OUT/fp8_gemv.so already built (not changing it). Use a different directory"; exit 1; }
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
  echo "- Date: $(date '+%Y-%m-%d %H:%M %Z')"
  echo "- Source: fp8_gemv.cu sha256 $(sha256sum fp8_gemv.cu | cut -c1-64)"
  echo "- nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1)"
  echo "- gcc: $(gcc --version | head -1)"
  echo "- torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  echo "- nvcc flags: $FLAGS (+ default flags of torch.utils.cpp_extension. TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
  # cuobjdump is missing from some CUDA distributions (e.g. ones that unpack only nvcc). Skip it if missing
  if [ -x "$CUDA_HOME/bin/cuobjdump" ]; then echo '```'; "$CUDA_HOME/bin/cuobjdump" --list-elf fp8_gemv.so 2>&1 | head; echo '```'
  else echo "- (cuobjdump not found, so the list of architectures in the .so was omitted)"; fi
} > BUILD.md
cat BUILD.md
