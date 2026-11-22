#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# build.sh <output directory> — builds fp4_gemv.cu into a PyTorch extension (.so).
#   Requires: nvcc from CUDA 12.9 or later (CUDA_HOME), ninja, and a Python with PyTorch (CUDA build) (PY, default python).
#   Default architecture is sm_121a (GB10). The "a" (architecture-specific) target is required: only then does the
#   FP4 -> FP16 conversion compile to the hardware instruction. Change with ARCH (e.g. ARCH=120a for sm_120 GPUs).
#   Output: <output directory>/fp4_gemv.so and BUILD.md (build conditions). Stops if it already exists (does not change what was built).
set -eu
OUT=${1:?output directory}
SRC=$(cd "$(dirname "$0")" && pwd)/fp4_gemv.cu
PY=${PY:-python}
: "${CUDA_HOME:?set CUDA_HOME}"
ARCH=${ARCH:-121a}
FLAGS="-O3 --use_fast_math -lineinfo -gencode=arch=compute_$ARCH,code=sm_$ARCH"
[ -e "$OUT/fp4_gemv.so" ] && { echo "$OUT/fp4_gemv.so already built (not changing it). Use a different directory"; exit 1; }
mkdir -p "$OUT"
cp "$SRC" "$OUT/"
export PATH=$CUDA_HOME/bin:$PATH
# Only the architecture above: with the default list, an sm_120 cubin is added too and it is unclear which one is loaded
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1a} MAX_JOBS=${MAX_JOBS:-4}
cd "$OUT"
$PY -I - "$PWD" "$FLAGS" <<'PYEOF'
import sys, torch
from torch.utils.cpp_extension import load
out, flags = sys.argv[1], sys.argv[2].split()
load(name="fp4_gemv", sources=[f"{out}/fp4_gemv.cu"], build_directory=out, extra_cuda_cflags=flags,
     extra_cflags=["-O3"], is_python_module=False, verbose=True)
torch.ops.load_library(f"{out}/fp4_gemv.so")
print("ops:", torch.ops.a2e4.fp4_gemv)
PYEOF
{
  echo "# BUILD — fp4_gemv"
  echo "- Date: $(date '+%Y-%m-%d %H:%M %Z')"
  echo "- Source: fp4_gemv.cu sha256 $(sha256sum fp4_gemv.cu | cut -c1-64)"
  echo "- nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1)"
  echo "- gcc: $(gcc --version | head -1)"
  echo "- torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  echo "- nvcc flags: $FLAGS (+ default flags of torch.utils.cpp_extension. TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
  # cuobjdump is missing from some CUDA distributions (e.g. ones that unpack only nvcc). Skip it if missing
  if [ -x "$CUDA_HOME/bin/cuobjdump" ]; then
    echo '```'; "$CUDA_HOME/bin/cuobjdump" --list-elf fp4_gemv.so 2>&1 | head; echo '```'
    # F2FP is the hardware FP4 conversion; 0 means the build fell back to the software conversion
    echo "- F2FP instructions in the SASS: $("$CUDA_HOME/bin/cuobjdump" -sass fp4_gemv.so | grep -c 'F2FP' || true)"
  else echo "- (cuobjdump not found, so the list of architectures in the .so was omitted)"; fi
} > BUILD.md
cat BUILD.md
