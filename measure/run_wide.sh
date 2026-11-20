#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_wide.sh — BF16 vs FP8 on wide scenes (run in tmux on the GPU machine)
#   On the same scenes and the same seeds (43-47), runs base (BF16) / fp8body / fp8all back to back (time is also recorded with the single-run measurement).
#   teacher forcing fixes base's seed 43 CoT and compares the token-by-token distributions of eager / fp8body / fp8all with base.
#   Added for AV2: for reference, BF16 with the navigation text always "Continue straight" (seeds 43-47).
#   Starts after the speed measurement (run_speed.sh) finishes (the GPU is not used concurrently).
set -u
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
while [ ! -f $N/raw/speed/DONE ]; do sleep 30; done
for i in $(seq 1 30); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 60 ] && break; sleep 10; done
echo "$(TZ=Asia/Tokyo date '+%F %T') start: GPU processes $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l), ${t}°C" | tee -a $N/logs/run_wide.log
$PY $(dirname "$0")/wide_eval.py --model $O/models/Alpamayo2-Edge --scenes $N/raw/wide/scenes_t0.csv \
  --variants base,fp8body,fp8all --tf-variants eager,fp8body,fp8all --seeds 43,44,45,46,47 \
  --straight-seeds 43,44,45,46,47 --av2-config wide_ft --out $N/raw/wide/wide.jsonl 2>&1 | grep -v -i warn >> $N/logs/run_wide.log
echo "$(TZ=Asia/Tokyo date '+%F %T') end rc=$?" | tee -a $N/logs/run_wide.log
$PY $(dirname "$0")/summarize_wide.py $N/raw/wide/wide.jsonl $N/raw/wide/scenes_t0.csv $N/raw/wide/summary_wide >> $N/logs/run_wide.log 2>&1
touch $N/raw/wide/DONE
