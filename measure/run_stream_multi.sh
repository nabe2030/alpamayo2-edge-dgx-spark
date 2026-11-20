#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_stream_multi.sh — continuous measurement over multiple logs (run in tmux on the GPU machine)
# Measured the same way as S10_* / S1_* in run_speed.sh. The difference is that, within one process (= 1 condition),
# it runs 12 logs in order (the input source is rebuilt per log, and a 5-step warm-up pass is run before the measured passes).
#   How the logs were chosen: from the 31 complete val logs, spread across cities (PIT/DTW/MIA/PAO/ATX/WDC) and driving (left turn, right turn, straight, stop),
#   and included logs containing scenes where the reasoning was long (around 30 tokens) in the wide-scene comparison.
#   d46ff5df is the log of a stop scene where FP8 output "go". 9e9bcfb7 is not included since it was measured separately.
# Conditions: the four optimizations (BF16) and + FP8 (FP8 body + output layer). 10Hz (0.1 s intervals, 1 measured pass) and 1Hz (1 s intervals, 3 measured passes).
set -u
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
OUT=$N/raw/speed_multi; mkdir -p $OUT
V=$N/data/av2/val
LOGS=""
for i in 0bae3b5e 36aec72e 0fb7276f 201fe83b 25e5c600 88f47a10 b87683ae 92b900b1 bdb9d309 a7636fca d46ff5df c8ec7be0; do
  LOGS="$LOGS,$(ls -d $V/${i}*)"; done
LOGS=${LOGS#,}
B="$PY $(dirname "$0")/a2e_bench2.py --model $O/models/Alpamayo2-Edge"
log(){ echo "$(TZ=Asia/Tokyo date '+%H:%M:%S') $*" | tee -a $OUT/matrix.log; }
cool(){ for i in $(seq 1 30); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 60 ] && break; sleep 10; done; log "temp ${t}C"; }
envrec(){ log "env $1: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) / GPU procs: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ') / mem: $(free -g | awk '/Mem/{print $3"G used "$7"G avail"}') / root free: $(df -h / | awk 'NR==2{print $4}')"; }
run(){ local L=$1; shift; [ -f $OUT/$L.jsonl ] && grep -q '"kind": "summary"' $OUT/$L.jsonl && { log "skip $L"; return; }
  rm -f $OUT/$L.jsonl $OUT/$L.log; cool; log "start $L: $*"; $B --label $L --out $OUT/$L.jsonl "$@" > $OUT/$L.stdout 2>&1; log "end $L rc=$?"; }
envrec start
S10="--mode stream --av2-log $LOGS --t0-start 2100000 --t0-stop 15800000 --stride 100000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 30 --prep-steps-next 5 --passes 1"
S1="--mode stream --av2-log $LOGS --t0-start 2100000 --t0-stop 15100000 --stride 1000000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 14 --prep-steps-next 5 --passes 3"
run M10_pgcv $S10 --accel tuned --feat p,g,c,v
run M10_all $S10 --accel tuned --feat p,g,c,v --q all
run M1_pgcv $S1 --accel tuned --feat p,g,c,v
run M1_all $S1 --accel tuned --feat p,g,c,v --q all
envrec end
log "ALL DONE"; touch $OUT/DONE
