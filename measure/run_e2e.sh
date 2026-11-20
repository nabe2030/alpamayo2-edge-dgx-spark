#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_e2e.sh — continuous measurement including re-rendering (run in tmux on the GPU machine)
# At each step, runs "read AV2 JPEGs → re-render into virtual cameras (av2_render_fast, GPU) → inference" back to back on the same GPU,
# and records re-rendering, inference and the total (a2e_bench2 --av2-render inline). Images rendered and frames read in the previous step are reused
# (corresponds to frames arriving one at a time from the cameras). Discarded at the start of each pass. Otherwise measured the same way as run_stream_multi.sh.
#   Logs: 9e9bcfb7 (the log of the speed measurement) and 3 of the 12 in §3.1 (long reasoning, stop, right turn)
#   The JPEG files were read in the preceding pass (the warm-up pass), so they are in the OS page cache (disk reads are
#   almost absent; corresponds to camera images having arrived in memory).
set -u
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
OUT=$N/raw/speed_e2e; mkdir -p $OUT
V=$N/data/av2/val
DEC=${DEC:-pil_threads}
OUTM=${OUTM:-pinned}  # how rendered images are held (cpu = .cpu one at a time, pinned = batched into reused pinned memory)
LOGS=""
for i in 9e9bcfb7 0bae3b5e 25e5c600 a7636fca; do LOGS="$LOGS,$(ls -d $V/${i}*)"; done
LOGS=${LOGS#,}
B="$PY $(dirname "$0")/a2e_bench2.py --model $O/models/Alpamayo2-Edge"
log(){ echo "$(TZ=Asia/Tokyo date '+%H:%M:%S') $*" | tee -a $OUT/matrix.log; }
cool(){ for i in $(seq 1 30); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 60 ] && break; sleep 10; done; log "temp ${t}C"; }
envrec(){ log "env $1: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) / GPU procs: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ') / mem: $(free -g | awk '/Mem/{print $3"G used "$7"G avail"}') / root free: $(df -h / | awk 'NR==2{print $4}')"; }
run(){ local L=$1; shift; [ -f $OUT/$L.jsonl ] && grep -q '"kind": "summary"' $OUT/$L.jsonl && { log "skip $L"; return; }
  rm -f $OUT/$L.jsonl $OUT/$L.log; cool; log "start $L: $*"; $B --label $L --out $OUT/$L.jsonl "$@" > $OUT/$L.stdout 2>&1; log "end $L rc=$?"; }
envrec start
S10="--mode stream --av2-log $LOGS --av2-render inline --av2-decode $DEC --av2-out $OUTM --t0-start 2100000 --t0-stop 15800000 --stride 100000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 30 --prep-steps-next 5 --passes 1"
S1="--mode stream --av2-log $LOGS --av2-render inline --av2-decode $DEC --av2-out $OUTM --t0-start 2100000 --t0-stop 15100000 --stride 1000000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 14 --prep-steps-next 5 --passes 3"
run E1_all_${DEC}_$OUTM $S1 --accel tuned --feat p,g,c,v --q all
run E10_all_${DEC}_$OUTM $S10 --accel tuned --feat p,g,c,v --q all
run E1_pgcv_${DEC}_$OUTM $S1 --accel tuned --feat p,g,c,v
run E10_pgcv_${DEC}_$OUTM $S10 --accel tuned --feat p,g,c,v
envrec end
log "ALL DONE"; touch $OUT/DONE_${DEC}_$OUTM
