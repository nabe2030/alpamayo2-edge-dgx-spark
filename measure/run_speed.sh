#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_speed.sh — main speed measurement (run in tmux on the GPU machine)
#   A separate process per condition (starting from model loading). single input = first run 1 + steady state 20 runs (fixed seed).
#   continuous = 1 warm-up pass + measured passes (seed changes per pass). Steps before 4.0 s are not counted (the period that fills the cache).
#   Between conditions, cool down to 60°C or below (up to 5 min). Before each call, the rule is to pause at 84°C (a2e_bench.thermal_gate).
# Input on AV2:
#   single input: log 9e9bcfb7 (val, PIT) at t0 = 8.5 s (left turn; navigation text is gt "Turn left in 10m"), wide_ft.
#   continuous: t0 of the same log from 2.1 → 15.8 s (10Hz: 138 steps at 0.1 s intervals), 2.1 → 15.1 s (1Hz: 14 steps at 1 s intervals).
#    Uses the log up to near its end. The navigation text is built from gt for each t0.
#   Re-rendering is done during input preparation (outside the measured interval) (av2_stream.py).
set -u
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
OUT=$N/raw/speed; mkdir -p $OUT
LOG=$N/data/av2/val/9e9bcfb7-601d-3d80-bc12-ef7025174beb
IN=$N/data/av2_inputs/single_9e9bcfb7_8.5.pt
B="$PY $(dirname "$0")/a2e_bench2.py --model $O/models/Alpamayo2-Edge"
log(){ echo "$(TZ=Asia/Tokyo date '+%H:%M:%S') $*" | tee -a $OUT/matrix.log; }
cool(){ for i in $(seq 1 30); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 60 ] && break; sleep 10; done; log "temp ${t}C"; }
run(){ local L=$1; shift; [ -f $OUT/$L.jsonl ] && grep -q '"kind": "summary"' $OUT/$L.jsonl && { log "skip $L"; return; }
  rm -f $OUT/$L.jsonl $OUT/$L.log; cool; log "start $L: $*"; $B --label $L --out $OUT/$L.jsonl "$@" > $OUT/$L.stdout 2>&1; log "end $L rc=$?"; }
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw --format=csv | tee -a $OUT/matrix.log
free -g | tee -a $OUT/matrix.log
[ -f $IN ] || { mkdir -p $(dirname $IN); $PY $(dirname "$0")/av2_make_input.py $LOG 8.5 $IN wide_ft gt | tee -a $OUT/matrix.log; }

SINGLE="--mode single --input pt:$IN --runs 20"
# A. The 3 levels of official acceleration. To capture the first run correctly, clear the compiler cache before the first compile / tuned run
for ACC in off compile tuned; do
  [ $ACC != off ] && { rm -rf $N/cache/inductor $N/cache/triton; log "cache cleared for $ACC"; }
  run A_${ACC}_s43 $SINGLE --accel $ACC --feat "" --seed 43
  run A_${ACC}_s45 $SINGLE --accel $ACC --feat "" --seed 45
done
# B. Add the four optimizations one at a time (on top of tuned)
for F in p pg pgc pgcv pgv; do
  case $F in p) FEAT="p";; pg) FEAT="p,g";; pgc) FEAT="p,g,c";; pgcv) FEAT="p,g,c,v";; pgv) FEAT="p,g,v";; esac
  run B_${F}_s43 $SINGLE --accel tuned --feat "$FEAT" --seed 43
  run B_${F}_s45 $SINGLE --accel tuned --feat "$FEAT" --seed 45
done
# C. FP8 (on top of tuned + the four optimizations). head is for reference
for Q in body all head; do
  run C_${Q}_s43 $SINGLE --accel tuned --feat p,g,c,v --q $Q --seed 43
  run C_${Q}_s45 $SINGLE --accel tuned --feat p,g,c,v --q $Q --seed 45
done

S10="--mode stream --av2-log $LOG --t0-start 2100000 --t0-stop 15800000 --stride 100000 --measure-from 4000000 --seed-mode vary --seed 43"
run S10_tuned $S10 --accel tuned --feat "" --passes 1
run S10_p $S10 --accel tuned --feat p --passes 1
run S10_pg $S10 --accel tuned --feat p,g --passes 2
run S10_pgc $S10 --accel tuned --feat p,g,c --passes 2
run S10_pgv $S10 --accel tuned --feat p,g,v --passes 2
run S10_pgcv $S10 --accel tuned --feat p,g,c,v --passes 5
run S10_body $S10 --accel tuned --feat p,g,c,v --q body --passes 5
run S10_all $S10 --accel tuned --feat p,g,c,v --q all --passes 5

S1="--mode stream --av2-log $LOG --t0-start 2100000 --t0-stop 15100000 --stride 1000000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 14"
run S1_pgv $S1 --accel tuned --feat p,g,v --passes 5
run S1_pgcv $S1 --accel tuned --feat p,g,c,v --passes 5
run S1_body $S1 --accel tuned --feat p,g,c,v --q body --passes 5
run S1_all $S1 --accel tuned --feat p,g,c,v --q all --passes 5

# Profile (torch.profiler on the last run only; excluded from the statistics)
run P_pgcv_s43 $SINGLE --accel tuned --feat p,g,c,v --seed 43 --runs 3 --torch-profile $OUT/prof
run P_all_s43 $SINGLE --accel tuned --feat p,g,c,v --q all --seed 43 --runs 3 --torch-profile $OUT/prof
log "ALL DONE"; touch $OUT/DONE
