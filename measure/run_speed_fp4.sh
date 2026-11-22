#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_speed_fp4.sh [method] — speed of the FP4 body decode (a2e_bench2.py --q fp4) with same-day references (run on the GPU machine).
#   method: FP4 quantization method of a2e_fp4q.py (default gptq:0.1; needs the calibration statistics A2E_CALIB_IN,
#           see run_calib_fp4.sh, unless $A2E_FP4_CACHE/body_<method>.pt already exists).
#   Same measurements as run_speed.sh / run_stream_multi.sh:
#   single: input 9e9bcfb7 t0 = 8.5 s, seed 43 / 45, 1 warm-up + 20 timed runs, the four optimizations (p,g,c,v), accel tuned
#   stream: the same 12 val logs; 10 Hz (0.1 s stride, 1 timed pass) and 1 Hz (1 s stride, 3 timed passes)
#   configs: BF16 (four optimizations only), FP8 body (--q body), FP4 body (--q fp4, output layer BF16), FP4 body + FP8 output layer (--q fp4_h8)
#   Between runs, cool down to 60°C or below (up to 5 min); the start of every run records GPU processes and temperature.
#   Summaries: summarize_speed.py $A2E_WORK/raw/speed_fp4 ... (single) and summarize_multi.py on the M* records (stream);
#   ours are results/fp4/summary_speed_fp4.md and summary_stream_fp4.md.
set -u
METHOD=${1:-gptq:0.1}
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
OUT=$N/raw/speed_fp4; mkdir -p $OUT
IN=$N/data/av2_inputs/single_9e9bcfb7_8.5.pt
V=$N/data/av2/val
export A2E_FP4_METHOD=$METHOD
B="$PY $(dirname "$0")/a2e_bench2.py --model $A2E_MODEL"
log(){ echo "$(TZ=Asia/Tokyo date '+%H:%M:%S') $*" | tee -a $OUT/matrix.log; }
cool(){ for i in $(seq 1 30); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 60 ] && break; sleep 10; done; log "temp ${t}C"; }
envrec(){ log "env $1: $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) / GPU procs: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ' ') / mem: $(free -g | awk '/Mem/{print $3"G used "$7"G avail"}')"; }
run(){ local L=$1; shift; [ -f $OUT/$L.jsonl ] && grep -q '"kind": "summary"' $OUT/$L.jsonl && { log "skip $L"; return; }
  rm -f $OUT/$L.jsonl; cool; envrec $L; log "start $L (method=$METHOD): $*"
  $B --label $L --out $OUT/$L.jsonl "$@" > $OUT/$L.stdout 2>&1; log "end $L rc=$?"; }
[ -f $IN ] || { mkdir -p $(dirname $IN); $PY $(dirname "$0")/../av2_input/av2_make_input.py $V/9e9bcfb7-601d-3d80-bc12-ef7025174beb 8.5 $IN wide_ft gt | tee -a $OUT/matrix.log; }
SINGLE="--mode single --input pt:$IN --runs 20 --accel tuned --feat p,g,c,v"
for s in 43 45; do
  run C_bf16_s$s $SINGLE --seed $s
  run C_fp8body_s$s $SINGLE --q body --seed $s
  run C_fp4_${METHOD}_s$s $SINGLE --q fp4 --seed $s
  run C_fp4h8_${METHOD}_s$s $SINGLE --q fp4_h8 --seed $s
done
LOGS=""
for i in 0bae3b5e 36aec72e 0fb7276f 201fe83b 25e5c600 88f47a10 b87683ae 92b900b1 bdb9d309 a7636fca d46ff5df c8ec7be0; do
  LOGS="$LOGS,$(ls -d $V/${i}*)"; done
LOGS=${LOGS#,}
S10="--mode stream --av2-log $LOGS --t0-start 2100000 --t0-stop 15800000 --stride 100000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 30 --prep-steps-next 5 --passes 1"
S1="--mode stream --av2-log $LOGS --t0-start 2100000 --t0-stop 15100000 --stride 1000000 --measure-from 4000000 --seed-mode vary --seed 43 --prep-steps 14 --prep-steps-next 5 --passes 3"
for M in 1 10; do
  S=$([ $M = 1 ] && echo "$S1" || echo "$S10")
  run M${M}_bf16 $S --accel tuned --feat p,g,c,v
  run M${M}_fp8body $S --accel tuned --feat p,g,c,v --q body
  run M${M}_fp4_$METHOD $S --accel tuned --feat p,g,c,v --q fp4
done
log "ALL DONE ($METHOD)"
