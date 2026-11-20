#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_stream_multi.sh —  連続の測定を複数のログで(GPU のマシン の tmux で実行)
# 測り方は run_speed.shの S10_* / S1_* と同じ。違いは、1 つのプロセス(= 1 条件)の中で
# 12 本のログを順に回すこと(ログごとに入力源を作り直し、準備のパスを 5 歩回してから計測のパス)。
#   ログの選び方: 丸ごとある val 31 本から、都市(PIT/DTW/MIA/PAO/ATX/WDC)、走り方(左折・右折・直進・停止)を散らし、
#    の広い場面の比較で判断理由が長かった場面(30 トークン前後)を含むログを入れた。
#   d46ff5df は  で FP8 が「進む」と出した停止の場面のログ。9e9bcfb7 は  で測ったので入れない。
# 条件: 4 つの手(BF16)と + FP8(本体 + 出力層)。10Hz(0.1 s きざみ、計測 1 パス)と 1Hz(1 s きざみ、計測 3 パス)。
set -u
O=${A2E_ROOT:?A2E_ROOT を設定する(モデル・venv・ビルドの置き場所)}
N=${A2E_WORK:?A2E_WORK を設定する(データ・記録の置き場所)}
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
