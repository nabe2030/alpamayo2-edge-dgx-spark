#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_e2e.sh —  描き直し込みの連続の測定(GPU のマシン の tmux で実行)
# 各歩で「AV2 の JPEG を読む → 仮想カメラに描き直す(av2_render_fast、GPU)→ 推論」を同じ GPU で続けて動かし、
# 描き直し・推論・合計を記録する(a2e_bench2 --av2-render inline)。前の歩で描いた絵と読んだコマは使い回す
# (カメラから 1 コマずつ届く流れにあたる)。パスの始めに捨てる。測り方のほかの点は run_stream_multi.sh と同じ。
#   ログ: 9e9bcfb7( の速さの測定のログ)と、§3.1 の 12 本から 3 本(判断理由の長いもの・停止・右折)
#   JPEG のファイルは 1 つ前のパス(準備のパス)で読んでいるので、OS のページキャッシュに載っている(ディスクの読み込みは
#   ほぼ入らない。カメラの絵がメモリに届いている状態にあたる)。
set -u
O=${A2E_ROOT:?A2E_ROOT を設定する(モデル・venv・ビルドの置き場所)}
N=${A2E_WORK:?A2E_WORK を設定する(データ・記録の置き場所)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
OUT=$N/raw/speed_e2e; mkdir -p $OUT
V=$N/data/av2/val
DEC=${DEC:-pil_threads}
OUTM=${OUTM:-pinned}  # 描いた絵の持ち方(cpu = 1 枚ずつ .cpu、pinned = 使い回しの pinned へまとめて)
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
