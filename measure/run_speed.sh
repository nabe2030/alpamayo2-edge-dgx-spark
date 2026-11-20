#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_speed.sh —  速さの本測定(GPU のマシン の tmux で実行)
#   条件ごとに別プロセス(モデルの読み込みから)。1 回だけ = 初回 1 + 安定 20 回(seed 固定)。
#   連続 = 準備のパス 1 回 + 計測のパス(パスごとに seed を変える)。4.0 s より前の歩はキャッシュを満たす期間として数えない。
#   条件の合間に 60℃ 以下まで冷ます(最大 5 分)。各呼び出しの前に 84℃ で休む決まり(a2e_bench.thermal_gate)。
# AV2 での入力:
#   1 回だけ: ログ 9e9bcfb7(val、PIT)の t0 = 8.5 s(左折。ナビ文は gt の "Turn left in 10m")、wide_ft。
#   連続: 同じログの t0 を 2.1 → 15.8 s(10Hz: 0.1 s きざみ 138 歩)、2.1 → 15.1 s(1Hz: 1 s きざみ 14 歩)。
#    ログの終わり近くまで使う。ナビ文は t0 ごとに gt で作る)
#   描き直しは入力の準備(測る区間の外)で済ませる(av2_stream.py)。
set -u
O=${A2E_ROOT:?A2E_ROOT を設定する(モデル・venv・ビルドの置き場所)}
N=${A2E_WORK:?A2E_WORK を設定する(データ・記録の置き場所)}
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
# A. 公式の加速の 3 段( にそろえる)。初回を正しく取るため、compile / tuned の最初の前にコンパイラのキャッシュを空にする
for ACC in off compile tuned; do
  [ $ACC != off ] && { rm -rf $N/cache/inductor $N/cache/triton; log "cache cleared for $ACC"; }
  run A_${ACC}_s43 $SINGLE --accel $ACC --feat "" --seed 43
  run A_${ACC}_s45 $SINGLE --accel $ACC --feat "" --seed 45
done
# B. 加速の 4 つの手を 1 つずつ足す(tuned の上。 にそろえる)
for F in p pg pgc pgcv pgv; do
  case $F in p) FEAT="p";; pg) FEAT="p,g";; pgc) FEAT="p,g,c";; pgcv) FEAT="p,g,c,v";; pgv) FEAT="p,g,v";; esac
  run B_${F}_s43 $SINGLE --accel tuned --feat "$FEAT" --seed 43
  run B_${F}_s45 $SINGLE --accel tuned --feat "$FEAT" --seed 45
done
# C. FP8(tuned + 4 つの手の上。 にそろえる)。head は参考
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

# プロファイル(最後の 1 回だけ torch.profiler、統計からは除く)
run P_pgcv_s43 $SINGLE --accel tuned --feat p,g,c,v --seed 43 --runs 3 --torch-profile $OUT/prof
run P_all_s43 $SINGLE --accel tuned --feat p,g,c,v --q all --seed 43 --runs 3 --torch-profile $OUT/prof
log "ALL DONE"; touch $OUT/DONE
