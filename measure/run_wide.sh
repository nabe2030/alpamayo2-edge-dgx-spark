#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_wide.sh —  広い場面での BF16 対 FP8(GPU のマシン の tmux で実行)
#   同じ場面・同じ seed(43〜47)で base(BF16)/ fp8body / fp8all を続けて回す(時間も 1 回だけの測り方で記録)。
#   teacher forcing は base の seed 43 の CoT を固定し、eager / fp8body / fp8all の 1 トークンずつの分布を base と比べる。
#   AV2 の追加: 参考として BF16 でナビ文をいつも "Continue straight" にしたとき(seed 43〜47)。
#   速さの測定(run_speed.sh)が終わってから始める(GPU を同時に使わない)。
set -u
O=${A2E_ROOT:?A2E_ROOT を設定する(モデル・venv・ビルドの置き場所)}
N=${A2E_WORK:?A2E_WORK を設定する(データ・記録の置き場所)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
while [ ! -f $N/raw/speed/DONE ]; do sleep 30; done
for i in $(seq 1 30); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 60 ] && break; sleep 10; done
echo "$(TZ=Asia/Tokyo date '+%F %T') 開始: GPU プロセス $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)、${t}℃" | tee -a $N/logs/run_wide.log
$PY $(dirname "$0")/wide_eval.py --model $O/models/Alpamayo2-Edge --scenes $N/raw/wide/scenes_t0.csv \
  --variants base,fp8body,fp8all --tf-variants eager,fp8body,fp8all --seeds 43,44,45,46,47 \
  --straight-seeds 43,44,45,46,47 --av2-config wide_ft --out $N/raw/wide/wide.jsonl 2>&1 | grep -v -i warn >> $N/logs/run_wide.log
echo "$(TZ=Asia/Tokyo date '+%F %T') 終わり rc=$?" | tee -a $N/logs/run_wide.log
$PY $(dirname "$0")/summarize_wide.py $N/raw/wide/wide.jsonl $N/raw/wide/scenes_t0.csv $N/raw/wide/summary_wide >> $N/logs/run_wide.log 2>&1
touch $N/raw/wide/DONE
