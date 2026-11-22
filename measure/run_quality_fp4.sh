#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_quality_fp4.sh — output-quality runs of the FP4 article (run on the GPU machine, after run_calib_fp4.sh).
#   Variant names (a2e_fp4q.py): fp4rtn = plain rounding, fp4mse = block-scale search, fp4gptq_d10 = GPTQ with damping 0.1
#   (the recommended configuration), fp4gptqm_d10 = GPTQ + scale search, _h8 / _h4 = output layer FP8 / FP4.
#   1. Rounding error on the real weights, with H of the calibration logs and of 48 other logs (held out)   -> raw/fp4/qerr.jsonl
#   2. 40 scenes x 3 seeds, rounding methods (article 4.1, figure 5)                                          -> raw/fp4/e40_methods.jsonl
#   3. 400 scenes x 5 seeds, FP8 body vs FP4 body GPTQ (article 4.3)                                           -> raw/fp4/e400_gptq.jsonl
#   4. 40 scenes x 3 seeds, output layer BF16 / FP8 / FP4 on top of the FP4 body (article 4.3, reference)      -> raw/fp4/e40_heads.jsonl
#   5. First reasoning token in the 139 stop/go scenes (article 4.3, figure 6)                                 -> raw/fp4/firsttoken_fp4.jsonl
#   6. Greedy reasoning on 20 scenes, PyTorch with quantized weights (fake quantization) vs BF16 (article 5.3) -> raw/fp4/greedy20/
#   Summaries: summarize_wide.py (2-4), summarize_firsttoken.py (5), cmp_greedy.py (6). Ours are in results/fp4/.
#   Needs: raw/wide/scenes_t0.csv (400 scenes, run_wide.sh), the plan and records of av2_stopgo.py (STOPGO_PLAN, STOPGO_JSONL), raw/calib/H_calib48.pt and
#   H_calibB48.pt (run_calib_fp4.sh, the second one with the held-out list).
set -u
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
source "${A2E_ENV:-$(dirname "$0")/../env.sh}"
HERE=$(cd "$(dirname "$0")" && pwd)
VM=$HERE/../accel/a2e_fp4q.py
F=$N/raw/fp4; mkdir -p $F
export A2E_CALIB_IN=${A2E_CALIB_IN:-$N/raw/calib/H_calib48.pt}
log(){ echo "$(TZ=Asia/Tokyo date '+%F %T') $*" | tee -a $F/run_quality_fp4.log; }
# the 40-scene subset used for the method comparison (the same draw as in the article)
[ -f $F/scenes40.csv ] || $PY -c "import pandas as pd; pd.read_csv('$N/raw/wide/scenes_t0.csv').sample(40, random_state=3).to_csv('$F/scenes40.csv', index=False)"
wide(){ local L=$1 SC=$2 SEEDS=$3 VAR=$4 TFV=$5
  [ -f $F/$L.jsonl ] && { log "skip $L"; return; }
  log "start $L; GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  $PY $HERE/wide_eval.py --model $A2E_MODEL --scenes $SC --variant-module $VM --variants $VAR --tf-variants $TFV \
    --seeds $SEEDS --straight-seeds "" --av2-config wide_ft --out $F/$L.jsonl > $F/$L.log 2>&1
  log "end $L rc=$?"; }
# 1. rounding error (also writes the quantized bodies to $A2E_FP4_CACHE)
[ -f $F/qerr.jsonl ] || $PY $HERE/fp4_qerr.py --model $A2E_MODEL --calib $A2E_CALIB_IN --held-out $N/raw/calib/H_calibB48.pt \
  --methods rtn,mseu,mse,gptq:0.01,gptq:0.1,gptq:0.5,gptqm:0.1 --out $F/qerr.jsonl
# 2-4
wide e40_methods $F/scenes40.csv 43,44,45 base,fp8body,fp4rtn,fp4mse,fp4gptq_d10,fp4gptqm_d10 eager,fp8body,fp4rtn,fp4mse,fp4gptq_d10,fp4gptqm_d10
wide e400_gptq $N/raw/wide/scenes_t0.csv 43,44,45,46,47 base,fp8body,fp4gptq_d10 eager,fp8body,fp4gptq_d10
wide e40_heads $F/scenes40.csv 43,44,45 base,fp4gptq_d10,fp4gptq_d10_h8,fp4gptq_d10_h4 eager,fp4gptq_d10,fp4gptq_d10_h8,fp4gptq_d10_h4
for L in e40_methods:$F/scenes40.csv e400_gptq:$N/raw/wide/scenes_t0.csv e40_heads:$F/scenes40.csv; do
  $PY $HERE/summarize_wide.py $F/${L%%:*}.jsonl ${L#*:} $F/summary_${L%%:*} > /dev/null; done
# 5. first token
[ -f $F/firsttoken_fp4.jsonl ] || $PY $HERE/av2_firsttoken.py ${STOPGO_PLAN:-$N/raw/stopgo_plan.csv} ${STOPGO_JSONL:-$N/raw/stopgo/stopgo.jsonl} $F/firsttoken_fp4.jsonl \
  base,fp8body,fp4rtn,fp4gptq_d10,fp4gptq_d10_h8 > $F/firsttoken_fp4.log 2>&1
$PY $HERE/summarize_firsttoken.py $F/firsttoken_fp4.jsonl $F/summary_firsttoken_fp4.md > /dev/null
# 6. greedy, 20 scenes (the first 20 of scenes40.csv). BF16 references first (also used for the TensorRT comparison, tensorrt/README.md)
G=$F/greedy20; mkdir -p $G
[ -d $N/trt/ref20 ] || $PY $HERE/../tensorrt/trt_ref_multi.py --model $A2E_MODEL --scenes $F/scenes40.csv --n 20 --out $N/trt/ref20
for q in fp8 fp4:gptq:0.1 fp4:rtn; do
  for mode in prefill dec; do
    od=$G/${mode}_${q//:/_}; flag=$([ $mode = dec ] && echo --decode-only)
    [ -d $od ] || $PY $HERE/greedy_fakeq.py --model $A2E_MODEL --scenes $F/scenes40.csv --n 20 --out $od --quant $q $flag > $od.log 2>&1
    $PY $HERE/cmp_greedy.py $N/trt/ref20 $od "$mode $q" | tee -a $G/summary.jsonl
  done
done
log "ALL DONE"
