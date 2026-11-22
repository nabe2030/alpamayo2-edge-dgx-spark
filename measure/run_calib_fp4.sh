#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# run_calib_fp4.sh — calibration statistics for the FP4 body (GPTQ), run on the GPU machine.
#   1. Pick 48 AV2 train logs that are not used by the 400 evaluation scenes (select_calib_scenes.py, seed 7).
#      The list we used is results/fp4/calib_logs_48.csv (set CALIB_LIST to use it instead of picking again).
#      Picking needs a pose scan of the train logs (av2_input/av2_scan_poses.py -> $A2E_WORK/raw/scan_train.csv).
#   2. Download only the frames needed for each scene (av2_fetch_frames.py; about 0.6 GB for 48 logs).
#   3. Run BF16 CoT decoding on them (wide_eval.py, variant "calib" of a2e_fp4q.py: 5 seeds + teacher forcing) and
#      save H = sum x x^T of the inputs of every projection to $A2E_WORK/raw/calib/H_calib48.pt (about 11 GB).
#   The FP4 body is then quantized on first use (a2e_fp4q.install / --q fp4 with A2E_FP4_METHOD=gptq:0.1 and
#   A2E_CALIB_IN=<H file>) and cached in $A2E_FP4_CACHE/body_gptq:0.1.pt.
#   For the held-out check of the article (4.2), run again with CALIB_LIST=results/fp4/calib_logs_48_heldout.csv TAG=calibB48.
set -u
O=${A2E_ROOT:?set A2E_ROOT (location of models, venv and builds)}
N=${A2E_WORK:?set A2E_WORK (location of data and records)}
# location of this script, taken before env.sh changes the directory to $A2E_WORK
HERE=$(cd "$(dirname "$0")" && pwd)
source "${A2E_ENV:-$HERE/../env.sh}"
TAG=${TAG:-calib48}
C=$N/raw/calib; mkdir -p $C $N/data/av2_calib
if [ -n "${CALIB_LIST:-}" ]; then cp "$CALIB_LIST" $C/$TAG.csv
elif [ ! -f $C/$TAG.csv ]; then
  $PY $HERE/select_calib_scenes.py --scan $N/raw/scan_train.csv --eval-scenes $N/raw/wide/scenes_t0.csv --n 48 --seed 7 --out $C/$TAG.csv
fi
$PY $HERE/../av2_input/av2_fetch_frames.py $C/$TAG.csv $N/data/av2_calib $N/data/av2_calib/full $C/${TAG}_t0.csv 1
# scene list in the wide_eval format (scene_id first)
$PY - $C/${TAG}_t0.csv $C/${TAG}_scenes.csv <<'PYEOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
d.insert(0, "scene_id", ["c_%s_%.1f" % (l[:8], t) for l, t in zip(d.log_id, d.t0_rel_s)])
d[["scene_id", "split", "log_id", "t0_rel_s", "kind", "v_mean", "dyaw_deg", "city", "t0_ns", "data_dir"]].to_csv(sys.argv[2], index=False)
PYEOF
echo "$(TZ=Asia/Tokyo date '+%F %T') start H collection; GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
A2E_CALIB_OUT=$C/H_$TAG.pt $PY $HERE/wide_eval.py --model $A2E_MODEL --scenes $C/${TAG}_scenes.csv \
  --variant-module $HERE/../accel/a2e_fp4q.py --variants base,calib --tf-variants calib --seeds 43,44,45,46,47 \
  --straight-seeds "" --av2-config wide_ft --out $C/${TAG}_run.jsonl
echo "$(TZ=Asia/Tokyo date '+%F %T') end rc=$? -> $C/H_$TAG.pt"
