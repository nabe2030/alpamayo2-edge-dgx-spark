#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_make_input.py <log_dir> <t0_rel_s> <out.pt> [config] [nav]
# 1 回だけの測定(a2e_bench2.py --mode single --input pt:<out.pt>)に渡す AV2 の入力を作って保存する。
# 中身は公式ローダと同じ形の dict(validate_sample に通る)。描き直しにかかった時間も、横に .json で残す。
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import av2_build as B  # noqa: E402
import av2_nav as N  # noqa: E402

log = B.AV2Log(sys.argv[1])
t0 = log.t0_from_rel(float(sys.argv[2]))
cfg = sys.argv[4] if len(sys.argv) > 4 else "wide_ft"
nav = N.variant(N.nav_from_future(log, t0), sys.argv[5] if len(sys.argv) > 5 else "gt")
d = B.build_sample(log, t0, cfg, nav)
d.pop("av2", None)
torch.save(d, sys.argv[3])
info = {"log_id": log.log_id, "t0_rel_s": float(sys.argv[2]), "t0_ns": t0, "config": cfg, "nav_text": nav,
        "shape": list(d["image_frames"].shape), "made": time.strftime("%Y-%m-%d %H:%M:%S")}
Path(sys.argv[3]).with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(info)
