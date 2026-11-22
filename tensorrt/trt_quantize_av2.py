#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-quantize the Alpamayo2-Edge reasoner LLM with TensorRT Edge-LLM's ModelOpt flow,
calibrated on AV2 prompts (trt_calib_av2.py) instead of the built-in web datasets.

The lm_head stays FP16; the visual tower stays FP16 unless --visual is given. Edge-LLM's multimodal calibration path is reused; only its
dataset loader is swapped for the precomputed AV2 batches (full 22-image Alpamayo prompts).
"""
import argparse
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default="nvfp4")
    ap.add_argument("--lm-head", default=None)
    ap.add_argument("--visual", default=None, help="e.g. fp8: also quantize the SigLIP2 tower (calibrated on the same AV2 batches)")
    a = ap.parse_args()
    import tensorrt_edgellm.quantization.quantize as Q

    batches = torch.load(a.calib, weights_only=False)["batches"]
    print("AV2 calibration batches:", len(batches), [int(b["input_ids"].shape[1]) for b in batches])
    Q._is_image_blind_calibration = lambda model, cfg: True
    Q._multimodal_calib_dataloader = lambda *args, **kw: batches
    t = time.perf_counter()
    Q.quantize_and_export(model_dir=a.model_dir, output_dir=a.out, quantization=a.quantization,
                          lm_head_quantization=a.lm_head, visual_quantization=a.visual, dtype="fp16", device="cuda")
    print(f"quantize+export {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
