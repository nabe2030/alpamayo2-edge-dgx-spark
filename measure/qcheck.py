#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# qcheck.py — checks, in a few minutes and in a way unaffected by sampling, the quality after swapping the decode implementation
#
# In one process, feeds the same input while switching between decode implementations (variants). Preprocessing, ViT, prefill and expert are shared.
#   base    : official tuned BF16 decode (starting point; the reference for comparison)
#   eager   : runs the same BF16 DecodeStep without compile (only the matmul splitting and fusion change = width of BF16 rounding jitter)
#   fp8body : only the projections and MLP of the body 28 layers in FP8
#   fp8all  : body + output layer in FP8
#
# What is checked (all with the same input, comparing each variant with base):
#  A. teacher forcing: feeds the reasoning (CoT) that base produced with seeds 43-47 as fixed input one token at a time, and compares the
#     next-token distribution at each position (KL(base‖variant) at temperature 1 and 0.6, top-1 agreement, difference in log-prob of the fixed token).
#     The same masks as official (trajectory tokens, EOS) are applied. top-p is not applied (it would make KL infinite).
#     Then runs through the expert with the same CoT and the same diffusion RNG (seed 1000) and checks the trajectory difference.
#     Yardstick: the trajectory difference on base when only the diffusion RNG is changed to 1001.
#  B. greedy (top_k=1): at which token it diverges from base, and the trajectory difference (diffusion RNG fixed to seed 43).
#  C. official sampling (temperature 0.6, top_p 0.98) with seeds 43-47: whether the CoT matches base for the same seed (diagnostic),
#     and the trajectory difference. Yardstick: the difference between base runs that differ only in seed.
import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2e_fp8 as Q  # noqa: E402
import a2e_stagea as S  # noqa: E402


def ade(a, b):
    d = [math.dist(p, r) for p, r in zip(a, b)]
    return sum(d) / len(d)


class ForcedSampler:
    """Instead of generate(custom_generate=...), feeds a given token sequence, writes the KV, and collects the logits at each position.

    Like the official forward (decode.py:755-772), replays DecodeAccelerator._step_graph (the CUDA graph of compiled_step)
    one token at a time. Feeds the last prompt token → each CoT token (up to the anchor <|traj_future_start|>) in order, so
    the KV read by the expert is written at the same positions as when official sampling produces the same CoT. No RNG is used.
    """

    def __init__(self, dec, keep):
        self.dec, self.keep = dec, keep
        self.tokens = None
        self.scores = None

    def __call__(self, model, input_ids, logits_processor, stopping_criteria, generation_config,
                 synced_gpus=False, streamer=None, **model_kwargs):
        from alpamayo2_edge.accel.decode import FlashKVCache
        from transformers.generation.utils import GenerateDecoderOnlyOutput

        cache = model_kwargs["past_key_values"]
        assert isinstance(cache, FlashKVCache) and not cache.frozen and input_ids.shape[1] == cache.length + 1
        masks = [p for p in logits_processor if type(p).__name__ in self.keep]
        graph = self.dec._step_graph(cache)
        forced = torch.tensor(self.tokens, dtype=torch.long, device=input_ids.device).view(1, -1)
        feed = [input_ids[:, -1:]] + [forced[:, i : i + 1] for i in range(forced.shape[1])]
        out = []
        for tok in feed:
            cache.input_ids.copy_(tok)
            cache.seqlens.fill_(cache.length)
            logits = graph()
            cache.length += 1
            scores = logits[:, -1, :].to(torch.float32).clone()
            for p in masks:
                scores = p(input_ids, scores)
            out.append(scores[0])
        self.scores = torch.stack(out[:-1])  # the distribution at position i predicts the i-th CoT token
        pad = torch.full((1, 1), model.generation_config.pad_token_id or 0, dtype=torch.long, device=input_ids.device)
        seq = torch.cat([input_ids, forced, pad], dim=1)
        return GenerateDecoderOnlyOutput(sequences=seq, past_key_values=cache)


def dist_metrics(sb, sv, tokens):
    """sb, sv: masked logits [L, V] (FP32). KL(base‖variant), etc."""
    fin = torch.isfinite(sb)
    out = {}
    for T, tag in ((1.0, "t1"), (0.6, "t06")):
        lb = torch.log_softmax(sb / T, -1)
        lv = torch.log_softmax(sv / T, -1)
        pb = lb.exp()
        kl = torch.where(fin, pb * (lb - lv), torch.zeros_like(pb)).sum(-1)
        out[f"kl_{tag}"] = kl.tolist()
        if tag == "t06":
            out["tv_t06"] = (0.5 * (pb - lv.exp()).abs().sum(-1)).tolist()
            idx = torch.tensor(tokens, device=sb.device)
            out["lp_ref_base"] = lb.gather(1, idx[:, None])[:, 0].tolist()
            out["lp_ref_var"] = lv.gather(1, idx[:, None])[:, 0].tolist()
            out["entropy_base"] = (-(torch.where(fin, pb * lb, torch.zeros_like(pb))).sum(-1)).tolist()
    out["top1_same"] = (sb.argmax(-1) == sv.argmax(-1)).tolist()
    return out


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


