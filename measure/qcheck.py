#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# qcheck.py — デコードの実装を差し替えたときの品質を、サンプリングの影響を受けない見方で数分で確かめる
#
# 1 つのプロセスで、デコードの実装(変種)を切り替えながら同じ入力を流す。前処理・ViT・プリフィル・expert は共通。
#   base    : 公式 tuned の BF16 デコード(出発点。比べる基準)
#   eager   : 同じ BF16 の DecodeStep を compile せずに動かす(行列積の分け方と融合が変わるだけ = BF16 の丸めの揺れの幅)
#   fp8body : 本体 28 層の射影と MLP だけ FP8
#   fp8all  : 本体 + 出力層を FP8
#
# 見るもの(どれも同じ入力で、変種と base を比べる):
#  A. teacher forcing: base が seed 43〜47 で出した判断理由(CoT)を固定の入力として 1 トークンずつ流し、各位置の
#     次トークンの分布を比べる(KL(base‖変種) を温度 1 と 0.6 で、上位 1 位の一致、固定トークンの対数確率の差)。
#     マスク(軌跡トークン・EOS)は公式と同じものを掛ける。top-p は掛けない(KL が無限大になるため)。
#     さらに同じ CoT・同じ拡散の乱数(seed 1000)で expert まで回し、軌跡の差を見る。
#     比べる幅: base で拡散の乱数だけを 1001 に変えたときの軌跡の差。
#  B. greedy(top_k=1): 何トークン目から base と分かれるか、軌跡の差(拡散の乱数は seed 43 に固定)。
#  C. 公式のサンプリング(温度 0.6、top_p 0.98)を seed 43〜47 で: 同じ seed で base と CoT が同じか(診断用)、
#     軌跡の差。比べる幅は base 同士で seed だけ違うときの差。
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
    """generate(custom_generate=...) の代わりに、決めたトークン列を流して KV を書き、各位置の logits を集める。

    公式の forward(decode.py:755-772)と同じく、DecodeAccelerator._step_graph(compiled_step の CUDA グラフ)を
    1 トークンずつ再生する。最後のプロンプトトークン → CoT の各トークン(アンカー <|traj_future_start|> まで)の順に流すので、
    expert が読む KV は公式のサンプリングで同じ CoT が出たときと同じ位置に書かれる。乱数は使わない。
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
        self.scores = torch.stack(out[:-1])  # 位置 i の分布が CoT の i 番目のトークンを予測する
        pad = torch.full((1, 1), model.generation_config.pad_token_id or 0, dtype=torch.long, device=input_ids.device)
        seq = torch.cat([input_ids, forced, pad], dim=1)
        return GenerateDecoderOnlyOutput(sequences=seq, past_key_values=cache)


def dist_metrics(sb, sv, tokens):
    """sb, sv: [L, V] のマスク済み logits(FP32)。KL(base‖変種) ほか。"""
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


