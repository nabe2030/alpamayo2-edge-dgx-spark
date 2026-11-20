# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030 (modifications)
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code modified from NVlabs/alpamayo2-edge (https://github.com/NVlabs/alpamayo2-edge,
# commit 02f55a7), licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt).
# Modified in 2026: DecodeStepQ.forward follows the CoT decode step of alpamayo2_edge/accel/decode.py; its linear layers call an FP8 (E4M3) GEMV kernel instead of BF16 weights.
# このファイルは、NVlabs/alpamayo2-edge(コミット 02f55a7、Apache License 2.0)のコードを変更したものを含む。
# 変更(2026 年): DecodeStepQ.forward は alpamayo2_edge/accel/decode.py の CoT デコードの 1 ステップを写し、線形層を FP8(E4M3)の GEMV カーネルに差し替えた。
#
# a2e_fp8.py — CoT デコードで重みを FP8 で読む差し替え
#
# 本家 alpamayo2-edge のファイルは変えない。公式の加速(enable_acceleration)が作った DecodeAccelerator の
# compiled_step(decode.py:563)を、最初の呼び出しより前に、量子化済みの重みを読む DecodeStepQ に入れ替える。
# プリフィル・expert・ViT は同じ nn.Linear を共有しているので、元の重みは書き換えず、FP8 の写しを別に持つ。
#
#   import a2e_fp8 as Q
#   handle = enable_acceleration(model, AccelConfig(mode="tuned"))
#   Q.install(handle, body=True, head=True)     # 本体 28 層と出力層を FP8 に
#
# 量子化: 出力チャネル(行)ごとに scale = max|W[n,:]| / 448、W8 = E4M3(W / scale)(最近接丸め)。活性化は BF16 のまま。
# 融合: q・k・v を 1 本の [4096, 2048] にまとめる。fc1 の後ろの relu² はカーネルの出口で FP32 のまま計算する。
from pathlib import Path

import os
import torch

P = Path(os.environ.get("A2E_ROOT", "."))
DEFAULT_SO = os.environ.get("A2E_FP8_SO", str(P / "build/fp8_gemv/v1/fp8_gemv.so"))
FP8_MAX = 448.0
# 形ごとの 1 ワープあたりの行数(bench_gemv.py の結果で決める)
DEFAULT_ROWS = {"qkv": 1, "o": 1, "fc1": 1, "fc2": 1, "head": 1}
_loaded: dict[str, str] = {}


def load_ops(so: str | Path | None = None):
    """ビルド済みの .so を読み、torch.compile 用の fake(出力の形と dtype だけを返す)を登録する。"""
    so = str(so or DEFAULT_SO)
    if not _loaded:
        torch.ops.load_library(so)

        @torch.library.register_fake("a2e::fp8_gemv")
        def _fp8_gemv_fake(x, w, scale, act, rows):
            del scale, act, rows
            return x.new_empty((*x.shape[:-1], w.shape[0]))

        _loaded["so"] = so
    elif _loaded["so"] != so:
        raise RuntimeError(f"別の .so がすでに読まれている: {_loaded['so']}")
    return torch.ops.a2e


def quantize_rowwise(weight: torch.Tensor, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] の重みを、行ごとのスケールの E4M3 にする。戻り値は (uint8 [N, K], FP32 [N])。

    行ごとに独立な計算なので、FP32 の一時領域を小さくするために chunk 行ずつ処理する(結果は一度に処理したときと同じ)。
    """
    w = weight.detach()
    q = torch.empty(w.shape, dtype=torch.uint8, device=w.device)
    scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for i in range(0, w.shape[0], chunk):
        wf = w[i : i + chunk].float()
        s = (wf.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
        q[i : i + chunk] = (wf / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        scale[i : i + chunk] = s
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.view(torch.float8_e4m3fn).float() * scale[:, None]


class QLinear(torch.nn.Module):
    """FP8 の重みを読む、バイアスなしの線形層(元の重みは触らない)。"""

    def __init__(self, weight: torch.Tensor, rows: int, act: int = 0) -> None:
        super().__init__()
        q, s = quantize_rowwise(weight)
        self.register_buffer("w", q, persistent=False)
        self.register_buffer("scale", s, persistent=False)
        self.rows, self.act = int(rows), int(act)
        self.out_features, self.in_features = q.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.a2e.fp8_gemv(x.to(torch.bfloat16), self.w, self.scale, self.act, self.rows)

    def nbytes(self) -> int:
        return self.w.numel() + self.scale.numel() * 4


class QLayer(torch.nn.Module):
    """1 層ぶんの FP8 の射影: qkv(融合)、o、fc1(+relu²)、fc2。"""

    def __init__(self, layer, rows: dict[str, int]) -> None:
        super().__init__()
        attn, mlp = layer.self_attn, layer.mlp
        if getattr(mlp.activation_fn, "__class__", None).__name__ != "ReLUSquaredActivation":
            raise RuntimeError(f"想定外の活性化関数: {mlp.activation_fn!r}")
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.fc1, mlp.fc2):
            if lin.bias is not None:
                raise RuntimeError("バイアス付きの射影は想定していない")
        qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
        self.nq, self.nkv = attn.q_proj.out_features, attn.k_proj.out_features
        self.qkv = QLinear(qkv, rows["qkv"])
        self.o = QLinear(attn.o_proj.weight, rows["o"])
        self.fc1 = QLinear(mlp.fc1.weight, rows["fc1"], act=1)
        self.fc2 = QLinear(mlp.fc2.weight, rows["fc2"])


def decoder_block_q(layer, ql: QLayer, hidden, cos, sin, k_cache, v_cache, seqlens, causal: bool):
    """本家 decode.decoder_block(decode.py:194-221)と同じ計算を、射影だけ FP8 の重みで行う。"""
    from alpamayo2_edge.accel.decode import apply_rotary_pos_emb, kvcache_attention

    attn = layer.self_attn
    rows, seq, _ = hidden.shape
    x = layer.input_layernorm(hidden)
    q, k, v = ql.qkv(x).split((ql.nq, ql.nkv, ql.nkv), dim=-1)
    q = q.reshape(rows, seq, -1, attn.head_dim)
    k = k.reshape(rows, seq, -1, attn.head_dim)
    v = v.reshape(rows, seq, -1, attn.head_dim)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    dtype = k_cache.dtype
    out = kvcache_attention(
        q.to(dtype).contiguous(), k_cache, v_cache, k.to(dtype).contiguous(), v.to(dtype).contiguous(),
        seqlens, attn.scaling, causal,
    )
    hidden = hidden + ql.o(out.reshape(rows, seq, -1))
    return hidden + ql.fc2(ql.fc1(layer.post_attention_layernorm(hidden)))


class DecodeStepQ(torch.nn.Module):
    """本家 DecodeStep(decode.py:257-281)の差し替え。qlayers / qhead が None の部分は BF16 のまま。"""

    def __init__(self, vlm, qlayers: torch.nn.ModuleList | None, qhead: QLinear | None) -> None:
        super().__init__()
        self.text = vlm.model.language_model
        self.lm_head = vlm.lm_head
        self.qlayers, self.qhead = qlayers, qhead

    def forward(self, input_ids, rope_delta, seqlens, k_caches, v_caches):
        from alpamayo2_edge.accel.decode import decoder_block

        text = self.text
        hidden = text.embed_tokens(input_ids)
        rows = hidden.shape[0]
        positions = (seqlens.to(torch.long) + rope_delta).view(1, rows, 1).expand(3, rows, 1)
        cos, sin = text.rotary_emb(hidden, positions)
        for index, layer in enumerate(text.layers):
            if self.qlayers is None:
                hidden = decoder_block(layer, hidden, cos, sin, k_caches[index], v_caches[index], seqlens, True)
            else:
                hidden = decoder_block_q(layer, self.qlayers[index], hidden, cos, sin, k_caches[index],
                                         v_caches[index], seqlens, True)
        normed = text.norm(hidden)
        logits = self.qhead(normed) if self.qhead is not None else self.lm_head(normed)
        seqlens.add_(1)
        return logits

    def weight_bytes(self) -> dict[str, int]:
        body = sum(m.nbytes() for m in self.qlayers.modules() if isinstance(m, QLinear)) if self.qlayers else 0
        return {"body_fp8": body, "head_fp8": self.qhead.nbytes() if self.qhead is not None else 0}


def build_step(vlm, body: bool, head: bool, rows: dict[str, int] | None = None) -> DecodeStepQ:
    rows = {**DEFAULT_ROWS, **(rows or {})}
    load_ops()
    with torch.no_grad():
        qlayers = torch.nn.ModuleList(QLayer(layer, rows) for layer in vlm.model.language_model.layers) if body else None
        qhead = QLinear(vlm.lm_head.weight, rows["head"]) if head else None
    return DecodeStepQ(vlm, qlayers, qhead)


def decode_accelerator(handle):
    from alpamayo2_edge.accel.decode import DecodeAccelerator

    for acc in handle.accelerators:
        if isinstance(acc, DecodeAccelerator):
            return acc
    raise RuntimeError("DecodeAccelerator が見つからない(decode の加速が入っていない)")


def compiled(step, handle):
    from alpamayo2_edge.accel import _DECODE_MODES
    from alpamayo2_edge.accel.graphs import maybe_compile

    return maybe_compile(step, handle.config.compile, _DECODE_MODES[handle.config.mode])


def install(handle, body: bool = True, head: bool = True, rows: dict[str, int] | None = None) -> DecodeStepQ:
    """公式の DecodeAccelerator の compiled_step を FP8 版に入れ替える。CUDA グラフを作る前に呼ぶ。"""
    dec = decode_accelerator(handle)
    if dec.step_graphs or dec.sampler.graphs:
        raise RuntimeError("デコードのグラフを作った後には入れ替えられない(最初の呼び出しより前に install する)")
    step = build_step(dec.vlm, body, head, rows)
    dec.compiled_step = compiled(step, handle)
    dec._a2e_fp8 = {"body": body, "head": head, "rows": {**DEFAULT_ROWS, **(rows or {})}, "so": _loaded.get("so")}
    return step


class DecodeVariants:
    """1 つのプロセスの中で、デコードの実装(BF16 / FP8 など)を切り替える(品質の確認用)。

    変種ごとに compiled_step、デコード 1 歩のグラフ(step_graphs)、サンプラー(GraphSampler と、その中のグラフ)を持ち、
    use() で DecodeAccelerator の 3 つの属性をまとめて差し替える。プリフィルと expert のグラフは共有する。
    """

    def __init__(self, handle) -> None:
        from alpamayo2_edge.accel.decode import GraphSampler

        self._GraphSampler = GraphSampler
        self.dec = decode_accelerator(handle)
        self.handle = handle
        self.items: dict[str, dict] = {}
        self.current = None
        self.add("base", self.dec.compiled_step)  # 公式の tuned(BF16)

    def add(self, name: str, compiled_step) -> None:
        dec = self.dec
        saved = (dec.compiled_step, dec.step_graphs, dec.sampler)
        dec.compiled_step = compiled_step
        sampler = self._GraphSampler(dec)  # GraphSampler はグラフを作るときに dec.compiled_step を読む
        dec.compiled_step, dec.step_graphs, dec.sampler = saved
        self.items[name] = {"compiled_step": compiled_step, "step_graphs": {}, "sampler": sampler}

    def use(self, name: str) -> None:
        it = self.items[name]
        self.dec.compiled_step, self.dec.step_graphs, self.dec.sampler = it["compiled_step"], it["step_graphs"], it["sampler"]
        self.current = name
