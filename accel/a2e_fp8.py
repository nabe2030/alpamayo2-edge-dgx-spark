# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030 (modifications)
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code modified from NVlabs/alpamayo2-edge (https://github.com/NVlabs/alpamayo2-edge,
# commit 02f55a7), licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt).
# Modified in 2026: DecodeStepQ.forward follows the CoT decode step of alpamayo2_edge/accel/decode.py; its linear layers call an FP8 (E4M3) GEMV kernel instead of BF16 weights.
#
# a2e_fp8.py — replacement that reads weights in FP8 during CoT decoding
#
# Upstream alpamayo2-edge files are not modified. The compiled_step (decode.py:563) of the DecodeAccelerator built by the official
# acceleration (enable_acceleration) is swapped, before the first call, for DecodeStepQ, which reads quantized weights.
# Prefill, expert and ViT share the same nn.Linear, so the original weights are not rewritten; a separate FP8 copy is kept.
#
#   import a2e_fp8 as Q
#   handle = enable_acceleration(model, AccelConfig(mode="tuned"))
#   Q.install(handle, body=True, head=True)     # body (28 layers) and output layer in FP8
#
# Quantization: per output channel (row) scale = max|W[n,:]| / 448, W8 = E4M3(W / scale) (round to nearest). Activations stay BF16.
# Fusion: q, k, v are merged into one [4096, 2048]. The relu² after fc1 is computed in FP32 at the kernel's exit.
from pathlib import Path

import os
import torch

P = Path(os.environ.get("A2E_ROOT", "."))
DEFAULT_SO = os.environ.get("A2E_FP8_SO", str(P / "build/fp8_gemv/v1/fp8_gemv.so"))
FP8_MAX = 448.0
# rows per warp for each shape (chosen from the bench_gemv.py results)
DEFAULT_ROWS = {"qkv": 1, "o": 1, "fc1": 1, "fc2": 1, "head": 1}
_loaded: dict[str, str] = {}


def load_ops(so: str | Path | None = None):
    """Load the built .so and register the fake for torch.compile (returns only the output shape and dtype)."""
    so = str(so or DEFAULT_SO)
    if not _loaded:
        torch.ops.load_library(so)

        @torch.library.register_fake("a2e::fp8_gemv")
        def _fp8_gemv_fake(x, w, scale, act, rows):
            del scale, act, rows
            return x.new_empty((*x.shape[:-1], w.shape[0]))

        _loaded["so"] = so
    elif _loaded["so"] != so:
        raise RuntimeError(f"a different .so is already loaded: {_loaded['so']}")
    return torch.ops.a2e


def quantize_rowwise(weight: torch.Tensor, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [N, K] weights to E4M3 with per-row scales. Returns (uint8 [N, K], FP32 [N]).

    Rows are independent, so they are processed chunk rows at a time to keep the FP32 scratch small (same result as processing all at once).
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
    """Bias-free linear layer that reads FP8 weights (the original weights are not touched)."""

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
    """FP8 projections for one layer: qkv (fused), o, fc1 (+relu²), fc2."""

    def __init__(self, layer, rows: dict[str, int]) -> None:
        super().__init__()
        attn, mlp = layer.self_attn, layer.mlp
        if getattr(mlp.activation_fn, "__class__", None).__name__ != "ReLUSquaredActivation":
            raise RuntimeError(f"unexpected activation function: {mlp.activation_fn!r}")
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.fc1, mlp.fc2):
            if lin.bias is not None:
                raise RuntimeError("projections with bias are not supported")
        qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
        self.nq, self.nkv = attn.q_proj.out_features, attn.k_proj.out_features
        self.qkv = QLinear(qkv, rows["qkv"])
        self.o = QLinear(attn.o_proj.weight, rows["o"])
        self.fc1 = QLinear(mlp.fc1.weight, rows["fc1"], act=1)
        self.fc2 = QLinear(mlp.fc2.weight, rows["fc2"])


def decoder_block_q(layer, ql: QLayer, hidden, cos, sin, k_cache, v_cache, seqlens, causal: bool):
    """Same computation as upstream decode.decoder_block (decode.py:194-221), with only the projections using FP8 weights."""
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
    """Replacement for upstream DecodeStep (decode.py:257-281). Parts where qlayers / qhead is None stay BF16."""

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
    raise RuntimeError("DecodeAccelerator not found (decode acceleration is not installed)")


def compiled(step, handle):
    from alpamayo2_edge.accel import _DECODE_MODES
    from alpamayo2_edge.accel.graphs import maybe_compile

    return maybe_compile(step, handle.config.compile, _DECODE_MODES[handle.config.mode])


def install(handle, body: bool = True, head: bool = True, rows: dict[str, int] | None = None) -> DecodeStepQ:
    """Swap the official DecodeAccelerator's compiled_step for the FP8 version. Call before CUDA graphs are built."""
    dec = decode_accelerator(handle)
    if dec.step_graphs or dec.sampler.graphs:
        raise RuntimeError("cannot swap after the decode graphs are built (install before the first call)")
    step = build_step(dec.vlm, body, head, rows)
    dec.compiled_step = compiled(step, handle)
    dec._a2e_fp8 = {"body": body, "head": head, "rows": {**DEFAULT_ROWS, **(rows or {})}, "so": _loaded.get("so")}
    return step


class DecodeVariants:
    """Switch the decode implementation (BF16 / FP8, etc.) within one process (for quality checks).

    Keeps per variant a compiled_step, the one-step decode graphs (step_graphs) and a sampler (GraphSampler and its graphs);
    use() swaps those 3 attributes of DecodeAccelerator together. Prefill and expert graphs are shared.
    """

    def __init__(self, handle) -> None:
        from alpamayo2_edge.accel.decode import GraphSampler

        self._GraphSampler = GraphSampler
        self.dec = decode_accelerator(handle)
        self.handle = handle
        self.items: dict[str, dict] = {}
        self.current = None
        self.add("base", self.dec.compiled_step)  # official tuned (BF16)

    def add(self, name: str, compiled_step) -> None:
        dec = self.dec
        saved = (dec.compiled_step, dec.step_graphs, dec.sampler)
        dec.compiled_step = compiled_step
        sampler = self._GraphSampler(dec)  # GraphSampler reads dec.compiled_step when it builds graphs
        dec.compiled_step, dec.step_graphs, dec.sampler = saved
        self.items[name] = {"compiled_step": compiled_step, "step_graphs": {}, "sampler": sampler}

    def use(self, name: str) -> None:
        it = self.items[name]
        self.dec.compiled_step, self.dec.step_graphs, self.dec.sampler = it["compiled_step"], it["step_graphs"], it["sampler"]
        self.current = name
