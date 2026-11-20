#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030 (modifications)
# SPDX-License-Identifier: Apache-2.0
#
# This file contains code modified from NVlabs/alpamayo2-edge (https://github.com/NVlabs/alpamayo2-edge,
# commit 02f55a7), licensed under the Apache License, Version 2.0 (see LICENSES/Apache-2.0.txt).
# Modified in 2026: GpuPrep follows alpamayo2_edge/helper.py prepare_model_inputs and moves image resizing, patching and normalization to the GPU; install_processor_once reads the processor once per checkpoint.
#
# a2e_stagea.py — applies the four acceleration optimizations from the outside, without rewriting
# upstream alpamayo2-edge (02f55a7) code. Each one is a replacement on a single model instance / module function,
# and is reversible (recreating the process restores the original).
#
#   #1 install_processor_once: the same change as upstream PR #11 (ea6c37c). helper.get_processor
#      calls AutoProcessor.from_pretrained only once per checkpoint and returns a shallow copy on each call.
#   #9 GpuPrep: builds the same inputs as prepare_model_inputs. Image resizing (bicubic, antialias), patchifying and normalization
#      run on the GPU. The prompt string and token sequence are built with the same functions as upstream (build_conversation, apply_chat_template,
#      processor tokenization).
#   #2 VitFeatureCache: wraps get_image_features and reuses features of images already processed in a previous call (same camera, capture time, grid and
#      pixel fingerprint). Only new images go through the ViT. Used together with GpuPrep;
#      reused images are neither resized nor transferred.
#   #3 VisionAccel: turns ViT (SigLIP2, 27 layers) + final LayerNorm + projector into a single attention computation over
#      images of the same grid, and uses torch.compile + CUDA graphs (one graph per grid and image count).
#      Ports the idea of accel/vision.py from upstream PR #2 (32685e4) (for Qwen3-VL, removed in PR #8) to Cosmos3-Edge.
#
# Replacement order: VisionAccel → VitFeatureCache → (timing wrapper). GpuPrep replaces the preprocessing function.

from __future__ import annotations

import copy
import functools
import hashlib
import io
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- #1 load the processor once (PR #11)
def install_processor_once() -> None:
    """Replace helper.get_processor exactly as in upstream PR #11 (lru_cache + copy.copy)."""
    from alpamayo2_edge import helper
    from alpamayo2_edge.config import resolve_checkpoint_name_or_path

    if getattr(helper.get_processor, "_stagea_pr11", False):
        return

    @functools.lru_cache(maxsize=None)
    def _load_processor(checkpoint: str):
        return helper.AutoProcessor.from_pretrained(checkpoint, fix_mistral_regex=True)

    def get_processor(tokenizer, model_config):
        checkpoint = resolve_checkpoint_name_or_path(model_config)
        if checkpoint is None:
            raise ValueError("model_config must include a checkpoint path or vlm_name_or_path")
        processor = copy.copy(_load_processor(str(checkpoint)))
        processor.tokenizer = tokenizer
        return processor

    get_processor._stagea_pr11 = True
    helper._load_processor = _load_processor
    helper.get_processor = get_processor


# ---------------------------------------------------------------- #2 reuse ViT results per image
def frame_fingerprint(frame: torch.Tensor) -> str:
    """128-bit fingerprint from subsampled pixels of a raw frame (uint8 [.., 3, H, W]) (guards against mixing up identical IDs)."""
    sub = frame.reshape(-1, *frame.shape[-3:])[0, :, ::37, ::41].contiguous()
    return hashlib.blake2b(sub.cpu().numpy().tobytes(), digest_size=16).hexdigest()  # also accepts images on the GPU


@dataclass
class _Plan:
    n_images: int
    keys: list
    misses: list[int]
    reused: dict[int, torch.Tensor] = field(default_factory=dict)


class VitFeatureCache:
    """Wrap ``vlm.model.get_image_features`` and reuse the features (projector output) of already processed images.

    The Cosmos3-Edge ViT is independent per image: attention is split per image by cu_seqlens
    (modeling_cosmos3_edge.py:588-598), and patch embedding, position embedding, LayerNorm, MLP, 2x2 merge and
    projector are per-patch (per-token) computations. So the features of one image do not depend on the other images in the same call.
    """

    def __init__(self, vlm_model: torch.nn.Module, max_images: int = 256):
        self.m = vlm_model
        self.inner = vlm_model.get_image_features
        self.max_images = max_images
        self.store: OrderedDict = OrderedDict()
        self.plan: _Plan | None = None
        self.stats = {"calls": 0, "images_encoded": 0, "images_reused": 0, "uncached_calls": 0}
        vlm_model.get_image_features = self.forward

    def clear(self):
        self.store.clear()
        self.plan = None

    def lookup(self, keys: list) -> list[int]:
        reused, misses = {}, []
        for i, k in enumerate(keys):
            f = self.store.get(k)
            if f is None:
                misses.append(i)
            else:
                self.store.move_to_end(k)
                reused[i] = f
        self.plan = _Plan(len(keys), keys, misses, reused)
        return misses

    def forward(self, pixel_values, image_grid_thw=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        plan, self.plan = self.plan, None
        self.stats["calls"] += 1
        if plan is None:
            self.stats["uncached_calls"] += 1
            return self.inner(pixel_values, image_grid_thw, **kwargs)
        if image_grid_thw.shape[0] != plan.n_images:
            raise RuntimeError("ViT cache: grid does not match the prepared plan")
        fresh = {}
        if plan.misses:
            idx = torch.tensor(plan.misses, device=image_grid_thw.device)
            out = self.inner(pixel_values, image_grid_thw.index_select(0, idx), **kwargs)
            for j, i in enumerate(plan.misses):
                feat = out.pooler_output[j].clone()  # may point to a graph's static output, so keep a copy
                fresh[i] = feat
                self.store[plan.keys[i]] = feat
                self.store.move_to_end(plan.keys[i])
        while len(self.store) > self.max_images:
            self.store.popitem(last=False)
        self.stats["images_encoded"] += len(plan.misses)
        self.stats["images_reused"] += plan.n_images - len(plan.misses)
        parts = tuple(fresh[i] if i in fresh else plan.reused[i] for i in range(plan.n_images))
        return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=parts)


# ---------------------------------------------------------------- #9 resize images on the GPU
class GpuPrep:
    """Build the same inputs as ``helper.prepare_model_inputs`` + ``to_device`` (only image processing on the GPU).

    The only difference from upstream is where images are resized (CPU uint8-only path → GPU float path);
    resized uint8 pixels can differ by up to 1. The prompt string,
    token sequence, mm_token_type_ids and image_grid_thw have been verified to match upstream.
    """

    def __init__(self, model, device="cuda", cache: VitFeatureCache | None = None, pin: bool = False):
        from alpamayo2_edge import helper

        self.helper = helper
        self.cfg = model.config
        self.tokenizer = model.tokenizer
        self.device = torch.device(device)
        self.cache = cache
        self.pin = pin
        self.processor = helper.get_processor(model.tokenizer, model.config)
        ip = self.processor.image_processor
        self.ip = ip
        self.factor = ip.patch_size * ip.merge_size
        self.last = {}

    def _canonical(self, data):
        from alpamayo2_edge.sample import canonical_frame_order, normalize_camera_pixel_bounds, CAMERA_PIXEL_BOUNDS, validate_sample

        # same checks as helper._canonical_inputs (history token count, image layout) and validate_sample
        cfg = self.cfg
        history_steps = getattr(cfg, "history_num_steps", 20)
        if cfg.tokens_per_history_traj != (history_steps - 1) * 3:
            raise ValueError("history token count mismatch")
        self.helper._check_image_layout(cfg)
        validate_sample(data, history_num_steps=history_steps)
        # same reordering as _sort_camera_inputs. Images are not copied; only a list of references is built (upstream copies 22 images on the CPU)
        order = canonical_frame_order(data["camera_indices"], data["absolute_timestamps"])
        result = dict(data)
        frames = data["image_frames"]
        result["image_frames"] = [frames[j] for j in order.tolist()]
        result["camera_indices"] = data["camera_indices"][order]
        result["absolute_timestamps"] = data["absolute_timestamps"].reshape(-1)[order]
        result["camera_pixel_bounds"] = {**CAMERA_PIXEL_BOUNDS,
                                         **normalize_camera_pixel_bounds(data.get("camera_pixel_bounds", {}))}
        return result

    def __call__(self, data, stream_id=None):
        from alpamayo2_edge.chat_template import CLOSED_THINKING, build_conversation
        from torchvision.transforms.functional import InterpolationMode, resize
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        cfg, proc, ip = self.cfg, self.processor, self.ip
        canonical = self._canonical(data)
        messages = build_conversation(
            canonical,
            num_tokens_per_history_traj=cfg.tokens_per_history_traj,
            include_camera_ids=cfg.include_camera_ids,
            system_prompt=getattr(cfg, "system_prompt", None),
        )
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        add_vision_id=False, continue_final_message=False,
                                        enable_thinking=False)
        if not text.endswith(CLOSED_THINKING):
            raise ValueError("chat template must end with the closed thinking block")
        entries = [e for m in messages for e in m["content"] if e["type"] == "image"]
        sizes, grids = [], []
        for e in entries:
            h, w = smart_resize(*e["image"].shape[-2:], factor=self.factor,
                                min_pixels=e["min_pixels"], max_pixels=e["max_pixels"])
            sizes.append((h, w))
            grids.append((1, h // ip.patch_size, w // ip.patch_size))
        # which images go through the ViT
        todo = list(range(len(entries)))
        if self.cache is not None:
            cams = canonical["camera_indices"].tolist()
            ts = canonical["absolute_timestamps"].tolist()
            sid = stream_id if stream_id is not None else data.get("clip_id")
            keys = [(sid, c, t, g, frame_fingerprint(e["image"]))
                    for c, t, g, e in zip(cams, ts, grids, entries)]
            todo = self.cache.lookup(keys)
        # images: transfer → resize → patchify (upstream image_processor.patchify) → normalize (upstream _normalize_patches)
        patches = []
        for i in todo:
            frame = entries[i]["image"].reshape(-1, *entries[i]["image"].shape[-3:])[0]
            if self.pin:
                frame = frame.pin_memory()
            g = frame.to(self.device, non_blocking=self.pin)
            r = resize(g, list(sizes[i]), interpolation=InterpolationMode.BICUBIC)
            p, _, _ = ip.patchify(r[None], patch_size=ip.patch_size, merge_size=ip.merge_size,
                                  temporal_patch_size=ip.temporal_patch_size)
            patches.append(p[0])
        if patches:
            raw = torch.cat(patches)
        else:
            dim = 3 * ip.temporal_patch_size * ip.patch_size ** 2
            raw = torch.empty(0, dim, dtype=torch.uint8, device=self.device)
        pixel_values = self.helper._normalize_patches(raw, ip)
        # string: expand the number of <|image_pad|> per image with the same rule as the upstream processor, then tokenize
        merge = ip.merge_size ** 2
        repl = [proc.image_token * (t * h * w // merge) for t, h, w in grids]
        expanded, _ = proc.get_text_with_replacements([text], images_replacements=repl)
        enc = proc.tokenizer(expanded, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"]
        mm = torch.zeros_like(input_ids)
        mm[input_ids == proc.image_token_id] = 1
        tokenized = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "mm_token_type_ids": mm.to(self.device),
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor(grids, dtype=torch.long).to(self.device),
        }
        if input_ids.shape[0] != 1:
            raise ValueError("one sample at a time")
        xyz, rot = canonical["ego_history_xyz"], canonical["ego_history_rot"]
        self.last = {"images_processed": len(todo), "images_total": len(entries)}
        return {
            "tokenized_data": tokenized,
            "ego_history_xyz": (xyz.unsqueeze(0) if xyz.ndim == 3 else xyz).to(self.device),
            "ego_history_rot": (rot.unsqueeze(0) if rot.ndim == 4 else rot).to(self.device),
        }


# ---------------------------------------------------------------- #3 compile the ViT / CUDA graphs
class _Tower(torch.nn.Module):
    """Run ``Cosmos3EdgeVisionModel.forward`` + ``projector`` on n images of the same grid.

    The upstream SDPA path splits per image and calls SDPA one image at a time. Images of the same grid have the same length,
    so one SDPA call on [n, heads, L, head_dim] gives the same computation. Each layer's modules (LayerNorm, Linear, MLP)
    are the upstream ones, called as is.
    """

    def __init__(self, visual, projector, n: int, L: int):
        super().__init__()
        self.visual, self.projector, self.n, self.L = visual, projector, n, L

    def forward(self, pixels, pos):
        v = self.visual
        emb = v.embeddings
        h = emb.patch_embedding(pixels.to(emb.patch_embedding.weight.dtype)) + pos
        n, L = self.n, self.L
        for layer in v.encoder.layers:
            a = layer.self_attn
            r = h
            x = layer.layer_norm1(h)
            q = a.q_proj(x).reshape(n, L, a.num_heads, a.head_dim).transpose(1, 2)
            k = a.k_proj(x).reshape(n, L, a.num_heads, a.head_dim).transpose(1, 2)
            val = a.v_proj(x).reshape(n, L, a.num_heads, a.head_dim).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, val, attn_mask=None, dropout_p=0.0,
                                               scale=a.scale, is_causal=False)
            o = o.transpose(1, 2).reshape(n * L, -1).contiguous()
            h = r + a.out_proj(o)
            r = h
            h = r + layer.mlp(layer.layer_norm2(h))
        h = v.post_layernorm(h)
        return self.projector(h)


class VisionAccel:
    """Replacement for ``vlm.model.get_image_features``. Groups images by grid and replays the graph for each (grid, count)."""

    def __init__(self, vlm_model, *, compile: bool = True, cuda_graphs: bool = True,
                 mode: str = "default", warmup: int = 2):
        from alpamayo2_edge.accel.graphs import AccelStats

        self.m = vlm_model
        self.visual = vlm_model.visual
        self.projector = vlm_model.projector
        self.inner = vlm_model.get_image_features
        self.compile, self.cuda_graphs, self.mode, self.warmup = compile, cuda_graphs, mode, warmup
        self.graphs: dict = {}
        self.index_cache: dict = {}
        self.stats = AccelStats()
        self.merge = int(self.projector.spatial_merge_size)
        vlm_model.get_image_features = self.forward

    def _pos(self, row, n):
        emb = self.visual.embeddings
        table = emb.position_embedding.weight.reshape(emb.position_embedding_size, emb.position_embedding_size, -1)
        grid = torch.tensor([row] * n, dtype=torch.long)
        return emb.resize_positional_embeddings(table, grid, self.visual.config.spatial_merge_size)

    def _graph(self, row, n, device):
        from alpamayo2_edge.accel.graphs import GraphedCall, autocast_state, maybe_compile

        key = (row, n, autocast_state(device.type))
        g = self.graphs.get(key)
        if g is None:
            t, hh, ww = row
            L = t * hh * ww
            tower = maybe_compile(_Tower(self.visual, self.projector, n, L), self.compile, self.mode)
            pix = torch.zeros(n * L, 3 * 16 * 16, dtype=torch.float32, device=device)
            pos = self._pos(row, n).to(device)
            g = GraphedCall("vision", (row, n), tower, (pix, pos), stats=self.stats,
                            warmup=self.warmup if self.cuda_graphs else 0,
                            cuda_graph=self.cuda_graphs, device=device)
            self.graphs[key] = g
        return g

    def forward(self, pixel_values, image_grid_thw=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        rows = [tuple(int(x) for x in r) for r in image_grid_thw.tolist()]
        patches = [t * h * w for t, h, w in rows]
        if sum(patches) != pixel_values.shape[0] or pixel_values.shape[-1] != 768:
            self.stats.fallback("vision_shape")
            return self.inner(pixel_values, image_grid_thw, **kwargs)
        device = pixel_values.device
        starts = [sum(patches[:i]) for i in range(len(rows))]
        groups: dict = {}
        for i, r in enumerate(rows):
            groups.setdefault(r, []).append(i)
        outs: list = [None] * len(rows)
        for r, members in groups.items():
            g = self._graph(r, len(members), device)
            L = patches[members[0]]
            ikey = (tuple(rows), r, device)
            idx = self.index_cache.get(ikey)
            if idx is None:
                idx = torch.cat([torch.arange(starts[i], starts[i] + L) for i in members]).to(device)
                self.index_cache[ikey] = idx
            torch.index_select(pixel_values, 0, idx, out=g.args[0])
            out = g()
            tok = L // self.merge ** 2
            for j, i in enumerate(members):
                outs[i] = out[j * tok:(j + 1) * tok]
        return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=tuple(outs))

    def describe(self):
        return self.stats.as_dict()


# (In the public version, the continuous-input part that reads clips from another dataset was removed. AV2 continuous input is in av2_stream.py)
