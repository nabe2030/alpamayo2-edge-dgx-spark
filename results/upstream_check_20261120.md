# Checking the latest upstream versions (2026-11-20)

## Versions

| | Measured version | Latest as of 2026-11-20 10:05 JST | Difference |
|---|---|---|---|
| Code NVlabs/alpamayo2-edge | `02f55a7` (2026-11-09) | main = `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` | none |
| Model nvidia/Alpamayo2-Edge | `f8951d095de3589b3face1dec3f4394c8ecedb38` (downloaded 2026-11-18) | main = `b92b6aa326bd3e2235db28f248ad5a30b5890e64` (2026-11-10, "Update the model card (#2)") | README.md (model card) and LICENSE only |

How the model contents were compared: the two weight files and tokenizer.json have matching LFS sha256 (`d13c1a42…`, `cfe424a7…`, `06a854e6…`).
config.json, generation_config.json, processor_config.json, chat_template.jinja, tokenizer_config.json and model.safetensors.index.json have
matching git blob hashes.

## Reloading the processor (same on the latest main)

- `src/alpamayo2_edge/helper.py` lines 55–62, `get_processor`: calls `AutoProcessor.from_pretrained(checkpoint, fix_mistral_regex=True)` every time it is called.
- In the same file, `prepare_model_inputs` (line 152) and `prepare_vqa_inputs` (line 179) call `get_processor` for every sample.
- The official CLI timing (`run_data` in `inference.py`, `elapsed_seconds`) starts before `prepare_model_inputs`, so it includes the reload time.
- The official acceleration (`docs/acceleration.md`) covers prefill, CoT decoding and the trajectory expert. "The vision tower and projector always run eagerly."
- Proposal to upstream: PR #11 "Read the processor once per checkpoint" (2026-11-14, OPEN, not merged). It loads the processor once per checkpoint with `functools.lru_cache`.
  This is the same change as optimization #1 in this repository (`install_processor_once`).

## Official tuned time without preprocessing (not newly measured; from the records)

Preprocessing = `prepare_model_inputs` (including the processor reload) + transfer to the GPU. Log 9e9bcfb7, t0 = 8.5 s, about 18 reasoning tokens.

| Measurement | Total (including preprocessing) | Preprocessing | Time without preprocessing |
|---|---|---|---|
| Single input (seed 43, 20 runs) | median 1927.3 ms | 1081.2 ms | median 846.5 ms (841.9–865.6) |
| Single input (seed 45, 20 runs) | median 1942.9 ms | 1082.9 ms | median 861.8 ms |
| Continuous, every 0.1 s (119 steps) | median 1927.3 ms | 1084.4 ms | median 849.2 / p90 897.6 / max 1030.8 ms |

For reference: Embedl (https://huggingface.co/embedl/Alpamayo2-Edge) reports a median of 603.7 ms (p90 840.6 ms) for upstream `--accel` tuned (BF16) on Jetson AGX Thor.
Their method is "the model's whole call from camera frames to trajectory, with the inputs on the GPU", running the 180 decisions of one clip (NVIDIA's public driving data;
one decision every 0.1 s of driving) in order. Weights `b92b6aa3`. The code commit, whether preprocessing is included, and the reasoning length are not stated.
If the timing starts with the inputs already on the GPU, the "time without preprocessing" above is the closer interval.
