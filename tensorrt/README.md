# Alpamayo 2 Edge on TensorRT Edge-LLM (DGX Spark)

Scripts that run the VLM part of Alpamayo 2 Edge (ViT, prefill, reasoning decode) on
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM), and the workaround for the NVFP4 W4A16 bug
([NVIDIA/TensorRT#4949](https://github.com/NVIDIA/TensorRT/issues/4949),
[NVIDIA/TensorRT-Edge-LLM#288](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/288)).
Results are in [`results/fp4/README.md`](../results/fp4/README.md); the write-up is section 5 and 6 of
[`docs/article_fp4_en.md`](../docs/article_fp4_en.md).

Limits (as of Edge-LLM 0.12.0):

- **The expert (trajectory generation) is not included.** Edge-LLM has no implementation of the Alpamayo 2 Edge expert, so only
  the VLM part is compared, and no trajectories come out.
- **No sampling comparison.** Alpamayo bans 4,000 trajectory tokens during the reasoning; Edge-LLM's `logit_bias` takes at most
  1,024 ids. Outputs are compared with greedy decoding.
- **ViT outputs are cached by image content** and the cache cannot be turned off from Python. Timed runs change one pixel per
  image (`trt_run.py --perturb 1`).

## Versions we used

| | |
|---|---|
| Machine | DGX Spark (GB10), DGX OS 7.6.0, driver 580.178.04 (no TensorRT installed on the host) |
| TensorRT Edge-LLM | 0.12.0 (PyPI aarch64 wheel `tensorrt-edgellm`; includes prebuilt plugins and runtime for `dgx-spark-cu13-sm121`, so no C++ build was needed) |
| TensorRT | 10.16.1.11 (pip `tensorrt-cu13`) |
| Quantization | nvidia-modelopt 0.45.0, torch 2.14.0+cu130, torchvision 0.29.0, transformers 5.17.0, `tensorrt-edgellm[quantize]` 0.12.0 |

The Edge-LLM documentation assumes quantization on an x86 GPU machine; on the Spark (aarch64) it ran as is (NVFP4,
32 AV2 calibration prompts, about 50 s including export).

## Steps

```bash
source env.sh     # A2E_EDGELLM_VENV, A2E_QUANT_VENV, A2E_TRT (see env.example.sh)

# 0. venvs (in addition to the upstream Alpamayo venv $PY)
python3 -m venv $A2E_EDGELLM_VENV && $A2E_EDGELLM_VENV/bin/pip install tensorrt-cu13==10.16.1.11 tensorrt-edgellm==0.12.0
python3 -m venv $A2E_QUANT_VENV && $A2E_QUANT_VENV/bin/pip install torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu130
$A2E_QUANT_VENV/bin/pip install nvidia-modelopt==0.45.0 transformers==5.17.0 tensorrt-cu13==10.16.1.11 "tensorrt-edgellm[quantize]==0.12.0"

# 1. Checkpoints: VLM part renamed for the builder (FP16), and NVFP4 quantized with AV2 calibration prompts.
#    Uses the FP4 calibration logs of measure/run_calib_fp4.sh ($A2E_WORK/raw/calib/calib48_scenes.csv).
bash $A2E_REPO/tensorrt/prepare_checkpoints.sh

# 2. Workaround for W4A16 (only needed for the nvfp4_w4a16 engine): patch the installed tensorrt_edgellm
SP=$($A2E_EDGELLM_VENV/bin/python -c "import tensorrt_edgellm, os; print(os.path.dirname(os.path.dirname(tensorrt_edgellm.__file__)))")
patch -p1 -d $SP < $A2E_REPO/tensorrt/pack_qkv_maxcopy.patch

# 3. Engines: fp16, nvfp4_w4a4, nvfp4_w4a16
bash $A2E_REPO/tensorrt/build_engines.sh

# 4. PyTorch references (BF16, plain Hugging Face path, greedy with Alpamayo's masks): one scene for timing, 20 scenes for outputs
$PY $A2E_REPO/tensorrt/trt_ref_dump.py --model $A2E_MODEL --input $A2E_WORK/data/av2_inputs/single_9e9bcfb7_8.5.pt --out $A2E_TRT/ref_9e9bcfb7
$PY $A2E_REPO/tensorrt/trt_ref_multi.py --model $A2E_MODEL --scenes $A2E_WORK/raw/fp4/scenes40.csv --n 20 --out $A2E_TRT/ref20

# 5. Outputs vs. PyTorch (20 scenes, greedy) and timing (single input)
source $A2E_EDGELLM_VENV/bin/activate
for e in fp16 nvfp4_w4a4 nvfp4_w4a16; do
  python $A2E_REPO/tensorrt/trt_eval_multi.py --engine-dir $A2E_TRT/engines/$e --model-dir $A2E_TRT/models/a2e-reasoner \
      --refs $A2E_TRT/ref20 --label $e --out $A2E_TRT/raw/eval20_$e.jsonl
  bash $A2E_REPO/tensorrt/trt_speed.sh $A2E_TRT/engines/$e $e
done

# 6. GPU-only stage times: capture with nsys (for example as below) and split per engine call
nsys profile -t cuda,nvtx -o $A2E_TRT/raw/w4a4_miss python $A2E_REPO/tensorrt/trt_run.py --engine-dir $A2E_TRT/engines/nvfp4_w4a4 \
    --model-dir $A2E_TRT/models/a2e-reasoner --ref $A2E_TRT/ref_9e9bcfb7 --mode greedy --warmup 2 --runs 3 --perturb 1
nsys export -t sqlite -o $A2E_TRT/raw/w4a4_miss.sqlite $A2E_TRT/raw/w4a4_miss.nsys-rep
python $A2E_REPO/tensorrt/nsys_stage_split.py $A2E_TRT/raw/w4a4_miss.sqlite --skip 1
```

- `scenes40.csv` is made by `measure/run_quality_fp4.sh` (40 of the 400 evaluation scenes); the first 20 are used here.
- `trt_ref_dump.py` takes the input file made by `av2_input/av2_make_input.py` (the same file as the single-input speed measurement).
- In step 6, the W4A4 engine makes one extra engine call at the start, hence `--skip 1`; omit it for the other engines.
- `trt_run.py` passes the prompt as text (the ViT runner of this model does not accept `pre_tokenized_input_ids`); we checked
  that re-tokenizing the text gives exactly the same ids as the PyTorch prompt.

## The W4A16 bug and the workaround

With `--no-quantize-activations` (NVFP4 W4A16: weights in 4 bits, activations in FP16), the engine produced meaningless output
from the first token on all 20 scenes, while W4A4 built from the same checkpoint worked.

Cause: TensorRT assigns the outputs of the three q / k / v projection plugins directly into the buffer of the concatenation that
follows (no copy). The plugins write their outputs densely, so with more than one row (prefill) the rows end up in the wrong
places: row r of the buffer holds the q projection of token 2r+1. Decoding (one row) is not affected, but it reads the KV cache
written during prefill. It reproduces with TensorRT alone (`w4a16_bug/issue_4949_repro.py`, three trivial Python plugins) on
TensorRT 10.15.1.29, 10.16.1.11 and 11.5.0.63. We have checked this only on DGX Spark.

Workaround (`pack_qkv_maxcopy.patch`, for `tensorrt_edgellm/builder/ops/transformer.py` of Edge-LLM **0.12.0**, tag `v0.12.0`;
it applies to the source tree and to the installed wheel): in `pack_qkv`, take `max(x, x)` of q, k and v before concatenating.
The value does not change, but TensorRT keeps this copy, so each plugin writes its own buffer. Things that do **not** help because
TensorRT removes them: an identity layer, a reshape to the same shape, multiplying by 1, flattening the nested concatenation.
Concatenating in FP32 and casting back to FP16 also works (same outputs on all 20 prompts) but moves more data.

`w4a16_bug/` also holds the scripts we used to narrow it down:

| File | Use |
|---|---|
| `issue_4949_repro.py` | The TensorRT-only reproducer attached to NVIDIA/TensorRT#4949 (unchanged) |
| `concat_alias_repro.py` | The same reproducer with options for the copy candidates (`--fix none / fp32 / identity / max / shuffle`) |
| `a16_micro.py` | Small TensorRT networks around Edge-LLM's `Nvfp4A16GemmPlugin` (synthetic or real NVFP4 weights) to find which graph breaks |
| `trt_dump.py` | One request with the runtime's LayerDebugger enabled (`EDGELLM_DUMP_LOGITS_KVCACHE_*`): per-layer KV cache after prefill |
| `cmp_dumps.py` | Compares two such dumps (cosine per layer and per token) |
| `trt_eval_mix.py` | `trt_eval_multi.py` with an LLM engine and a visual engine from different builds (to find which engine is broken) |

## Files in this folder

| File | Use |
|---|---|
| `make_cosmos3_reasoner_ckpt.py` | Copies the VLM part of the Alpamayo 2 Edge checkpoint into the Cosmos3-Edge (reasoner) layout read by the Edge-LLM builder (names only; tensors copied byte for byte; expert dropped). `--hf` keeps the Transformers layout for quantization |
| `hf_to_edge_names.py` | Renames the quantized Transformers-layout checkpoint to the builder layout |
| `trt_calib_av2.py` | AV2 calibration batches for ModelOpt (full Alpamayo prompts: 22 images + trajectory tokens) |
| `trt_quantize_av2.py` | NVFP4 quantization through Edge-LLM's ModelOpt flow, with the AV2 batches instead of the built-in datasets |
| `prepare_checkpoints.sh` | Runs the four scripts above |
| `build_engines.sh` | `trtedge build` of the FP16, NVFP4 W4A4 and NVFP4 W4A16 engines |
| `trt_ref_dump.py`, `trt_ref_multi.py` | PyTorch BF16 references (prompt, resized images, greedy ids and top-20 log-probabilities per step) |
| `trt_run.py` | Runs one reference prompt on an engine (greedy or sampled; timing with `--runs`, `--perturb`) |
| `trt_eval_multi.py` | Greedy outputs of an engine vs. the PyTorch references (identical reasoning, KL per step) |
| `trt_speed.sh` | Stage timing from three wall-clock runs (ViT miss / ViT hit / one token) |
| `nsys_stage_split.py` | Splits an nsys capture into ViT, prefill and per-token decode GPU windows |
| `pack_qkv_maxcopy.patch` | The W4A16 workaround above |
