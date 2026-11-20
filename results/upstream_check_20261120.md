# 上流の最新版の確かめ(2026-11-20)

## 版

| | 測った版 | 2026-11-20 10:05 JST の最新 | 違い |
|---|---|---|---|
| コード NVlabs/alpamayo2-edge | `02f55a7`(2026-11-09) | main = `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` | なし |
| モデル nvidia/Alpamayo2-Edge | `f8951d095de3589b3face1dec3f4394c8ecedb38`(2026-11-18 取得) | main = `b92b6aa326bd3e2235db28f248ad5a30b5890e64`(2026-11-10「Update the model card (#2)」) | README.md(モデルカード)と LICENSE だけ |

モデルの中身の比べ方: 重み 2 つと tokenizer.json は LFS の sha256 が一致(`d13c1a42…`、`cfe424a7…`、`06a854e6…`)。
config.json、generation_config.json、processor_config.json、chat_template.jinja、tokenizer_config.json、model.safetensors.index.json は
git の blob のハッシュが一致。

## プロセッサの読み直し(最新の main でも同じ)

- `src/alpamayo2_edge/helper.py` 55〜62 行 `get_processor`: 呼ばれるたびに `AutoProcessor.from_pretrained(checkpoint, fix_mistral_regex=True)` を呼ぶ。
- 同 152 行 `prepare_model_inputs`、179 行 `prepare_vqa_inputs` が、サンプルごとに `get_processor` を呼ぶ。
- 公式 CLI の時間(`inference.py` の `run_data`、`elapsed_seconds`)は `prepare_model_inputs` の前から測り始めるので、読み直しの時間を含む。
- 公式の加速(`docs/acceleration.md`): 対象はプリフィル・CoT デコード・軌跡のエキスパート。"The vision tower and projector always run eagerly."
- 上流への提案: PR #11「Read the processor once per checkpoint」(2026-11-14、OPEN、未マージ)。`functools.lru_cache` でチェックポイントごとに 1 回だけ読む。
  このリポジトリの手 #1(`install_processor_once`)と同じ変更。

## 公式 tuned の前処理を除いた時間(新しく測っていない。記録から)

前処理 = `prepare_model_inputs`(プロセッサの読み直しを含む)+ GPU への転送。ログ 9e9bcfb7、t0 = 8.5 s、判断理由 18 トークン前後。

| 測り方 | 全体(前処理込み) | 前処理 | 前処理を除いた時間 |
|---|---|---|---|
| 1 回だけ(seed 43、20 回) | 中央値 1927.3 ms | 1081.2 ms | 中央値 846.5 ms(841.9〜865.6) |
| 1 回だけ(seed 45、20 回) | 中央値 1942.9 ms | 1082.9 ms | 中央値 861.8 ms |
| 連続 0.1 秒おき(119 歩) | 中央値 1927.3 ms | 1084.4 ms | 中央値 849.2 / p90 897.6 / 最大 1030.8 ms |

参考: Embedl(https://huggingface.co/embedl/Alpamayo2-Edge)は、Jetson AGX Thor で上流の `--accel` tuned(BF16)を中央値 603.7 ms(p90 840.6 ms)と公表している。
測り方は「the model's whole call from camera frames to trajectory, with the inputs on the GPU」と、1 本のクリップ(NVIDIA の公開の運転データ)の
180 回の判断(走行の 0.1 秒ごとに 1 回)を順に動かしたもの。重みは `b92b6aa3`。コードのコミット、前処理を含むか、判断理由の長さは書かれていない。
入力が GPU に載った状態から測るなら、上の「前処理を除いた時間」が近い区間になる。
