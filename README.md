# Alpamayo 2 Edge を Argoverse 2 の映像で動かし、1 台の DGX Spark(GB10)で 1Hz に収める

[English](README.en.md)

**記事:** 経過と結果の解説は Qiita の記事「[自動運転 AI「Alpamayo 2 Edge」DGX Spark でソースコード解析＋FP8 カーネルチューニング](https://qiita.com/nabe2030/items/f220eb860dda549e966d)」(日本語)にまとめた。英語版の記事は [`docs/article_en.md`](docs/article_en.md)。

NVIDIA の運転用 VLA モデル [Alpamayo 2 Edge](https://huggingface.co/nvidia/Alpamayo2-Edge) を、
公開データセット [Argoverse 2 Sensor](https://www.argoverse.org/av2.html)(AV2)のカメラ映像で動かした。
推論を速くする 4 つの手と、自作の FP8 デコードカーネルを使い、1 台の NVIDIA GB10 で連続 1Hz の推論が収まるかを測った。
あわせて、FP8 にしたときの出力の変わり方(軌跡・判断理由)を BF16 と比べた。そのためのコード一式と結果。

モデルの重み、上流のコード、AV2 のデータは含まない。それぞれの配布元から取る。

## 中身

| フォルダ | 中身 |
|---|---|
| `kernels/fp8_gemv/` | FP8(E4M3)の重み × BF16 の行列×ベクトル積の CUDA カーネル(`fp8_gemv.cu`)、ビルドの手順(`build.sh`)、データセットもモデルも使わない単体のベンチ(`bench_fp8_gemv.py`、合成データ) |
| `accel/` | 加速の 4 つの手(`a2e_stagea.py`)と、CoT デコードの重みを FP8 で読む差し替え(`a2e_fp8.py`)。どちらも上流のコードを書き換えず、読み込んだあとのモデルに差し込む |
| `av2_input/` | AV2 のログから、Alpamayo 2 Edge の入力(公式ローダと同じ形の dict)を組み立てるスクリプト。データの取得、場面選び、仮想カメラへの描き直し、ナビ文の作り方 |
| `measure/` | 速さの測定(1 回だけ / 連続 10Hz・1Hz / 描き直し込み)、BF16 と FP8 の出力の比較、止まる場面での比較、集計と図のスクリプト |
| `results/` | AV2 で測った結果の表と図(AV2 の画像を含まないもの) |
| `results_av2_images/` | AV2 の画像を含む図(帰属表示つき。CC BY-NC-SA 4.0。一覧は `results_av2_images/LIST.md`) |
| `docs/` | 英語版の記事(`article_en.md`)と、その図。`docs/figs_av2/` は記事の図 6・7・11(AV2 の画像を含む。英語ラベルの版と、`_ja` を付けた日本語ラベルの版。CC BY-NC-SA 4.0。出典・加工・ライセンスは `docs/figs_av2/README.md`)。描くスクリプトは `measure/article_figs.py`(`A2E_FIG_LANG=en` で英語ラベル、既定は日本語ラベル) |
| `LICENSES/` | Apache-2.0(このリポジトリのコードと、上流のコード)と CC BY-NC-SA 4.0(AV2)のライセンスの本文 |

## 加速の 4 つの手と FP8

番号(#1 など)は作業のときの呼び名で、`accel/a2e_stagea.py` の節の見出しと同じ。記事では ①〜④ と呼んでいる(#1 = ①、#9 = ②、#2 = ③、#3 = ④)。

1. **#1 プロセッサを 1 回だけ読む**(`install_processor_once`): 公式の前処理は、呼ぶたびに `AutoProcessor.from_pretrained` を読み直す(2026-11-20 の上流の main `02f55a7` でも同じ)。これを 1 回だけにする。上流に出ている提案 PR #11 と同じ変更。
2. **#9 前処理の縮小を GPU で**(`GpuPrep`): 22 枚の縮小・パッチ化・正規化を GPU で行う。縮小後の uint8 の画素が上流と最大 1 違うことがある。プロンプト、トークン列、画像の格子の大きさは上流と一致する。
3. **#2 ViT の結果を画像ごとに使い回す**(`VitFeatureCache`): 連続入力では、同じコマが次の時刻でも履歴として入る。そのため、前に ViT に通した画像の結果を使う(キーはカメラ・時刻・格子・画素の指紋)。
4. **#3 ViT を compile / CUDA グラフにする**(`VisionAccel`)。上流の公式の加速では ViT は eager のまま。
- **FP8 デコード**(`a2e_fp8.py` + `kernels/fp8_gemv`): CoT のデコード(バッチ 1)で、言語モデルの線形層(本体 28 層)の重みを、行ごとのスケールの E4M3 で読む。出力層も FP8 にできる(`--q all`)が、判断理由の 1 トークン目が「進む」寄りにわずかに偏るので、**本体だけ(`--q body`)を推す**。プリフィル・ViT・軌跡のエキスパートは BF16 のまま。

上流の公式の加速(`alpamayo2_edge.accel`、`mode="tuned"`)の上に重ねて使う。

## 動かし方

### 1. 用意するもの

- 上流のコード: https://github.com/NVlabs/alpamayo2-edge (測ったのはコミット `02f55a7`)を `$A2E_ROOT/src/alpamayo2-edge` に置き、上流の README の手順で `.venv` を作る(FlashAttention 2 を含む。GB10 では flash-attn 2.8.3 を sm_121 向けにビルドした)。
- モデル: https://huggingface.co/nvidia/Alpamayo2-Edge を `$A2E_ROOT/models/Alpamayo2-Edge` に置く(利用条件はモデルのページで確かめる)。測ったリビジョンは `f8951d0`。2026-11-20 の main `b92b6aa` は、重みと設定が同じでモデルカードだけが違う。
  例: `hf download nvidia/Alpamayo2-Edge --local-dir $A2E_ROOT/models/Alpamayo2-Edge`
- FP8 カーネルのビルド: sm_121 に対応した nvcc(CUDA 12.9 以降)と ninja(例: `$PY -m pip install ninja`、または ninja のバイナリを PATH に置く。無いと `Ninja is required to load C++ extensions` で止まる)。測ったのは CUDA 12.9(PyTorch の cu129 と同じ)。CUDA 13.0 の nvcc でもビルドと単体ベンチは通った(2026-11-20、DGX OS 7.6.0 の `/usr/local/cuda`)。
- AV2 のデータ: 下の手順 4 で、場面ごとに要るコマだけを取る(1 場面 10 MB 前後)。

### 2. 環境変数

```bash
cp env.example.sh env.sh      # A2E_ROOT・A2E_WORK(と、必要なら CUDA_HOME)を書き換える
source env.sh                 # PY(上流の venv の python)、PYTHONPATH、A2E_REPO などが決まり、A2E_WORK に移る
```

`env.sh` が決めるもの: `A2E_ROOT`、`A2E_WORK`、`A2E_MODEL`、`A2E_FP8_SO`(カーネルの .so の置き場所)、`A2E_REPO`(このリポジトリ)、
`PY`、`CUDA_HOME`、`PYTHONPATH`(上流の `src` と、このリポジトリの `accel/`・`av2_input/`・`measure/`)、コンパイルのキャッシュの場所。

### 3. FP8 カーネル(データもモデルも使わない)

```bash
PY=$PY $A2E_REPO/kernels/fp8_gemv/build.sh $(dirname $A2E_FP8_SO)    # 数分。既にあれば止まる
$PY $A2E_REPO/kernels/fp8_gemv/bench_fp8_gemv.py $A2E_FP8_SO --out $A2E_WORK/fp8_gemv_bench.json
```

既定のアーキテクチャは sm_120 / sm_121(GB10)。ほかの GPU では `ARCH_FLAGS` を変える(FP8 の変換命令は sm_89 以降)。
ベンチは、正しさ(同じ FP8 の重みを FP32 に戻した計算との相対誤差。0.2% 未満なら正常)と、カーネル・BF16 の `F.linear`・読むだけのカーネルの時間を出す。

### 4. AV2 の 1 場面で推論する(最小の確認)

```bash
# 場面の一覧(CSV。列は scene_id, split, log_id, t0_rel_s。t0_rel_s はログの最初のコマからの秒)
mkdir -p $A2E_WORK/data/av2 $A2E_WORK/inputs $A2E_WORK/raw/speed
printf "scene_id,split,log_id,t0_rel_s\nv_9e9bcfb7_8.5,val,9e9bcfb7-601d-3d80-bc12-ef7025174beb,8.5\n" > $A2E_WORK/scenes.csv
# 要るコマだけを公開 S3 から取る(匿名の HTTPS)。3 つ目の引数は「丸ごと取ったログの置き場所」(無ければ空のフォルダでよい)
$PY $A2E_REPO/av2_input/av2_fetch_frames.py $A2E_WORK/scenes.csv $A2E_WORK/data/av2 $A2E_WORK/data/av2/full $A2E_WORK/scenes_t0.csv 1
# 入力を作る(仮想カメラへの描き直しとナビ文。組み立て方 wide_ft、ナビ文は記録された走行から)
$PY $A2E_REPO/av2_input/av2_make_input.py $A2E_WORK/data/av2/val/9e9bcfb7-601d-3d80-bc12-ef7025174beb 8.5 $A2E_WORK/inputs/9e9bcfb7_8.5.pt
# 1 回だけの測定: 公式 tuned + 4 つの手 + 本体だけ FP8。初回はコンパイルで数分かかる
$PY $A2E_REPO/measure/a2e_bench2.py --model $A2E_MODEL --mode single --input pt:$A2E_WORK/inputs/9e9bcfb7_8.5.pt \
    --accel tuned --feat p,g,c,v --q body --runs 5 --label C_body_s43 --out $A2E_WORK/raw/speed/C_body_s43.jsonl
# 集計(表は Markdown と JSON)
$PY $A2E_REPO/measure/summarize_speed.py $A2E_WORK/raw/speed $A2E_WORK/raw/speed/summary_speed.md $A2E_WORK/raw/speed/summary_speed.json
```

`--feat` は 4 つの手(p = #1、g = #9、c = #2、v = #3)、`--q` は FP8 の範囲(`none` / `body` / `head` / `all`)。
公式のままの時間は `--accel tuned --feat "" --q none`(または `--accel off`)。
この場面(seed 43、判断理由 18 トークン)で、作者の環境では公式 tuned 1927 ms、4 つの手 + 本体だけ FP8 695 ms(20 回の中央値)。

### 5. 測定の一式

- 速さ: `measure/run_speed.sh`(1 回だけ・連続)、`measure/run_stream_multi.sh`(連続を 12 本のログで)、`measure/run_e2e.sh`(描き直し込み)
- BF16 と FP8 の比較: `measure/run_wide.sh`(`wide_eval.py`)、`measure/av2_stopgo.py`(止まる場面で seed を増やす)
- 集計: `summarize_speed.py`、`summarize_multi.py`、`summarize_wide.py`、`summarize_stopgo.py`
- `run_*.sh` は、このリポジトリの `env.sh` を読む(別の場所の env.sh を使うときは `A2E_ENV=/path/to/env.sh`)。ログは `$A2E_WORK/data/av2/val/` に丸ごと置く(`av2_input/av2_fetch.py`)。

## AV2 の入力の作り方(大事な前提)

- **カメラの組み直し**: Alpamayo 2 Edge は、NVIDIA の車のカメラ構成(前方広角 120°、前方望遠 30°、横、後ろ)を前提にしている。AV2 は向きと画角が違うリングカメラ 7 台。どのスロットも、キャリブレーションを使ってリングカメラから「仮想カメラ」(1920×1080)に描き直した。
  - 前方と横: 120° の等距離射影
  - 後ろ: 70°
  - 望遠: 前方中央のカメラの 30° を切り出す
  - 元のカメラで覆えない部分は黒にした。灰色や端の色で埋めても、結果に差は出なかった。
- **ナビ文は、記録された走行(その先の実際の自車の動き)から作っている。** 例: "Turn left in 10m"、"Continue straight"。本来は経路の計画から来る入力で、モデルに先の答えの一部を渡していることになる。参考に、ナビ文を常に "Continue straight" にした比較も載せた。
- **描き直しの時間は、推論の時間と分けて示している。** 「推論だけ」の数字には描き直しが入っていない。描き直し込みの数字は別に示した(`results/`)。
- 時刻の揺れ: AV2 のコマの時刻は 50 ms きざみから ±数 ns 揺れる。そのため「求めた時刻と同じかそれより前のコマ」を選ぶときに 1 ms の許しを入れた。

## 測った環境と版

| | |
|---|---|
| マシン | NVIDIA DGX Spark(GB10、compute capability 12.1)、DGX OS 7.6.0、ドライバ 580.178.04 |
| PyTorch | 2.8.0+cu129(CUDA 12.9)、torchvision 0.23.0、triton 3.4.0 |
| transformers | 5.17.0 |
| flash-attn | 2.8.3(sm_121 向けに自前でビルド) |
| alpamayo2-edge | `02f55a7`(2026-11-20 の main と同じ) |
| モデル | nvidia/Alpamayo2-Edge リビジョン `f8951d095de3589b3face1dec3f4394c8ecedb38`(2026-11-18 に取得。2026-11-20 の main `b92b6aa` と重み・設定は同じ) |
| FP8 カーネル | `kernels/fp8_gemv/fp8_gemv.cu`、nvcc 12.9、`-O3 --use_fast_math`、sm_120/sm_121 |
| 推論の設定 | 公式の `inference` と同じ: 温度 0.6、top_p 0.98、拡散 10 ステップ、候補 1 本 |

## AV2 の入手とライセンス

- 入手元: `s3://argoverse/datasets/av2/sensor/`(匿名で取れる。手順は https://argoverse.github.io/user-guide/getting_started.html )
- ライセンス: **CC BY-NC-SA 4.0**(https://creativecommons.org/licenses/by-nc-sa/4.0/ )。利用規約: https://www.argoverse.org/about.html#terms-of-use
- 帰属の書き方(図に入れているもの): `画像: Argoverse 2 (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)`
- 引用: Wilson et al., "Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting", NeurIPS Datasets and Benchmarks 2021
- **AV2 の画像を含む図**(`results_av2_images/` と `docs/figs_av2/`)は AV2 の画像の翻案物なので、CC BY-NC-SA 4.0(非営利、同じライセンスで継承)で出す。このリポジトリのコードのライセンスとは別。
- AV2 の画像そのもの、生のテンソル、AV2 のファイルは入れていない。

## ライセンス

- **コード: Apache License 2.0**(`LICENSES/Apache-2.0.txt`)。Copyright (c) 2026 [nabe2030](https://github.com/nabe2030)。
- 自作のコード(`.py`・`.sh`・`.cu`)の頭には `SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030` と `SPDX-License-Identifier: Apache-2.0` を入れた。
- 上流のコードを変更したものを含む 2 ファイル(それぞれの頭に、元の著作権表示、nabe2030 の変更の著作権表示、変更した旨を書いた):
  - `accel/a2e_fp8.py` の `DecodeStepQ.forward`: 上流の `alpamayo2_edge/accel/decode.py` の CoT デコードの 1 ステップを写し、線形層を FP8 のカーネルに差し替えた。
  - `accel/a2e_stagea.py` の `GpuPrep`: 上流の `alpamayo2_edge/helper.py` の `prepare_model_inputs` の処理の順を写し、画像の処理を GPU に移した。
  - 上流: NVlabs/alpamayo2-edge、Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES、Apache-2.0。
- `results_av2_images/` と `docs/figs_av2/` の図: CC BY-NC-SA 4.0(上記)。
- 上流のファイルそのもの、モデルの重み(NVIDIA のライセンス)、AV2 のデータは含まない。

## 結果(AV2、NVIDIA GB10 1 台)

表の元は `results/summary_*.md/json` と `results/compare_3configs.md`、図は `results/fig_*.png`。

### 1 回だけ(ログ 9e9bcfb7、t0 = 8.5 s、seed 43、18 トークン、20 回の中央値)

| 状態 | 時間 |
|---|---|
| 公式の加速なし | 2524 ms |
| 公式 tuned | 1927 ms(前処理 1081 ms を除くと 847 ms) |
| + プロセッサを 1 回だけ読む | 932 ms |
| + 加速の 4 つの手 | 790 ms |
| + 本体だけ FP8(推す構成) | 695 ms |
| + 本体と出力層 FP8(参考) | 678 ms |

### 連続(12 本のログ、`compare_3configs.md`)

| 構成 | 推論だけ 10Hz 中央値 / 最大 | 推論だけ 1Hz 中央値 / 最大 | 描き直し込み 1Hz 中央値 / 最大 | 1 秒超え | 1 トークンあたり |
|---|---|---|---|---|---|
| 4 つの手(BF16) | 641 / 867 ms | 757 / 977 ms | 849 / 1077 ms | 推論だけ 0%、描き直し込み 1.4% | 15.4 ms |
| + 本体だけ FP8(推す構成) | 528 / 685 ms | 638 / 780 ms | 736 / 878 ms | 0% | 10.0 ms |
| + 本体と出力層 FP8(参考) | 512 / 644 ms | 620 / 752 ms | 715 / 852 ms | 0% | 9.2 ms |

- 推論だけ: 10Hz は 1428 歩、1Hz は 432 歩(条件ごと)。描き直し込み: 1Hz で 432 歩。
- 1 秒に収まる判断理由の長さ(1Hz の当てはめ): BF16 37 トークン、本体だけ FP8 58 トークン、本体と出力層 FP8 63 トークン。12 本のログで出た判断理由は最長 37 トークン。

- 遅い回は、判断理由の長さでほぼ説明できる(時間 ≈ 固定分 + 1 トークンあたりの時間 × トークン数)。
- 描き直しは、出力が画素まで同じ速くした版(`av2_render_fast.py`)で測った。1Hz で約 94 ms。

### BF16 と FP8 の出力(400 場面 × 5 seed、`summary_wide.md`)
- 同じ seed での軌跡の差は、BF16 どうしで seed を変えたときの幅の 1.8%(本体だけ)/ 2.0%(本体と出力層)。
- 記録された走行との誤差の差は ±0.004 m 以内。走り方・都市で分けた傾向は無い。

### 止まるべき場面(22 場面 × 64 seed、ほか 117 場面 × 16 seed、`summary_stopgo.md`)
- 軌跡が動き出す割合(3 s で 2 m 以上)は、BF16 と FP8 で変わらない(差 −0.1 pt、95% 区間 [−0.5, +0.4])。
- 出力層まで FP8 にすると、判断理由の 1 トークン目が「進む」系になる確率が上がる(22 場面で +1.45 pt、22 場面すべてで同じ向き)。出力層だけ FP8 でも同じ向きに出るので、出どころは主に出力層。
- 本体だけ FP8 なら +0.37 pt(117 場面では −0.05 pt)。
