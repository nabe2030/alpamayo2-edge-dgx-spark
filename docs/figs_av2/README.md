# Figures with Argoverse 2 images / AV2 の画像を含む図

## Source, modifications, license

- Source: Argoverse 2 Sensor Dataset, © 2021 Argo AI, LLC (https://www.argoverse.org/av2.html).
- Modifications: the original camera images were re-rendered to virtual cameras, cropped and resized, and predicted / recorded trajectories and text were overlaid. No unmodified AV2 image is included.
- License: these figures are adaptations of the AV2 images and are provided under **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/ , full text: `../../LICENSES/CC-BY-NC-SA-4.0.txt`). This is separate from the Apache-2.0 license of the code in this repository.
- The attribution and the modifications are also written at the bottom of each figure.

## 出典・加工・ライセンス

- 出典: Argoverse 2 Sensor Dataset、© 2021 Argo AI, LLC(https://www.argoverse.org/av2.html )。
- 加工: 元のカメラ画像を仮想カメラに描き直し、切り出し・縮小し、予測と記録の軌跡と文字を重ねた。加工していない AV2 の画像は入れていない。
- ライセンス: AV2 の画像の翻案物なので **CC BY-NC-SA 4.0**(https://creativecommons.org/licenses/by-nc-sa/4.0/ 、本文は `../../LICENSES/CC-BY-NC-SA-4.0.txt`)で提供する。このリポジトリのコードの Apache-2.0 とは別。
- 帰属と加工した旨は、それぞれの図の下にも入れている。

## Files / ファイル

`<name>.jpg` has English labels (used by `../article_en.md`); `<name>_ja.jpg` has Japanese labels (used by the Japanese article). The two versions differ only in the text; the scenes, seeds, numbers and colors are the same.

`<名前>.jpg` は英語ラベルの版(`../article_en.md` が参照)、`<名前>_ja.jpg` は日本語ラベルの版(日本語の記事に載せたもの)。違うのは文字だけで、場面・seed・数字・色は同じ。

| File | Content | 中身 |
|---|---|---|
| `fig06_left_turn.jpg`, `fig06_left_turn_ja.jpg` | Article figure 6: left turn. Re-rendered input (front 120°) vs. AV2 cameras as is (front approx. 47°): trajectories of 5 seeds, top view, reasoning | 記事の図 6: 左折の場面。描き直した入力(前方 120°)と、AV2 のカメラをそのまま入れた入力(前方 約 47°)の比較(5 seed の軌跡、上から見た図、判断理由) |
| `fig07_input_layouts.jpg`, `fig07_input_layouts_ja.jpg` | Article figure 7: one scene as input built in 3 layouts (front 3 slots), and the other 3 slots of the adopted layout | 記事の図 7: 同じ場面を 3 通りの組み立て方で作った入力(前の 3 スロット)と、採用した組み立て方の残りの 3 スロット |
| `fig11_pedestrian_stop.jpg`, `fig11_pedestrian_stop_ja.jpg` | Article figure 11: pedestrian crossing. Trajectories and reasoning of 5 seeds with BF16, FP8 body only, and FP8 body + output layer | 記事の図 11: 歩行者が横断中の場面。BF16・本体だけ FP8・本体と出力層 FP8 の 5 seed の軌跡と判断理由 |
