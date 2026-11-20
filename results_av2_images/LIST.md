# AV2 の画像を含む図の一覧

どの図にも帰属表示「Argoverse 2 (CC BY-NC-SA 4.0, © 2021 Argo AI, LLC)」を入れている(図ごとに目で確かめた)。
AV2 の画像の翻案物なので、このフォルダの図は CC BY-NC-SA 4.0(https://creativecommons.org/licenses/by-nc-sa/4.0/ 、
本文は ../LICENSES/CC-BY-NC-SA-4.0.txt)で提供する。変更点: 元のカメラ画像を仮想カメラに描き直し、軌跡・文字を重ねた。

| ファイル | 中身 |
|---|---|
| preview_fill_9e9bcfb7.png | 22 枚の入力の組み立て(wide_ft)と、覆えない部分の埋め方の比較(ログ 9e9bcfb7)(帰属の帯は公開の候補で足した) |
| wide_v_d46ff5df_8.5_fp8all.png | 停止・歩行者の場面で、本体 + 出力層 FP8 の 1 回が「進む」と言った場面(BF16 と FP8 の 5 seed の軌跡と判断理由) |
| wide_v_ba67827f_8.0_fp8all.png | BF16 と FP8 の差が大きかった場面(本体 + 出力層 FP8) |
| wide_v_214e388e_7.0_fp8body.png | BF16 と FP8 の差が大きかった場面(本体だけ FP8) |
| wide_v_e1d68dde_6.5_fp8body.png | BF16 と FP8 の差が大きかった場面(本体だけ FP8) |

記事の図 6・7・11(英語ラベルの版と日本語ラベルの版)は `../docs/figs_av2/` に置いた(一覧と表示は `../docs/figs_av2/README.md`)。
