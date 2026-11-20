記録 3280 件(場面 × seed)。各件で BF16・本体だけ FP8・本体 + 出力層 FP8 を同じ seed で回した。

## 集合 A(場面 22、件数 1408)

| 変種 | 軌跡で進む [95%] | 文で go [95%] | 文 go かつ軌跡で進む [95%] | 文の分類(stop/go/creep/slow/nominal/other) |
|---|---|---|---|---|
| BF16 | 26.3% [14.3%, 40.3%] | 11.9% [6.1%, 18.8%] | 6.9% [2.8%, 11.8%] | 1147 / 167 / 7 / 60 / 3 / 24 |
| 本体だけ FP8 | 26.3% [14.3%, 40.4%] | 12.4% [6.3%, 19.6%] | 7.0% [3.1%, 12.0%] | 1140 / 174 / 8 / 58 / 4 / 24 |
| 本体 + 出力層 FP8 | 26.2% [14.3%, 40.2%] | 13.3% [7.0%, 20.7%] | 7.3% [3.2%, 12.6%] | 1121 / 187 / 10 / 62 / 3 / 25 |

対にした差(同じ場面・同じ seed。FP8 − BF16、ポイント)と、食い違いの向き:

| 変種 | 軌跡で進む の差 [95%] | BF16 止まる→FP8 進む | BF16 進む→FP8 止まる | 文 go の差 [95%] | 文 stop→go | 文 go→stop |
|---|---|---|---|---|---|---|
| 本体だけ FP8 | +0.1 [-0.4, +0.4] | 6 | 5 | +0.5 [+0.0, +1.1] | 9 | 3 |
| 本体 + 出力層 FP8 | -0.1 [-0.5, +0.4] | 7 | 8 | +1.4 [+0.7, +2.2] | 20 | 2 |

## 集合 B(場面 117、件数 1872)

| 変種 | 軌跡で進む [95%] | 文で go [95%] | 文 go かつ軌跡で進む [95%] | 文の分類(stop/go/creep/slow/nominal/other) |
|---|---|---|---|---|
| BF16 | 92.8% [89.3%, 95.7%] | 9.5% [7.2%, 12.1%] | 9.1% [6.9%, 11.6%] | 1037 / 178 / 8 / 411 / 235 / 3 |
| 本体だけ FP8 | 93.0% [89.5%, 95.9%] | 9.2% [6.9%, 11.8%] | 8.8% [6.7%, 11.3%] | 1033 / 172 / 8 / 397 / 259 / 3 |
| 本体 + 出力層 FP8 | 92.8% [89.4%, 95.8%] | 10.3% [7.9%, 13.0%] | 9.9% [7.6%, 12.4%] | 1015 / 192 / 7 / 403 / 252 / 3 |

対にした差(同じ場面・同じ seed。FP8 − BF16、ポイント)と、食い違いの向き:

| 変種 | 軌跡で進む の差 [95%] | BF16 止まる→FP8 進む | BF16 進む→FP8 止まる | 文 go の差 [95%] | 文 stop→go | 文 go→stop |
|---|---|---|---|---|---|---|
| 本体だけ FP8 | +0.2 [+0.0, +0.3] | 3 | 0 | -0.3 [-0.7, +0.1] | 3 | 4 |
| 本体 + 出力層 FP8 | +0.0 [-0.2, +0.2] | 2 | 2 | +0.7 [+0.3, +1.2] | 14 | 0 |

## 集合 A+B(場面 139、件数 3280)

| 変種 | 軌跡で進む [95%] | 文で go [95%] | 文 go かつ軌跡で進む [95%] | 文の分類(stop/go/creep/slow/nominal/other) |
|---|---|---|---|---|
| BF16 | 64.3% [54.8%, 73.7%] | 10.5% [7.6%, 13.9%] | 8.2% [6.0%, 10.7%] | 2184 / 345 / 15 / 471 / 238 / 27 |
| 本体だけ FP8 | 64.4% [54.9%, 73.8%] | 10.5% [7.5%, 14.0%] | 8.0% [5.9%, 10.6%] | 2173 / 346 / 16 / 455 / 263 / 27 |
| 本体 + 出力層 FP8 | 64.2% [54.8%, 73.7%] | 11.6% [8.5%, 15.2%] | 8.8% [6.5%, 11.5%] | 2136 / 379 / 17 / 465 / 255 / 28 |

対にした差(同じ場面・同じ seed。FP8 − BF16、ポイント)と、食い違いの向き:

| 変種 | 軌跡で進む の差 [95%] | BF16 止まる→FP8 進む | BF16 進む→FP8 止まる | 文 go の差 [95%] | 文 stop→go | 文 go→stop |
|---|---|---|---|---|---|---|
| 本体だけ FP8 | +0.1 [-0.1, +0.3] | 9 | 5 | +0.0 [-0.3, +0.4] | 12 | 7 |
| 本体 + 出力層 FP8 | -0.0 [-0.2, +0.2] | 9 | 10 | +1.0 [+0.6, +1.4] | 34 | 2 |

## 文が BF16 で stop、FP8 で go に分かれた対の、分かれた位置

- **本体だけ FP8**: 12 対。分かれたトークンの位置 {0: 12}。語の入れ替わり(上位): Decelerate→Proceed 3、Remain→Proceed 2、Yield→Proceed 1、Remain→Pull 1、Hold→Proceed 1、Remain→The 1
  - v_20bcd747_3.0 s82: 「Decelerate to stop at the stop line for the all-way stop sign before turning right to foll」→「Proceed into the intersection and turn right to follow our route, as the crossing vehicle 」(軌跡 stop→go)
  - v_20bcd747_3.0 s86: 「Decelerate to stop at the stop line for the all-way stop sign at the intersection, prepari」→「Proceed into the intersection and turn right to follow our route, as the crossing vehicle 」(軌跡 go→go)
  - v_22052525_7.0 s50: 「Yield to oncoming traffic before turning left to follow our route, as the oncoming vehicle」→「Proceed with the left turn to follow our route, as oncoming traffic has cleared our path u」(軌跡 stop→stop)
  - v_7e4d67b3_3.5 s64: 「Remain stopped behind the parked police car as traffic ahead is stationary in our lane.」→「Pull out from the curb into the travel lane to follow our route, as the passing traffic ha」(軌跡 stop→stop)
- **本体 + 出力層 FP8**: 34 対。分かれたトークンの位置 {0: 34}。語の入れ替わり(上位): Decelerate→Proceed 9、Wait→Proceed 7、Remain→Proceed 4、Yield→Proceed 3、Decelerate→Turn 2、Remain→Pull 1
  - v_1f434d15_2.5 s67: 「Wait behind the stationary lead vehicle at the red traffic light to follow our route.」→「Proceed forward in the right lane toward the intersection to follow our route, as the cros」(軌跡 stop→stop)
  - v_1f434d15_2.5 s90: 「Wait in queue behind the stopped lead vehicle at the intersection for the red traffic ligh」→「Proceed forward and turn right at the intersection to follow our route, as the lead vehicl」(軌跡 go→go)
  - v_1f434d15_2.5 s106: 「Wait in the queue behind the stopped lead vehicle for the red traffic light to turn green 」→「Proceed forward and turn right to follow our route, as the lead vehicle departs and the in」(軌跡 go→go)
  - v_20bcd747_3.0 s82: 「Decelerate to stop at the stop line for the all-way stop sign before turning right to foll」→「Proceed into the intersection and turn right to follow our route, as the crossing vehicle 」(軌跡 stop→go)

## BF16 では「進む」が一度も出ないのに FP8 で出た場面(軌跡で。文で go の回数も併記)

| 場面 | 集合 | seed 数 | 軌跡で進む BF16 / 本体 / 本体+出力層 | 文で go BF16 / 本体 / 本体+出力層 |
|---|---|---|---|---|

(逆向き: BF16 だけで「進む」が出た場面は 0。)

## 1 トークン目の確率(teacher forcing。BF16 の最初の seed の判断理由を流し、位置 0 の分布を温度 0.6 で)

「進む」系の頭のトークン: Acc、Adv、Appro、Change、Complete、Enter、Execute、Finish、Follow、Go、Merge、Move、Pro、Pull、Res、Ste、Take、Turn。「止まる」系: Bring、Come、Hold、Rem、Stay、Stop、Wait、Y。Dec(Decelerate)は止まる/ゆるめるの両方に使われ、Cre(Creep)はゆっくり前に出るので、どちらにも入れない。上位 30 トークンの中の確率の和。

| 集合 | 場面 | 変種 | 進む系 平均 | 止まる系 平均 | Decelerate 平均 | 進む系の差(FP8 − BF16)[95%] |
|---|---|---|---|---|---|---|
| A | 22 | BF16 | 14.5% | 61.7% | 18.2% | — |
| A | 22 | 本体だけ FP8 | 14.8% | 61.5% | 17.9% | +0.37 [+0.03, +0.72] pt(上がった場面 15 / 下がった場面 7) |
| A | 22 | 本体 + 出力層 FP8 | 15.9% | 60.6% | 17.4% | +1.45 [+0.90, +2.07] pt(上がった場面 22 / 下がった場面 0) |
| B | 117 | BF16 | 11.0% | 7.0% | 61.6% | — |
| B | 117 | 本体だけ FP8 | 10.9% | 6.9% | 60.3% | -0.05 [-0.22, +0.12] pt(上がった場面 62 / 下がった場面 55) |
| B | 117 | 本体 + 出力層 FP8 | 11.8% | 6.9% | 59.3% | +0.79 [+0.59, +1.00] pt(上がった場面 101 / 下がった場面 16) |
| A+B | 139 | BF16 | 11.5% | 15.7% | 54.7% | — |
| A+B | 139 | 本体だけ FP8 | 11.5% | 15.6% | 53.6% | +0.02 [-0.13, +0.17] pt(上がった場面 77 / 下がった場面 62) |
| A+B | 139 | 本体 + 出力層 FP8 | 12.4% | 15.4% | 52.7% | +0.90 [+0.71, +1.11] pt(上がった場面 123 / 下がった場面 16) |

進む系の確率が最も上がった場面(本体 + 出力層 FP8): v_20bcd747_3.0 60.5%→65.3%; v_ff0dbfc5_8.5 37.6%→41.7%; v_1f434d15_2.5 24.7%→28.8%; v_ba67827f_8.0 15.3%→19.4%; v_fbd62533_4.5 19.5%→23.5%
