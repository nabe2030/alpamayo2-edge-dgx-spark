3280 records (scene × seed). Each record ran BF16, FP8 body only and FP8 body + output layer with the same seed.

## Set A (22 scenes, 1408 records)

| Variant | Trajectory go [95%] | Text go [95%] | Text go and trajectory go [95%] | Text class (stop/go/creep/slow/nominal/other) |
|---|---|---|---|---|
| BF16 | 26.3% [14.3%, 40.3%] | 11.9% [6.1%, 18.8%] | 6.9% [2.8%, 11.8%] | 1147 / 167 / 7 / 60 / 3 / 24 |
| FP8 body only | 26.3% [14.3%, 40.4%] | 12.4% [6.3%, 19.6%] | 7.0% [3.1%, 12.0%] | 1140 / 174 / 8 / 58 / 4 / 24 |
| FP8 body + output layer | 26.2% [14.3%, 40.2%] | 13.3% [7.0%, 20.7%] | 7.3% [3.2%, 12.6%] | 1121 / 187 / 10 / 62 / 3 / 25 |

Paired difference (same scene and seed; FP8 − BF16, points) and direction of disagreement:

| Variant | Trajectory go difference [95%] | BF16 stop→FP8 go | BF16 go→FP8 stop | Text go difference [95%] | Text stop→go | Text go→stop |
|---|---|---|---|---|---|---|
| FP8 body only | +0.1 [-0.4, +0.4] | 6 | 5 | +0.5 [+0.0, +1.1] | 9 | 3 |
| FP8 body + output layer | -0.1 [-0.5, +0.4] | 7 | 8 | +1.4 [+0.7, +2.2] | 20 | 2 |

## Set B (117 scenes, 1872 records)

| Variant | Trajectory go [95%] | Text go [95%] | Text go and trajectory go [95%] | Text class (stop/go/creep/slow/nominal/other) |
|---|---|---|---|---|
| BF16 | 92.8% [89.3%, 95.7%] | 9.5% [7.2%, 12.1%] | 9.1% [6.9%, 11.6%] | 1037 / 178 / 8 / 411 / 235 / 3 |
| FP8 body only | 93.0% [89.5%, 95.9%] | 9.2% [6.9%, 11.8%] | 8.8% [6.7%, 11.3%] | 1033 / 172 / 8 / 397 / 259 / 3 |
| FP8 body + output layer | 92.8% [89.4%, 95.8%] | 10.3% [7.9%, 13.0%] | 9.9% [7.6%, 12.4%] | 1015 / 192 / 7 / 403 / 252 / 3 |

Paired difference (same scene and seed; FP8 − BF16, points) and direction of disagreement:

| Variant | Trajectory go difference [95%] | BF16 stop→FP8 go | BF16 go→FP8 stop | Text go difference [95%] | Text stop→go | Text go→stop |
|---|---|---|---|---|---|---|
| FP8 body only | +0.2 [+0.0, +0.3] | 3 | 0 | -0.3 [-0.7, +0.1] | 3 | 4 |
| FP8 body + output layer | +0.0 [-0.2, +0.2] | 2 | 2 | +0.7 [+0.3, +1.2] | 14 | 0 |

## Set A+B (139 scenes, 3280 records)

| Variant | Trajectory go [95%] | Text go [95%] | Text go and trajectory go [95%] | Text class (stop/go/creep/slow/nominal/other) |
|---|---|---|---|---|
| BF16 | 64.3% [54.8%, 73.7%] | 10.5% [7.6%, 13.9%] | 8.2% [6.0%, 10.7%] | 2184 / 345 / 15 / 471 / 238 / 27 |
| FP8 body only | 64.4% [54.9%, 73.8%] | 10.5% [7.5%, 14.0%] | 8.0% [5.9%, 10.6%] | 2173 / 346 / 16 / 455 / 263 / 27 |
| FP8 body + output layer | 64.2% [54.8%, 73.7%] | 11.6% [8.5%, 15.2%] | 8.8% [6.5%, 11.5%] | 2136 / 379 / 17 / 465 / 255 / 28 |

Paired difference (same scene and seed; FP8 − BF16, points) and direction of disagreement:

| Variant | Trajectory go difference [95%] | BF16 stop→FP8 go | BF16 go→FP8 stop | Text go difference [95%] | Text stop→go | Text go→stop |
|---|---|---|---|---|---|---|
| FP8 body only | +0.1 [-0.1, +0.3] | 9 | 5 | +0.0 [-0.3, +0.4] | 12 | 7 |
| FP8 body + output layer | -0.0 [-0.2, +0.2] | 9 | 10 | +1.0 [+0.6, +1.4] | 34 | 2 |

## Divergence position for pairs whose text is stop in BF16 and go in FP8

- **FP8 body only**: 12 pairs. Divergent token position {0: 12}. Word swaps (top): Decelerate→Proceed 3, Remain→Proceed 2, Yield→Proceed 1, Remain→Pull 1, Hold→Proceed 1, Remain→The 1
  - v_20bcd747_3.0 s82: "Decelerate to stop at the stop line for the all-way stop sign before turning right to foll" → "Proceed into the intersection and turn right to follow our route, as the crossing vehicle " (trajectory stop→go)
  - v_20bcd747_3.0 s86: "Decelerate to stop at the stop line for the all-way stop sign at the intersection, prepari" → "Proceed into the intersection and turn right to follow our route, as the crossing vehicle " (trajectory go→go)
  - v_22052525_7.0 s50: "Yield to oncoming traffic before turning left to follow our route, as the oncoming vehicle" → "Proceed with the left turn to follow our route, as oncoming traffic has cleared our path u" (trajectory stop→stop)
  - v_7e4d67b3_3.5 s64: "Remain stopped behind the parked police car as traffic ahead is stationary in our lane." → "Pull out from the curb into the travel lane to follow our route, as the passing traffic ha" (trajectory stop→stop)
- **FP8 body + output layer**: 34 pairs. Divergent token position {0: 34}. Word swaps (top): Decelerate→Proceed 9, Wait→Proceed 7, Remain→Proceed 4, Yield→Proceed 3, Decelerate→Turn 2, Remain→Pull 1
  - v_1f434d15_2.5 s67: "Wait behind the stationary lead vehicle at the red traffic light to follow our route." → "Proceed forward in the right lane toward the intersection to follow our route, as the cros" (trajectory stop→stop)
  - v_1f434d15_2.5 s90: "Wait in queue behind the stopped lead vehicle at the intersection for the red traffic ligh" → "Proceed forward and turn right at the intersection to follow our route, as the lead vehicl" (trajectory go→go)
  - v_1f434d15_2.5 s106: "Wait in the queue behind the stopped lead vehicle for the red traffic light to turn green " → "Proceed forward and turn right to follow our route, as the lead vehicle departs and the in" (trajectory go→go)
  - v_20bcd747_3.0 s82: "Decelerate to stop at the stop line for the all-way stop sign before turning right to foll" → "Proceed into the intersection and turn right to follow our route, as the crossing vehicle " (trajectory stop→go)

## Scenes where BF16 never produced "go" but FP8 did (by trajectory; text go counts also shown)

| Scene | Set | Seeds | Trajectory go BF16 / body / body+output layer | Text go BF16 / body / body+output layer |
|---|---|---|---|---|

(Reverse direction: scenes where only BF16 produced "go": 0.)

## First-token probability (teacher forcing: feed the reasoning of the first BF16 seed, distribution at position 0 at temperature 0.6)

"Go" leading tokens: Acc, Adv, Appro, Change, Complete, Enter, Execute, Finish, Follow, Go, Merge, Move, Pro, Pull, Res, Ste, Take, Turn. "Stop" tokens: Bring, Come, Hold, Rem, Stay, Stop, Wait, Y. Dec (Decelerate) is used both for stopping and slowing down, and Cre (Creep) means moving forward slowly, so neither is counted in either group. Sum of probabilities within the top 30 tokens.

| Set | Scenes | Variant | Go mean | Stop mean | Decelerate mean | Go difference (FP8 − BF16) [95%] |
|---|---|---|---|---|---|---|
| A | 22 | BF16 | 14.5% | 61.7% | 18.2% | — |
| A | 22 | FP8 body only | 14.8% | 61.5% | 17.9% | +0.37 [+0.03, +0.72] pt (scenes up 15 / scenes down 7) |
| A | 22 | FP8 body + output layer | 15.9% | 60.6% | 17.4% | +1.45 [+0.90, +2.07] pt (scenes up 22 / scenes down 0) |
| B | 117 | BF16 | 11.0% | 7.0% | 61.6% | — |
| B | 117 | FP8 body only | 10.9% | 6.9% | 60.3% | -0.05 [-0.22, +0.12] pt (scenes up 62 / scenes down 55) |
| B | 117 | FP8 body + output layer | 11.8% | 6.9% | 59.3% | +0.79 [+0.59, +1.00] pt (scenes up 101 / scenes down 16) |
| A+B | 139 | BF16 | 11.5% | 15.7% | 54.7% | — |
| A+B | 139 | FP8 body only | 11.5% | 15.6% | 53.6% | +0.02 [-0.13, +0.17] pt (scenes up 77 / scenes down 62) |
| A+B | 139 | FP8 body + output layer | 12.4% | 15.4% | 52.7% | +0.90 [+0.71, +1.11] pt (scenes up 123 / scenes down 16) |

Scenes with the largest rise in go probability (FP8 body + output layer): v_20bcd747_3.0 60.5%→65.3%; v_ff0dbfc5_8.5 37.6%→41.7%; v_1f434d15_2.5 24.7%→28.8%; v_ba67827f_8.0 15.3%→19.4%; v_fbd62533_4.5 19.5%→23.5%
