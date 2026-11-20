## 条件ごと(計測の歩。1 秒を超えた割合は推論だけ、E* は描き直し込みの合計)

| 条件 | 済み | 歩数 | ログ | 中央値 ms | p90 | 最大 | 1 秒超え | トークン 中央値/最大 | 当てはめ ms + ms/トークン |
|---|---|---|---|---|---|---|---|---|---|
| E10_all_pil_threads_cpu | 済 | 476 | 4 | **573** | 659 | 893 | **0.0%** | 21 / 36 | 311 + 9.28 |
| E10_all_pil_threads_pinned | 済 | 476 | 4 | **550** | 595 | 684 | **0.0%** | 21 / 36 | 309 + 9.25 |
| E10_pgcv_pil_threads_cpu | 済 | 476 | 4 | **716** | 828 | 1732 | **1.7%** | 21 / 36 | 314 + 15.49 |
| E10_pgcv_pil_threads_pinned | 済 | 476 | 4 | **685** | 776 | 913 | **0.0%** | 21 / 36 | 312 + 15.50 |
| E1_all_pil_threads_cpu | 済 | 144 | 4 | **740** | 846 | 998 | **0.0%** | 22 / 36 | 413 + 9.13 |
| E1_all_pil_threads_pinned | 済 | 144 | 4 | **716** | 760 | 845 | **0.0%** | 22 / 36 | 414 + 9.16 |
| E1_pgcv_pil_threads_cpu | 済 | 144 | 4 | **889** | 1038 | 1446 | **13.2%** | 23 / 36 | 416 + 15.45 |
| E1_pgcv_pil_threads_pinned | 済 | 144 | 4 | **861** | 933 | 1063 | **2.8%** | 23 / 36 | 415 + 15.44 |

- E10_all_pil_threads_cpu: 描き直し 中央値 60.4 / p90 123.1 / 最大 364.2 ms、推論 中央値 507 / 最大 644 ms、合計 中央値 573 / 最大 893 ms(1 秒超え 0.0%)

- E10_all_pil_threads_pinned: 描き直し 中央値 43.8 / p90 46.4 / 最大 51.2 ms、推論 中央値 505 / 最大 641 ms、合計 中央値 550 / 最大 684 ms(1 秒超え 0.0%)

- E10_pgcv_pil_threads_cpu: 描き直し 中央値 60.9 / p90 172.0 / 最大 1109.9 ms、推論 中央値 642 / 最大 871 ms、合計 中央値 716 / 最大 1732 ms(1 秒超え 1.7%)

- E10_pgcv_pil_threads_pinned: 描き直し 中央値 44.0 / p90 48.1 / 最大 51.1 ms、推論 中央値 641 / 最大 867 ms、合計 中央値 685 / 最大 913 ms(1 秒超え 0.0%)

- E1_all_pil_threads_cpu: 描き直し 中央値 115.6 / p90 223.1 / 最大 360.3 ms、推論 中央値 618 / 最大 742 ms、合計 中央値 740 / 最大 998 ms(1 秒超え 0.0%)

- E1_all_pil_threads_pinned: 描き直し 中央値 94.2 / p90 102.1 / 最大 107.1 ms、推論 中央値 620 / 最大 743 ms、合計 中央値 716 / 最大 845 ms(1 秒超え 0.0%)

- E1_pgcv_pil_threads_cpu: 描き直し 中央値 113.0 / p90 253.6 / 最大 610.1 ms、推論 中央値 767 / 最大 973 ms、合計 中央値 889 / 最大 1446 ms(1 秒超え 13.2%)

- E1_pgcv_pil_threads_pinned: 描き直し 中央値 94.8 / p90 101.4 / 最大 104.9 ms、推論 中央値 768 / 最大 969 ms、合計 中央値 861 / 最大 1063 ms(1 秒超え 2.8%)

## ログごと(中央値 / 最大 / 1 秒超え、トークン数の中央値/最大)

| ログ | E10_all_pil_threads_cpu | E10_all_pil_threads_pinned | E10_pgcv_pil_threads_cpu | E10_pgcv_pil_threads_pinned | E1_all_pil_threads_cpu | E1_all_pil_threads_pinned | E1_pgcv_pil_threads_cpu | E1_pgcv_pil_threads_pinned |
|---|---|---|---|---|---|---|---|---|
| 9e9bcfb7 | 547 / 791 / 0% (18/30) | 520 / 628 / 0% (18/30) | 666 / 1035 / 1% (18/30) | 634 / 813 / 0% (18/30) | 698 / 906 / 0% (19/30) | 676 / 781 / 0% (19/30) | 846 / 1042 / 3% (19/30) | 799 / 975 / 0% (19/30) |
| 0bae3b5e | 582 / 879 / 0% (22/36) | 563 / 684 / 0% (22/36) | 729 / 1024 / 2% (22/36) | 702 / 913 / 0% (22/36) | 774 / 998 / 0% (24/34) | 729 / 817 / 0% (24/34) | 909 / 1186 / 14% (24/36) | 882 / 1061 / 6% (24/36) |
| 25e5c600 | 607 / 893 / 0% (25/27) | 586 / 611 / 0% (25/27) | 767 / 1138 / 3% (25/27) | 749 / 784 / 0% (25/27) | 750 / 936 / 0% (24/36) | 735 / 845 / 0% (24/36) | 911 / 1446 / 17% (24/36) | 886 / 1063 / 3% (24/36) |
| a7636fca | 555 / 861 / 0% (20/35) | 537 / 676 / 0% (20/35) | 682 / 1732 / 2% (20/35) | 664 / 894 / 0% (20/35) | 711 / 754 / 0% (20/24) | 690 / 731 / 0% (20/24) | 840 / 1243 / 19% (20/35) | 817 / 1045 / 3% (20/35) |

## 最も遅い回(条件ごとに 5 回)

- **E10_all_pil_threads_cpu**
  - 25e5c600 t0=5.3s pass1: 893 ms(描き直し 350)、25 トークン、ViT 6 枚、76℃、vit 53 / prefill 182 / decode 242 / expert 54
  - 0bae3b5e t0=5.7s pass1: 879 ms(描き直し 364)、22 トークン、ViT 6 枚、69℃、vit 52 / prefill 182 / decode 214 / expert 54
  - a7636fca t0=5.1s pass1: 861 ms(描き直し 358)、21 トークン、ViT 6 枚、67℃、vit 52 / prefill 180 / decode 205 / expert 55
  - a7636fca t0=4.7s pass1: 854 ms(描き直し 348)、21 トークン、ViT 6 枚、65℃、vit 52 / prefill 183 / decode 206 / expert 53
  - 0bae3b5e t0=6.3s pass1: 852 ms(描き直し 266)、30 トークン、ViT 6 枚、70℃、vit 52 / prefill 180 / decode 288 / expert 55
  - 当てはめから最も上に外れた回(トークン数で説明できない分): 0bae3b5e t0=6.9s +6 ms(23 トークン、vit 52 / prefill 187 / decode 223 / expert 55); 9e9bcfb7 t0=13.7s +5 ms(21 トークン、vit 52 / prefill 186 / decode 206 / expert 54); 0bae3b5e t0=4.4s +5 ms(22 トークン、vit 53 / prefill 186 / decode 214 / expert 55)
- **E10_all_pil_threads_pinned**
  - 0bae3b5e t0=8.5s pass1: 684 ms(描き直し 43)、36 トークン、ViT 6 枚、66℃、vit 52 / prefill 181 / decode 342 / expert 53
  - 0bae3b5e t0=8.3s pass1: 683 ms(描き直し 44)、36 トークン、ViT 6 枚、70℃、vit 52 / prefill 182 / decode 341 / expert 53
  - a7636fca t0=8.7s pass1: 676 ms(描き直し 41)、35 トークン、ViT 6 枚、66℃、vit 53 / prefill 181 / decode 336 / expert 53
  - 0bae3b5e t0=8.7s pass1: 675 ms(描き直し 42)、35 トークン、ViT 6 枚、63℃、vit 52 / prefill 182 / decode 332 / expert 54
  - 0bae3b5e t0=9.3s pass1: 667 ms(描き直し 45)、34 トークン、ViT 6 枚、71℃、vit 52 / prefill 182 / decode 323 / expert 53
  - 当てはめから最も上に外れた回(トークン数で説明できない分): a7636fca t0=4.7s +6 ms(21 トークン、vit 53 / prefill 185 / decode 207 / expert 53); 0bae3b5e t0=11.5s +6 ms(22 トークン、vit 52 / prefill 186 / decode 215 / expert 53); a7636fca t0=13.5s +6 ms(20 トークン、vit 52 / prefill 186 / decode 196 / expert 53)
- **E10_pgcv_pil_threads_cpu**
  - a7636fca t0=6.1s pass1: 1732 ms(描き直し 1110)、20 トークン、ViT 6 枚、67℃、vit 52 / prefill 178 / decode 328 / expert 52
  - 25e5c600 t0=5.3s pass1: 1138 ms(描き直し 435)、25 トークン、ViT 6 枚、73℃、vit 52 / prefill 182 / decode 405 / expert 53
  - 25e5c600 t0=6.1s pass1: 1078 ms(描き直し 346)、27 トークン、ViT 6 枚、70℃、vit 52 / prefill 179 / decode 436 / expert 53
  - a7636fca t0=5.3s pass1: 1074 ms(描き直し 436)、21 トークン、ViT 6 枚、68℃、vit 52 / prefill 178 / decode 343 / expert 52
  - 9e9bcfb7 t0=5.8s pass1: 1035 ms(描き直し 262)、30 トークン、ViT 6 枚、61℃、vit 52 / prefill 179 / decode 478 / expert 52
  - 当てはめから最も上に外れた回(トークン数で説明できない分): 9e9bcfb7 t0=15.6s +7 ms(6 トークン、vit 53 / prefill 184 / decode 114 / expert 52); a7636fca t0=4.4s +7 ms(18 トークン、vit 55 / prefill 181 / decode 299 / expert 52); 25e5c600 t0=4.6s +6 ms(26 トークン、vit 52 / prefill 183 / decode 423 / expert 52)
- **E10_pgcv_pil_threads_pinned**
  - 0bae3b5e t0=8.5s pass1: 913 ms(描き直し 45)、36 トークン、ViT 6 枚、66℃、vit 51 / prefill 177 / decode 575 / expert 52
  - 0bae3b5e t0=8.7s pass1: 900 ms(描き直し 46)、35 トークン、ViT 6 枚、67℃、vit 51 / prefill 179 / decode 559 / expert 52
  - 0bae3b5e t0=8.3s pass1: 896 ms(描き直し 45)、35 トークン、ViT 6 枚、66℃、vit 52 / prefill 177 / decode 557 / expert 52
  - a7636fca t0=8.7s pass1: 894 ms(描き直し 42)、35 トークン、ViT 6 枚、73℃、vit 52 / prefill 177 / decode 558 / expert 53
  - 0bae3b5e t0=9.4s pass1: 883 ms(描き直し 45)、34 トークン、ViT 6 枚、66℃、vit 52 / prefill 177 / decode 543 / expert 54
  - 当てはめから最も上に外れた回(トークン数で説明できない分): a7636fca t0=4.0s +14 ms(20 トークン、vit 52 / prefill 192 / decode 328 / expert 53); 25e5c600 t0=4.2s +8 ms(26 トークン、vit 53 / prefill 181 / decode 427 / expert 52); 25e5c600 t0=8.7s +8 ms(27 トークン、vit 52 / prefill 179 / decode 436 / expert 54)
- **E1_all_pil_threads_cpu**
  - 0bae3b5e t0=7.1s pass1: 998 ms(描き直し 350)、26 トークン、ViT 14 枚、58℃、vit 151 / prefill 177 / decode 250 / expert 54
  - 0bae3b5e t0=6.1s pass2: 973 ms(描き直し 360)、22 トークン、ViT 14 枚、59℃、vit 152 / prefill 177 / decode 213 / expert 54
  - 0bae3b5e t0=8.1s pass2: 938 ms(描き直し 353)、19 トークン、ViT 14 枚、66℃、vit 152 / prefill 177 / decode 185 / expert 53
  - 25e5c600 t0=7.1s pass1: 936 ms(描き直し 279)、27 トークン、ViT 14 枚、70℃、vit 152 / prefill 176 / decode 258 / expert 53
  - 25e5c600 t0=10.1s pass1: 927 ms(描き直し 287)、25 トークン、ViT 14 枚、69℃、vit 150 / prefill 178 / decode 242 / expert 54
  - 当てはめから最も上に外れた回(トークン数で説明できない分): a7636fca t0=4.1s +6 ms(20 トークン、vit 152 / prefill 183 / decode 196 / expert 55); a7636fca t0=8.1s +6 ms(20 トークン、vit 155 / prefill 179 / decode 196 / expert 53); 25e5c600 t0=15.1s +5 ms(27 トークン、vit 152 / prefill 184 / decode 259 / expert 54)
- **E1_all_pil_threads_pinned**
  - 25e5c600 t0=4.1s pass2: 845 ms(描き直し 102)、36 トークン、ViT 14 枚、71℃、vit 152 / prefill 181 / decode 342 / expert 53
  - 0bae3b5e t0=9.1s pass2: 817 ms(描き直し 94)、34 トークン、ViT 14 枚、63℃、vit 152 / prefill 180 / decode 321 / expert 54
  - 0bae3b5e t0=12.1s pass3: 794 ms(描き直し 94)、31 トークン、ViT 14 枚、70℃、vit 152 / prefill 181 / decode 296 / expert 56
  - 9e9bcfb7 t0=9.1s pass3: 781 ms(描き直し 90)、30 トークン、ViT 14 枚、69℃、vit 154 / prefill 180 / decode 288 / expert 54
  - 0bae3b5e t0=10.1s pass3: 778 ms(描き直し 98)、29 トークン、ViT 14 枚、71℃、vit 152 / prefill 182 / decode 277 / expert 54
  - 当てはめから最も上に外れた回(トークン数で説明できない分): a7636fca t0=6.1s +5 ms(23 トークン、vit 151 / prefill 183 / decode 224 / expert 57); a7636fca t0=9.1s +5 ms(24 トークン、vit 153 / prefill 184 / decode 232 / expert 54); a7636fca t0=4.1s +5 ms(20 トークン、vit 156 / prefill 180 / decode 196 / expert 54)
- **E1_pgcv_pil_threads_cpu**
  - 25e5c600 t0=6.1s pass1: 1446 ms(描き直し 610)、27 トークン、ViT 14 枚、65℃、vit 152 / prefill 179 / decode 437 / expert 52
  - a7636fca t0=7.1s pass2: 1243 ms(描き直し 470)、23 トークン、ViT 14 枚、68℃、vit 149 / prefill 179 / decode 373 / expert 55
  - 25e5c600 t0=7.1s pass1: 1217 ms(描き直し 415)、25 トークン、ViT 14 枚、64℃、vit 149 / prefill 181 / decode 403 / expert 52
  - 0bae3b5e t0=7.1s pass1: 1186 ms(描き直し 369)、26 トークン、ViT 14 枚、63℃、vit 149 / prefill 181 / decode 419 / expert 52
  - 25e5c600 t0=7.1s pass3: 1139 ms(描き直し 354)、24 トークン、ViT 14 枚、66℃、vit 150 / prefill 179 / decode 388 / expert 52
  - 当てはめから最も上に外れた回(トークン数で説明できない分): a7636fca t0=10.1s +9 ms(20 トークン、vit 151 / prefill 180 / decode 331 / expert 56); 25e5c600 t0=15.1s +7 ms(27 トークン、vit 150 / prefill 179 / decode 433 / expert 54); a7636fca t0=6.1s +6 ms(6 トークン、vit 151 / prefill 183 / decode 112 / expert 52)
- **E1_pgcv_pil_threads_pinned**
  - 25e5c600 t0=4.1s pass2: 1063 ms(描き直し 94)、36 トークン、ViT 14 枚、65℃、vit 148 / prefill 179 / decode 574 / expert 53
  - 0bae3b5e t0=8.1s pass3: 1061 ms(描き直し 94)、36 トークン、ViT 14 枚、64℃、vit 148 / prefill 181 / decode 571 / expert 52
  - a7636fca t0=8.1s pass2: 1045 ms(描き直し 90)、35 トークン、ViT 14 枚、67℃、vit 151 / prefill 178 / decode 557 / expert 53
  - 0bae3b5e t0=9.1s pass2: 1026 ms(描き直し 101)、33 トークン、ViT 14 枚、64℃、vit 149 / prefill 180 / decode 528 / expert 52
  - 0bae3b5e t0=12.1s pass3: 989 ms(描き直し 97)、31 トークン、ViT 14 枚、64℃、vit 148 / prefill 179 / decode 497 / expert 52
  - 当てはめから最も上に外れた回(トークン数で説明できない分): a7636fca t0=13.1s +5 ms(6 トークン、vit 150 / prefill 184 / decode 112 / expert 51); a7636fca t0=4.1s +5 ms(20 トークン、vit 149 / prefill 183 / decode 329 / expert 52); 25e5c600 t0=7.1s +5 ms(23 トークン、vit 150 / prefill 180 / decode 376 / expert 53)
