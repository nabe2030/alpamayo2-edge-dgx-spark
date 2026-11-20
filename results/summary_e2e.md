## Per condition (measured steps; share over 1 s is inference only, for E* it is the total including re-rendering)

| Condition | Done | Steps | Logs | Median ms | p90 | Max | Over 1 s | Tokens median/max | Fit ms + ms per token |
|---|---|---|---|---|---|---|---|---|---|
| E10_all_pil_threads_cpu | done | 476 | 4 | **573** | 659 | 893 | **0.0%** | 21 / 36 | 311 + 9.28 |
| E10_all_pil_threads_pinned | done | 476 | 4 | **550** | 595 | 684 | **0.0%** | 21 / 36 | 309 + 9.25 |
| E10_pgcv_pil_threads_cpu | done | 476 | 4 | **716** | 828 | 1732 | **1.7%** | 21 / 36 | 314 + 15.49 |
| E10_pgcv_pil_threads_pinned | done | 476 | 4 | **685** | 776 | 913 | **0.0%** | 21 / 36 | 312 + 15.50 |
| E1_all_pil_threads_cpu | done | 144 | 4 | **740** | 846 | 998 | **0.0%** | 22 / 36 | 413 + 9.13 |
| E1_all_pil_threads_pinned | done | 144 | 4 | **716** | 760 | 845 | **0.0%** | 22 / 36 | 414 + 9.16 |
| E1_pgcv_pil_threads_cpu | done | 144 | 4 | **889** | 1038 | 1446 | **13.2%** | 23 / 36 | 416 + 15.45 |
| E1_pgcv_pil_threads_pinned | done | 144 | 4 | **861** | 933 | 1063 | **2.8%** | 23 / 36 | 415 + 15.44 |

- E10_all_pil_threads_cpu: re-rendering median 60.4 / p90 123.1 / max 364.2 ms, inference median 507 / max 644 ms, total median 573 / max 893 ms (over 1 s 0.0%)

- E10_all_pil_threads_pinned: re-rendering median 43.8 / p90 46.4 / max 51.2 ms, inference median 505 / max 641 ms, total median 550 / max 684 ms (over 1 s 0.0%)

- E10_pgcv_pil_threads_cpu: re-rendering median 60.9 / p90 172.0 / max 1109.9 ms, inference median 642 / max 871 ms, total median 716 / max 1732 ms (over 1 s 1.7%)

- E10_pgcv_pil_threads_pinned: re-rendering median 44.0 / p90 48.1 / max 51.1 ms, inference median 641 / max 867 ms, total median 685 / max 913 ms (over 1 s 0.0%)

- E1_all_pil_threads_cpu: re-rendering median 115.6 / p90 223.1 / max 360.3 ms, inference median 618 / max 742 ms, total median 740 / max 998 ms (over 1 s 0.0%)

- E1_all_pil_threads_pinned: re-rendering median 94.2 / p90 102.1 / max 107.1 ms, inference median 620 / max 743 ms, total median 716 / max 845 ms (over 1 s 0.0%)

- E1_pgcv_pil_threads_cpu: re-rendering median 113.0 / p90 253.6 / max 610.1 ms, inference median 767 / max 973 ms, total median 889 / max 1446 ms (over 1 s 13.2%)

- E1_pgcv_pil_threads_pinned: re-rendering median 94.8 / p90 101.4 / max 104.9 ms, inference median 768 / max 969 ms, total median 861 / max 1063 ms (over 1 s 2.8%)

## Per log (median / max / over 1 s, token count median/max)

| Log | E10_all_pil_threads_cpu | E10_all_pil_threads_pinned | E10_pgcv_pil_threads_cpu | E10_pgcv_pil_threads_pinned | E1_all_pil_threads_cpu | E1_all_pil_threads_pinned | E1_pgcv_pil_threads_cpu | E1_pgcv_pil_threads_pinned |
|---|---|---|---|---|---|---|---|---|
| 9e9bcfb7 | 547 / 791 / 0% (18/30) | 520 / 628 / 0% (18/30) | 666 / 1035 / 1% (18/30) | 634 / 813 / 0% (18/30) | 698 / 906 / 0% (19/30) | 676 / 781 / 0% (19/30) | 846 / 1042 / 3% (19/30) | 799 / 975 / 0% (19/30) |
| 0bae3b5e | 582 / 879 / 0% (22/36) | 563 / 684 / 0% (22/36) | 729 / 1024 / 2% (22/36) | 702 / 913 / 0% (22/36) | 774 / 998 / 0% (24/34) | 729 / 817 / 0% (24/34) | 909 / 1186 / 14% (24/36) | 882 / 1061 / 6% (24/36) |
| 25e5c600 | 607 / 893 / 0% (25/27) | 586 / 611 / 0% (25/27) | 767 / 1138 / 3% (25/27) | 749 / 784 / 0% (25/27) | 750 / 936 / 0% (24/36) | 735 / 845 / 0% (24/36) | 911 / 1446 / 17% (24/36) | 886 / 1063 / 3% (24/36) |
| a7636fca | 555 / 861 / 0% (20/35) | 537 / 676 / 0% (20/35) | 682 / 1732 / 2% (20/35) | 664 / 894 / 0% (20/35) | 711 / 754 / 0% (20/24) | 690 / 731 / 0% (20/24) | 840 / 1243 / 19% (20/35) | 817 / 1045 / 3% (20/35) |

## Slowest runs (5 per condition)

- **E10_all_pil_threads_cpu**
  - 25e5c600 t0=5.3s pass1: 893 ms (re-rendering 350), 25 tokens, ViT 6 images, 76°C, vit 53 / prefill 182 / decode 242 / expert 54
  - 0bae3b5e t0=5.7s pass1: 879 ms (re-rendering 364), 22 tokens, ViT 6 images, 69°C, vit 52 / prefill 182 / decode 214 / expert 54
  - a7636fca t0=5.1s pass1: 861 ms (re-rendering 358), 21 tokens, ViT 6 images, 67°C, vit 52 / prefill 180 / decode 205 / expert 55
  - a7636fca t0=4.7s pass1: 854 ms (re-rendering 348), 21 tokens, ViT 6 images, 65°C, vit 52 / prefill 183 / decode 206 / expert 53
  - 0bae3b5e t0=6.3s pass1: 852 ms (re-rendering 266), 30 tokens, ViT 6 images, 70°C, vit 52 / prefill 180 / decode 288 / expert 55
  - Runs furthest above the fit (the part not explained by token count): 0bae3b5e t0=6.9s +6 ms (23 tokens, vit 52 / prefill 187 / decode 223 / expert 55); 9e9bcfb7 t0=13.7s +5 ms (21 tokens, vit 52 / prefill 186 / decode 206 / expert 54); 0bae3b5e t0=4.4s +5 ms (22 tokens, vit 53 / prefill 186 / decode 214 / expert 55)
- **E10_all_pil_threads_pinned**
  - 0bae3b5e t0=8.5s pass1: 684 ms (re-rendering 43), 36 tokens, ViT 6 images, 66°C, vit 52 / prefill 181 / decode 342 / expert 53
  - 0bae3b5e t0=8.3s pass1: 683 ms (re-rendering 44), 36 tokens, ViT 6 images, 70°C, vit 52 / prefill 182 / decode 341 / expert 53
  - a7636fca t0=8.7s pass1: 676 ms (re-rendering 41), 35 tokens, ViT 6 images, 66°C, vit 53 / prefill 181 / decode 336 / expert 53
  - 0bae3b5e t0=8.7s pass1: 675 ms (re-rendering 42), 35 tokens, ViT 6 images, 63°C, vit 52 / prefill 182 / decode 332 / expert 54
  - 0bae3b5e t0=9.3s pass1: 667 ms (re-rendering 45), 34 tokens, ViT 6 images, 71°C, vit 52 / prefill 182 / decode 323 / expert 53
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=4.7s +6 ms (21 tokens, vit 53 / prefill 185 / decode 207 / expert 53); 0bae3b5e t0=11.5s +6 ms (22 tokens, vit 52 / prefill 186 / decode 215 / expert 53); a7636fca t0=13.5s +6 ms (20 tokens, vit 52 / prefill 186 / decode 196 / expert 53)
- **E10_pgcv_pil_threads_cpu**
  - a7636fca t0=6.1s pass1: 1732 ms (re-rendering 1110), 20 tokens, ViT 6 images, 67°C, vit 52 / prefill 178 / decode 328 / expert 52
  - 25e5c600 t0=5.3s pass1: 1138 ms (re-rendering 435), 25 tokens, ViT 6 images, 73°C, vit 52 / prefill 182 / decode 405 / expert 53
  - 25e5c600 t0=6.1s pass1: 1078 ms (re-rendering 346), 27 tokens, ViT 6 images, 70°C, vit 52 / prefill 179 / decode 436 / expert 53
  - a7636fca t0=5.3s pass1: 1074 ms (re-rendering 436), 21 tokens, ViT 6 images, 68°C, vit 52 / prefill 178 / decode 343 / expert 52
  - 9e9bcfb7 t0=5.8s pass1: 1035 ms (re-rendering 262), 30 tokens, ViT 6 images, 61°C, vit 52 / prefill 179 / decode 478 / expert 52
  - Runs furthest above the fit (the part not explained by token count): 9e9bcfb7 t0=15.6s +7 ms (6 tokens, vit 53 / prefill 184 / decode 114 / expert 52); a7636fca t0=4.4s +7 ms (18 tokens, vit 55 / prefill 181 / decode 299 / expert 52); 25e5c600 t0=4.6s +6 ms (26 tokens, vit 52 / prefill 183 / decode 423 / expert 52)
- **E10_pgcv_pil_threads_pinned**
  - 0bae3b5e t0=8.5s pass1: 913 ms (re-rendering 45), 36 tokens, ViT 6 images, 66°C, vit 51 / prefill 177 / decode 575 / expert 52
  - 0bae3b5e t0=8.7s pass1: 900 ms (re-rendering 46), 35 tokens, ViT 6 images, 67°C, vit 51 / prefill 179 / decode 559 / expert 52
  - 0bae3b5e t0=8.3s pass1: 896 ms (re-rendering 45), 35 tokens, ViT 6 images, 66°C, vit 52 / prefill 177 / decode 557 / expert 52
  - a7636fca t0=8.7s pass1: 894 ms (re-rendering 42), 35 tokens, ViT 6 images, 73°C, vit 52 / prefill 177 / decode 558 / expert 53
  - 0bae3b5e t0=9.4s pass1: 883 ms (re-rendering 45), 34 tokens, ViT 6 images, 66°C, vit 52 / prefill 177 / decode 543 / expert 54
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=4.0s +14 ms (20 tokens, vit 52 / prefill 192 / decode 328 / expert 53); 25e5c600 t0=4.2s +8 ms (26 tokens, vit 53 / prefill 181 / decode 427 / expert 52); 25e5c600 t0=8.7s +8 ms (27 tokens, vit 52 / prefill 179 / decode 436 / expert 54)
- **E1_all_pil_threads_cpu**
  - 0bae3b5e t0=7.1s pass1: 998 ms (re-rendering 350), 26 tokens, ViT 14 images, 58°C, vit 151 / prefill 177 / decode 250 / expert 54
  - 0bae3b5e t0=6.1s pass2: 973 ms (re-rendering 360), 22 tokens, ViT 14 images, 59°C, vit 152 / prefill 177 / decode 213 / expert 54
  - 0bae3b5e t0=8.1s pass2: 938 ms (re-rendering 353), 19 tokens, ViT 14 images, 66°C, vit 152 / prefill 177 / decode 185 / expert 53
  - 25e5c600 t0=7.1s pass1: 936 ms (re-rendering 279), 27 tokens, ViT 14 images, 70°C, vit 152 / prefill 176 / decode 258 / expert 53
  - 25e5c600 t0=10.1s pass1: 927 ms (re-rendering 287), 25 tokens, ViT 14 images, 69°C, vit 150 / prefill 178 / decode 242 / expert 54
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=4.1s +6 ms (20 tokens, vit 152 / prefill 183 / decode 196 / expert 55); a7636fca t0=8.1s +6 ms (20 tokens, vit 155 / prefill 179 / decode 196 / expert 53); 25e5c600 t0=15.1s +5 ms (27 tokens, vit 152 / prefill 184 / decode 259 / expert 54)
- **E1_all_pil_threads_pinned**
  - 25e5c600 t0=4.1s pass2: 845 ms (re-rendering 102), 36 tokens, ViT 14 images, 71°C, vit 152 / prefill 181 / decode 342 / expert 53
  - 0bae3b5e t0=9.1s pass2: 817 ms (re-rendering 94), 34 tokens, ViT 14 images, 63°C, vit 152 / prefill 180 / decode 321 / expert 54
  - 0bae3b5e t0=12.1s pass3: 794 ms (re-rendering 94), 31 tokens, ViT 14 images, 70°C, vit 152 / prefill 181 / decode 296 / expert 56
  - 9e9bcfb7 t0=9.1s pass3: 781 ms (re-rendering 90), 30 tokens, ViT 14 images, 69°C, vit 154 / prefill 180 / decode 288 / expert 54
  - 0bae3b5e t0=10.1s pass3: 778 ms (re-rendering 98), 29 tokens, ViT 14 images, 71°C, vit 152 / prefill 182 / decode 277 / expert 54
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=6.1s +5 ms (23 tokens, vit 151 / prefill 183 / decode 224 / expert 57); a7636fca t0=9.1s +5 ms (24 tokens, vit 153 / prefill 184 / decode 232 / expert 54); a7636fca t0=4.1s +5 ms (20 tokens, vit 156 / prefill 180 / decode 196 / expert 54)
- **E1_pgcv_pil_threads_cpu**
  - 25e5c600 t0=6.1s pass1: 1446 ms (re-rendering 610), 27 tokens, ViT 14 images, 65°C, vit 152 / prefill 179 / decode 437 / expert 52
  - a7636fca t0=7.1s pass2: 1243 ms (re-rendering 470), 23 tokens, ViT 14 images, 68°C, vit 149 / prefill 179 / decode 373 / expert 55
  - 25e5c600 t0=7.1s pass1: 1217 ms (re-rendering 415), 25 tokens, ViT 14 images, 64°C, vit 149 / prefill 181 / decode 403 / expert 52
  - 0bae3b5e t0=7.1s pass1: 1186 ms (re-rendering 369), 26 tokens, ViT 14 images, 63°C, vit 149 / prefill 181 / decode 419 / expert 52
  - 25e5c600 t0=7.1s pass3: 1139 ms (re-rendering 354), 24 tokens, ViT 14 images, 66°C, vit 150 / prefill 179 / decode 388 / expert 52
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=10.1s +9 ms (20 tokens, vit 151 / prefill 180 / decode 331 / expert 56); 25e5c600 t0=15.1s +7 ms (27 tokens, vit 150 / prefill 179 / decode 433 / expert 54); a7636fca t0=6.1s +6 ms (6 tokens, vit 151 / prefill 183 / decode 112 / expert 52)
- **E1_pgcv_pil_threads_pinned**
  - 25e5c600 t0=4.1s pass2: 1063 ms (re-rendering 94), 36 tokens, ViT 14 images, 65°C, vit 148 / prefill 179 / decode 574 / expert 53
  - 0bae3b5e t0=8.1s pass3: 1061 ms (re-rendering 94), 36 tokens, ViT 14 images, 64°C, vit 148 / prefill 181 / decode 571 / expert 52
  - a7636fca t0=8.1s pass2: 1045 ms (re-rendering 90), 35 tokens, ViT 14 images, 67°C, vit 151 / prefill 178 / decode 557 / expert 53
  - 0bae3b5e t0=9.1s pass2: 1026 ms (re-rendering 101), 33 tokens, ViT 14 images, 64°C, vit 149 / prefill 180 / decode 528 / expert 52
  - 0bae3b5e t0=12.1s pass3: 989 ms (re-rendering 97), 31 tokens, ViT 14 images, 64°C, vit 148 / prefill 179 / decode 497 / expert 52
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=13.1s +5 ms (6 tokens, vit 150 / prefill 184 / decode 112 / expert 51); a7636fca t0=4.1s +5 ms (20 tokens, vit 149 / prefill 183 / decode 329 / expert 52); 25e5c600 t0=7.1s +5 ms (23 tokens, vit 150 / prefill 180 / decode 376 / expert 53)
