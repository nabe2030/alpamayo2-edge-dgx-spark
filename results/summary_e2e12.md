## Per condition (measured steps; share over 1 s is inference only, for E* it is the total including re-rendering)

| Condition | Done | Steps | Logs | Median ms | p90 | Max | Over 1 s | Tokens median/max | Fit ms + ms per token |
|---|---|---|---|---|---|---|---|---|---|
| F1_all | done | 432 | 12 | **715** | 767 | 852 | **0.0%** | 22 / 36 | 419 + 9.16 |
| F1_pgcv | done | 432 | 12 | **849** | 941 | 1077 | **1.4%** | 22 / 36 | 416 + 15.43 |

- F1_all: re-rendering median 94.4 / p90 99.2 / max 105.2 ms, inference median 621 / max 749 ms, total median 715 / max 852 ms (over 1 s 0.0%)

- F1_pgcv: re-rendering median 93.9 / p90 98.8 / max 106.0 ms, inference median 756 / max 975 ms, total median 849 / max 1077 ms (over 1 s 1.4%)

## Per log (median / max / over 1 s, token count median/max)

| Log | F1_all | F1_pgcv |
|---|---|---|
| 0bae3b5e | 728 / 819 / 0% (24/34) | 877 / 1064 / 6% (24/36) |
| 36aec72e | 751 / 814 / 0% (26/33) | 911 / 1022 / 3% (26/33) |
| 0fb7276f | 721 / 802 / 0% (22/31) | 861 / 994 / 0% (22/31) |
| 201fe83b | 766 / 796 / 0% (28/30) | 933 / 979 / 0% (28/30) |
| 25e5c600 | 740 / 852 / 0% (24/36) | 890 / 1077 / 3% (24/36) |
| 88f47a10 | 715 / 764 / 0% (22/27) | 856 / 893 / 0% (22/25) |
| b87683ae | 698 / 782 / 0% (21/29) | 835 / 962 / 0% (21/29) |
| 92b900b1 | 691 / 786 / 0% (20/30) | 821 / 973 / 0% (20/30) |
| bdb9d309 | 705 / 746 / 0% (21/25) | 798 / 895 / 0% (18/25) |
| a7636fca | 696 / 738 / 0% (20/24) | 818 / 1053 / 3% (20/35) |
| d46ff5df | 712 / 765 / 0% (22/27) | 848 / 928 / 0% (22/27) |
| c8ec7be0 | 695 / 766 / 0% (20/27) | 788 / 1033 / 3% (18/34) |

## Slowest runs (5 per condition)

- **F1_all**
  - 25e5c600 t0=4.1s pass2: 852 ms (re-rendering 102), 36 tokens, ViT 14 images, 63°C, vit 154 / prefill 184 / decode 343 / expert 54
  - 0bae3b5e t0=9.1s pass2: 819 ms (re-rendering 93), 34 tokens, ViT 14 images, 65°C, vit 151 / prefill 182 / decode 324 / expert 54
  - 36aec72e t0=4.1s pass3: 814 ms (re-rendering 92), 33 tokens, ViT 14 images, 69°C, vit 153 / prefill 183 / decode 316 / expert 55
  - 0fb7276f t0=4.1s pass3: 802 ms (re-rendering 102), 31 tokens, ViT 14 images, 72°C, vit 152 / prefill 185 / decode 295 / expert 53
  - 0bae3b5e t0=12.1s pass3: 800 ms (re-rendering 95), 31 tokens, ViT 14 images, 68°C, vit 154 / prefill 186 / decode 296 / expert 53
  - Runs furthest above the fit (the part not explained by token count): d46ff5df t0=8.1s +7 ms (16 tokens, vit 154 / prefill 189 / decode 159 / expert 54); 25e5c600 t0=5.1s +7 ms (24 tokens, vit 153 / prefill 191 / decode 233 / expert 54); a7636fca t0=9.1s +5 ms (19 tokens, vit 157 / prefill 185 / decode 187 / expert 54)
- **F1_pgcv**
  - 25e5c600 t0=4.1s pass2: 1077 ms (re-rendering 102), 36 tokens, ViT 14 images, 67°C, vit 152 / prefill 182 / decode 573 / expert 52
  - 0bae3b5e t0=8.1s pass3: 1064 ms (re-rendering 94), 36 tokens, ViT 14 images, 63°C, vit 150 / prefill 179 / decode 573 / expert 52
  - a7636fca t0=8.1s pass2: 1053 ms (re-rendering 98), 35 tokens, ViT 14 images, 69°C, vit 149 / prefill 178 / decode 561 / expert 52
  - c8ec7be0 t0=9.1s pass3: 1033 ms (re-rendering 91), 34 tokens, ViT 14 images, 69°C, vit 148 / prefill 180 / decode 544 / expert 55
  - 0bae3b5e t0=9.1s pass2: 1023 ms (re-rendering 98), 33 tokens, ViT 14 images, 64°C, vit 153 / prefill 177 / decode 528 / expert 52
  - Runs furthest above the fit (the part not explained by token count): 88f47a10 t0=14.1s +8 ms (21 tokens, vit 152 / prefill 184 / decode 345 / expert 52); 88f47a10 t0=7.1s +7 ms (24 tokens, vit 151 / prefill 181 / decode 393 / expert 54); b87683ae t0=5.1s +6 ms (22 tokens, vit 152 / prefill 180 / decode 362 / expert 53)
