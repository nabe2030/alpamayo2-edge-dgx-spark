## Per condition (measured steps; share over 1 s is inference only, for E* it is the total including re-rendering)

| Condition | Done | Steps | Logs | Median ms | p90 | Max | Over 1 s | Tokens median/max | Fit ms + ms per token |
|---|---|---|---|---|---|---|---|---|---|
| F1_body | done | 432 | 12 | **736** | 790 | 878 | **0.0%** | 22 / 36 | 420 + 9.96 |

- F1_body: re-rendering median 95.5 / p90 99.9 / max 105.0 ms, inference median 638 / max 778 ms, total median 736 / max 878 ms (over 1 s 0.0%)

## Per log (median / max / over 1 s, token count median/max)

| Log | F1_body |
|---|---|
| 0bae3b5e | 754 / 870 / 0% (24/36) |
| 36aec72e | 770 / 842 / 0% (26/33) |
| 0fb7276f | 740 / 826 / 0% (22/31) |
| 201fe83b | 783 / 819 / 0% (28/30) |
| 25e5c600 | 760 / 878 / 0% (24/36) |
| 88f47a10 | 747 / 782 / 0% (24/27) |
| b87683ae | 726 / 811 / 0% (21/29) |
| 92b900b1 | 714 / 822 / 0% (20/30) |
| bdb9d309 | 720 / 770 / 0% (20/25) |
| a7636fca | 709 / 762 / 0% (20/25) |
| d46ff5df | 734 / 784 / 0% (22/27) |
| c8ec7be0 | 693 / 783 / 0% (18/27) |

## Slowest runs (5 per condition)

- **F1_body**
  - 25e5c600 t0=4.1s pass2: 878 ms (re-rendering 101), 36 tokens, ViT 14 images, 65°C, vit 152 / prefill 185 / decode 372 / expert 53
  - 0bae3b5e t0=8.1s pass3: 870 ms (re-rendering 92), 36 tokens, ViT 14 images, 61°C, vit 150 / prefill 185 / decode 373 / expert 53
  - 0bae3b5e t0=9.1s pass2: 851 ms (re-rendering 95), 34 tokens, ViT 14 images, 60°C, vit 154 / prefill 183 / decode 351 / expert 53
  - 36aec72e t0=4.1s pass3: 842 ms (re-rendering 93), 33 tokens, ViT 14 images, 70°C, vit 152 / prefill 183 / decode 343 / expert 54
  - 0fb7276f t0=12.1s pass3: 826 ms (re-rendering 98), 31 tokens, ViT 14 images, 68°C, vit 156 / prefill 182 / decode 322 / expert 53
  - Runs furthest above the fit (the part not explained by token count): 88f47a10 t0=14.1s +7 ms (24 tokens, vit 157 / prefill 184 / decode 254 / expert 54); a7636fca t0=10.1s +6 ms (20 tokens, vit 154 / prefill 186 / decode 215 / expert 54); bdb9d309 t0=10.1s +6 ms (25 tokens, vit 154 / prefill 188 / decode 264 / expert 53)
