## Per condition (measured steps; share over 1 s is inference only, for E* it is the total including re-rendering)

| Condition | Done | Steps | Logs | Median ms | p90 | Max | Over 1 s | Tokens median/max | Fit ms + ms per token |
|---|---|---|---|---|---|---|---|---|---|
| M10_body | done | 1428 | 12 | **528** | 586 | 685 | **0.0%** | 21 / 37 | 314 + 10.05 |
| M1_body | done | 432 | 12 | **638** | 694 | 780 | **0.0%** | 22 / 36 | 419 + 9.98 |

## Per log (median / max / over 1 s, token count median/max)

| Log | M10_body | M1_body |
|---|---|---|
| 0bae3b5e | 536 / 673 / 0% (22/36) | 658 / 780 / 0% (24/36) |
| 36aec72e | 573 / 588 / 0% (26/27) | 677 / 745 / 0% (26/33) |
| 0fb7276f | 526 / 596 / 0% (21/28) | 640 / 729 / 0% (22/31) |
| 201fe83b | 594 / 621 / 0% (28/30) | 690 / 719 / 0% (28/30) |
| 25e5c600 | 566 / 589 / 0% (25/27) | 659 / 779 / 0% (24/36) |
| 88f47a10 | 524 / 590 / 0% (21/27) | 651 / 688 / 0% (24/27) |
| b87683ae | 517 / 656 / 0% (20/34) | 628 / 711 / 0% (21/29) |
| 92b900b1 | 504 / 685 / 0% (19/37) | 617 / 717 / 0% (20/30) |
| bdb9d309 | 552 / 584 / 0% (24/27) | 623 / 673 / 0% (20/25) |
| a7636fca | 514 / 643 / 0% (20/33) | 618 / 670 / 0% (20/25) |
| d46ff5df | 535 / 666 / 0% (22/35) | 638 / 688 / 0% (22/27) |
| c8ec7be0 | 493 / 595 / 0% (18/28) | 600 / 692 / 0% (18/27) |

## Slowest runs (5 per condition)

- **M10_body**
  - 92b900b1 t0=14.4s pass1: 685 ms, 37 tokens, ViT 6 images, 65°C, vit 52 / prefill 181 / decode 386 / expert 54
  - 0bae3b5e t0=8.5s pass1: 673 ms, 36 tokens, ViT 6 images, 56°C, vit 52 / prefill 181 / decode 376 / expert 53
  - 0bae3b5e t0=8.4s pass1: 673 ms, 36 tokens, ViT 6 images, 63°C, vit 53 / prefill 182 / decode 372 / expert 54
  - d46ff5df t0=5.3s pass1: 666 ms, 35 tokens, ViT 6 images, 61°C, vit 52 / prefill 181 / decode 364 / expert 57
  - d46ff5df t0=6.2s pass1: 666 ms, 35 tokens, ViT 6 images, 62°C, vit 52 / prefill 186 / decode 362 / expert 54
  - Runs furthest above the fit (the part not explained by token count): 201fe83b t0=8.9s +8 ms (28 tokens, vit 52 / prefill 183 / decode 295 / expert 55); b87683ae t0=10.8s +7 ms (21 tokens, vit 52 / prefill 187 / decode 225 / expert 54); a7636fca t0=8.4s +7 ms (20 tokens, vit 54 / prefill 185 / decode 215 / expert 55)
- **M1_body**
  - 0bae3b5e t0=8.1s pass3: 780 ms, 36 tokens, ViT 14 images, 64°C, vit 155 / prefill 181 / decode 374 / expert 54
  - 25e5c600 t0=4.1s pass2: 779 ms, 36 tokens, ViT 14 images, 63°C, vit 152 / prefill 184 / decode 372 / expert 53
  - 0bae3b5e t0=9.1s pass2: 757 ms, 34 tokens, ViT 14 images, 67°C, vit 151 / prefill 182 / decode 352 / expert 54
  - 36aec72e t0=4.1s pass3: 745 ms, 33 tokens, ViT 14 images, 62°C, vit 154 / prefill 179 / decode 342 / expert 54
  - 0fb7276f t0=5.1s pass3: 729 ms, 31 tokens, ViT 14 images, 64°C, vit 156 / prefill 181 / decode 322 / expert 54
  - Runs furthest above the fit (the part not explained by token count): d46ff5df t0=15.1s +8 ms (22 tokens, vit 153 / prefill 180 / decode 233 / expert 55); a7636fca t0=13.1s +6 ms (19 tokens, vit 157 / prefill 183 / decode 204 / expert 54); b87683ae t0=11.1s +5 ms (21 tokens, vit 154 / prefill 182 / decode 226 / expert 54)
