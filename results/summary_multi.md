## Per condition (measured steps; share over 1 s is inference only, for E* it is the total including re-rendering)

| Condition | Done | Steps | Logs | Median ms | p90 | Max | Over 1 s | Tokens median/max | Fit ms + ms per token |
|---|---|---|---|---|---|---|---|---|---|
| M10_all | done | 1428 | 12 | **512** | 560 | 644 | **0.0%** | 21 / 36 | 312 + 9.23 |
| M10_pgcv | done | 1428 | 12 | **641** | 731 | 867 | **0.0%** | 21 / 36 | 313 + 15.43 |
| M1_all | done | 432 | 12 | **620** | 670 | 752 | **0.0%** | 22 / 36 | 417 + 9.23 |
| M1_pgcv | done | 432 | 12 | **757** | 848 | 977 | **0.0%** | 22 / 36 | 417 + 15.43 |

## Per log (median / max / over 1 s, token count median/max)

| Log | M10_all | M10_pgcv | M1_all | M1_pgcv |
|---|---|---|---|---|
| 0bae3b5e | 516 / 644 / 0% (22/36) | 653 / 867 / 0% (22/36) | 636 / 730 / 0% (24/34) | 787 / 971 / 0% (24/36) |
| 36aec72e | 549 / 563 / 0% (26/27) | 714 / 734 / 0% (26/27) | 652 / 721 / 0% (26/33) | 816 / 926 / 0% (26/33) |
| 0fb7276f | 505 / 568 / 0% (21/28) | 638 / 748 / 0% (21/28) | 623 / 704 / 0% (22/31) | 764 / 897 / 0% (22/31) |
| 201fe83b | 566 / 592 / 0% (28/30) | 743 / 779 / 0% (28/30) | 671 / 696 / 0% (28/30) | 846 / 882 / 0% (28/30) |
| 25e5c600 | 542 / 564 / 0% (25/27) | 701 / 734 / 0% (25/27) | 639 / 752 / 0% (24/36) | 790 / 977 / 0% (24/36) |
| 88f47a10 | 503 / 564 / 0% (21/27) | 635 / 761 / 0% (21/29) | 621 / 665 / 0% (22/27) | 766 / 804 / 0% (22/25) |
| b87683ae | 510 / 627 / 0% (20/32) | 622 / 837 / 0% (20/34) | 607 / 690 / 0% (21/29) | 740 / 869 / 0% (21/29) |
| 92b900b1 | 487 / 561 / 0% (19/27) | 607 / 730 / 0% (19/27) | 600 / 694 / 0% (20/30) | 727 / 879 / 0% (20/30) |
| bdb9d309 | 531 / 562 / 0% (24/27) | 642 / 733 / 0% (21/27) | 607 / 653 / 0% (21/25) | 706 / 806 / 0% (18/25) |
| a7636fca | 496 / 634 / 0% (20/35) | 621 / 854 / 0% (20/35) | 601 / 644 / 0% (20/24) | 724 / 958 / 0% (20/35) |
| d46ff5df | 515 / 634 / 0% (22/35) | 653 / 855 / 0% (22/35) | 620 / 644 / 0% (22/25) | 755 / 834 / 0% (22/27) |
| c8ec7be0 | 477 / 563 / 0% (18/27) | 580 / 732 / 0% (17/27) | 602 / 667 / 0% (20/27) | 699 / 943 / 0% (18/34) |

## Slowest runs (5 per condition)

- **M10_all**
  - 0bae3b5e t0=8.5s pass1: 644 ms, 36 tokens, ViT 6 images, 68°C, vit 52 / prefill 181 / decode 346 / expert 53
  - 0bae3b5e t0=8.3s pass1: 642 ms, 36 tokens, ViT 6 images, 61°C, vit 52 / prefill 181 / decode 343 / expert 54
  - d46ff5df t0=5.5s pass1: 634 ms, 35 tokens, ViT 6 images, 63°C, vit 52 / prefill 181 / decode 335 / expert 54
  - a7636fca t0=8.7s pass1: 634 ms, 35 tokens, ViT 6 images, 64°C, vit 52 / prefill 184 / decode 333 / expert 53
  - 0bae3b5e t0=8.7s pass1: 633 ms, 35 tokens, ViT 6 images, 62°C, vit 53 / prefill 181 / decode 335 / expert 53
  - Runs furthest above the fit (the part not explained by token count): b87683ae t0=5.5s +43 ms (21 tokens, vit 64 / prefill 210 / decode 207 / expert 55); b87683ae t0=12.5s +38 ms (20 tokens, vit 62 / prefill 212 / decode 195 / expert 53); b87683ae t0=10.3s +34 ms (20 tokens, vit 70 / prefill 181 / decode 204 / expert 64)
- **M10_pgcv**
  - 0bae3b5e t0=8.5s pass1: 867 ms, 36 tokens, ViT 6 images, 61°C, vit 52 / prefill 177 / decode 574 / expert 53
  - d46ff5df t0=5.3s pass1: 855 ms, 35 tokens, ViT 6 images, 65°C, vit 52 / prefill 177 / decode 561 / expert 52
  - a7636fca t0=8.7s pass1: 854 ms, 35 tokens, ViT 6 images, 66°C, vit 51 / prefill 176 / decode 558 / expert 56
  - d46ff5df t0=5.4s pass1: 852 ms, 35 tokens, ViT 6 images, 65°C, vit 52 / prefill 176 / decode 560 / expert 53
  - d46ff5df t0=5.6s pass1: 852 ms, 35 tokens, ViT 6 images, 65°C, vit 51 / prefill 177 / decode 558 / expert 53
  - Runs furthest above the fit (the part not explained by token count): 0fb7276f t0=9.9s +9 ms (21 tokens, vit 53 / prefill 178 / decode 350 / expert 53); 88f47a10 t0=11.8s +7 ms (18 tokens, vit 52 / prefill 180 / decode 298 / expert 56); 201fe83b t0=14.6s +7 ms (20 tokens, vit 52 / prefill 179 / decode 333 / expert 52)
- **M1_all**
  - 25e5c600 t0=4.1s pass2: 752 ms, 36 tokens, ViT 14 images, 64°C, vit 151 / prefill 185 / decode 344 / expert 54
  - 0bae3b5e t0=9.1s pass2: 730 ms, 34 tokens, ViT 14 images, 68°C, vit 149 / prefill 184 / decode 327 / expert 54
  - 36aec72e t0=4.1s pass3: 721 ms, 33 tokens, ViT 14 images, 65°C, vit 155 / prefill 180 / decode 315 / expert 54
  - 0bae3b5e t0=12.1s pass3: 705 ms, 31 tokens, ViT 14 images, 69°C, vit 152 / prefill 183 / decode 299 / expert 54
  - 0fb7276f t0=12.1s pass3: 704 ms, 31 tokens, ViT 14 images, 67°C, vit 151 / prefill 183 / decode 298 / expert 54
  - Runs furthest above the fit (the part not explained by token count): bdb9d309 t0=7.1s +7 ms (16 tokens, vit 156 / prefill 182 / decode 160 / expert 56); d46ff5df t0=4.1s +7 ms (19 tokens, vit 154 / prefill 183 / decode 189 / expert 56); d46ff5df t0=8.1s +6 ms (19 tokens, vit 156 / prefill 182 / decode 188 / expert 55)
- **M1_pgcv**
  - 25e5c600 t0=4.1s pass2: 977 ms, 36 tokens, ViT 14 images, 66°C, vit 153 / prefill 179 / decode 574 / expert 53
  - 0bae3b5e t0=8.1s pass3: 971 ms, 36 tokens, ViT 14 images, 64°C, vit 150 / prefill 176 / decode 576 / expert 53
  - a7636fca t0=8.1s pass2: 958 ms, 35 tokens, ViT 14 images, 69°C, vit 150 / prefill 176 / decode 562 / expert 53
  - c8ec7be0 t0=9.1s pass3: 943 ms, 34 tokens, ViT 14 images, 70°C, vit 150 / prefill 176 / decode 544 / expert 56
  - 36aec72e t0=4.1s pass3: 926 ms, 33 tokens, ViT 14 images, 72°C, vit 155 / prefill 176 / decode 527 / expert 52
  - Runs furthest above the fit (the part not explained by token count): 92b900b1 t0=4.1s +7 ms (22 tokens, vit 152 / prefill 179 / decode 363 / expert 53); 25e5c600 t0=10.1s +7 ms (24 tokens, vit 151 / prefill 180 / decode 392 / expert 54); a7636fca t0=6.1s +6 ms (23 tokens, vit 155 / prefill 177 / decode 376 / expert 53)
