# Continuous runs over 12 logs: three configurations

- Same 12 logs and same method (same arguments; only the FP8 scope differs). All times in ms.
- Inference only = re-rendering is outside the measured interval. Including re-rendering = the total from reading the JPEGs to the trajectory, run back to back on the same GPU.
- The four optimizations and FP8 body + output layer were measured on the night of 2026-11-19, FP8 body only on 2026-11-20.

## Median / p90 / max, share over 1 s (steps)

| Configuration | Inference only, every 0.1 s | Inference only, every 1 s | Including re-rendering, every 1 s | Re-rendering (median / max) |
|---|---|---|---|---|
| Four optimizations (BF16) | 641 / 731 / 867, 0.0% (1428) | 757 / 848 / 977, 0.0% (432) | 849 / 941 / 1077, 1.4% (432) | 93.9 / 106.0 |
| + FP8 body only (recommended) | 528 / 586 / 685, 0.0% (1428) | 638 / 694 / 780, 0.0% (432) | 736 / 790 / 878, 0.0% (432) | 95.5 / 105.0 |
| + FP8 body + output layer (reference) | 512 / 560 / 644, 0.0% (1428) | 620 / 670 / 752, 0.0% (432) | 715 / 767 / 852, 0.0% (432) | 94.4 / 105.2 |

## Fit to reasoning length (time = a + b × tokens)

| Configuration | Every 0.1 s a + b | Every 1 s a + b | Length that fits in 1 s (every 1 s, inference only) | Inference in the run including re-rendering a + b | Length that fits in 1 s (adding the median re-rendering) | Tokens median / max |
|---|---|---|---|---|---|---|
| Four optimizations (BF16) | 313 + 15.43 | 417 + 15.43 | 37 | 416 + 15.43 | 31 | 22.0 / 36 |
| + FP8 body only (recommended) | 314 + 10.05 | 419 + 9.98 | 58 | 420 + 9.96 | 48 | 22.0 / 36 |
| + FP8 body + output layer (reference) | 312 + 9.23 | 417 + 9.23 | 63 | 419 + 9.16 | 53 | 22.0 / 36 |

## Per log (inference only, every 1 s, median / max)

| Log | Four optimizations (BF16) | + FP8 body only (recommended) | + FP8 body + output layer (reference) |
|---|---|---|---|
| 0bae3b5e | 787 / 971 | 658 / 780 | 636 / 730 |
| 36aec72e | 816 / 926 | 677 / 745 | 652 / 721 |
| 0fb7276f | 764 / 897 | 640 / 729 | 623 / 704 |
| 201fe83b | 846 / 882 | 690 / 719 | 671 / 696 |
| 25e5c600 | 790 / 977 | 659 / 779 | 639 / 752 |
| 88f47a10 | 766 / 804 | 651 / 688 | 621 / 665 |
| b87683ae | 740 / 869 | 628 / 711 | 607 / 690 |
| 92b900b1 | 727 / 879 | 617 / 717 | 600 / 694 |
| bdb9d309 | 706 / 806 | 623 / 673 | 607 / 653 |
| a7636fca | 724 / 958 | 618 / 670 | 601 / 644 |
| d46ff5df | 755 / 834 | 638 / 688 | 620 / 644 |
| c8ec7be0 | 699 / 943 | 600 / 692 | 602 / 667 |

## Slow steps with FP8 body only (the 5 slowest, and the largest positive residuals from the fit)

- **M10_body**
  - 92b900b1 t0=14.4s pass1: 685 ms, 37 tokens, ViT 6 images, preprocessing 9 / vit 52 / prefill 181 / decode 386 / expert 54
  - 0bae3b5e t0=8.5s pass1: 673 ms, 36 tokens, ViT 6 images, preprocessing 8 / vit 52 / prefill 181 / decode 376 / expert 53
  - 0bae3b5e t0=8.4s pass1: 673 ms, 36 tokens, ViT 6 images, preprocessing 8 / vit 53 / prefill 182 / decode 372 / expert 54
  - d46ff5df t0=5.3s pass1: 666 ms, 35 tokens, ViT 6 images, preprocessing 9 / vit 52 / prefill 181 / decode 364 / expert 57
  - d46ff5df t0=6.2s pass1: 666 ms, 35 tokens, ViT 6 images, preprocessing 9 / vit 52 / prefill 186 / decode 362 / expert 54
  - residual +8 ms: 201fe83b t0=8.9s pass1: 603 ms, 28 tokens, ViT 6 images, preprocessing 9 / vit 52 / prefill 183 / decode 295 / expert 55
  - residual +7 ms: b87683ae t0=10.8s pass1: 532 ms, 21 tokens, ViT 6 images, preprocessing 10 / vit 52 / prefill 187 / decode 225 / expert 54
  - residual +7 ms: a7636fca t0=8.4s pass1: 522 ms, 20 tokens, ViT 6 images, preprocessing 9 / vit 54 / prefill 185 / decode 215 / expert 55
  - residual +6 ms: 25e5c600 t0=12.7s pass1: 572 ms, 25 tokens, ViT 6 images, preprocessing 10 / vit 53 / prefill 188 / decode 264 / expert 54
  - residual +6 ms: b87683ae t0=11.9s pass1: 531 ms, 21 tokens, ViT 6 images, preprocessing 9 / vit 52 / prefill 187 / decode 224 / expert 55
- **M1_body**
  - 0bae3b5e t0=8.1s pass3: 780 ms, 36 tokens, ViT 14 images, preprocessing 13 / vit 155 / prefill 181 / decode 374 / expert 54
  - 25e5c600 t0=4.1s pass2: 779 ms, 36 tokens, ViT 14 images, preprocessing 14 / vit 152 / prefill 184 / decode 372 / expert 53
  - 0bae3b5e t0=9.1s pass2: 757 ms, 34 tokens, ViT 14 images, preprocessing 14 / vit 151 / prefill 182 / decode 352 / expert 54
  - 36aec72e t0=4.1s pass3: 745 ms, 33 tokens, ViT 14 images, preprocessing 13 / vit 154 / prefill 179 / decode 342 / expert 54
  - 0fb7276f t0=5.1s pass3: 729 ms, 31 tokens, ViT 14 images, preprocessing 13 / vit 156 / prefill 181 / decode 322 / expert 54
  - residual +8 ms: d46ff5df t0=15.1s pass1: 646 ms, 22 tokens, ViT 14 images, preprocessing 13 / vit 153 / prefill 180 / decode 233 / expert 55
  - residual +6 ms: a7636fca t0=13.1s pass1: 614 ms, 19 tokens, ViT 14 images, preprocessing 13 / vit 157 / prefill 183 / decode 204 / expert 54
  - residual +5 ms: b87683ae t0=11.1s pass2: 634 ms, 21 tokens, ViT 14 images, preprocessing 14 / vit 154 / prefill 182 / decode 226 / expert 54
  - residual +5 ms: d46ff5df t0=8.1s pass1: 644 ms, 22 tokens, ViT 14 images, preprocessing 14 / vit 155 / prefill 185 / decode 233 / expert 54
  - residual +5 ms: b87683ae t0=8.1s pass3: 664 ms, 24 tokens, ViT 14 images, preprocessing 14 / vit 153 / prefill 188 / decode 253 / expert 53
- **F1_body**
  - 25e5c600 t0=4.1s pass2: 777 ms (re-rendering 101), 36 tokens, ViT 14 images, preprocessing 13 / vit 152 / prefill 185 / decode 372 / expert 53
  - 0bae3b5e t0=8.1s pass3: 778 ms (re-rendering 92), 36 tokens, ViT 14 images, preprocessing 13 / vit 150 / prefill 185 / decode 373 / expert 53
  - 0bae3b5e t0=9.1s pass2: 756 ms (re-rendering 95), 34 tokens, ViT 14 images, preprocessing 13 / vit 154 / prefill 183 / decode 351 / expert 53
  - 36aec72e t0=4.1s pass3: 748 ms (re-rendering 93), 33 tokens, ViT 14 images, preprocessing 13 / vit 152 / prefill 183 / decode 343 / expert 54
  - 0fb7276f t0=12.1s pass3: 729 ms (re-rendering 98), 31 tokens, ViT 14 images, preprocessing 13 / vit 156 / prefill 182 / decode 322 / expert 53
  - residual +7 ms: 88f47a10 t0=14.1s pass2: 665 ms, 24 tokens, ViT 14 images, preprocessing 13 / vit 157 / prefill 184 / decode 254 / expert 54
  - residual +6 ms: a7636fca t0=10.1s pass1: 625 ms, 20 tokens, ViT 14 images, preprocessing 13 / vit 154 / prefill 186 / decode 215 / expert 54
  - residual +6 ms: bdb9d309 t0=10.1s pass1: 675 ms, 25 tokens, ViT 14 images, preprocessing 13 / vit 154 / prefill 188 / decode 264 / expert 53
  - residual +6 ms: a7636fca t0=12.1s pass1: 625 ms, 20 tokens, ViT 14 images, preprocessing 12 / vit 156 / prefill 188 / decode 213 / expert 53
  - residual +6 ms: 92b900b1 t0=4.1s pass3: 645 ms, 22 tokens, ViT 14 images, preprocessing 13 / vit 154 / prefill 189 / decode 233 / expert 53
