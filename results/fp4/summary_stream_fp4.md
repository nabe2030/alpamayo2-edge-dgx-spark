## Per condition (measured steps; share over 1 s is inference only, for E* it is the total including re-rendering)

| Condition | Done | Steps | Logs | Median ms | p90 | Max | Over 1 s | Tokens median/max | Fit ms + ms per token |
|---|---|---|---|---|---|---|---|---|---|
| M10_bf16 | done | 1428 | 12 | **643** | 733 | 870 | **0.0%** | 21 / 36 | 315 + 15.46 |
| M10_fp4_gptq:0.1 | done | 1428 | 12 | **475** | 521 | 609 | **0.0%** | 21 / 36 | 305 + 7.92 |
| M10_fp8body | done | 1428 | 12 | **536** | 593 | 692 | **0.0%** | 21 / 37 | 317 + 10.18 |
| M1_bf16 | done | 432 | 12 | **758** | 848 | 973 | **0.0%** | 22 / 36 | 417 + 15.53 |
| M1_fp4_gptq:0.1 | done | 432 | 12 | **581** | 622 | 693 | **0.0%** | 22 / 36 | 407 + 7.88 |
| M1_fp8body | done | 432 | 12 | **636** | 696 | 782 | **0.0%** | 22 / 36 | 409 + 10.34 |

## Per log (median / max / over 1 s, token count median/max)

| Log | M10_bf16 | M10_fp4_gptq:0.1 | M10_fp8body | M1_bf16 | M1_fp4_gptq:0.1 | M1_fp8body |
|---|---|---|---|---|---|---|
| 0bae3b5e | 656 / 870 / 0% (22/36) | 479 / 587 / 0% (22/36) | 543 / 687 / 0% (22/36) | 801 / 940 / 0% (24/33) | 588 / 671 / 0% (24/34) | 656 / 782 / 0% (24/36) |
| 36aec72e | 715 / 736 / 0% (26/27) | 515 / 537 / 0% (26/27) | 581 / 595 / 0% (26/27) | 815 / 923 / 0% (26/33) | 610 / 665 / 0% (26/33) | 675 / 751 / 0% (26/33) |
| 0fb7276f | 640 / 747 / 0% (21/28) | 472 / 527 / 0% (21/28) | 531 / 602 / 0% (21/28) | 762 / 897 / 0% (22/31) | 584 / 653 / 0% (23/31) | 637 / 732 / 0% (22/31) |
| 201fe83b | 745 / 783 / 0% (28/30) | 523 / 546 / 0% (28/30) | 600 / 626 / 0% (28/30) | 843 / 884 / 0% (28/30) | 617 / 647 / 0% (26/30) | 696 / 720 / 0% (28/30) |
| 25e5c600 | 703 / 738 / 0% (25/27) | 505 / 550 / 0% (25/27) | 572 / 596 / 0% (25/27) | 790 / 973 / 0% (24/36) | 598 / 693 / 0% (24/36) | 657 / 782 / 0% (24/36) |
| 88f47a10 | 637 / 763 / 0% (21/29) | 469 / 546 / 0% (21/29) | 529 / 611 / 0% (21/29) | 768 / 807 / 0% (22/25) | 586 / 605 / 0% (22/25) | 641 / 689 / 0% (22/27) |
| b87683ae | 626 / 813 / 0% (20/32) | 464 / 575 / 0% (20/34) | 521 / 665 / 0% (20/34) | 741 / 871 / 0% (21/29) | 573 / 639 / 0% (21/29) | 624 / 712 / 0% (21/29) |
| 92b900b1 | 610 / 734 / 0% (19/27) | 455 / 550 / 0% (19/27) | 513 / 692 / 0% (19/37) | 725 / 882 / 0% (20/30) | 564 / 626 / 0% (20/28) | 616 / 715 / 0% (20/30) |
| bdb9d309 | 643 / 730 / 0% (21/27) | 474 / 522 / 0% (21/26) | 558 / 595 / 0% (21/27) | 731 / 807 / 0% (20/25) | 564 / 610 / 0% (20/25) | 618 / 671 / 0% (20/25) |
| a7636fca | 623 / 857 / 0% (20/35) | 460 / 565 / 0% (20/33) | 519 / 650 / 0% (20/33) | 725 / 960 / 0% (20/35) | 562 / 683 / 0% (20/35) | 614 / 667 / 0% (20/25) |
| d46ff5df | 656 / 857 / 0% (22/35) | 479 / 609 / 0% (22/35) | 541 / 676 / 0% (22/35) | 755 / 835 / 0% (22/27) | 578 / 620 / 0% (22/27) | 634 / 678 / 0% (22/26) |
| c8ec7be0 | 584 / 736 / 0% (17/27) | 446 / 534 / 0% (18/28) | 498 / 602 / 0% (18/28) | 711 / 944 / 0% (19/34) | 556 / 674 / 0% (19/34) | 598 / 689 / 0% (18/27) |

## Slowest runs (5 per condition)

- **M10_bf16**
  - 0bae3b5e t0=8.5s pass1: 870 ms, 36 tokens, ViT 6 images, 64°C, vit 52 / prefill 178 / decode 576 / expert 52
  - d46ff5df t0=5.6s pass1: 857 ms, 35 tokens, ViT 6 images, 64°C, vit 52 / prefill 179 / decode 561 / expert 53
  - a7636fca t0=8.7s pass1: 857 ms, 35 tokens, ViT 6 images, 65°C, vit 53 / prefill 178 / decode 562 / expert 52
  - d46ff5df t0=5.0s pass1: 857 ms, 35 tokens, ViT 6 images, 63°C, vit 52 / prefill 179 / decode 561 / expert 52
  - d46ff5df t0=5.4s pass1: 857 ms, 35 tokens, ViT 6 images, 61°C, vit 52 / prefill 179 / decode 559 / expert 53
  - Runs furthest above the fit (the part not explained by token count): d46ff5df t0=15.8s +11 ms (22 tokens, vit 52 / prefill 181 / decode 366 / expert 54); c8ec7be0 t0=15.1s +6 ms (21 tokens, vit 52 / prefill 184 / decode 345 / expert 52); 25e5c600 t0=8.4s +6 ms (27 tokens, vit 53 / prefill 181 / decode 440 / expert 53)
- **M10_fp4_gptq:0.1**
  - d46ff5df t0=5.8s pass1: 609 ms, 35 tokens, ViT 6 images, 67°C, vit 52 / prefill 178 / decode 295 / expert 70
  - d46ff5df t0=5.6s pass1: 589 ms, 35 tokens, ViT 6 images, 64°C, vit 53 / prefill 175 / decode 295 / expert 52
  - 0bae3b5e t0=8.5s pass1: 587 ms, 36 tokens, ViT 6 images, 61°C, vit 54 / prefill 174 / decode 295 / expert 51
  - d46ff5df t0=5.5s pass1: 581 ms, 35 tokens, ViT 6 images, 63°C, vit 52 / prefill 175 / decode 289 / expert 51
  - d46ff5df t0=5.4s pass1: 577 ms, 35 tokens, ViT 6 images, 63°C, vit 53 / prefill 173 / decode 286 / expert 51
  - Runs furthest above the fit (the part not explained by token count): b87683ae t0=15.0s +55 ms (6 tokens, vit 53 / prefill 232 / decode 59 / expert 51); 92b900b1 t0=4.0s +47 ms (25 tokens, vit 52 / prefill 227 / decode 207 / expert 51); 92b900b1 t0=4.6s +42 ms (11 tokens, vit 53 / prefill 219 / decode 99 / expert 51)
- **M10_fp8body**
  - 92b900b1 t0=14.4s pass1: 692 ms, 37 tokens, ViT 6 images, 71°C, vit 52 / prefill 180 / decode 395 / expert 53
  - 0bae3b5e t0=8.4s pass1: 687 ms, 36 tokens, ViT 6 images, 67°C, vit 54 / prefill 184 / decode 383 / expert 53
  - 0bae3b5e t0=8.5s pass1: 685 ms, 36 tokens, ViT 6 images, 60°C, vit 52 / prefill 181 / decode 387 / expert 53
  - d46ff5df t0=6.2s pass1: 676 ms, 35 tokens, ViT 6 images, 64°C, vit 52 / prefill 185 / decode 373 / expert 53
  - d46ff5df t0=5.4s pass1: 676 ms, 35 tokens, ViT 6 images, 62°C, vit 52 / prefill 186 / decode 371 / expert 53
  - Runs furthest above the fit (the part not explained by token count): a7636fca t0=15.8s +47 ms (6 tokens, vit 54 / prefill 231 / decode 75 / expert 53); 92b900b1 t0=10.5s +38 ms (22 tokens, vit 54 / prefill 213 / decode 247 / expert 52); 92b900b1 t0=8.5s +37 ms (22 tokens, vit 53 / prefill 211 / decode 251 / expert 52)
- **M1_bf16**
  - 25e5c600 t0=4.1s pass2: 973 ms, 36 tokens, ViT 14 images, 67°C, vit 148 / prefill 179 / decode 577 / expert 52
  - a7636fca t0=8.1s pass2: 960 ms, 35 tokens, ViT 14 images, 67°C, vit 149 / prefill 181 / decode 562 / expert 52
  - c8ec7be0 t0=9.1s pass3: 944 ms, 34 tokens, ViT 14 images, 67°C, vit 149 / prefill 179 / decode 547 / expert 52
  - 0bae3b5e t0=9.1s pass2: 940 ms, 33 tokens, ViT 14 images, 57°C, vit 149 / prefill 177 / decode 546 / expert 52
  - 36aec72e t0=4.1s pass3: 923 ms, 33 tokens, ViT 14 images, 66°C, vit 148 / prefill 180 / decode 527 / expert 52
  - Runs furthest above the fit (the part not explained by token count): 0bae3b5e t0=8.1s +55 ms (26 tokens, vit 171 / prefill 199 / decode 428 / expert 62); 0bae3b5e t0=4.1s +51 ms (22 tokens, vit 169 / prefill 175 / decode 382 / expert 66); 0bae3b5e t0=9.1s +36 ms (24 tokens, vit 148 / prefill 200 / decode 407 / expert 54)
- **M1_fp4_gptq:0.1**
  - 25e5c600 t0=4.1s pass2: 693 ms, 36 tokens, ViT 14 images, 73°C, vit 151 / prefill 176 / decode 295 / expert 52
  - a7636fca t0=8.1s pass2: 683 ms, 35 tokens, ViT 14 images, 66°C, vit 150 / prefill 176 / decode 287 / expert 52
  - c8ec7be0 t0=9.1s pass3: 674 ms, 34 tokens, ViT 14 images, 67°C, vit 151 / prefill 177 / decode 278 / expert 51
  - 0bae3b5e t0=9.1s pass2: 671 ms, 34 tokens, ViT 14 images, 62°C, vit 151 / prefill 174 / decode 278 / expert 51
  - 36aec72e t0=4.1s pass3: 665 ms, 33 tokens, ViT 14 images, 64°C, vit 151 / prefill 175 / decode 271 / expert 51
  - Runs furthest above the fit (the part not explained by token count): 201fe83b t0=6.1s +8 ms (28 tokens, vit 150 / prefill 181 / decode 233 / expert 53); 201fe83b t0=6.1s +6 ms (26 tokens, vit 154 / prefill 179 / decode 216 / expert 51); bdb9d309 t0=14.1s +6 ms (17 tokens, vit 152 / prefill 179 / decode 146 / expert 52)
- **M1_fp8body**
  - 25e5c600 t0=4.1s pass2: 782 ms, 36 tokens, ViT 14 images, 64°C, vit 150 / prefill 176 / decode 386 / expert 52
  - 0bae3b5e t0=8.1s pass3: 782 ms, 36 tokens, ViT 14 images, 60°C, vit 149 / prefill 180 / decode 385 / expert 52
  - 0bae3b5e t0=9.1s pass2: 760 ms, 34 tokens, ViT 14 images, 65°C, vit 153 / prefill 175 / decode 363 / expert 52
  - 36aec72e t0=4.1s pass3: 751 ms, 33 tokens, ViT 14 images, 60°C, vit 149 / prefill 176 / decode 357 / expert 52
  - 0fb7276f t0=5.1s pass3: 732 ms, 31 tokens, ViT 14 images, 69°C, vit 151 / prefill 178 / decode 333 / expert 53
  - Runs furthest above the fit (the part not explained by token count): 0bae3b5e t0=7.1s +6 ms (20 tokens, vit 151 / prefill 180 / decode 223 / expert 52); a7636fca t0=8.1s +6 ms (19 tokens, vit 150 / prefill 178 / decode 212 / expert 53); 36aec72e t0=6.1s +6 ms (26 tokens, vit 154 / prefill 178 / decode 282 / expert 54)
