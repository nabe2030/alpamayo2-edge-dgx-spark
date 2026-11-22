# Figures with Argoverse 2 images

## Source, modifications, license

- Source: Argoverse 2 Sensor Dataset, © 2021 Argo AI, LLC (https://www.argoverse.org/av2.html).
- Modifications: the original camera images were re-rendered to virtual cameras, cropped and resized, and predicted / recorded trajectories and text were overlaid. No unmodified AV2 image is included.
- License: these figures are adaptations of the AV2 images and are provided under **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/ , full text: `../../LICENSES/CC-BY-NC-SA-4.0.txt`). This is separate from the Apache-2.0 license of the code in this repository.
- The attribution and the modifications are also written at the bottom of each figure.

## How to redraw

`../../measure/article_figs.py` draws these figures from the recorded runs and the AV2 logs (after `source env.sh`; the records it reads are listed at the top of the script):

```bash
$PY $A2E_REPO/measure/article_figs.py out    # out/fig06_left_turn.jpg, out/fig07_input_layouts.jpg, out/fig11_pedestrian_stop.jpg
```

Redrawn from our records with this script, all three JPEG files match the ones here byte for byte. Fonts: Noto Sans CJK (`fonts-noto-cjk` on Ubuntu).

`demo_realtime.gif` is made in three steps: `../../measure/demo_realtime.py --mode realtime` runs the real-time loop on log 9e9bcfb7 for each setup (examples at the top of the script), `../../measure/demo_frames.py --layout gif` draws the frames from those records, and ffmpeg turns them into the GIF (command at the top of `demo_frames.py`). The inference times depend on the machine and on what else runs on it; ours were measured on 2026-11-21 on one GB10 with no other GPU work running.

`demo_realtime_fp4.gif` is made the same way, with `demo_realtime.py --q fp4` (`A2E_FP4_METHOD=gptq:0.1`, FP4 body with GPTQ, BF16 output layer) for the right side and `../../measure/demo_frames_fp4.py --layout gif` for the frames (labels, the orange color of the right side and one line on output quality differ from `demo_frames.py`). Its inference times were measured on 2026-11-22 on one GB10 with no other GPU work running; the frames were drawn on the CPU.

## Files

| File | Content |
|---|---|
| `fig06_left_turn.jpg` | Article figure 6: left turn. Re-rendered input (front 120°) vs. AV2 cameras as is (front approx. 47°): trajectories of 5 seeds, top view, reasoning |
| `fig07_input_layouts.jpg` | Article figure 7: one scene as input built in 3 layouts (front 3 slots), and the other 3 slots of the adopted layout |
| `fig11_pedestrian_stop.jpg` | Article figure 11: pedestrian crossing. Front image and top views of the trajectories of 5 seeds with BF16, FP8 body only, and FP8 body + output layer (the reasoning per seed is in the article text) |
| `demo_realtime.gif` | Real-time comparison on log 9e9bcfb7 (4–15 s, 1x): official tuned vs. the four optimizations + FP8 body only. Each prediction appears when its inference finishes, moved to the current pose with the recorded ego motion. 960×540, 7 fps, 11 s |
| `demo_realtime_fp4.gif` | Real-time comparison on the same log and interval (4–15 s, 1x): official tuned (gray) vs. the four optimizations + FP4 body (orange; GPTQ 4-bit body, BF16 output layer). Shown as in `demo_realtime.gif`. The bottom band also gives the output quality of the FP4 body measured offline (400 scenes × 5 seeds). 960×540, 7 fps, 11 s |
