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

## Files

| File | Content |
|---|---|
| `fig06_left_turn.jpg` | Article figure 6: left turn. Re-rendered input (front 120°) vs. AV2 cameras as is (front approx. 47°): trajectories of 5 seeds, top view, reasoning |
| `fig07_input_layouts.jpg` | Article figure 7: one scene as input built in 3 layouts (front 3 slots), and the other 3 slots of the adopted layout |
| `fig11_pedestrian_stop.jpg` | Article figure 11: pedestrian crossing. Trajectories and reasoning of 5 seeds with BF16, FP8 body only, and FP8 body + output layer |
