# LA691 FPV tour

A 45 second "indoor FPV drone" fly-through of the Lagos villa listed as LA691
on yourluxuryproperty.com, built only from the listing's real photos.

No generative video, no AI imagery. Each photo gets a monocular depth map and
a virtual camera films it with real parallax. Shots are linked by
fly-through, whip and rise transitions with true motion blur (temporal
supersampling), then graded, titled and scored.

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `fpv/fetch.py` | Pulls the gallery (full size, no thumbnails or logos) and the listing text |
| 2 | `fpv/depth.py` | Depth Anything V2 (ONNX, CPU), mirrored pass averaged, guided-filter edges |
| 3 | `fpv/warp.py` | 2.5D camera: ray / depth-surface intersection by damped fixed-point iteration |
| 4 | `fpv/engine.py` | Shot timeline on the beat grid, FPV speed ramps, banking, transitions, motion blur |
| 5 | `fpv/finish.py` | Highlight roll-off, bloom, split tone, vignette, fine grain |
| 6 | `fpv/graphics.py` | Title, room labels with mask reveals, specs, progress hairline, end card |
| 7 | `fpv/audio.py` | Synthesized deep-house bed plus whooshes and hits locked to every cut |
| 8 | `fpv/render.py` | Parallel render, H.264 encode, audio mux |

The edit lives in `config/la691.json`: photo order, beats per shot, camera
start and end pose, transition type and the doorway each fly-through aims at.

## Run

```bash
pip install -r requirements.txt
./run.sh            # fetch photos, depth, soundtrack, render
```

Output: `output/LA691_fpv_tour.mp4` (1920x1080, 30 fps, H.264 + AAC).

Listing photos are not committed: they belong to the agency. `run.sh`
downloads them. Fonts are Cormorant Garamond and Montserrat (SIL OFL, see
`assets/fonts`).
