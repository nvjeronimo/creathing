# YLP FPV tours

"Indoor FPV drone" fly-throughs for yourluxuryproperty.com listings, built
only from the listing's real photos.

No generative video and no AI imagery. Each photo gets a monocular depth
map and a virtual camera films it with real parallax. Shots are linked by
fly-through, whip and rise transitions with true motion blur (temporal
supersampling), then graded, titled and scored.

| Project | Config | Format |
|---------|--------|--------|
| VM687, Vilamoura | `config/vm687_reel.json` | 1080x1920 reel, 34 s |
| LA691, Lagos | `config/la691.json` | 1920x1080, 45 s (auto edit until the photos are in) |

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `fpv/fetch.py` | Pulls a listing gallery (full size, no thumbnails or logos) and its text |
| 2 | `fpv/depth.py` | Depth Anything V2 (ONNX, CPU), mirrored pass averaged, guided-filter edges |
| 3 | `fpv/warp.py` | Photo clean-up, depth floor and simplification, 2.5D camera by ray / depth-surface intersection |
| 4 | `fpv/engine.py` | Beat-grid timeline, keyframed FPV paths that fly where they look, banking, transitions, motion blur |
| 5 | `fpv/finish.py` | Highlight roll-off, bloom, split tone, vignette, fine grain |
| 6 | `fpv/graphics.py` | Title, room labels, specs, end card; light or dark theme, Reels-safe layout in 9:16 |
| 7 | `fpv/audio.py` | Synthesized deep-house bed plus whooshes and hits locked to every cut |
| 8 | `fpv/render.py` | Parallel render, H.264 encode, audio mux, contact sheets for review |

A config is the whole edit: photo order, beats per shot, camera keys,
transition type and the doorway each fly-through aims at.

Two settings matter most for interiors:

- `dfloor` (per photo): nothing is farther than the room's own walls, so the
  view through a window behaves as a plane in the wall and mullions stay
  straight. Read it off the depth map at the wall next to the window.
- `heading` (per shot, default on): the camera moves where it looks. Sliding
  sideways relative to the view is what bends thin verticals.

## Run

```bash
pip install -r requirements.txt
# put the photos in the folder named by the config ("photos"), then:
./run.sh config/vm687_reel.json
# review stills without a full render:
python3 -m fpv.render config/vm687_reel.json sheet.jpg --scale 0.5 --cols 6 --stills 2,8,16,25
```

Listing photos and renders are not committed: they belong to the agency and
this repository is public. Fonts are Cormorant Garamond and Montserrat (SIL
OFL, see `assets/fonts`).
