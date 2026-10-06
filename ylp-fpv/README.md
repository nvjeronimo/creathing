# YLP FPV tours

"Indoor FPV drone" fly-throughs for yourluxuryproperty.com listings, built
only from the listing's real photos.

No generative video and no AI imagery. Each photo gets a monocular depth
map and a virtual camera films it with real parallax. Shots are linked by
fly-through, whip and rise transitions with true motion blur (temporal
supersampling), then graded, titled and scored.

| Project | Config | Format |
|---------|--------|--------|
| VM687, Vilamoura, cinematic preset | `config/vm687_cine_16x9.json` | 1920x1080, 28 s, hard cuts on 108 BPM |
| VM687, Vilamoura, FPV reel | `config/vm687_reel.json` | 1080x1920, 34 s, fly-through and whip transitions |
| VM687, Vilamoura, FPV wide | `config/vm687_wide.json` | 1920x1080, 34 s |
| LA691, Lagos | `config/la691.json` | 1920x1080, 45 s (auto edit until the photos are in) |

The cinematic preset copies the editing language of two reference films,
measured frame by frame and on the spectrogram: see
`docs/reference_breakdown.md`. Hard cuts on the music grid, a flash-cut hook,
a wordmark over a texture, a speed-ramped dive on the drop, gimbal moves with
parallax, macro inserts with depth of field, a low-key warm grade matched by
numbers, and sound design (downlifter, ticks, impact, dropout, reverse swell).
`tools/ref_*.py` analyse new references the same way.

A test of generative motion (Higgsfield), with the rules a generated clip must
pass before it can enter a listing video, is planned in `docs/higgsfield_test.md`.

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `fpv/fetch.py` | Pulls a listing gallery (full size, no thumbnails or logos) and its text |
| 2 | `fpv/depth.py` | Depth Anything V2 (ONNX, CPU), mirrored pass averaged, guided-filter edges |
| 3 | `fpv/warp.py` | Photo clean-up, depth floor and simplification, 2.5D camera by ray / depth-surface intersection |
| 4 | `fpv/engine.py` | Beat-grid timeline, keyframed FPV paths that fly where they look, banking, transitions, motion blur |
| 5 | `fpv/finish.py` | Highlight roll-off, bloom, split tone, vignette, fine grain |
| 6 | `fpv/graphics.py` | Title, room labels, specs, end card, wordmark lockups; light or dark theme, Reels-safe layout in 9:16 |
| 7 | `fpv/audio.py` | Synthesized deep-house bed plus whooshes and hits locked to every cut |
| 8 | `fpv/render.py` | Parallel render, H.264 encode, audio mux, contact sheets for review |

A config is the whole edit: photo order, beats per shot, camera keys,
transition type and the doorway each fly-through aims at.

Two settings matter most for interiors:

- `dfloor` (per photo): nothing is farther than the room's own walls, so the
  view through a window behaves as a plane in the wall and mullions stay
  straight. Read it off the depth map at the window head, where the ceiling
  meets the glazing: a floor set at the glass leaves a step at the head and
  the top of every mullion bends.
- `heading` (per shot, default on): the camera moves where it looks. Sliding
  sideways relative to the view is what bends thin verticals.

## Run

```bash
pip install -r requirements.txt
# put the photos in the folder named by the config ("photos"), then:
# brand logo for the end card (SVG to transparent PNG):
python3 tools/svg2png.py assets/brand/ylp-logo.svg assets/brand/ylp-logo.png
./run.sh config/vm687_reel.json
# Higgsfield API check, one billable Seedance 2.5 clip (needs HF_KEY=key-id:key-secret
# in .env.local or the environment, and api.higgsfield.ai reachable):
python3 main.py
# review stills without a full render:
python3 -m fpv.render config/vm687_reel.json sheet.jpg --scale 0.5 --cols 6 --stills 2,8,16,25
```

Listing photos and renders are not committed: they belong to the agency and
this repository is public. Fonts are Cormorant Garamond and Montserrat (SIL
OFL, see `assets/fonts`).
