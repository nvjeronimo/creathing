#!/usr/bin/env bash
# Fetch the listing photos, estimate depth, synthesize the soundtrack, render.
set -euo pipefail
cd "$(dirname "$0")"

URL="https://www.yourluxuryproperty.com/property-detail/lagosvilla-for-saleria-do-alvorviews-la691"
CONFIG="config/la691.json"
MODEL="models/depth_anything_v2_vitb_dynamic.onnx"

if [ ! -f "$MODEL" ]; then
  mkdir -p models
  curl -L -o "$MODEL" \
    https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitb_dynamic.onnx
fi

if ! ls assets/photos/*.jpg >/dev/null 2>&1; then
  python3 -m fpv.fetch "$URL" assets/photos
fi

python3 -m fpv.depth assets/photos assets/depth --model "$MODEL"
python3 -m fpv.audio "$CONFIG" output/soundtrack.wav
python3 -m fpv.render "$CONFIG" output/LA691_fpv_tour.mp4 --audio output/soundtrack.wav \
  --crf 20 --maxrate 9M
