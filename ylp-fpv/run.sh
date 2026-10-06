#!/usr/bin/env bash
# Usage: ./run.sh config/vm687_reel.json [output.mp4]
# Depth for every photo of the project, soundtrack, render.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-config/vm687_reel.json}"
NAME="$(basename "$CONFIG" .json)"
OUT="${2:-output/${NAME}.mp4}"
MODEL="models/depth_anything_v2_vitb_dynamic.onnx"

if [ ! -f "$MODEL" ]; then
  mkdir -p models
  curl -L -o "$MODEL" \
    https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitb_dynamic.onnx
fi

PHOTOS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('photos','assets/photos'))" "$CONFIG")
DEPTH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('depth','assets/depth'))" "$CONFIG")

mkdir -p output
python3 -m fpv.depth "$PHOTOS" "$DEPTH" --model "$MODEL"
python3 -m fpv.audio "$CONFIG" "output/${NAME}.wav"
python3 -m fpv.render "$CONFIG" "$OUT" --audio "output/${NAME}.wav" --crf 19 --maxrate 10M
