"""Contact sheet of a video: one tile every `step` seconds, timestamped."""
import sys, cv2, numpy as np
video, out, step, tile_w, cols = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
t0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
t1 = float(sys.argv[7]) if len(sys.argv) > 7 else 1e9
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = n / fps
tiles = []
t = t0
while t < min(dur, t1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    if not ok: break
    h = int(f.shape[0] * tile_w / f.shape[1])
    f = cv2.resize(f, (tile_w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(f, (0, 0), (62, 20), (0, 0, 0), -1)
    cv2.putText(f, f"{t:.1f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(f); t += step
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
print(out, len(tiles), "tiles")
