"""Every frame between t0 and t1 as one row, for transition anatomy."""
import sys, cv2, numpy as np
video, out = sys.argv[1], sys.argv[2]
spans = [tuple(map(float, s.split(":"))) for s in sys.argv[3].split(",")]
tile_h = int(sys.argv[4])
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS)
rows = []
for t0, t1 in spans:
    tiles = []
    for fi in range(int(round(t0 * fps)), int(round(t1 * fps)) + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
        if not ok: break
        w = int(f.shape[1] * tile_h / f.shape[0])
        f = cv2.resize(f, (w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(f, f"{fi / fps:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    rows.append(np.hstack(tiles))
W = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 4, 0, W - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
