"""Cut detection and per-shot camera motion from optical flow.

Cuts: HSV histogram distance spikes (hard cuts) plus a sliding test for
dissolves. Motion: Farneback flow on a 192 px proxy, fitted per frame to a
similarity model (pan dx, tilt dy, zoom, roll), plus a parallax measure
(residual flow after the global fit, high when the camera translates in depth).
"""
import json, sys
import cv2, numpy as np

video, out_json = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
frames, hists, lum, sat, sharp = [], [], [], [], []
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    pw = 192
    ph = int(h * pw / w)
    small = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    hists.append(cv2.normalize(hist, None).flatten())
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    frames.append(g)
    lum.append(float(g.mean()))
    sat.append(float(hsv[..., 1].mean()))
    sharp.append(float(cv2.Laplacian(cv2.cvtColor(cv2.resize(f, (480, int(h * 480 / w))), cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()))
n = len(frames)
d = np.array([cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(1, n)])
med = np.median(d)
cuts = []
for i in range(len(d)):
    lo, hi = max(0, i - 6), min(len(d), i + 7)
    local = np.median(np.r_[d[lo:i], d[i + 1:hi]])
    if d[i] > max(0.28, 3.5 * local) and (not cuts or i + 1 - cuts[-1] > 4):
        cuts.append(i + 1)
# motion per frame
motion = []
ph, pw = frames[0].shape
yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
xc, yc = xx - pw / 2, yy - ph / 2
for i in range(1, n):
    fl = cv2.calcOpticalFlowFarneback(frames[i - 1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = fl[..., 0].ravel(), fl[..., 1].ravel()
    X, Y = xc.ravel(), yc.ravel()
    # similarity: u = a*x - b*y + tx, v = b*x + a*y + ty
    A = np.zeros((2 * len(u), 4), np.float32)
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = X, -Y, 1
    A[1::2, 0], A[1::2, 1], A[1::2, 3] = Y, X, 1
    rhs = np.empty(2 * len(u), np.float32); rhs[0::2], rhs[1::2] = u, v
    sol, *_ = np.linalg.lstsq(A[::7], rhs[::7], rcond=None)
    a, b, tx, ty = sol
    res = rhs - A @ sol
    motion.append(dict(pan=float(tx) / pw, tilt=float(ty) / ph, zoom=float(a), roll=float(np.degrees(b)),
                       parallax=float(np.sqrt(np.mean(res ** 2))) / pw, speed=float(np.sqrt(np.mean(u ** 2 + v ** 2))) / pw))
bounds = [0] + cuts + [n]
shots = []
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    ms = motion[a:b - 1] if b - 1 > a else motion[a:a + 1]
    arr = {key: np.array([m[key] for m in ms]) for key in ms[0]}
    sp = arr["speed"] * fps
    shots.append(dict(i=k + 1, t0=round(a / fps, 2), t1=round(b / fps, 2), dur=round((b - a) / fps, 2),
                      pan=round(float(arr["pan"].mean() * fps), 3), tilt=round(float(arr["tilt"].mean() * fps), 3),
                      zoom=round(float(arr["zoom"].mean() * fps), 4), roll=round(float(arr["roll"].mean() * fps), 2),
                      parallax=round(float(arr["parallax"].mean() * fps), 4),
                      speed=round(float(sp.mean()), 3), speed_peak=round(float(sp.max()), 3),
                      speed_ramp=round(float(sp[-max(1, len(sp) // 4):].mean() / (sp[:max(1, len(sp) // 4)].mean() + 1e-6)), 2),
                      lum=round(float(np.mean(lum[a:b])), 1), sat=round(float(np.mean(sat[a:b])), 1),
                      sharp=round(float(np.median(sharp[a:b])), 1)))
json.dump(dict(fps=fps, frames=n, cuts=[round(c / fps, 3) for c in cuts], shots=shots,
               speed=[round(m["speed"] * fps, 4) for m in motion], lum=lum), open(out_json, "w"))
durs = np.array([s["dur"] for s in shots])
print(f"{video}: {n} frames, {len(shots)} shots, mean {durs.mean():.2f}s, median {np.median(durs):.2f}s, min {durs.min():.2f}s, max {durs.max():.2f}s")
for s in shots:
    print(s)
