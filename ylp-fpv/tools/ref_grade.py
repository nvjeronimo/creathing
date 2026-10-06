"""Grade fingerprint: black/white points, contrast, saturation, warmth (Lab), per video."""
import sys, cv2, numpy as np
for video in sys.argv[1:]:
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    stats = []
    for fi in np.linspace(0, n - 1, 60).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (320, int(f.shape[0] * 320 / f.shape[1])))
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
        if L.mean() < 3: continue  # skip black cards
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        stats.append([np.percentile(L, 1), np.percentile(L, 99), L.mean(), L.std(), hsv[..., 1].mean() / 2.55,
                      a.mean(), b.mean(), np.percentile(L, 50)])
    s = np.array(stats).mean(axis=0)
    print(f"{video.split('/')[-1]:32s} black {s[0]:5.1f}  white {s[1]:5.1f}  mean {s[2]:5.1f}  median {s[7]:5.1f}  contrast(std) {s[3]:5.1f}  sat {s[4]:5.1f}%  a* {s[5]:+5.1f}  b*(warm+) {s[6]:+5.1f}")
