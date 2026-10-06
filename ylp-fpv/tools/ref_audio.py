"""Tempo, beat grid, cut alignment and sound-effect candidates of a video's audio."""
import json, subprocess, sys
import numpy as np, cv2
from scipy.signal import stft, butter, sosfilt, find_peaks

video, shots_json, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
SR = 22050
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", video, "-ac", "1", "-ar", str(SR),
                      "-f", "f32le", "-"], capture_output=True).stdout
x = np.frombuffer(raw, np.float32)
dur = len(x) / SR
info = json.load(open(shots_json))
cuts = info["cuts"]

hop = 256
f, t, Z = stft(x, SR, nperseg=2048, noverlap=2048 - hop)
mag = np.abs(Z)
logm = np.log1p(100 * mag)
flux = np.maximum(np.diff(logm, axis=1), 0).sum(axis=0)
flux = np.r_[0, flux]
flux = (flux - flux.mean()) / (flux.std() + 1e-9)
fr = SR / hop
# tempo by autocorrelation of the onset envelope
ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
lags = np.arange(len(ac)) / fr
bpms = 60 / np.maximum(lags, 1e-9)
mask = (bpms > 70) & (bpms < 180)
cand = np.argsort(ac[mask])[::-1]
best = []
for c in cand[:40]:
    b = bpms[mask][c]
    if all(abs(b - o) > 2 for o in best):
        best.append(b)
    if len(best) == 4:
        break
# refine around the strongest with phase search
def score(bpm):
    period = 60 / bpm * fr
    best_s, best_ph = -1e9, 0
    for ph in np.linspace(0, period, 48, endpoint=False):
        idx = np.arange(ph, len(flux) - 1, period).astype(int)
        s = flux[idx].mean()
        if s > best_s:
            best_s, best_ph = s, ph
    return best_s, best_ph / fr
grid = np.arange(best[0] - 3, best[0] + 3, 0.1)
scores = [(score(b)[0], b) for b in grid]
bpm = max(scores)[1]
_, phase = score(bpm)
beat = 60 / bpm
# cut alignment to the beat grid
def nearest(tc, div):
    step = beat / div
    k = np.round((tc - phase) / step)
    return abs(tc - (phase + k * step))
al = []
for c in cuts:
    al.append((c, round(nearest(c, 1) * 1000), round(nearest(c, 2) * 1000), round(nearest(c, 4) * 1000)))
on_beat = np.mean([a[1] < 60 for a in al]) if al else 0
on_half = np.mean([a[2] < 60 for a in al]) if al else 0
on_16th = np.mean([a[3] < 45 for a in al]) if al else 0
print(f"{video}: {dur:.1f}s, tempo candidates {[round(b, 1) for b in best]}, refined {bpm:.1f} BPM, beat {beat:.3f}s, phase {phase:.3f}s")
print(f"cuts on beat (<60ms): {on_beat:.0%}, on half-beat: {on_half:.0%}, on 16th (<45ms): {on_16th:.0%}")
# sound-effect candidates: band energies over time
def band(lo, hi):
    sel = (f >= lo) & (f < hi)
    e = (mag[sel] ** 2).sum(axis=0)
    return 10 * np.log10(e + 1e-12)
hi_noise = band(3000, 10000)
low = band(25, 90)
mid = band(300, 2000)
hi_s = cv2.GaussianBlur(hi_noise.reshape(1, -1).astype(np.float32), (0, 0), fr * 0.12).ravel()
lo_s = cv2.GaussianBlur(low.reshape(1, -1).astype(np.float32), (0, 0), fr * 0.03).ravel()
# whoosh/swish: slow swell of high band (peaks of smoothed hi band above its rolling median)
med_hi = np.array([np.median(hi_s[max(0, i - int(fr * 2)):i + int(fr * 2)]) for i in range(0, len(hi_s))])
pk, prop = find_peaks(hi_s - med_hi, height=4, distance=int(fr * 0.4), prominence=3)
whoosh = [(round(t[p], 2), round(float((hi_s - med_hi)[p]), 1)) for p in pk]
# impacts: low-band transients well above the local level
med_lo = np.array([np.median(lo_s[max(0, i - int(fr)):i + int(fr)]) for i in range(len(lo_s))])
pk2, _ = find_peaks(lo_s - med_lo, height=8, distance=int(fr * 0.3))
impacts = [(round(t[p], 2), round(float((lo_s - med_lo)[p]), 1)) for p in pk2]
def near_cut(tt, tol=0.35):
    return min((abs(tt - c) for c in cuts), default=9) < tol
print("high-band swells (whoosh/swish candidates):", [(a, b, "CUT" if near_cut(a) else "") for a, b in whoosh])
print("low-band hits (impact/boom candidates):", [(a, b, "CUT" if near_cut(a) else "") for a, b in impacts])
# spectral centroid trend (risers)
cent = (f[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-9)
cent_s = cv2.GaussianBlur(cent.reshape(1, -1).astype(np.float32), (0, 0), fr * 0.25).ravel()
# spectrogram picture with cut lines (white) and beat ticks (cyan)
H, W = 360, 1800
fl = np.geomspace(30, 11000, H)
img = np.array([np.interp(fl, f, 20 * np.log10(mag[:, j] + 1e-9)) for j in range(0, mag.shape[1], max(1, mag.shape[1] // W))]).T[::-1]
img = np.clip((img + 90) / 75, 0, 1)
img = cv2.applyColorMap((cv2.resize(img, (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
for c in cuts:
    xx = int(c / dur * W); cv2.line(img, (xx, 0), (xx, H), (255, 255, 255), 1)
bt = phase
while bt < dur:
    xx = int(bt / dur * W); cv2.line(img, (xx, H - 12), (xx, H), (255, 255, 0), 1); bt += beat
for s in range(0, int(dur) + 1, 5):
    xx = int(s / dur * W); cv2.putText(img, str(s), (xx + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
for fr_ in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
    yy = int(H - 1 - np.interp(np.log(fr_), np.log(fl), np.arange(H)))
    cv2.putText(img, str(fr_), (2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1)
# envelope strip
env = np.abs(x); step = len(env) // W
pk_env = np.array([env[i * step:(i + 1) * step].max() for i in range(W)])
strip = np.zeros((90, W, 3), np.uint8)
for i, v in enumerate(pk_env):
    hgt = int(min(1, v) * 88); cv2.line(strip, (i, 45 - hgt // 2), (i, 45 + hgt // 2), (200, 200, 200), 1)
cv2.imwrite(out_png, np.vstack([img, strip]))
json.dump(dict(bpm=bpm, beat=beat, phase=phase, cuts_alignment=al, whoosh=whoosh, impacts=impacts),
          open(out_png.replace(".png", "_audio.json"), "w"))
