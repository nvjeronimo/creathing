"""Soundtrack: warm deep-house bed on the project tempo plus sound design
locked to every transition. Pure synthesis, no samples, nothing to license.

python -m fpv.audio config/project.json output/soundtrack.wav
"""
import argparse
import math
import os

import numpy as np
from scipy.signal import butter, fftconvolve, istft, sosfilt, stft

from .engine import Timeline, load_project

SR = 48000


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def env_adsr(n, a, d, s, r, hold):
    """Sample envelope: attack, decay to sustain, hold, release (seconds)."""
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), 0.0)
    dec = (t >= a) & (t < a + d)
    e = np.where(dec, 1 - (1 - s) * (t - a) / max(d, 1e-4), e)
    e = np.where((t >= a + d) & (t < hold), s, e)
    rel = t >= hold
    e = np.where(rel, s * np.exp(-(t - hold) / max(r / 4.6, 1e-4)), e)
    return e


class Bus:
    def __init__(self, n):
        self.x = np.zeros((2, n), np.float64)

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= self.x.shape[1]:
            return
        if sig.ndim == 1:
            # equal power pan
            a = (pan + 1) * math.pi / 4
            sig = np.vstack([sig * math.cos(a), sig * math.sin(a)]) * math.sqrt(2)
        j0 = max(0, i)
        k0 = j0 - i
        n = min(sig.shape[1] - k0, self.x.shape[1] - j0)
        if n > 0:
            self.x[:, j0:j0 + n] += gain * sig[:, k0:k0 + n]


def reverb_ir(rt60=2.4, predelay=0.022, seed=3, bright=3200):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    env = 10 ** (-3 * t / rt60)
    out = []
    for ch in range(2):
        nz = rng.standard_normal(n)
        dark = lp(nz, 1400)
        w = np.clip(t / (rt60 * 0.6), 0, 1)
        tail = lp(nz, bright) * (1 - w) + dark * w * 1.6
        ir = tail * env
        pad = np.zeros(int(predelay * SR))
        out.append(np.concatenate([pad, ir]))
    ir = np.vstack(out)
    return ir / np.sqrt((ir ** 2).sum() / 2)


def apply_reverb(x, ir, wet):
    y = np.vstack([fftconvolve(x[0], ir[0])[: x.shape[1]],
                   fftconvolve(x[1], ir[1])[: x.shape[1]]])
    return y * wet


def delay(x, time, fb=0.35, mix=0.3, n_taps=6, pingpong=True):
    d = int(time * SR)
    y = np.zeros_like(x)
    g = 1.0
    for k in range(1, n_taps + 1):
        g *= fb
        sh = d * k
        if sh >= x.shape[1]:
            break
        src = x[:, :-sh]
        if pingpong and k % 2:
            src = src[::-1]
        y[:, sh:] += g * lp(src, 5000 - 500 * k)
    return x + mix * y


# ----------------------------------------------------------------- voices
def kick():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 44 + 88 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    click = hp(np.random.default_rng(1).standard_normal(n), 2500) * np.exp(-t * 500) * 0.12
    return np.tanh(1.8 * (body + click)) * 0.85


def hat(open_=False, seed=0):
    n = int((0.32 if open_ else 0.07) * SR)
    t = np.arange(n) / SR
    nz = np.random.default_rng(100 + seed).standard_normal(n)
    x = hp(nz, 7500, 3)
    return x * np.exp(-t * (11 if open_ else 70)) * 0.3


def shaker(seed=0):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    nz = np.random.default_rng(200 + seed).standard_normal(n)
    x = bp(nz, 4500, 11000)
    e = np.minimum(t / 0.012, 1) * np.exp(-t * 45)
    return x * e * 0.16


def clap(seed=0):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    nz = np.random.default_rng(300 + seed).standard_normal(n)
    x = bp(nz, 900, 4200)
    e = np.zeros(n)
    for k, o in enumerate((0.0, 0.009, 0.018)):
        e += np.where(t >= o, np.exp(-(t - o) * (160 if k < 2 else 16)), 0)
    return x * e * 0.22


def sub(freq, dur, release=0.08):
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(4 * np.pi * freq * t)
    e = env_adsr(n, 0.006, 0.12, 0.75, release, dur)
    return np.tanh(1.3 * x) * e * 0.45


def saw(freq, n, phase=0.0):
    """Band-limited sawtooth (PolyBLEP), phase in cycles."""
    dt = freq / SR
    ph = (phase + dt * np.arange(n)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    x = ph[m] / dt
    y[m] -= x + x - x * x - 1
    m = ph > 1 - dt
    x = (ph[m] - 1) / dt
    y[m] -= x * x + x + x + 1
    return y * 0.5


def pad_note(freq, dur, seed):
    rng = np.random.default_rng(seed)
    n = int((dur + 1.6) * SR)
    left = np.zeros(n)
    right = np.zeros(n)
    for det in (-7, 0, 7):
        f = freq * 2 ** (det / 1200.0)
        v = saw(f, n, rng.uniform(0, 1))
        pan = det / 10.0
        left += v * (1 - pan) * 0.5
        right += v * (1 + pan) * 0.5
    e = env_adsr(n, 0.9, 0.6, 0.85, 1.5, dur)
    left = lp(left * e, 2100, 2)
    right = lp(right * e, 2100, 2)
    return np.vstack([left, right]) * 0.11


def pluck(freq, seed=0):
    """Soft additive pluck, a little like a muted Rhodes or kalimba."""
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k in range(1, 9):
        fk = freq * k * (1 + 0.0007 * k * k)
        a = 1.0 / k ** 1.4
        tau = 0.75 / (1 + 0.9 * (k - 1))
        x += a * np.sin(2 * np.pi * fk * t) * np.exp(-t / tau)
    x *= np.minimum(t / 0.004, 1)
    return lp(x, 6000) * 0.16


def noise_sweep(dur, fc, bw, amp, seed=0, stereo=True):
    """Filtered noise whose band centre and level follow fc(t), amp(t)."""
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    chans = []
    for ch in range(2 if stereo else 1):
        x = rng.standard_normal(n)
        f, tt, Z = stft(x, SR, nperseg=1024, noverlap=768)
        c = fc(tt)
        mask = np.exp(-0.5 * (np.log2(np.maximum(f[:, None], 1) / c[None, :]) / bw) ** 2)
        Z = Z * mask * amp(tt)[None, :]
        _, y = istft(Z, SR, nperseg=1024, noverlap=768)
        chans.append(y[:n])
    return np.vstack(chans) if stereo else chans[0]


def whoosh(pre, post, f_lo=260, f_hi=2600, f_end=700, bw=0.9, seed=0):
    """Swell that peaks at `pre` seconds, then falls away for `post`."""
    dur = pre + post

    def fc(t):
        up = f_lo * (f_hi / f_lo) ** np.clip(t / pre, 0, 1) ** 1.6
        down = f_hi * (f_end / f_hi) ** np.clip((t - pre) / post, 0, 1) ** 0.7
        return np.where(t < pre, up, down)

    def amp(t):
        a = np.where(t < pre, np.clip(t / pre, 0, 1) ** 2.4,
                     np.exp(-(t - pre) / (post / 3.2)))
        return a

    return noise_sweep(dur, fc, bw, amp, seed=seed)


def impact(seed=0):
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    f = 30 + 34 * np.exp(-t * 6)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    nz = np.random.default_rng(seed).standard_normal(n)
    crack = lp(nz, 2600) * np.exp(-t * 18) * 0.35
    x = np.tanh(1.5 * boom) * 0.8 + crack
    return np.vstack([x, x])


def riser(dur, seed=0):
    def fc(t):
        return 300 * (7000 / 300) ** np.clip(t / dur, 0, 1) ** 1.3

    def amp(t):
        return np.clip(t / dur, 0, 1) ** 2.6

    x = noise_sweep(dur + 0.05, fc, 0.7, amp, seed=seed)
    return x * 0.5


def tick(seed=0):
    """Crisp shutter tick for flash cuts."""
    n = int(0.07 * SR)
    t = np.arange(n) / SR
    nz = np.random.default_rng(400 + seed).standard_normal(n)
    click = bp(nz, 2500, 9500) * np.exp(-t * 900) * 0.7
    tone = (np.sin(2 * np.pi * 2100 * t) * np.exp(-t * 120) * 0.22
            + np.sin(2 * np.pi * 3350 * t) * np.exp(-t * 170) * 0.1)
    return (click + tone) * 0.55


def downlifter(dur, f0=1300.0, f1=190.0, seed=0):
    """Falling harmonic tone over falling filtered noise (Ref 2 opening)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    env = (1 - t / dur) ** 1.1 * np.minimum(t / 0.06, 1)
    tone = lp(tone * env, 3500) * 0.22
    nz = noise_sweep(dur, lambda tt: 3200 * (260 / 3200) ** np.clip(tt / dur, 0, 1), 0.8,
                     lambda tt: np.clip(1 - tt / dur, 0, 1) ** 1.6, seed=seed) * 0.35
    return np.vstack([tone, tone]) + nz


def shimmer(seed=0):
    """High bell cluster, the 'ting' on reveals."""
    n = int(2.6 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(500 + seed)
    out = np.zeros((2, n))
    for k, fq in enumerate((2093.0, 3136.0, 4186.0, 5274.0)):
        for ch in range(2):
            det = fq * (1 + rng.uniform(-0.002, 0.002))
            out[ch] += np.sin(2 * np.pi * det * t + rng.uniform(0, 6.28)) * np.exp(-t * (1.1 + 0.45 * k)) / (1 + 0.6 * k)
    return out * np.minimum(t / 0.004, 1) * 0.075


def reverse_swell(dur, seed=0):
    """Reversed-cymbal swell that ends exactly on its target time."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    nz = hp(np.random.default_rng(600 + seed).standard_normal((2, n)), 2600)
    return nz * (t / dur) ** 3.2 * 0.32


def air(seed=0):
    """Soft air whoosh for hard cuts into moving shots."""
    return whoosh(0.32, 0.42, f_lo=480, f_hi=3400, f_end=900, bw=0.85, seed=700 + seed) * 0.6


SFX = {"tick": lambda i: (tick(i), 0.0), "air": lambda i: (air(i), 0.32),
       "impact": lambda i: (impact(i), 0.0), "shimmer": lambda i: (shimmer(i), 0.0)}


# --------------------------------------------------------------- arrangement
CHORDS = {
    # name: (bass midi, pad voicing midi, arp tones midi)
    "Dm9": (38, [50, 53, 57, 60, 64], [69, 72, 76, 74, 77, 76, 72, 74]),
    "Bbmaj7": (34, [46, 50, 53, 57, 60], [65, 69, 72, 74, 77, 74, 72, 69]),
    "Fmaj7": (41, [53, 57, 60, 64, 67], [69, 72, 76, 79, 77, 76, 72, 76]),
    "C69": (36, [48, 55, 62, 64, 69], [67, 72, 74, 76, 79, 76, 74, 72]),
    "Fmaj9": (41, [53, 57, 64, 67, 72], [72, 76, 79, 81]),
}
PROGRESSION = ["Dm9", "Bbmaj7", "Fmaj7", "C69"]


def build(project_path, root, out_path, with_music=True, with_sfx=True):
    proj = load_project(project_path)
    tl = Timeline(proj, root)
    m = proj.get("music", {})
    bpm = tl.bpm
    beat = 60.0 / bpm
    bar = 4 * beat
    dur = tl.duration
    n = int((dur + 0.5) * SR)
    drop = m.get("drop", 8.0)  # drums come in
    outro = m.get("outro", dur - 5.0)  # drums go out, final chord
    off = tl.offset

    drums, bass, pad, keys, fx = (Bus(n) for _ in range(5))
    k_times = []
    if with_music:
        # pads: one chord per two bars from the start, final chord at outro
        t = off
        ci = 0
        while t < outro - 0.01:
            name = PROGRESSION[ci % len(PROGRESSION)]
            length = min(2 * bar, outro - t)
            for j, note in enumerate(CHORDS[name][1]):
                pad.add(pad_note(midi(note), length, seed=ci * 10 + j), t)
            t += 2 * bar
            ci += 1
        final = CHORDS["Fmaj9"]
        fin_len = dur - outro + 0.4
        for j, note in enumerate(final[1]):
            pad.add(pad_note(midi(note), fin_len, seed=900 + j), outro, gain=1.15)
        for j, note in enumerate(final[2]):
            keys.add(pluck(midi(note)), outro + j * beat / 2, gain=0.8, pan=(-0.4, 0.4)[j % 2])

        # arp: 8ths, from bar 2 (or "arp_start") to the outro, softer before the drop
        t = m.get("arp_start", off + bar)
        step = beat / 2
        i = 0
        while t < outro - 0.01:
            ci = int((t - off) // (2 * bar))
            name = PROGRESSION[ci % len(PROGRESSION)]
            arp = CHORDS[name][2]
            g = 0.55 if t < drop else 0.85
            keys.add(pluck(midi(arp[i % len(arp)])), t, gain=g, pan=0.35 * math.sin(i * 0.7))
            t += step
            i += 1

        # drums and bass from the drop
        kk = kick()
        b = 0
        t = drop
        while t < outro - 0.01:
            drums.add(kk, t, gain=0.85)
            k_times.append(t)
            drums.add(hat(seed=b), t + beat / 2, gain=0.8, pan=0.25)
            if b % 2 == 1:
                drums.add(clap(seed=b), t, gain=0.75, pan=-0.05)
            for s16 in range(4):
                sw = 0.018 if s16 % 2 else 0.0
                drums.add(shaker(seed=b * 4 + s16), t + s16 * beat / 4 + sw,
                          gain=0.6 if s16 % 2 else 0.35, pan=-0.3)
            if b % 8 == 7:
                drums.add(hat(open_=True, seed=b), t + beat / 2, gain=0.55, pan=0.3)
            ci = int((t - off) // (2 * bar))
            root_note = CHORDS[PROGRESSION[ci % len(PROGRESSION)]][0]
            # off-beat bass, deep house style
            bass.add(sub(midi(root_note), beat * 0.42), t + beat / 2)
            if b % 4 == 3:
                bass.add(sub(midi(root_note + 12), beat * 0.2), t + beat * 0.75, gain=0.5)
            t += beat
            b += 1
        bass.add(sub(midi(CHORDS["Fmaj9"][0]), 1.2, release=2.2), outro, gain=0.85)
        drums.add(kk, outro, gain=0.9)

        # build into the drop
        if m.get("riser", True):
            fx.add(riser(2 * bar - 0.05, seed=7), drop - 2 * bar, gain=0.7)

    if with_sfx:
        shots = tl.shots
        for i, sh in enumerate(shots[:-1]):
            o = sh.out
            typ = o.get("type", "fly")
            L = o.get("len", 0.5)
            c = sh.cut_out
            if typ == "fly":
                w = whoosh(0.55 + L / 2, 0.7, seed=i)
                fx.add(w, c - 0.55 - L / 2, gain=0.9 * o.get("sfx", 1.0))
            elif typ in ("whip_r", "whip_l"):
                w = whoosh(0.22 + L / 2, 0.38, f_lo=600, f_hi=4200, f_end=1400, bw=0.7, seed=i)
                d = 1 if typ == "whip_r" else -1
                ramp = np.linspace(-0.8 * d, 0.8 * d, w.shape[1])
                a = (ramp + 1) * math.pi / 4
                w = np.vstack([w[0] * np.cos(a), w[1] * np.sin(a)]) * math.sqrt(2)
                fx.add(w, c - 0.22 - L / 2, gain=0.8 * o.get("sfx", 1.0))
            elif typ in ("rise", "drop"):
                up = typ == "rise"
                w = whoosh(0.4 + L / 2, 0.6, f_lo=200 if up else 3000, f_hi=4800 if up else 400,
                           f_end=1200 if up else 160, bw=0.8, seed=i)
                fx.add(w, c - 0.4 - L / 2, gain=0.8 * o.get("sfx", 1.0))
            if o.get("hit"):
                fx.add(impact(seed=i), c, gain=0.55 * o["hit"])
            kind = o.get("sfx_type")
            if kind in SFX:
                sig, lead = SFX[kind](i)
                fx.add(sig, c - lead, gain=o.get("sfx_gain", 1.0))
        for hit_t in m.get("hits", []):
            fx.add(impact(seed=int(hit_t * 10)), hit_t, gain=0.5)
        for k, ev in enumerate(m.get("events", [])):
            typ, at, g = ev["type"], ev["t"], ev.get("gain", 1.0)
            if typ == "downlifter":
                fx.add(downlifter(ev.get("dur", 2.0), seed=k), at, gain=g)
            elif typ == "riser":
                fx.add(riser(ev.get("dur", 2.0), seed=k), at - ev.get("dur", 2.0), gain=g)
            elif typ == "reverse":
                fx.add(reverse_swell(ev.get("dur", 1.0), seed=k), at - ev.get("dur", 1.0), gain=g)
            elif typ in SFX:
                sig, lead = SFX[typ](k + 50)
                fx.add(sig, at - lead, gain=g)

    # ------------------------------------------------------------ mixdown
    t = np.arange(n) / SR
    duck = np.ones(n)
    for kt in k_times:
        i0 = int(kt * SR)
        seg = np.arange(max(0, min(n - i0, int(0.6 * SR)))) / SR
        d = 1 - 0.55 * np.exp(-seg / 0.11) * np.minimum(seg / 0.004, 1)
        duck[i0:i0 + len(seg)] = np.minimum(duck[i0:i0 + len(seg)], d)
    ir_big = reverb_ir(2.8, seed=3)
    ir_small = reverb_ir(1.1, seed=5, bright=5000)
    keys_x = delay(keys.x, beat * 0.75, fb=0.38, mix=0.32)
    music = (pad.x * duck + apply_reverb(pad.x, ir_big, 0.35) * duck
             + keys_x * duck * 0.9 + apply_reverb(keys_x, ir_big, 0.3) * duck
             + bass.x * duck * 0.95
             + drums.x + apply_reverb(drums.x, ir_small, 0.1))
    for t0m, t1m in m.get("mute", []):
        g = np.ones(n)
        i0, i1 = int(t0m * SR), int(t1m * SR)
        dn, up = int(0.04 * SR), int(0.03 * SR)
        floor = m.get("mute_floor", 0.05)
        g[i0:i1] = floor
        g[max(0, i0 - dn):i0] = np.linspace(1, floor, i0 - max(0, i0 - dn))
        g[i1:i1 + up] = np.linspace(floor, 1, len(g[i1:i1 + up]))
        music = music * g
    # no sub in the effects reverb: a reverberated boom turns into mud
    sfx = fx.x + apply_reverb(hp(fx.x, 250), ir_big, 0.25)
    mix = hp(music, 28) + hp(sfx, 40) * m.get("sfx_gain", 1.0)
    # fade in/out
    fade_in = np.clip(t / 0.25, 0, 1)
    end = dur
    fade_out = np.clip((end - t) / m.get("fade", 1.6), 0, 1) ** 1.5
    mix *= fade_in * fade_out
    return master(mix, out_path, target_lufs=m.get("lufs", -14.0))


def master(mix, out_path, target_lufs=-14.0):
    import pyloudnorm as pyln
    from scipy.io import wavfile
    # gentle glue compression on the sum
    mix = mix / np.sqrt(np.mean(mix ** 2) + 1e-12) * 10 ** (-20 / 20)
    lvl = np.sqrt(lp(np.mean(mix ** 2, axis=0), 8, 1).clip(1e-12))
    thr = 10 ** (-22 / 20)
    gain = np.where(lvl > thr, (lvl / thr) ** (1 / 2.0 - 1), 1.0)
    mix = mix * gain
    meter = pyln.Meter(SR)
    loud = meter.integrated_loudness(mix.T)
    mix = mix * 10 ** ((target_lufs - loud) / 20)
    # soft ceiling at -1 dBFS
    ceil = 10 ** (-1.0 / 20)
    mix = np.tanh(mix / ceil) * ceil
    loud2 = meter.integrated_loudness(mix.T)
    peak = 20 * np.log10(np.abs(mix).max())
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wavfile.write(out_path, SR, (mix.T * 32767).astype(np.int16))
    print(f"{out_path}: {mix.shape[1] / SR:.2f}s, {loud2:.1f} LUFS, peak {peak:.1f} dBFS")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--no-sfx", action="store_true")
    args = ap.parse_args()
    build(args.project, args.root, args.out, not args.no_music, not args.no_sfx)


if __name__ == "__main__":
    main()
