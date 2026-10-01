"""Shot timeline, FPV camera motion, transitions and motion blur.

A project is a list of shots cut on a musical grid. Each shot films one
photo with a virtual camera that moves from `from` to `to`. Cuts are
transitions with a length centred on a beat:

  fly    the camera turns toward `focus` (a doorway, a window), accelerates
         and the next room opens from that point, like flying through it
  whip_r / whip_l  fast yaw with the next shot continuing the same turn
  rise / drop      the same on the vertical axis
  dip    fade through black, cut  hard cut

Everything is evaluated as a pure function of time, so motion blur is real
temporal supersampling of the whole composite.
"""
import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .depth import load_depth
from .warp import Cam, Plate

CAM_KEYS = ("x", "y", "z", "yaw", "pitch", "roll", "zoom", "sx", "sy")


def clamp01(x):
    return min(1.0, max(0.0, x))


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_in(x, k=2.0):
    return clamp01(x) ** k


def ease_out(x, k=2.0):
    return 1 - (1 - clamp01(x)) ** k


def scurve(p, k=7.0):
    """Logistic 0 -> 1 on [0, 1], steepest in the middle."""
    p = clamp01(p)
    lo = 1 / (1 + math.exp(k / 2))
    hi = 1 / (1 + math.exp(-k / 2))
    return (1 / (1 + math.exp(-k * (p - 0.5))) - lo) / (hi - lo)


def hermite01(u, m0, m1):
    """0 -> 1 with end speeds m0, m1 (1 = linear). Monotone if m0^2 + m1^2 <= 9."""
    u = clamp01(u)
    u2, u3 = u * u, u * u * u
    return (-2 * u3 + 3 * u2) + (u3 - 2 * u2 + u) * m0 + (u3 - u2) * m1


def cam_from(d, base=None):
    c = Cam() if base is None else Cam(*base.astuple())
    for k, v in (d or {}).items():
        setattr(c, k, float(v))
    return c


@dataclass
class Shot:
    photo: str
    beats: float
    cam_from: Cam
    cam_to: Cam
    speed: tuple = (1.6, 1.6)
    out: dict = field(default_factory=lambda: {"type": "fly", "len": 0.5})
    label: str = ""
    plate: dict = field(default_factory=dict)
    exposure: float = 1.0
    drift: float = 1.0
    seed: int = 0
    # filled by Timeline
    t0: float = 0.0
    t1: float = 0.0
    cut_in: float = 0.0
    cut_out: float = 0.0
    zoom_fix: float = 1.0


MOVES = [
    ({"z": 0, "x": -0.03, "yaw": -1.5, "zoom": 1.03}, {"z": 0.2, "x": 0.03, "yaw": 1.5, "zoom": 1.1}),
    ({"z": 0, "x": 0.03, "yaw": 1.5, "zoom": 1.03}, {"z": 0.2, "x": -0.03, "yaw": -1.5, "zoom": 1.1}),
    ({"z": 0, "y": 0.02, "pitch": -1, "zoom": 1.03}, {"z": 0.22, "y": -0.02, "pitch": 1, "zoom": 1.1}),
]
TRANSITIONS = [{"type": "fly", "len": 0.5}, {"type": "whip_r", "len": 0.4},
               {"type": "fly", "len": 0.5}, {"type": "whip_l", "len": 0.4},
               {"type": "fly", "len": 0.5}, {"type": "rise", "len": 0.45}]


def auto_shots(photos, total_beats, intro=8, outro=10, max_shots=20):
    """A reasonable first cut: photo order, even beats, varied moves."""
    photos = photos[:max_shots]
    n = len(photos)
    if n == 1:
        beats = [total_beats]
    else:
        mid = max(0, total_beats - intro - outro)
        k = max(1, n - 2)
        beats = [intro] + [mid // k + (1 if i < mid % k else 0) for i in range(k)][:n - 2] + [outro]
        beats = beats[:n]
    shots = []
    for i, (ph, b) in enumerate(zip(photos, beats)):
        a, z = MOVES[i % len(MOVES)]
        shots.append({"photo": ph, "beats": b, "from": a, "to": z, "speed": [2.0, 2.0],
                      "out": dict(TRANSITIONS[i % len(TRANSITIONS)])})
    return shots


class Timeline:
    def __init__(self, project, root):
        self.root = root
        self.fps = project.get("fps", 30)
        self.bpm = project.get("bpm", 120)
        self.w, self.h = project.get("size", [1920, 1080])
        self.beat = 60.0 / self.bpm
        self.offset = project.get("offset", 0.0)
        self.photos_dir = os.path.join(root, project.get("photos", "assets/photos"))
        self.depth_dir = os.path.join(root, project.get("depth", "assets/depth"))
        self.plate_defaults = project.get("plate", {})
        self.shots = []
        shots_cfg = project["shots"]
        if shots_cfg == "auto":
            photos = sorted(n for n in os.listdir(self.photos_dir)
                            if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
            total = project.get("duration", 45.0) / self.beat
            shots_cfg = auto_shots(photos, int(round(total)))
        for i, s in enumerate(shots_cfg):
            a = cam_from(s.get("from"))
            b = cam_from(s.get("to"), a)
            self.shots.append(Shot(
                photo=s["photo"], beats=s["beats"], cam_from=a, cam_to=b,
                speed=tuple(s.get("speed", (1.6, 1.6))),
                out=s.get("out", {"type": "fly", "len": 0.5}),
                label=s.get("label", ""), plate=s.get("plate", {}),
                exposure=s.get("exposure", 1.0), drift=s.get("drift", 1.0),
                seed=s.get("seed", i * 7 + 3)))
        t = self.offset
        prev_len = 0.0
        for i, s in enumerate(self.shots):
            s.cut_in = t
            s.cut_out = t + s.beats * self.beat
            s.t0 = 0.0 if i == 0 else s.cut_in - prev_len / 2
            last = i == len(self.shots) - 1
            out_len = 0.0 if last else s.out.get("len", 0.5)
            s.t1 = s.cut_out + out_len / 2
            prev_len = out_len
            t = s.cut_out
        self.duration = project.get("duration", self.shots[-1].cut_out)
        self.shots[-1].t1 = self.duration
        self._plates = {}
        self._fix_zoom()

    # ---------------------------------------------------------------- plates
    def plate(self, shot):
        key = shot.photo
        if key not in self._plates:
            if len(self._plates) >= 4:
                self._plates.pop(next(iter(self._plates)))
            bgr = cv2.imread(os.path.join(self.photos_dir, shot.photo), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            d = load_depth(os.path.join(self.depth_dir, os.path.splitext(shot.photo)[0] + ".png"))
            if d.shape != rgb.shape[:2]:
                d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            kw = dict(self.plate_defaults)
            kw.update(shot.plate)
            self._plates[key] = Plate(rgb, d, **kw)
        return self._plates[key]

    # ---------------------------------------------------------------- camera
    def _drift(self, shot, t):
        """Slow organic wander, a pilot's hands are never perfectly still."""
        rng = np.random.default_rng(shot.seed)
        ph = rng.uniform(0, 2 * math.pi, 8)
        fr = rng.uniform(0.18, 0.45, 8)
        n = [math.sin(2 * math.pi * fr[i] * t + ph[i]) for i in range(8)]
        a = shot.drift
        return dict(yaw=a * (0.45 * n[0] + 0.2 * n[1]),
                    pitch=a * (0.3 * n[2] + 0.12 * n[3]),
                    roll=a * (0.5 * n[4] + 0.25 * n[5]),
                    x=a * 0.006 * n[6], y=a * 0.005 * n[7])

    def _base(self, shot, t):
        u = (t - shot.t0) / max(1e-6, shot.t1 - shot.t0)
        s = hermite01(u, *shot.speed)
        return shot.cam_from.lerp(shot.cam_to, s)

    def camera(self, shot, t, with_fix=True):
        cam = self._base(shot, t)
        # Bank into the turn: roll follows yaw rate and sideways speed.
        dt = 1.0 / 60
        c0, c1 = self._base(shot, t - dt), self._base(shot, t + dt)
        yaw_rate = (c1.yaw - c0.yaw) / (2 * dt)
        side = (c1.x - c0.x) / (2 * dt)
        cam.roll += 0.35 * yaw_rate + 9.0 * side
        for k, v in self._drift(shot, t).items():
            setattr(cam, k, getattr(cam, k) + v)
        if with_fix:
            cam.zoom *= shot.zoom_fix
        i = self.shots.index(shot)
        # outgoing transition
        if i < len(self.shots) - 1:
            L = shot.out.get("len", 0.5)
            p = (t - (shot.cut_out - L / 2)) / max(L, 1e-6)
            if p > 0:
                cam = self._trans_out(shot, cam, p)
        # incoming transition
        if i > 0:
            prev = self.shots[i - 1]
            L = prev.out.get("len", 0.5)
            q = (t - (shot.cut_in - L / 2)) / max(L, 1e-6)
            if q < 1:
                cam = self._trans_in(prev, cam, q, t)
        return cam

    def _focus_angles(self, shot):
        plate = self.plate(shot)
        fx, fy = shot.out.get("focus", (0.5, 0.5))
        a = math.atan((fx * plate.w - plate.cx) / plate.f)
        b = math.atan((fy * plate.h - plate.cy) / plate.f)
        return a, b

    def _trans_out(self, shot, cam, p):
        o = shot.out
        typ = o.get("type", "fly")
        S = scurve(p, o.get("k", 7.0))
        bump = 4 * S * (1 - S)
        if typ == "fly":
            a, b = self._focus_angles(shot)
            look = smoothstep(p) * o.get("look", 0.7)
            cam.yaw += (math.degrees(a) - cam.yaw) * look
            cam.pitch += (-math.degrees(b) - cam.pitch) * look
            dist = o.get("push", 0.22) * S
            d = np.array([math.tan(a), math.tan(b), 1.0])
            d /= np.linalg.norm(d)
            cam.x += d[0] * dist
            cam.y += d[1] * dist
            cam.z += d[2] * dist
            cam.zoom *= math.exp(math.log(o.get("zoom", 3.0)) * S)
            cam.roll += o.get("roll", 0.0) * S
        elif typ in ("whip_r", "whip_l"):
            sgn = 1 if typ == "whip_r" else -1
            cam.yaw += sgn * 2 * o.get("angle", 16) * S
            cam.roll += sgn * o.get("bank", 5) * bump
            cam.zoom *= 1 + 0.08 * bump
        elif typ in ("rise", "drop"):
            sgn = 1 if typ == "rise" else -1
            cam.pitch += sgn * 2 * o.get("angle", 8) * S
            cam.y -= sgn * o.get("lift", 0.12) * S
            cam.zoom *= 1 + 0.1 * bump
        return cam

    def _trans_in(self, prev, cam, q, t):
        o = prev.out
        typ = o.get("type", "fly")
        S = scurve(q, o.get("k", 7.0))
        bump = 4 * S * (1 - S)
        if typ in ("whip_r", "whip_l"):
            sgn = 1 if typ == "whip_r" else -1
            cam.yaw += sgn * 2 * o.get("angle", 16) * (S - 1)
            cam.roll += sgn * o.get("bank", 5) * bump
            cam.zoom *= 1 + 0.08 * bump
        elif typ in ("rise", "drop"):
            sgn = 1 if typ == "rise" else -1
            cam.pitch += sgn * 2 * o.get("angle", 8) * (S - 1)
            cam.y -= sgn * o.get("lift", 0.12) * (S - 1)
            cam.zoom *= 1 + 0.1 * bump
        elif typ == "fly":
            # The next room is born small inside the doorway and grows to
            # fill the frame, so the forward motion never reverses. It is
            # full size by 3/4 of the transition, before the reveal reaches
            # the frame corners.
            Sb = scurve(q / 0.75, o.get("k", 7.0))
            cam.zoom *= math.exp(math.log(o.get("enter", 0.45)) * (1 - Sb))
            px, py = self.portal(prev, t)
            cam.sx += (px - self.w / 2) / self.h * (1 - Sb)
            cam.sy += (py - self.h / 2) / self.h * (1 - Sb)
        return cam

    def portal(self, shot, t):
        """Screen position (timeline pixels) of the doorway we fly through."""
        px, py = self.plate(shot).project_point(self.camera(shot, t), self.w, self.h,
                                                *shot.out.get("focus", (0.5, 0.5)))
        if not (math.isfinite(px) and math.isfinite(py)):
            return self.w / 2, self.h / 2
        return min(max(px, 0.0), self.w), min(max(py, 0.0), self.h)

    def _fix_zoom(self):
        """Per-shot constant zoom so roll, yaw and drift never show borders."""
        for i, s in enumerate(self.shots):
            plate_w = plate_h = None
            need = 1.0
            ts = np.linspace(s.t0, s.t1, 24)
            for t in map(float, ts):
                # skip transition windows, the blur and the mix hide edges there
                if i > 0 and t < s.cut_in + self.shots[i - 1].out.get("len", 0.5) / 2:
                    continue
                if i < len(self.shots) - 1 and t > s.cut_out - s.out.get("len", 0.5) / 2:
                    continue
                cam = self.camera(s, t, with_fix=False)
                need = max(need, self._cover_need(s, cam))
            s.zoom_fix = need

    def _cover_need(self, shot, cam):
        """Smallest zoom multiplier that keeps the frame inside the photo."""
        plate = self.plate(shot)
        z = 1.0
        for _ in range(40):
            c = Cam(*cam.astuple())
            c.zoom *= z
            su, sv = plate._maps(c, self.w, self.h, 24, 14, iters=4)
            m = 2.0
            inside = (su.min() >= m and sv.min() >= m and
                      su.max() <= plate.w - m and sv.max() <= plate.h - m)
            if inside:
                return z
            z *= 1.01
        return z

    # ---------------------------------------------------------------- frames
    def active(self, t):
        t = min(max(t, 0.0), self.duration - 1e-6)
        return [s for s in self.shots if s.t0 <= t < s.t1] or [self.shots[-1]]

    def render_shot(self, shot, t, w, h, map_scale=0.5, coverage=False):
        cam = self.camera(shot, t)
        res = self.plate(shot).render(cam, w, h, map_scale=map_scale, coverage=coverage)
        img, cov = res if coverage else (res, None)
        if shot.exposure != 1.0:
            img = img * shot.exposure
        return (img, cov) if coverage else img

    def composite(self, t, w, h, map_scale=0.5):
        act = self.active(t)
        if len(act) == 1:
            return self.render_shot(act[0], t, w, h, map_scale)
        a, b = act[0], act[1]
        o = a.out
        L = o.get("len", 0.5)
        p = clamp01((t - (a.cut_out - L / 2)) / L)
        typ = o.get("type", "fly")
        ia = self.render_shot(a, t, w, h, map_scale)
        if typ == "fly":
            ib, cov = self.render_shot(b, t, w, h, map_scale, coverage=True)
            px, py = self.portal(a, t)
            px, py = px * w / self.w, py * h / self.h
            yy, xx = np.ogrid[0:h, 0:w]
            dist = np.sqrt(((xx - px) * (h / w) * 1.35) ** 2 + (yy - py) ** 2, dtype=np.float32)
            feather = 0.5 * h
            reach = math.hypot(max(px, w - px) * (h / w) * 1.35, max(py, h - py))
            r = -0.2 * feather + (reach + 1.2 * feather) * smoothstep((p - 0.04) / 0.76)
            radial = np.clip((r - dist) / feather, 0, 1)
            radial = radial * radial * (3 - 2 * radial)
            m = (cov * radial)[..., None]
            out = ia * (1 - m) + ib * m
            g = math.exp(-((p - 0.5) / 0.17) ** 2)
            glow = np.exp(-(dist / (0.55 * h)) ** 2)[..., None]
            return out * (1 + o.get("flash", 0.28) * g * (0.45 + glow))
        ib = self.render_shot(b, t, w, h, map_scale)
        if typ == "dip":
            k = 1 - smoothstep(p * 2) if p < 0.5 else smoothstep(p * 2 - 1)
            return (ia if p < 0.5 else ib) * k
        if typ == "cut":
            return ia if p < 0.5 else ib
        m = smoothstep((p - 0.38) / 0.24)
        return ia * (1 - m) + ib * m

    def motion_extent(self, t, shutter):
        """Largest on-screen travel (px) of a probe grid during the shutter."""
        best = 0.0
        for s in self.active(t):
            plate = self.plate(s)
            c0 = self.camera(s, t - shutter / 2)
            c1 = self.camera(s, t + shutter / 2)
            for fx in (0.08, 0.5, 0.92):
                for fy in (0.1, 0.5, 0.9):
                    x0, y0 = plate.project_point(c0, self.w, self.h, fx, fy)
                    x1, y1 = plate.project_point(c1, self.w, self.h, fx, fy)
                    if all(map(math.isfinite, (x0, y0, x1, y1))):
                        if -self.w < x0 < 2 * self.w and -self.h < y0 < 2 * self.h:
                            best = max(best, math.hypot(x1 - x0, y1 - y0))
        return best

    def frame(self, i, shutter_deg=180.0, max_samples=64, w=None, h=None):
        """Linear RGB frame i with motion blur (temporal supersampling)."""
        w = w or self.w
        h = h or self.h
        t = i / self.fps
        shutter = shutter_deg / 360.0 / self.fps
        ext = self.motion_extent(t, shutter) * (w / self.w)
        n = int(min(max_samples, max(1, math.ceil(ext / 2.0))))
        if n == 1:
            return self.composite(t, w, h)
        # Heavily blurred frames carry no fine detail: render them smaller.
        div = 1 if n <= 6 else (2 if n <= 20 else 4)
        rw, rh = w // div, h // div
        acc = np.zeros((rh, rw, 3), np.float32)
        for k in range(n):
            tk = t + shutter * ((k + 0.5) / n - 0.5)
            acc += self.composite(tk, rw, rh, map_scale=1.0 if div > 1 else 0.5)
        acc /= n
        gap = ext / n / div
        if gap > 0.8:
            acc = cv2.GaussianBlur(acc, (0, 0), 0.5 * gap)
        if div > 1:
            acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_CUBIC)
        return acc


def load_project(path):
    with open(path) as f:
        return json.load(f)
