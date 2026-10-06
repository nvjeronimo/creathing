"""2.5D camera: re-projects a photo with its depth map from a moving camera.

Every output pixel shoots a ray from the virtual camera. We look for the point
where that ray meets the depth surface of the photo with a damped fixed-point
iteration, then sample the photo there. The map is solved at reduced
resolution (it is smooth) and upsampled before the colour lookup.
"""
from dataclasses import dataclass, replace
import math

import cv2
import numpy as np

# sRGB <-> linear lookups. Blending and motion blur happen in linear light.
_LIN_LUT = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)


def to_linear(rgb_u8):
    return _LIN_LUT[rgb_u8]


def to_srgb_u8(lin):
    return (np.clip(lin, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


@dataclass
class Cam:
    """Virtual camera relative to the photo's own camera.

    x, y, z: translation (z forward) in units of the nearest surface depth.
    yaw, pitch, roll: degrees. zoom: focal multiplier over a cover fit.
    sx, sy: lens shift as a fraction of output height (reframing without
    perspective change).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    zoom: float = 1.0
    sx: float = 0.0
    sy: float = 0.0

    def lerp(self, other, t):
        return Cam(*[a + (b - a) * t for a, b in zip(self.astuple(), other.astuple())])

    def astuple(self):
        return (self.x, self.y, self.z, self.yaw, self.pitch, self.roll,
                self.zoom, self.sx, self.sy)

    def offset(self, **kw):
        vals = {k: getattr(self, k) + v for k, v in kw.items() if k != "zoom_mul"}
        c = replace(self, **vals)
        if "zoom_mul" in kw:
            c.zoom *= kw["zoom_mul"]
        return c


def rot_matrix(yaw, pitch, roll):
    """World-to-camera rotation. Yaw about y, pitch about x, roll about z."""
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), math.sin(r), 0], [-math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def _ellipse(k):
    k = max(3, int(k) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def prep_photo(rgb_u8, denoise=0.0, upscale=1.0, sharpen=0.0):
    """Classical clean-up for compressed listing photos.

    Non-local means removes JPEG mosquito noise and blocking on flat walls,
    Lanczos enlarges, an unsharp mask restores crispness. No generative fill.
    """
    out = rgb_u8
    if denoise > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None, denoise, denoise, 5, 15)
    if upscale and abs(upscale - 1.0) > 1e-3:
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(round(w * upscale)), int(round(h * upscale))),
                         interpolation=cv2.INTER_LANCZOS4)
    if sharpen > 0:
        f = out.astype(np.float32)
        blur = cv2.GaussianBlur(f, (0, 0), 1.3 * max(1.0, upscale))
        out = np.clip(f + sharpen * (f - blur), 0, 255).astype(np.uint8)
    return out


def simplify_depth(disp, rgb_u8, k_frac=0.02):
    """Drop thin structures (slats, railings, leaves) from the depth map.

    They would each get their own parallax and wobble like liquid. A median
    at quarter resolution keeps big shapes, then a guided filter snaps the
    remaining edges back onto the photo.
    """
    from .depth import guided_filter
    h, w = disp.shape
    sw, sh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(disp, (sw, sh), interpolation=cv2.INTER_AREA)
    k = max(3, int(w * k_frac / 4) | 1)
    med = cv2.medianBlur((small * 255).astype(np.uint8), k).astype(np.float32) / 255
    med = cv2.resize(med, (w, h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return np.clip(guided_filter(gray, med, max(2, w // 400), 2e-3), 0, 1)


class Plate:
    """A photo plus its depth, ready to be filmed by a virtual camera.

    By default a single layer on a simplified depth map: thin structures are
    flattened so they cannot wobble, and big shapes keep their parallax.

    `layered=True` adds a back layer: the same photo with the rim of every
    foreground object painted out (classic Telea inpainting, no generative
    fill) on a smooth background depth. Where the front layer stretches to
    uncover something, those pixels are swapped for the back layer. It helps
    on large moves but can ghost on glass and frames, so it is opt-in.
    """

    def __init__(self, rgb_u8, disp, hfov=72.0, znear=1.0, zfar=14.0,
                 band=0.035, edge_thr=0.07, layered=False, simplify=0.02, dfloor=0.0):
        self.h, self.w = rgb_u8.shape[:2]
        if dfloor:
            # Nothing is farther than the room's own walls: views through the
            # glass become a plane in the wall, so mullions stay straight.
            disp = np.maximum(disp, dfloor)
        if simplify:
            disp = simplify_depth(disp, rgb_u8, simplify)
        self.disp = disp
        self.znear, self.zfar = znear, zfar
        self.lin = to_linear(rgb_u8)
        # Pre-filtered copies for frames rendered far below photo resolution.
        self.mips = [self.lin]
        for _ in range(3):
            self.mips.append(cv2.pyrDown(self.mips[-1]))
        self.f = (self.w / 2) / math.tan(math.radians(hfov) / 2)
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        w = self.w
        d_front = cv2.dilate(disp, _ellipse(w * 0.004))
        d_front = cv2.GaussianBlur(d_front, (0, 0), w * 0.0012)
        self.Z = self._to_z(d_front)
        self.layered = layered
        if not layered:
            return
        # Foreground pixels that sit within `band` of a farther surface.
        bpx = int(w * band)
        d_min = cv2.erode(disp, _ellipse(bpx))
        mask = ((disp - d_min) > edge_thr).astype(np.uint8)
        mask = cv2.dilate(mask, _ellipse(w * 0.006))
        self.back_mask = mask
        # Paint the rim out at half resolution, keep full detail elsewhere.
        small = cv2.resize(rgb_u8, (w // 2, self.h // 2), interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(mask, (w // 2, self.h // 2), interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), msmall, 6,
                             cv2.INPAINT_TELEA)
        filled = cv2.cvtColor(cv2.resize(filled, (w, self.h), interpolation=cv2.INTER_CUBIC),
                              cv2.COLOR_BGR2RGB)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)[..., None]
        back = (rgb_u8 * (1 - soft) + filled * soft).astype(np.uint8)
        self.lin_back = to_linear(back)
        d_back = np.where(mask > 0, d_min, disp)
        d_back = cv2.GaussianBlur(d_back, (0, 0), w * 0.006)
        self.Zb = self._to_z(d_back)

    def _to_z(self, d):
        inv = np.clip(d, 0, 1) * (1.0 / self.znear - 1.0 / self.zfar) + 1.0 / self.zfar
        return (1.0 / inv).astype(np.float32)

    def _maps(self, cam, ow, oh, mw, mh, iters=7, Z=None):
        Z = self.Z if Z is None else Z
        cam = Cam(*[float(v) for v in cam.astuple()])
        cover = max(ow / self.w, oh / self.h)
        fo = self.f * cover * cam.zoom * (mw / ow)
        u = (np.arange(mw, dtype=np.float32) + 0.5) - mw / 2 - cam.sx * mh
        v = (np.arange(mh, dtype=np.float32) + 0.5) - mh / 2 - cam.sy * mh
        dx = (u / fo)[None, :].repeat(mh, 0)
        dy = (v / fo)[:, None].repeat(mw, 1)
        R = rot_matrix(cam.yaw, cam.pitch, cam.roll).T.astype(np.float32)
        wx = R[0, 0] * dx + R[0, 1] * dy + R[0, 2]
        wy = R[1, 0] * dx + R[1, 1] * dy + R[1, 2]
        wz = R[2, 0] * dx + R[2, 1] * dy + R[2, 2]
        wz = np.maximum(wz, 1e-3)
        cx, cy, cz = cam.x, cam.y, cam.z
        f, pcx, pcy = self.f, self.cx, self.cy
        # Start from the pure rotation solution (camera at origin).
        su = f * wx / wz + pcx
        sv = f * wy / wz + pcy
        zk = cv2.remap(Z, su, sv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if abs(cx) + abs(cy) + abs(cz) > 1e-6:
            for _ in range(iters):
                s = np.maximum(zk - cz, 0.05) / wz
                px = cx + s * wx
                py = cy + s * wy
                pz = np.maximum(zk, 1e-3)
                su = f * px / pz + pcx
                sv = f * py / pz + pcy
                zn = cv2.remap(Z, su, sv, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
                zk = zk + 0.65 * (zn - zk)
            s = np.maximum(zk - cz, 0.05) / wz
            pz = np.maximum(zk, 1e-3)
            su = f * (cx + s * wx) / pz + pcx
            sv = f * (cy + s * wy) / pz + pcy
        return su.astype(np.float32), sv.astype(np.float32)

    @staticmethod
    def _jacobian(su, sv):
        dux = np.gradient(su, axis=1)
        duy = np.gradient(su, axis=0)
        dvx = np.gradient(sv, axis=1)
        dvy = np.gradient(sv, axis=0)
        return np.abs(dux * dvy - duy * dvx)

    @staticmethod
    def _up(m, ow, oh):
        if m.shape[1] == ow and m.shape[0] == oh:
            return m
        return cv2.resize(m, (ow, oh), interpolation=cv2.INTER_LINEAR)

    def coverage(self, su, sv, ow, oh, feather=0.12):
        """1 inside the photo, fading out over `feather` beyond its border.

        Beyond the border the image is mirrored, which reads as a plausible
        continuation once blurred, so the edge of the photo never shows as a
        straight line and full coverage is exactly 1 everywhere.
        """
        f = feather * self.w
        d = np.minimum(np.minimum(su, self.w - su), np.minimum(sv, self.h - sv))
        c = np.clip(1 + d / f, 0, 1)
        c = c * c * (3 - 2 * c)
        return self._up(c.astype(np.float32), ow, oh)

    def render(self, cam, ow, oh, map_scale=0.5, interp=cv2.INTER_CUBIC,
               debug=False, coverage=False):
        """Linear RGB float32 image of size (oh, ow)."""
        mw, mh = max(8, int(ow * map_scale)), max(8, int(oh * map_scale))
        su, sv = self._maps(cam, ow, oh, mw, mh)
        moving = abs(cam.x) + abs(cam.y) + abs(cam.z) > 1e-6
        scale = max(ow / self.w, oh / self.h) * cam.zoom  # output px per photo px
        level = int(min(len(self.mips) - 1, max(0, math.floor(math.log2(1 / max(scale, 1e-3)) + 0.25))))
        k = 0.5 ** level
        front = cv2.remap(self.mips[level], self._up(su, ow, oh) * k - 0.5,
                          self._up(sv, ow, oh) * k - 0.5,
                          interp, borderMode=cv2.BORDER_REFLECT_101)
        if coverage:
            return front, self.coverage(su, sv, ow, oh)
        if not (self.layered and moving):
            return front
        bu, bv = self._maps(cam, ow, oh, mw, mh, Z=self.Zb)
        # Where the front layer is stretched far more than the smooth back
        # layer, it is smearing an edge across uncovered space.
        ratio = self._jacobian(su, sv) / (self._jacobian(bu, bv) + 1e-9)
        alpha = np.clip((ratio - 0.22) / (0.55 - 0.22), 0, 1)
        alpha = cv2.erode(alpha, _ellipse(3))
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
        if alpha.min() > 0.999:
            return front
        back = cv2.remap(self.lin_back, self._up(bu, ow, oh) - 0.5, self._up(bv, ow, oh) - 0.5,
                         interp, borderMode=cv2.BORDER_REFLECT_101)
        a = self._up(alpha, ow, oh)[..., None]
        out = front * a + back * (1 - a)
        if debug:
            return out, a[..., 0]
        return out

    def project_point(self, cam, ow, oh, nx, ny):
        """Where a photo point (normalised 0..1) lands in the output frame."""
        u = nx * self.w
        v = ny * self.h
        uu = np.array([[u]], np.float32)
        vv = np.array([[v]], np.float32)
        z = float(cv2.remap(self.Z, uu, vv, cv2.INTER_LINEAR)[0, 0])
        p = np.array([(u - self.cx) * z / self.f, (v - self.cy) * z / self.f, z])
        R = rot_matrix(cam.yaw, cam.pitch, cam.roll)
        q = R @ (p - np.array([cam.x, cam.y, cam.z]))
        cover = max(ow / self.w, oh / self.h)
        fo = self.f * cover * cam.zoom
        return (fo * q[0] / q[2] + ow / 2 + cam.sx * oh,
                fo * q[1] / q[2] + oh / 2 + cam.sy * oh)
