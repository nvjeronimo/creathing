"""Film finishing: highlight roll-off, soft bloom, split tone, vignette, grain."""
import cv2
import numpy as np


class Finisher:
    def __init__(self, w, h, grain=0.009, vignette=0.16, bloom=0.07,
                 warmth=0.025, contrast=0.06, sharpen=0.25, seed=7, knee=0.72,
                 exposure=1.0):
        self.w, self.h = w, h
        self.knee, self.exposure = knee, exposure
        self.grain, self.bloom = grain, bloom
        self.warmth, self.contrast, self.sharpen = warmth, contrast, sharpen
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        self.vig = (1 - vignette * np.clip(r, 0, 1) ** 2.2)[..., None].astype(np.float32)
        rng = np.random.default_rng(seed)
        # A few grain plates, re-used with random offsets. Slightly soft so it
        # reads as film, not as sensor noise.
        self.grains = []
        for _ in range(6):
            g = rng.standard_normal((h + 64, w + 64)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.65)
            self.grains.append(g / g.std())
        self.rng = rng

    @staticmethod
    def _shoulder(x, knee=0.72):
        """Linear below the knee, smooth roll-off to 1.0 above it."""
        over = np.maximum(x - knee, 0)
        return np.minimum(x, knee) + over / (1 + over / (1 - knee))

    def apply(self, lin, i, sharp=True):
        x = lin
        if self.bloom > 0:
            hi = np.maximum(x - 0.6, 0)
            small = cv2.resize(hi, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 9)
            x = x + self.bloom * cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if self.exposure != 1.0:
            x = x * self.exposure
        x = self._shoulder(x, self.knee) * self.vig
        y = np.clip(x, 0, 1) ** (1 / 2.2)
        if self.contrast:
            y = y + self.contrast * (y - 0.5) * (1 - np.abs(2 * y - 1))
        if self.warmth:
            lum = y.mean(axis=2, keepdims=True)
            t = np.clip((lum - 0.35) / 0.5, 0, 1)
            warm = np.array([1.0, 0.35, -0.6], np.float32) * self.warmth
            cool = np.array([-0.4, 0.0, 0.7], np.float32) * self.warmth * 0.6
            y = y + t * warm * lum + (1 - t) * cool * (1 - lum) * 0.5
        if sharp and self.sharpen:
            blur = cv2.GaussianBlur(y, (0, 0), 1.1)
            y = y + self.sharpen * (y - blur)
        if self.grain:
            g = self.grains[i % len(self.grains)]
            ox, oy = self.rng.integers(0, 64, 2)
            g = g[oy:oy + self.h, ox:ox + self.w, None]
            lum = y.mean(axis=2, keepdims=True)
            amp = self.grain * (0.35 + 0.65 * 4 * lum * (1 - lum))
            y = y + g * amp
        return (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
