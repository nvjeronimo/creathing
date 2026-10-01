"""Motion graphics drawn over the finished frames.

Restrained on purpose: thin serif display type, wide-tracked caps, a gold
hairline, mask reveals and a progress hairline. Everything is a pure
function of time.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def clamp01(x):
    return min(1.0, max(0.0, x))


def ease_out(x, k=3):
    return 1 - (1 - clamp01(x)) ** k


def ease_in_out(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def window(t, t0, t1, fade_in=0.5, fade_out=0.35):
    """0..1 envelope: rises after t0, falls before t1."""
    if t <= t0 or t >= t1:
        return 0.0
    return min(ease_out((t - t0) / fade_in), ease_in_out((t1 - t) / fade_out))


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Typeface:
    def __init__(self, path, size, weight=None, tracking=0.0):
        self.font = ImageFont.truetype(path, int(round(size)))
        if weight is not None:
            try:
                self.font.set_variation_by_axes([weight])
            except OSError:
                pass
        self.size = size
        self.tracking = tracking * size  # em -> px

    def layout(self, text):
        xs = []
        for i in range(len(text)):
            xs.append(self.font.getlength(text[:i], features=["lnum"]) + self.tracking * i)
        width = (self.font.getlength(text, features=["lnum"])
                 + self.tracking * max(0, len(text) - 1))
        return xs, width

    def ascent(self):
        return self.font.getmetrics()[0]


class Graphics:
    def __init__(self, project, root, w, h, timeline):
        g = project["graphics"]
        self.cfg = g
        self.tl = timeline
        self.w, self.h = w, h
        s = h / 1080.0
        self.s = s
        fonts = os.path.join(root, g.get("fonts", "assets/fonts"))
        serif = os.path.join(fonts, "CormorantGaramond.ttf")
        serif_i = os.path.join(fonts, "CormorantGaramond-Italic.ttf")
        sans = os.path.join(fonts, "Montserrat.ttf")
        self.white = hex_rgb(g.get("white", "#F7F3EC"))
        self.gold = hex_rgb(g.get("accent", "#C9A66B"))
        self.f_kicker = Typeface(sans, 19 * s, 500, 0.42)
        self.f_title = Typeface(serif, 118 * s, 300, 0.01)
        self.f_sub = Typeface(serif_i, 40 * s, 400, 0.01)
        self.f_index = Typeface(sans, 17 * s, 600, 0.3)
        self.f_room = Typeface(serif_i, 62 * s, 400, 0.0)
        self.f_room_caps = Typeface(sans, 15 * s, 500, 0.45)
        self.f_stat = Typeface(serif, 84 * s, 300, 0.0)
        self.f_stat_cap = Typeface(sans, 15 * s, 500, 0.38)
        self.f_price = Typeface(serif, 64 * s, 400, 0.02)
        self.f_small = Typeface(sans, 18 * s, 500, 0.4)
        self.margin = int(96 * s)
        self.labels = self._label_schedule()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Soft shade in the lower-left for room labels.
        d = np.sqrt(((xx - 0) / (0.55 * w)) ** 2 + ((yy - h) / (0.42 * h)) ** 2)
        self.shade_label = np.clip(1 - d, 0, 1) ** 1.6
        d = np.sqrt(((xx - w / 2) / (0.62 * w)) ** 2 + ((yy - h / 2) / (0.5 * h)) ** 2)
        self.shade_center = np.clip(1.15 - d, 0, 1) ** 1.2

    # ------------------------------------------------------------- schedule
    def _label_schedule(self):
        """(t0, t1, index, label) for every labelled shot long enough to read."""
        out = []
        shots = self.tl.shots
        k = 0
        for i, sh in enumerate(shots):
            if not sh.label:
                continue
            k += 1
            in_len = shots[i - 1].out.get("len", 0.5) if i > 0 else 0.0
            out_len = sh.out.get("len", 0.5) if i < len(shots) - 1 else 0.0
            t0 = sh.cut_in + in_len / 2 + 0.05
            t1 = sh.cut_out - out_len / 2 - 0.05
            outro = self.cfg.get("outro")
            if outro:
                t1 = min(t1, outro["t0"] - 0.1)
            if t1 - t0 > 0.9:
                out.append((t0, t1, k, sh.label))
        return out

    # --------------------------------------------------------------- drawing
    def _text(self, draw, face, text, x, y, color, alpha, per_char=None, anchor="l"):
        xs, width = face.layout(text)
        if anchor == "c":
            x = x - width / 2
        elif anchor == "r":
            x = x - width
        for i, ch in enumerate(text):
            a = alpha
            dy = 0.0
            if per_char is not None:
                a, dy = per_char(i, len(text))
                a *= alpha
            if a <= 0.002 or ch == " ":
                continue
            draw.text((x + xs[i], y + dy), ch, font=face.font,
                      fill=color + (int(255 * clamp01(a)),), features=["lnum"])
        return width

    def _masked(self, layer, face, text, x, y, color, progress, alpha=1.0, anchor="l"):
        """Text that slides up into view from behind a line mask."""
        asc, desc = face.font.getmetrics()
        hgt = asc + desc
        xs, width = face.layout(text)
        if anchor == "c":
            x -= width / 2
        tile = Image.new("RGBA", (int(width + face.size), hgt + 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        self._text(d, face, text, 0, 0, color, alpha)
        off = int(round((1 - ease_out(progress, 4)) * hgt))
        if off >= hgt:
            return
        crop = tile.crop((0, 0, tile.width, hgt - off))
        layer.alpha_composite(crop, (int(x), int(y + off)))

    def draw(self, rgb, t):
        layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        shade = np.zeros((self.h, self.w), np.float32)
        shade = self._intro(layer, draw, t, shade)
        shade = self._labels(layer, draw, t, shade)
        shade = self._specs(layer, draw, t, shade)
        shade = self._outro(layer, draw, t, shade)
        self._progress(draw, t)
        out = rgb.astype(np.float32)
        if shade.max() > 0:
            out *= (1 - shade)[..., None]
        arr = np.asarray(layer).astype(np.float32)
        if arr[..., 3].max() > 0:
            a = arr[..., 3:4] / 255.0
            # soft shadow for legibility on bright interiors
            sh = layer.split()[3].filter(ImageFilter.GaussianBlur(5 * self.s))
            sa = np.asarray(sh).astype(np.float32)[..., None] / 255.0 * 0.45
            out = out * (1 - sa)
            out = out * (1 - a) + arr[..., :3] * a
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- pieces
    def _intro(self, layer, draw, t, shade):
        c = self.cfg.get("intro")
        if not c:
            return shade
        t0, t1 = c["t0"], c["t1"]
        env = window(t, t0, t1, 0.6, 0.45)
        if env <= 0:
            return shade
        shade = np.maximum(shade, self.shade_center * 0.42 * env)
        cx, cy = self.w / 2, self.h / 2
        lt = t - t0
        # kicker
        k_al = ease_out(lt / 0.7) * env
        self._text(draw, self.f_kicker, c["kicker"], cx, cy - 130 * self.s, self.white, k_al,
                   anchor="c")
        # title, letters rise one by one
        title = c["title"]

        def per_char(i, n):
            st = 0.25 + i * 0.045
            p = clamp01((lt - st) / 0.7)
            return ease_out(p), (1 - ease_out(p, 4)) * 36 * self.s
        self._text(draw, self.f_title, title, cx, cy - 92 * self.s, self.white, env,
                   per_char=per_char, anchor="c")
        # gold hairline
        lp = ease_out((lt - 0.7) / 0.9)
        half = 120 * self.s * lp
        yl = cy + 62 * self.s
        if half > 1:
            draw.line([(cx - half, yl), (cx + half, yl)], fill=self.gold + (int(255 * env),),
                      width=max(1, int(2 * self.s)))
        sub_al = ease_out((lt - 1.0) / 0.8) * env
        self._text(draw, self.f_sub, c.get("subtitle", ""), cx, cy + 82 * self.s, self.white,
                   sub_al, anchor="c")
        return shade

    def _labels(self, layer, draw, t, shade):
        for t0, t1, k, label in self.labels:
            env = window(t, t0, t1, 0.35, 0.3)
            if env <= 0:
                continue
            lt = t - t0
            shade = np.maximum(shade, self.shade_label * 0.5 * env)
            x = self.margin
            y = self.h - self.margin - 70 * self.s
            idx = f"{k:02d}"
            self._text(draw, self.f_index, idx, x, y - 30 * self.s, self.gold,
                       env * ease_out(lt / 0.4))
            ln = 56 * self.s * ease_out((lt - 0.05) / 0.5)
            lx = x + self.f_index.layout(idx)[1] + 16 * self.s
            ly = y - 30 * self.s + self.f_index.ascent() * 0.62
            if ln > 1:
                draw.line([(lx, ly), (lx + ln, ly)], fill=self.gold + (int(220 * env),),
                          width=max(1, int(1.6 * self.s)))
            self._masked(layer, self.f_room, label, x, y, self.white,
                         clamp01((lt - 0.08) / 0.6), alpha=env)
        return shade

    def _specs(self, layer, draw, t, shade):
        c = self.cfg.get("specs")
        if not c:
            return shade
        t0, t1 = c["t0"], c["t1"]
        env = window(t, t0, t1, 0.5, 0.4)
        if env <= 0:
            return shade
        lt = t - t0
        shade = np.maximum(shade, self.shade_center * 0.5 * env)
        items = c["items"]
        n = len(items)
        widest = max(max(self.f_stat.layout(b)[1], self.f_stat_cap.layout(cap.upper())[1])
                     for b, cap in items)
        cw = min((self.w - 2 * self.margin) / n, widest + 110 * self.s)
        span = cw * n
        x0 = self.w / 2 - span / 2
        cy = self.h / 2
        for i, (big, cap) in enumerate(items):
            p = clamp01((lt - 0.12 * i) / 0.7)
            a = ease_out(p) * env
            dy = (1 - ease_out(p, 4)) * 28 * self.s
            cx = x0 + cw * (i + 0.5)
            self._text(draw, self.f_stat, big, cx, cy - 70 * self.s + dy, self.white, a,
                       anchor="c")
            self._text(draw, self.f_stat_cap, cap.upper(), cx, cy + 28 * self.s + dy, self.white,
                       a * 0.9, anchor="c")
            if i > 0:
                xl = x0 + cw * i
                hl = 46 * self.s * ease_out((lt - 0.3) / 0.6)
                if hl > 1:
                    draw.line([(xl, cy - 10 * self.s - hl), (xl, cy - 10 * self.s + hl)],
                              fill=self.gold + (int(200 * env),), width=max(1, int(1.4 * self.s)))
        return shade

    def _outro(self, layer, draw, t, shade):
        c = self.cfg.get("outro")
        if not c:
            return shade
        t0 = c["t0"]
        t1 = c.get("t1", self.tl.duration + 1)
        env = window(t, t0, t1, 0.8, 0.6)
        if env <= 0:
            return shade
        lt = t - t0
        shade = np.maximum(shade, (0.25 + 0.4 * self.shade_center) * env)
        cx, cy = self.w / 2, self.h / 2

        def fade(st, d=0.7):
            return ease_out((lt - st) / d) * env

        self._text(draw, self.f_kicker, c.get("kicker", ""), cx, cy - 170 * self.s,
                   self.white, fade(0.1), anchor="c")

        def per_char(i, n):
            st = 0.3 + i * 0.035
            p = clamp01((lt - st) / 0.7)
            return ease_out(p), (1 - ease_out(p, 4)) * 30 * self.s
        self._text(draw, self.f_title, c.get("title", ""), cx, cy - 135 * self.s, self.white, env,
                   per_char=per_char, anchor="c")
        lp = ease_out((lt - 0.8) / 0.8)
        half = 110 * self.s * lp
        yl = cy + 22 * self.s
        if half > 1:
            draw.line([(cx - half, yl), (cx + half, yl)], fill=self.gold + (int(255 * env),),
                      width=max(1, int(2 * self.s)))
        if c.get("price"):
            self._text(draw, self.f_price, c["price"], cx, cy + 40 * self.s, self.white,
                       fade(1.0), anchor="c")
        self._text(draw, self.f_small, c.get("ref", ""), cx, cy + 140 * self.s, self.gold,
                   fade(1.3), anchor="c")
        self._text(draw, self.f_small, c.get("url", ""), cx, self.h - self.margin, self.white,
                   fade(1.6) * 0.85, anchor="c")
        return shade

    def _progress(self, draw, t):
        c = self.cfg.get("progress")
        if not c:
            return
        t0, t1 = c.get("t0", 0), c.get("t1", self.tl.duration)
        env = window(t, t0, t1, 0.6, 0.5)
        if env <= 0:
            return
        y = self.h - int(44 * self.s)
        x0, x1 = self.margin, self.w - self.margin
        wl = max(1, int(1.5 * self.s))
        draw.line([(x0, y), (x1, y)], fill=self.white + (int(70 * env),), width=wl)
        p = clamp01((t - t0) / (t1 - t0))
        draw.line([(x0, y), (x0 + (x1 - x0) * p, y)], fill=self.gold + (int(230 * env),), width=wl)
