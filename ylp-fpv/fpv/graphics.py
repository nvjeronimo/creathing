"""Motion graphics drawn over the finished frames.

Restrained on purpose: thin serif display type, wide-tracked caps, a gold
hairline, mask reveals. Two themes: "dark" (light ink over a shaded image)
and "light" (dark ink over a white veil, for bright minimal interiors).
Works in 16:9 and 9:16; in vertical the layout keeps clear of the Reels UI
(caption band at the bottom, buttons on the right). Everything is a pure
function of time.
"""
import os

import cv2
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
        self.path, self.weight, self.em_tracking = path, weight, tracking
        self.font = ImageFont.truetype(path, max(4, int(round(size))))
        if weight is not None:
            try:
                self.font.set_variation_by_axes([weight])
            except OSError:
                pass
        self.size = size
        self.tracking = tracking * size  # em -> px

    def layout(self, text):
        xs = [self.font.getlength(text[:i], features=["lnum"]) + self.tracking * i
              for i in range(len(text))]
        width = (self.font.getlength(text, features=["lnum"])
                 + self.tracking * max(0, len(text) - 1))
        return xs, width

    def ascent(self):
        return self.font.getmetrics()[0]

    def fit(self, text, max_w):
        """This face, shrunk if needed so every line of `text` fits max_w."""
        widest = max(self.layout(line)[1] for line in text.split("\n"))
        if widest <= max_w:
            return self
        return Typeface(self.path, self.size * max_w / widest * 0.98, self.weight,
                        self.em_tracking)


class Graphics:
    def __init__(self, project, root, w, h, timeline):
        g = project["graphics"]
        self.cfg = g
        self.tl = timeline
        self.w, self.h = w, h
        self.vertical = h > w
        s = min(w, h) / 1080.0
        self.s = s
        fonts = os.path.join(root, g.get("fonts", "assets/fonts"))
        serif = os.path.join(fonts, "CormorantGaramond.ttf")
        serif_i = os.path.join(fonts, "CormorantGaramond-Italic.ttf")
        sans = os.path.join(fonts, "Montserrat.ttf")
        self.theme = g.get("theme", "dark")
        light = self.theme == "light"
        self.ink = hex_rgb(g.get("ink", "#1E1A15" if light else "#F7F3EC"))
        self.gold = hex_rgb(g.get("accent", "#A07C45" if light else "#C9A66B"))
        self.veil = np.array(hex_rgb(g.get("veil", "#FFFFFF" if light else "#000000")),
                             np.float32)
        self.glow = 255.0 if light else 0.0
        sz = g.get("sizes", {})
        self.f_kicker = Typeface(sans, sz.get("kicker", 19) * s, 500, 0.42)
        self.f_title = Typeface(serif, sz.get("title", 118) * s, 300, 0.01)
        self.f_sub = Typeface(serif_i, sz.get("sub", 40) * s, 400, 0.01)
        self.f_index = Typeface(sans, 17 * s, 600, 0.3)
        self.f_room = Typeface(serif_i, sz.get("room", 62) * s, 400, 0.0)
        self.f_room_caps = Typeface(sans, sz.get("room_caps", 15) * s, 500, 0.42)
        self.f_stat = Typeface(serif, 84 * s, 300, 0.0)
        self.f_stat_cap = Typeface(sans, 15 * s, 500, 0.38)
        self.f_price = Typeface(serif, 64 * s, 400, 0.02)
        self.f_small = Typeface(sans, sz.get("small", 18) * s, 500, 0.4)
        self.margin = int(g.get("margin", 72 if self.vertical else 96) * s)
        # usable width: in vertical keep clear of the Reels buttons on the right
        right_gap = 130 * s if self.vertical else 0
        self.text_w = self.w - 2 * self.margin - right_gap
        self.labels = self._label_schedule()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        lab_y = self._label_y()
        d = np.sqrt(((xx - 0) / (0.75 * w if self.vertical else 0.55 * w)) ** 2
                    + ((yy - lab_y) / (0.2 * h if self.vertical else 0.42 * h)) ** 2)
        self.shade_label = np.clip(1 - d, 0, 1) ** 1.4
        cy_title = self._title_y()
        d = np.sqrt(((xx - w / 2) / (0.75 * w if self.vertical else 0.62 * w)) ** 2
                    + ((yy - cy_title) / (0.24 * h if self.vertical else 0.5 * h)) ** 2)
        self.shade_title = np.clip(1.15 - d, 0, 1) ** 1.2
        d = np.sqrt(((xx - w / 2) / (0.8 * w)) ** 2 + ((yy - h * 0.45) / (0.45 * h)) ** 2)
        self.shade_center = np.clip(1.2 - d, 0, 1) ** 1.1

    # ---------------------------------------------------------------- layout
    def _title_y(self):
        return self.h * self.cfg.get("title_y", 0.21 if self.vertical else 0.5)

    def _label_y(self):
        return self.h * 0.66 if self.vertical else self.h - self.margin - 70 * self.s

    # ------------------------------------------------------------- schedule
    def _label_schedule(self):
        """(t0, t1, index, label, sub) for every labelled shot long enough to read."""
        out = []
        shots = self.tl.shots
        k = 0
        outro = self.cfg.get("outro")
        intro = self.cfg.get("intro")
        for i, sh in enumerate(shots):
            if not sh.label:
                continue
            k += 1
            in_len = shots[i - 1].out.get("len", 0.5) if i > 0 else 0.0
            out_len = sh.out.get("len", 0.5) if i < len(shots) - 1 else 0.0
            t0 = sh.cut_in + in_len / 2 + 0.05
            t1 = sh.cut_out - out_len / 2 - 0.05
            if sh.label_t:
                t0, t1 = sh.label_t
            if intro:
                t0 = max(t0, intro["t1"] + 0.1)
            if outro:
                t1 = min(t1, outro["t0"] - 0.1)
            if t1 - t0 > 0.9:
                out.append((t0, t1, k, sh.label, sh.sub))
        return out

    # --------------------------------------------------------------- drawing
    def _text(self, draw, face, text, x, y, color, alpha, per_char=None, anchor="l",
              index0=0):
        xs, width = face.layout(text)
        if anchor == "c":
            x = x - width / 2
        elif anchor == "r":
            x = x - width
        for i, ch in enumerate(text):
            a = alpha
            dy = 0.0
            if per_char is not None:
                a, dy = per_char(index0 + i)
                a *= alpha
            if a <= 0.002 or ch == " ":
                continue
            draw.text((x + xs[i], y + dy), ch, font=face.font,
                      fill=color + (int(255 * clamp01(a)),), features=["lnum"])
        return width

    def _lines(self, draw, face, text, cx, y, color, alpha, per_char=None, leading=1.02):
        """Centred multi-line text; returns the y below the last line."""
        k = 0
        for line in text.split("\n"):
            self._text(draw, face, line, cx, y, color, alpha, per_char, anchor="c", index0=k)
            k += len(line)
            y += face.size * leading
        return y

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

    def _rise(self, lt, start, step, dist):
        def per_char(i):
            p = clamp01((lt - start - i * step) / 0.7)
            return ease_out(p), (1 - ease_out(p, 4)) * dist * self.s
        return per_char

    def draw(self, rgb, t):
        layer = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        veil = np.zeros((self.h, self.w), np.float32)
        out = rgb.astype(np.float32)
        out, veil = self._outro_defocus(out, t, veil)
        veil = self._intro(layer, draw, t, veil)
        veil = self._labels(layer, draw, t, veil)
        veil = self._specs(layer, draw, t, veil)
        veil = self._outro(layer, draw, t, veil)
        self._lockups(layer, draw, t)
        self._watermark(draw, t)
        self._progress(draw, t)
        if veil.max() > 0:
            v = veil[..., None]
            out = out * (1 - v) + self.veil * v
        arr = np.asarray(layer).astype(np.float32)
        if arr[..., 3].max() > 0:
            a = arr[..., 3:4] / 255.0
            # soft glow (light theme) or shadow (dark theme) for legibility
            sh = layer.split()[3].filter(ImageFilter.GaussianBlur(6 * self.s))
            sa = np.asarray(sh).astype(np.float32)[..., None] / 255.0 * 0.5
            out = out * (1 - sa) + self.glow * sa
            out = out * (1 - a) + arr[..., :3] * a
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- pieces
    def _intro(self, layer, draw, t, veil):
        c = self.cfg.get("intro")
        if not c:
            return veil
        env = window(t, c["t0"], c["t1"], 0.6, 0.45)
        if env <= 0:
            return veil
        veil = np.maximum(veil, self.shade_title * c.get("veil", 0.42) * env)
        cx, cy = self.w / 2, self._title_y()
        lt = t - c["t0"]
        title_face = self.f_title.fit(c["title"], self.text_w)
        n_lines = c["title"].count("\n") + 1
        block = title_face.size * 1.02 * n_lines
        top = cy - block / 2
        self._text(draw, self.f_kicker, c["kicker"], cx, top - 52 * self.s, self.ink,
                   ease_out(lt / 0.7) * env, anchor="c")
        y = self._lines(draw, title_face, c["title"], cx, top, self.ink, env,
                        per_char=self._rise(lt, 0.25, 0.045, 36))
        lp = ease_out((lt - 0.7) / 0.9)
        half = 110 * self.s * lp
        yl = y + 26 * self.s
        if half > 1:
            draw.line([(cx - half, yl), (cx + half, yl)], fill=self.gold + (int(255 * env),),
                      width=max(1, int(2 * self.s)))
        if c.get("subtitle"):
            sub_face = self.f_sub.fit(c["subtitle"], self.text_w)
            self._text(draw, sub_face, c["subtitle"], cx, yl + 18 * self.s, self.ink,
                       ease_out((lt - 1.0) / 0.8) * env, anchor="c")
        return veil

    def _labels(self, layer, draw, t, veil):
        for t0, t1, k, label, sub in self.labels:
            env = window(t, t0, t1, 0.35, 0.3)
            if env <= 0:
                continue
            lt = t - t0
            veil = np.maximum(veil, self.shade_label * self.cfg.get("label_veil", 0.5) * env)
            x = self.margin
            y = self._label_y()
            idx = f"{k:02d}"
            self._text(draw, self.f_index, idx, x, y - 30 * self.s, self.gold,
                       env * ease_out(lt / 0.4))
            ln = 56 * self.s * ease_out((lt - 0.05) / 0.5)
            lx = x + self.f_index.layout(idx)[1] + 16 * self.s
            ly = y - 30 * self.s + self.f_index.ascent() * 0.62
            if ln > 1:
                draw.line([(lx, ly), (lx + ln, ly)], fill=self.gold + (int(230 * env),),
                          width=max(1, int(1.6 * self.s)))
            room = self.f_room.fit(label, self.text_w)
            self._masked(layer, room, label, x, y, self.ink, clamp01((lt - 0.08) / 0.6),
                         alpha=env)
            if sub:
                caps = self.f_room_caps.fit(sub.upper(), self.text_w)
                self._text(draw, caps, sub.upper(), x + 2 * self.s, y + room.size * 1.08,
                           self.ink, env * ease_out((lt - 0.45) / 0.6) * 0.85)
        return veil

    def _specs(self, layer, draw, t, veil):
        c = self.cfg.get("specs")
        if not c:
            return veil
        env = window(t, c["t0"], c["t1"], 0.5, 0.4)
        if env <= 0:
            return veil
        lt = t - c["t0"]
        veil = np.maximum(veil, self.shade_center * 0.5 * env)
        items = c["items"]
        n = len(items)
        widest = max(max(self.f_stat.layout(b)[1], self.f_stat_cap.layout(cap.upper())[1])
                     for b, cap in items)
        cw = min((self.w - 2 * self.margin) / n, widest + 110 * self.s)
        x0 = self.w / 2 - cw * n / 2
        cy = self.h / 2
        for i, (big, cap) in enumerate(items):
            p = clamp01((lt - 0.12 * i) / 0.7)
            a = ease_out(p) * env
            dy = (1 - ease_out(p, 4)) * 28 * self.s
            cx = x0 + cw * (i + 0.5)
            self._text(draw, self.f_stat, big, cx, cy - 70 * self.s + dy, self.ink, a, anchor="c")
            self._text(draw, self.f_stat_cap, cap.upper(), cx, cy + 28 * self.s + dy, self.ink,
                       a * 0.9, anchor="c")
            if i > 0:
                xl = x0 + cw * i
                hl = 46 * self.s * ease_out((lt - 0.3) / 0.6)
                if hl > 1:
                    draw.line([(xl, cy - 10 * self.s - hl), (xl, cy - 10 * self.s + hl)],
                              fill=self.gold + (int(200 * env),), width=max(1, int(1.4 * self.s)))
        return veil

    def _outro_env(self, t):
        c = self.cfg.get("outro")
        if not c:
            return 0.0, None
        return window(t, c["t0"], c.get("t1", self.tl.duration + 1), 0.9, 0.6), c

    def _outro_defocus(self, out, t, veil):
        env, c = self._outro_env(t)
        if env <= 0 or not c.get("defocus"):
            return out, veil
        sigma = c["defocus"] * self.s * ease_in_out(env)
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sigma)
        return out, veil

    def _outro(self, layer, draw, t, veil):
        env, c = self._outro_env(t)
        if env <= 0:
            return veil
        lt = t - c["t0"]
        veil = np.maximum(veil, (c.get("veil", 0.25) + 0.35 * self.shade_center) * env)
        cx = self.w / 2
        cy = self.h * (0.42 if self.vertical else 0.5)

        def fade(st, d=0.7):
            return ease_out((lt - st) / d) * env

        title_face = self.f_title.fit(c.get("title", ""), self.text_w)
        n_lines = c.get("title", "").count("\n") + 1
        top = cy - title_face.size * 1.02 * n_lines / 2 - 30 * self.s
        self._text(draw, self.f_kicker, c.get("kicker", ""), cx, top - 52 * self.s, self.ink,
                   fade(0.1), anchor="c")
        y = self._lines(draw, title_face, c.get("title", ""), cx, top, self.ink, env,
                        per_char=self._rise(lt, 0.3, 0.035, 30))
        lp = ease_out((lt - 0.8) / 0.8)
        half = 100 * self.s * lp
        yl = y + 24 * self.s
        if half > 1:
            draw.line([(cx - half, yl), (cx + half, yl)], fill=self.gold + (int(255 * env),),
                      width=max(1, int(2 * self.s)))
        y = yl + 22 * self.s
        if c.get("subtitle"):
            self._text(draw, self.f_sub, c["subtitle"], cx, y, self.ink, fade(0.9), anchor="c")
            y += self.f_sub.size * 1.3
        if c.get("price"):
            self._text(draw, self.f_price, c["price"], cx, y, self.ink, fade(1.0), anchor="c")
            y += self.f_price.size * 1.3
        self._text(draw, self.f_small, c.get("ref", ""), cx, y + 18 * self.s, self.gold,
                   fade(1.3), anchor="c")
        url_y = self.h * 0.70 if self.vertical else self.h - self.margin
        self._text(draw, self.f_small, c.get("url", ""), cx, url_y, self.ink, fade(1.6) * 0.85,
                   anchor="c")
        return veil

    def _lockups(self, layer, draw, t):
        """Wordmark lockup: serif name, hairline divider, two stacked caps lines.

        Used over an abstract texture at the open and on black at the close,
        the way luxury developers sign their films.
        """
        for c in self.cfg.get("lockups", []):
            env = window(t, c["t0"], c.get("t1", self.tl.duration + 1),
                         c.get("fade_in", 0.7), c.get("fade_out", 0.4))
            if env <= 0:
                continue
            lt = t - c["t0"]
            ink = hex_rgb(c["ink"]) if c.get("ink") else self.ink
            serif = os.path.join(os.path.dirname(self.f_title.path), "CormorantGaramond.ttf")
            main = Typeface(serif, c.get("size", 78) * self.s, c.get("weight", 400), 0.01)
            caps = Typeface(self.f_kicker.path, c.get("caps_size", 13) * self.s, 500, 0.34)
            mw = main.layout(c["main"])[1]
            stack = c.get("stack", [])
            sw = max((caps.layout(x)[1] for x in stack), default=0)
            gap = 24 * self.s
            total = mw + (2 * gap + sw if stack else 0)
            x0 = self.w / 2 - total / 2
            cy = self.h * c.get("y", 0.5)
            drift = (1 - ease_out(lt / 1.2, 3)) * 6 * self.s
            asc = main.ascent()
            top = cy - asc * 0.62 + drift
            self._text(draw, main, c["main"], x0, top, ink, env)
            if stack:
                xh_mid = top + asc * 0.62
                half = asc * 0.36 * ease_out((lt - 0.15) / 0.6)
                xd = x0 + mw + gap
                if half > 0.5:
                    draw.line([(xd, xh_mid - half), (xd, xh_mid + half)],
                              fill=ink + (int(200 * env),), width=max(1, int(1.3 * self.s)))
                lh = caps.size * 1.45
                ys = xh_mid - lh * len(stack) / 2 + (lh - caps.size) / 2
                for k, line in enumerate(stack):
                    self._text(draw, caps, line, xd + gap, ys + k * lh, ink,
                               env * ease_out((lt - 0.25 - 0.08 * k) / 0.6))
            y = cy + asc * 0.6
            for k, extra in enumerate(c.get("lines", [])):
                face = Typeface(self.f_kicker.path, extra.get("size", 15) * self.s, 500,
                                extra.get("tracking", 0.3))
                color = self.gold if extra.get("color") == "accent" else ink
                y += extra.get("dy", 40) * self.s
                self._text(draw, face, extra["text"], self.w / 2, y, color,
                           env * ease_out((lt - 0.6 - 0.15 * k) / 0.7) * extra.get("alpha", 0.9),
                           anchor="c")

    def _watermark(self, draw, t):
        """Small brand mark fixed at the top for the whole film (Ref 2 style)."""
        c = self.cfg.get("watermark")
        if not c:
            return
        env = window(t, c.get("t0", 0.0), c.get("t1", self.tl.duration + 1),
                     c.get("fade_in", 0.3), c.get("fade_out", 0.3)) * c.get("alpha", 0.9)
        if env <= 0:
            return
        ink = hex_rgb(c["ink"]) if c.get("ink") else self.ink
        y = self.h * c.get("y", 0.045)
        for line in c["lines"]:
            face = Typeface(self.f_kicker.path, line.get("size", 18) * self.s,
                            line.get("weight", 600), line.get("tracking", 0.28))
            self._text(draw, face, line["text"], self.w / 2, y, ink, env * line.get("alpha", 1.0),
                       anchor="c")
            y += face.size * line.get("leading", 1.55)

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
        draw.line([(x0, y), (x1, y)], fill=self.ink + (int(70 * env),), width=wl)
        p = clamp01((t - t0) / (t1 - t0))
        draw.line([(x0, y), (x0 + (x1 - x0) * p, y)], fill=self.gold + (int(230 * env),), width=wl)
