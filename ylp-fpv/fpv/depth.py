"""Monocular depth for each photo with Depth Anything V2 (ONNX, CPU).

The network predicts relative inverse depth (disparity). We run it on the
photo and on its mirror, average both, upsample to the photo size and snap
the edges to the photo with a guided filter. Output is a 16-bit PNG where
65535 is the nearest surface and 0 the farthest.
"""
import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """He et al. guided filter on single channel float32 images."""
    mean_i = _box(guide, r)
    mean_p = _box(src, r)
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    var_i = _box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


class DepthModel:
    def __init__(self, model_path, short_side=518, threads=None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or os.cpu_count()
        self.sess = ort.InferenceSession(model_path, opts,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.short = short_side

    def _infer(self, rgb):
        h, w = rgb.shape[:2]
        s = self.short / min(h, w)
        nh = max(14, int(round(h * s / 14)) * 14)
        nw = max(14, int(round(w * s / 14)) * 14)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        x = x.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: x})[0][0]
        return out

    def predict(self, rgb):
        """rgb uint8 HxWx3 -> disparity float32 HxW normalised to [0, 1]."""
        h, w = rgb.shape[:2]
        d = self._infer(rgb)
        d_flip = self._infer(np.ascontiguousarray(rgb[:, ::-1]))[:, ::-1]
        d = 0.5 * (d + d_flip)
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(d, [0.5, 99.7])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        r = max(2, int(round(min(h, w) / 300)))
        d = guided_filter(gray, d, r, 1e-3)
        return np.clip(d, 0, 1)


def save_depth(path, d):
    cv2.imwrite(path, (np.clip(d, 0, 1) * 65535).astype(np.uint16))


def load_depth(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = DepthModel(args.model)
    names = sorted(n for n in os.listdir(args.photos)
                   if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    for n in names:
        dst = os.path.join(args.out, os.path.splitext(n)[0] + ".png")
        if os.path.exists(dst) and not args.force:
            continue
        t = time.time()
        bgr = cv2.imread(os.path.join(args.photos, n), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        save_depth(dst, model.predict(rgb))
        print(f"{n}: {bgr.shape[1]}x{bgr.shape[0]} depth in {time.time() - t:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
