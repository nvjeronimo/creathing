"""Render a project to MP4 (or to a contact sheet of stills).

python -m fpv.render config/project.json output/video.mp4 [--scale 0.5]
python -m fpv.render config/project.json sheet.jpg --stills 1.0,2.5,4.0
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from .engine import Timeline, load_project
from .finish import Finisher

_G = {}


def _init(project_path, root, scale, with_graphics):
    proj = load_project(project_path)
    tl = Timeline(proj, root)
    w, h = int(tl.w * scale) // 2 * 2, int(tl.h * scale) // 2 * 2
    _G.update(tl=tl, w=w, h=h, fin=Finisher(w, h, **proj.get("finish", {})))
    if with_graphics:
        from .graphics import Graphics
        _G["gfx"] = Graphics(proj, root, w, h, tl)


def _render(i):
    tl, w, h = _G["tl"], _G["w"], _G["h"]
    lin = tl.frame(i, w=w, h=h)
    rgb = _G["fin"].apply(lin, i)
    if "gfx" in _G:
        rgb = _G["gfx"].draw(rgb, i / tl.fps)
    return i, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--frames", default="")
    ap.add_argument("--stills", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--no-graphics", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--maxrate", default="11M")
    args = ap.parse_args()

    proj = load_project(args.project)
    gfx = not args.no_graphics and "graphics" in proj
    fps = proj.get("fps", 30)

    if args.stills:
        _init(args.project, args.root, args.scale, gfx)
        times = [float(x) for x in args.stills.split(",")]
        tiles = []
        for t in times:
            _, rgb = _render(int(round(t * fps)))
            img = rgb.copy()
            cv2.putText(img, f"{t:.2f}s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2, cv2.LINE_AA)
            tiles.append(img)
        cols = 2 if len(tiles) > 1 else 1
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
        sheet = np.vstack(rows)
        cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        return

    tl = Timeline(proj, args.root)
    n_total = int(round(tl.duration * fps))
    if args.frames:
        a, b = args.frames.split(":")
        frames = list(range(int(a), int(b)))
    else:
        frames = list(range(n_total))
    w, h = int(tl.w * args.scale) // 2 * 2, int(tl.h * args.scale) // 2 * 2

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if args.audio:
        start = frames[0] / fps
        cmd += ["-ss", f"{start:.4f}", "-i", args.audio, "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "320k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
            "-maxrate", args.maxrate, "-bufsize", "22M",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-tune", "film",
            "-x264-params", "keyint=60:min-keyint=30",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(args.project, args.root, args.scale, gfx)) as pool:
        for k, (i, rgb) in enumerate(pool.imap(_render, frames, chunksize=4)):
            enc.stdin.write(rgb.tobytes())
            if k % 30 == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (len(frames) - k - 1)
                print(f"frame {i} ({k + 1}/{len(frames)}) {el:.0f}s elapsed, eta {eta:.0f}s",
                      flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")
    sys.exit(enc.returncode)


if __name__ == "__main__":
    main()
