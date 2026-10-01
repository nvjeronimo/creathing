"""Collect the listing photos and facts from a property page.

python -m fpv.fetch <listing url> assets/photos

Reads the static HTML first. If the gallery is built by JavaScript, it
renders the page in headless Chromium and also records every image the page
downloads. Keeps full-size photos only (no logos, thumbnails or duplicates),
in gallery order, and saves the page text to listing.txt for the facts.
"""
import argparse
import html
import json
import os
import re
from urllib.parse import urljoin

import cv2
import numpy as np
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0 Safari/537.36")
IMG_RE = re.compile(r"""(?:https?:)?//[^\s"'<>()\\]+?\.(?:jpe?g|png|webp)(?:\?[^\s"'<>()\\]*)?""", re.I)
SKIP = ("logo", "icon", "favicon", "sprite", "avatar", "flag", "placeholder", "whatsapp",
        "facebook", "instagram", "linkedin", "youtube", "badge", "award", "team", "agent",
        "staff", "map", "marker")
SIZE_SUFFIX = re.compile(r"(-\d{2,4}x\d{2,4}|_thumb|_small|_medium|-thumb|-small)(?=\.\w+$)", re.I)


def static_page(url):
    r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en"}, timeout=40)
    r.raise_for_status()
    return r.text, [], ""


def rendered_page(url):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    from playwright.sync_api import sync_playwright
    seen = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1680, "height": 1050}, user_agent=UA)
        pg.on("response", lambda r: seen.append(r.url)
              if r.request.resource_type == "image" else None)
        pg.goto(url, wait_until="networkidle", timeout=120000)
        for _ in range(16):
            pg.mouse.wheel(0, 1400)
            pg.wait_for_timeout(350)
        pg.wait_for_timeout(1500)
        text = pg.inner_text("body")
        content = pg.content()
        b.close()
    return content, seen, text


def candidates(page_html, extra, base):
    blob = html.unescape(page_html).replace("\\/", "/")
    urls = IMG_RE.findall(blob) + list(extra)
    out, keys = [], set()
    for u in urls:
        if u.startswith("//"):
            u = "https:" + u
        u = urljoin(base, u)
        low = u.lower()
        if any(s in low for s in SKIP):
            continue
        key = SIZE_SUFFIX.sub("", low.split("?")[0])
        if key in keys:
            continue
        keys.add(key)
        out.append(u)
    return out


def ahash(img):
    g = cv2.cvtColor(cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (g > g.mean()).flatten()


def facts(page_html, text):
    info = {}
    for name in ("og:title", "og:description", "description"):
        m = re.search(r'<meta[^>]+(?:property|name)="%s"[^>]+content="([^"]*)"' % re.escape(name),
                      page_html, re.I)
        if m:
            info[name] = html.unescape(m.group(1))
    m = re.search(r"<title>(.*?)</title>", page_html, re.I | re.S)
    if m:
        info["title"] = html.unescape(m.group(1)).strip()
    info["ld_json"] = [html.unescape(x) for x in re.findall(
        r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page_html, re.I | re.S)]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--min-width", type=int, default=1000)
    ap.add_argument("--render", action="store_true", help="force headless browser")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    page, seen, text = ("", [], "")
    if not args.render:
        try:
            page, seen, text = static_page(args.url)
        except requests.RequestException as e:
            print("static fetch failed:", e)
    cands = candidates(page, seen, args.url)
    if args.render or len(cands) < 6:
        page, seen, text = rendered_page(args.url)
        cands = candidates(page, seen, args.url)
    print(f"{len(cands)} candidate images")

    kept, hashes = [], []
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Referer": args.url})
    for u in cands:
        try:
            r = sess.get(u, timeout=60)
            r.raise_for_status()
        except requests.RequestException:
            continue
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[1] < args.min_width or img.shape[0] < 600:
            continue
        hsh = ahash(img)
        dup = next((i for i, h in enumerate(hashes) if (h != hsh).sum() <= 12), None)
        if dup is not None:
            if img.shape[1] > kept[dup]["img"].shape[1]:
                kept[dup].update(img=img, url=u)
            continue
        hashes.append(hsh)
        kept.append({"img": img, "url": u})

    manifest = []
    for i, k in enumerate(kept, 1):
        name = f"{i:02d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), k["img"], [cv2.IMWRITE_JPEG_QUALITY, 96])
        manifest.append({"file": name, "url": k["url"],
                         "w": k["img"].shape[1], "h": k["img"].shape[0]})
        print(f"{name}  {k['img'].shape[1]}x{k['img'].shape[0]}  {k['url']}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"source": args.url, "photos": manifest, "facts": facts(page, text)}, f, indent=1)
    if text:
        with open(os.path.join(args.out, "listing.txt"), "w") as f:
            f.write(text)
    print(f"kept {len(manifest)} photos in {args.out}")


if __name__ == "__main__":
    main()
