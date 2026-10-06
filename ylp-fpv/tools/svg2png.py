"""Render an SVG logo to a transparent PNG with headless Chromium.

    python3 tools/svg2png.py assets/brand/ylp-logo.svg assets/brand/ylp-logo.png [size]

size is the long side in pixels (default 2048). The configs reference the PNG.
"""
import base64
import os
import re
import sys

from playwright.sync_api import sync_playwright


def main():
    src, dst = sys.argv[1], sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    svg = open(src, "rb").read()
    m = re.search(rb'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)', svg)
    vw, vh = (float(m.group(1)), float(m.group(2))) if m else (1.0, 1.0)
    k = size / max(vw, vh)
    w, h = max(1, round(vw * k)), max(1, round(vh * k))
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    html = (f'<html><body style="margin:0;background:transparent">'
            f'<img src="{uri}" style="display:block;width:{w}px;height:{h}px"></body></html>')
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": w, "height": h})
        page.set_content(html)
        page.wait_for_function("document.images[0].complete")
        page.screenshot(path=dst, omit_background=True)
        browser.close()
    print(f"{dst}: {w}x{h}")


if __name__ == "__main__":
    main()
