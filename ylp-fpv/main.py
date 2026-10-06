"""Higgsfield API check: one Seedance 2.5 text-to-video generation.

    python3 main.py

Reads HF_KEY (key-id:key-secret) from .env.local next to this file, or from the
environment. The key is never printed. Each run is one billable request.
Prints the video URL and exits 0 only when the generation completed.
"""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))

import higgsfield_client  # noqa: E402  (reads HF_KEY lazily, on the first request)

MODEL = "bytedance/seedance-2.5/text-to-video"
ARGUMENTS = {
    "prompt": "A cinematic scene at sunset",
    "duration": 5,
    "resolution": "720p",
    "aspect_ratio": "16:9",
}


def redact(text):
    """Strip the credential from any text before it reaches the terminal."""
    text = str(text)
    for name in ("HF_KEY", "HF_API_KEY", "HF_API_SECRET"):
        for part in (os.getenv(name) or "").split(":"):
            if len(part) >= 8:
                text = text.replace(part, "***")
    return text


def video_url(node):
    """First video URL in a result: video.url, videos[0].url, or any .mp4 link."""
    if isinstance(node, dict):
        video = node.get("video")
        if isinstance(video, dict) and isinstance(video.get("url"), str):
            return video["url"]
        videos = node.get("videos")
        if isinstance(videos, list) and videos and isinstance(videos[0], dict):
            if isinstance(videos[0].get("url"), str):
                return videos[0]["url"]
        nodes = node.values()
    elif isinstance(node, list):
        nodes = node
    elif isinstance(node, str) and node.startswith("http") and node.split("?")[0].endswith(".mp4"):
        return node
    else:
        return None
    for child in nodes:
        url = video_url(child)
        if url:
            return url
    return None


def main():
    if not (os.getenv("HF_KEY") or (os.getenv("HF_API_KEY") and os.getenv("HF_API_SECRET"))):
        print("HF_KEY is missing: add it to .env.local as key-id:key-secret", file=sys.stderr)
        return 2

    last = []

    def on_update(status):
        name = type(status).__name__
        if last != [name]:
            last[:] = [name]
            print(f"status: {name}", flush=True)

    try:
        result = higgsfield_client.subscribe(
            MODEL,
            arguments=ARGUMENTS,
            on_enqueue=lambda request_id: print(f"request: {request_id}", flush=True),
            on_queue_update=on_update,
        )
    except higgsfield_client.HiggsfieldClientError as e:
        print(f"rejected by the API: {redact(e)}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"could not reach the API: {type(e).__name__}: {redact(e)}", file=sys.stderr)
        return 1

    status = result.get("status")
    if status != "completed":
        detail = result.get("error") or result.get("detail") or result.get("message") or ""
        print(f"generation did not complete: status={status} {redact(detail)}".rstrip(), file=sys.stderr)
        return 1

    url = video_url(result)
    if not url:
        print(f"completed, but no video URL in the response (keys: {sorted(result)})", file=sys.stderr)
        return 1
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
