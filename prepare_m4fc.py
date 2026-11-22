"""Prepare English M4FC annotations and retrieve their public image URLs.

M4FC annotations are CC-BY-SA-4.0. Images remain at their source URLs and may
have separate rights; this script downloads only to the user's local project.
"""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "M4FC.json"
OUT = ROOT / "data" / "m4fc"
IMAGES = OUT / "images"
MANIFEST = OUT / "manifest.csv"
HEADERS = {"User-Agent": "CrossCheckAI educational research project"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def build_manifest():
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    manifest = []
    for idx, row in enumerate(rows):
        if str(row.get("claim_language", "")).lower() != "english":
            continue
        verdict = str(row.get("verdict", "")).lower()
        if verdict == "true":
            factual = "true"
            consistency = "consistent"
        elif verdict == "false (out-of-context)":
            factual = "false"
            consistency = "mismatched"
        elif verdict.startswith("false"):
            factual = "false"
            consistency = "unknown"
        else:
            factual = "uncertain"
            consistency = "unknown"
        url = row.get("image_url", "")
        suffix = Path(unquote(urlparse(url).path)).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".img"
        manifest.append({
            "sample_id": f"m4fc_{idx:05d}", "image_file": f"m4fc_{idx:05d}{suffix}",
            "image_url": url, "claim": row.get("claim", ""),
            "true_caption": row.get("true_caption", ""),
            "factual_label": factual, "consistency_label": consistency,
            "verdict": verdict, "official_split": row.get("split", ""),
            "article_url": row.get("article_url", ""),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(MANIFEST, index=False)
    return manifest


def fetch(item):
    path = IMAGES / item["image_file"]
    if path.exists() and path.stat().st_size:
        try:
            with Image.open(path) as im:
                im.verify()
            return "cached"
        except (OSError, UnidentifiedImageError):
            path.unlink(missing_ok=True)
    try:
        with requests.get(item["image_url"], headers=HEADERS, timeout=15, stream=True) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > MAX_IMAGE_BYTES:
                    return "too_large"
                chunks.append(chunk)
        data = b"".join(chunks)
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        path.write_bytes(data)
        return "downloaded"
    except Exception:
        return "failed"


def download_images(limit=None, workers=12):
    IMAGES.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(MANIFEST).fillna("").to_dict("records")
    if limit:
        records = records[:limit]
    counts = {"downloaded": 0, "cached": 0, "failed": 0, "too_large": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in records]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            counts[future.result()] += 1
            if i % 100 == 0 or i == len(futures):
                print(f"Images {i}/{len(futures)}: {counts}", flush=True)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit image retrieval for a quick pilot.")
    args = parser.parse_args()
    manifest = build_manifest()
    print(f"Prepared {len(manifest)} English labeled examples at {MANIFEST}")
    print(pd.Series([r["factual_label"] for r in manifest]).value_counts().to_dict())
    print(pd.Series([r["consistency_label"] for r in manifest]).value_counts().to_dict())
    if args.download:
        download_images(args.limit)
