"""Create a small paired-image subset from official Fakeddit TSV files.

Run only after downloading the official train and validation TSVs. It samples rows,
downloads their image_url assets, and writes compact TSVs for train.py.
"""
import argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests

def prepare(src, out, image_dir, limit, timeout, workers):
    if str(src).lower().endswith(".parquet"):
        df = pd.read_parquet(src)
    else:
        df = pd.read_csv(src, sep="\t")
    label = next((c for c in ("6_way_label", "3_way_label", "2_way_label") if c in df), None)
    text = "clean_title" if "clean_title" in df else ("title" if "title" in df else ("text" if "text" in df else None))
    if not label or not text or "image_url" not in df or "id" not in df:
        raise ValueError("Expected Fakeddit columns id, image_url, clean_title/title and a *_way_label column")
    split = getattr(prepare, "split", "all")
    split_col = {"train":"general_train", "test":"general_test"}.get(split)
    if split_col and split_col in df: df = df[df[split_col] == True]
    df = df.dropna(subset=["id", "image_url", text, label]).sample(frac=1, random_state=42).head(limit)
    image_dir.mkdir(parents=True, exist_ok=True)
    def fetch(row):
        ident = str(row["id"])
        dest = image_dir / f"{ident}.jpg"
        if not dest.exists():
            try:
                resp = requests.get(str(row["image_url"]), timeout=timeout, headers={"User-Agent":"CrossCheckAI-course-project/1.0"})
                resp.raise_for_status()
                if not resp.headers.get("content-type", "").startswith("image/") or len(resp.content) > 12_000_000: return None
                dest.write_bytes(resp.content)
            except requests.RequestException: return None
        copied=row.copy(); copied["image_id"]=ident
        return copied
    kept=[]
    rows=[row for _,row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,row) for row in rows]
        for n,fut in enumerate(as_completed(futures),1):
            result=fut.result()
            if result is not None: kept.append(result)
            if n % 100 == 0: print(f"Checked {n}/{len(rows)} rows; paired images: {len(kept)}",flush=True)
    if not kept: raise SystemExit("No images downloaded. Check internet access and image_url availability.")
    result=pd.DataFrame(kept)
    out.parent.mkdir(parents=True, exist_ok=True); result.to_csv(out, sep="\t", index=False)
    print(f"Wrote {len(result)} paired samples to {out}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_tsv"); p.add_argument("output_tsv"); p.add_argument("images_dir")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--limit", type=int, default=3000); p.add_argument("--timeout", type=int, default=6); p.add_argument("--workers",type=int,default=20)
    a=p.parse_args(); prepare.split=a.split; prepare(a.source_tsv, Path(a.output_tsv), Path(a.images_dir), a.limit, a.timeout, a.workers)
if __name__=="__main__": main()
