"""Build the local Simple English Wikipedia evidence index from a pinned digest."""
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile

import pyarrow.parquet as pq
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
DB = DATA / "simplewiki.sqlite"
SNAPSHOT = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet"
SHA256 = "31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0"


def main():
    with tempfile.TemporaryDirectory(dir=DATA) as temp:
        source = Path(temp) / "snapshot.parquet"
        digest = hashlib.sha256()
        with requests.get(SNAPSHOT,stream=True,timeout=(10,60)) as r, source.open("wb") as out:
            r.raise_for_status()
            for chunk in r.iter_content(1024*1024):
                if out.tell()+len(chunk)>160_000_000:
                    raise ValueError("Unexpected snapshot size")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest()!=SHA256:
            raise ValueError("Snapshot digest mismatch")
        path = Path(temp) / "index.sqlite"
        conn = sqlite3.connect(path)
        try:
            conn.execute("pragma journal_mode=off")
            conn.execute("pragma max_page_count=400000")
            conn.execute("create virtual table pages using fts5(title, body, url UNINDEXED, tokenize='porter unicode61')")
            for batch in pq.ParquetFile(source).iter_batches(batch_size=512,columns=["title","text","url"]):
                conn.executemany("insert into pages values(?,?,?)",
                                 ((r["title"],r["text"][:2800],r["url"]) for r in batch.to_pylist()))
            conn.commit()
            count = conn.execute("select count(*) from pages").fetchone()[0]
        finally:
            conn.close()
        os.replace(path,DB)
        print(f"Indexed {count:,} pages in {DB.stat().st_size:,} bytes.")


if __name__=="__main__":
    main()
