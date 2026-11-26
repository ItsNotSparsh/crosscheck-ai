"""Build and search a bounded local index of original publisher ClaimReview records."""
from contextlib import closing
from datetime import date
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from urllib.parse import urlsplit
import zipfile

DATA = Path(__file__).resolve().parent / "data"
INDEX = DATA / "publisher_reviews.sqlite"
SNAPSHOT = "2026_11_25"
DIGEST = "cedc71a2e8adb283f8b8201dabc70b79291a5f53f5f609846461be4bc6274369"


def clean(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ",value)).split()) if isinstance(value,str) else ""


def build(archive):
    import ijson
    with open(archive,"rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != DIGEST:
            raise ValueError("Unexpected publisher snapshot checksum")
    with tempfile.TemporaryDirectory(dir=DATA) as temp, zipfile.ZipFile(archive) as bundle:
        path = Path(temp)/"reviews.sqlite"
        with bundle.open(f"{SNAPSHOT}/ifcn_sources.json") as file:
            domains = {r["domain"].lower().removeprefix("www.") for r in json.load(file) if r.get("domain")}
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE records(claim TEXT,rating TEXT,url TEXT,date TEXT, UNIQUE(url,claim,rating,date))")
            for name in ["claim_reviews_raw.json","claim_reviews_raw_recollected.json"]:
                with bundle.open(f"{SNAPSHOT}/{name}") as file:
                    for row in ijson.items(file,"item"):
                        url = row.get("url")
                        if not isinstance(url,str) or not url.startswith("https://"):
                            continue
                        try:
                            host = (urlsplit(url).hostname or "").lower()
                            when = date.fromisoformat(str(row.get("datePublished",""))[:10]).isoformat()
                        except ValueError:
                            continue
                        if not any(".".join(host.split(".")[i:]) in domains for i in range(host.count(".")+1)):
                            continue
                        claim = clean(row.get("claimReviewed"))
                        raw_rating = row.get("reviewRating")
                        rating = clean(raw_rating.get("alternateName")) if isinstance(raw_rating,dict) else ""
                        if 5<=len(claim)<=2000 and 0<len(rating)<=2500:
                            db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?)",(claim,rating,url,when))
            db.execute("CREATE VIRTUAL TABLE words USING fts5(claim,content='records',content_rowid='rowid',tokenize='porter unicode61')")
            db.execute("INSERT INTO words(words) VALUES('rebuild')")
            db.commit()
            print("Indexed reviews:", db.execute("SELECT count(*) FROM records").fetchone()[0],flush=True)
        path.replace(INDEX)


def candidates(terms):
    if not INDEX.is_file() or not terms:
        return []
    query = " OR ".join('"'+t+'"' for t in sorted(terms) if re.fullmatch(r"[a-z0-9]+",t))
    if not query:
        return []
    with closing(sqlite3.connect(INDEX.as_uri()+"?mode=ro",uri=True)) as db:
        db.row_factory = sqlite3.Row
        hits = db.execute("SELECT records.* FROM words JOIN records ON words.rowid=records.rowid "
                          "WHERE words MATCH ? ORDER BY rank LIMIT 35",(query,))
        return [dict(row) for row in hits]


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv)>1 else DATA/"publisher_snapshot.zip")
