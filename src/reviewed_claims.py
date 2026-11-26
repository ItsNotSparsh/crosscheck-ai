"""A bounded, refreshable index of publisher-written ClaimReview records."""
import concurrent.futures
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
import time
from urllib.parse import urljoin, urlsplit

import requests

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "published_reviews.json"
HEADERS = {"User-Agent": "CrossCheckAI/1.1 (educational claim review app)"}
_refresh_lock = Lock()
_last_attempt = 0.0


def refresh_if_stale():
    """Keep serving the previous snapshot during a refresh. Retry failures hourly."""
    global _last_attempt
    now = time.time()
    if INDEX.exists() and now - INDEX.stat().st_mtime < 6*3600:
        return
    if not _refresh_lock.acquire(blocking=False):
        return
    if now - _last_attempt < 3600:
        _refresh_lock.release()
        return
    _last_attempt = now
    def worker():
        try:
            refresh()
        except (RuntimeError, OSError, ValueError):
            pass
        finally:
            _refresh_lock.release()
    Thread(target=worker, daemon=True, name="publisher-review-refresh").start()


def page(url):
    with requests.get(url, headers=HEADERS, stream=True, timeout=(4, 7)) as response:
        response.raise_for_status()
        buf = bytearray()
        for chunk in response.iter_content(65536):
            buf.extend(chunk)
            if len(buf) > 3_000_000:
                raise ValueError("Publisher page exceeded size limit")
        return buf.decode("utf-8", errors="replace")


def review(url):
    try:
        article = page(url)
        for block in re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', article, re.I | re.S):
            try:
                obj = json.loads(block)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("@type") == "ClaimReview":
                claim = obj.get("claimReviewed")
                rating = obj.get("reviewRating", {}).get("alternateName")
                if isinstance(claim, str) and isinstance(rating, str) and claim.strip() and rating.strip():
                    return dict(claim=html.unescape(claim)[:1800], rating=rating[:100],
                                url=url, date=obj.get("datePublished", "")[:10])
    except (requests.RequestException, ValueError, TypeError):
        pass
    return None


def refresh():
    existing = read_index()
    records = {r["url"]: r for r in existing}
    sources = [
        ("https://www.snopes.com/fact-check/", r'href="(https://www\.snopes\.com/fact-check/[\w-]+/)"'),
        ("https://www.indiatoday.in/fact-check", r'href="(/fact-check/story/[\w-]+)"'),
    ]
    links = []
    for url, pattern in sources:
        try:
            links.extend(urljoin(url, p) for p in list(dict.fromkeys(re.findall(pattern, page(url))))[:26])
        except (requests.RequestException, ValueError):
            continue
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        latest = [row for row in pool.map(review, links) if row]
    if not latest:
        raise RuntimeError("No usable publisher reviews returned. Existing index retained.")
    records.update({r["url"]: r for r in latest})
    rows = sorted(records.values(), key=lambda r: r["date"], reverse=True)[:1000]
    temp = INDEX.with_suffix(".tmp")
    temp.write_text(json.dumps({"updated": datetime.now(timezone.utc).isoformat(), "reviews": rows},
                               ensure_ascii=False), encoding="utf-8")
    temp.replace(INDEX)
    print(f"Saved {len(rows)} dated reviews ({len(latest)} fetched).")


def read_index():
    if not INDEX.exists():
        return []
    return json.loads(INDEX.read_text(encoding="utf-8"))["reviews"]


def search_reviews(query):
    """Read dated English ClaimReview entries returned by public Fact Check Explorer."""
    try:
        response = requests.get("https://toolbox.google.com/factcheck/api/search",
                                params={"query":query}, headers=HEADERS, timeout=(3,5))
        response.raise_for_status()
        payload = json.loads(response.text.removeprefix(")]}'").strip())
        rows = []
        for entry in payload[0][1][:25]:
            document = entry[0]
            if len(document)<4 or not isinstance(document[0],str):
                continue
            claim, reviews = document[0][:2_000], document[3] or []
            for review in reviews:
                if len(review)<7 or review[6] != "en" or not isinstance(review[2],(float,int)) or not isinstance(review[3],str) or not str(review[1]).startswith("https://"):
                    continue
                date = datetime.fromtimestamp(review[2], timezone.utc).date().isoformat()
                rows.append(dict(claim=claim, url=review[1], rating=review[3],date=date))
        return rows
    except (requests.RequestException, ValueError, TypeError, KeyError, IndexError, AttributeError, OSError):
        return []


if __name__ == "__main__":
    refresh()
