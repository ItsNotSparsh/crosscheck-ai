"""Conservative live-event checks using dated publisher articles."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from html.parser import HTMLParser
from urllib.parse import urlsplit

import requests

from evidence_search import _ddgs
from reviewed_claims import HEADERS

PUBLISHERS = ("bbc.com","bbc.co.uk","theguardian.com","reuters.com","apnews.com",
              "npr.org","indiatoday.in","indianexpress.com","thehindu.com","pib.gov.in")

FEEDS = ("https://feeds.bbci.co.uk/news/world/rss.xml","https://www.theguardian.com/world/rss",
         "https://indianexpress.com/section/world/feed/","https://feeds.npr.org/1004/rss.xml")


@lru_cache(maxsize=2)
def headlines(quarter_hour):
    def fetch(url):
        try:
            response = requests.get(url,headers=HEADERS,timeout=(3,5))
            response.raise_for_status()
            return [(item.findtext("title") or "", item.findtext("link") or "")
                    for item in ET.fromstring(response.content).findall(".//item")[:80]]
        except (requests.RequestException,ET.ParseError):
            return []
    with ThreadPoolExecutor(max_workers=4) as pool:
        return [row for feed in pool.map(fetch,FEEDS) for row in feed]


def relevant_headlines(claim):
    from live_screening import tokens
    terms=tokens(claim)
    ranked=[]
    for title,url in headlines(int(time.time()//900)):
        overlap=len(terms & tokens(title))
        if overlap>=3 and overlap>=.4*len(terms) and publisher(url):
            ranked.append((overlap,url))
    return [url for _,url in sorted(ranked,reverse=True)[:6]]


def publisher(url):
    parts = urlsplit(url)
    if parts.scheme!="https" or parts.username or parts.password or parts.port not in (None,443):
        return None
    host=parts.hostname or ""
    return next((p for p in PUBLISHERS if host==p or host.endswith("."+p)),None)


class ArticleParagraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.current = None
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        if tag == "p" and self.in_article:
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            self.paragraphs.append("".join(self.current))
            self.current = None
        if tag == "article":
            self.in_article = False
            self.current = None


def article(url, max_age_days=7):
    try:
        domain=publisher(url)
        if not domain:
            return None
        with requests.get(url,headers=HEADERS,allow_redirects=False,stream=True,timeout=(3,5)) as response:
            if response.status_code!=200:
                return None
            buf=bytearray()
            for chunk in response.iter_content(65536):
                buf.extend(chunk)
                if len(buf)>2_500_000:
                    return None
        raw=buf.decode("utf8","replace")
        for block in re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>',raw,re.I|re.S):
            try:
                obj=json.loads(block)
            except ValueError:
                continue
            nodes=obj if isinstance(obj,list) else obj.get("@graph",[obj]) if isinstance(obj,dict) else []
            for node in nodes:
                if not isinstance(node,dict) or "Article" not in str(node.get("@type")):
                    continue
                date=str(node.get("datePublished",""))[:10]
                age=(datetime.now(timezone.utc).date()-datetime.fromisoformat(date).date()).days
                if age<0 or (max_age_days is not None and age>max_age_days):
                    continue
                body=node.get("articleBody")
                if not body:
                    parser = ArticleParagraphs()
                    parser.feed(raw)
                    body = " ".join(parser.paragraphs) or node.get("description")
                if not isinstance(body,str):
                    continue
                body=" ".join(html.unescape(re.sub("<[^>]+>"," ",body)).split())
                return dict(title=str(node.get("headline","Article")),body=body[:15000],source=domain,url=url,date=date)
    except (ValueError,TypeError,requests.RequestException):
        return None
    return None


def check_article(claim, url):
    from live_screening import UNCERTAIN, tokens, outcome, relevant_passages, compare
    item = article(url,max_age_days=None)
    if not item:
        return outcome(UNCERTAIN,"Could not read a dated article at this URL.")
    body = item.pop("body")
    terms = tokens(claim)
    passages = []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z“\x22])",body)
    # Keep adjacent statements together when a name appears in one and its
    # action or number in the next.
    windows = sentences + [first+" "+second for first,second in zip(sentences,sentences[1:])]
    for sentence in windows:
        if not 20<=len(sentence)<=1200 or len(terms & tokens(sentence))<min(2,len(terms)):
            continue
        passages.append(dict(item,snippet=sentence))
    passages.sort(key=lambda p:-len(terms & tokens(p["snippet"])))
    relevant = relevant_passages(claim,passages[:16])
    positive,negative = [],[]
    if relevant:
        for p,(support,refute) in zip(relevant,compare(claim,[p["snippet"] for p in relevant])):
            if re.search(r"\b(rumou?r|unverified|allegedly|purport(?:s|ed)|falsely claimed|without evidence)\b",p["snippet"],re.I):
                continue
            if support>=.98:
                positive.append(p)
            if refute>=.98:
                negative.append(p)
    if bool(positive)!=bool(negative):
        hits=positive or negative
        return outcome("Real" if positive else "Fake","Supported by the linked article" if positive
                       else "Contradicted by the linked article",hits,
                       "This comparison applies to what the publisher reported on "+item["date"]+".")
    return outcome(UNCERTAIN,"The linked article does not settle the wording.",
                   relevant[:2] or [dict(item,snippet="No passage directly addresses the submitted claim.")])


def screen_recent(claim, feed_urls=()):
    from live_screening import tokens, compare, outcome, UNCERTAIN
    if re.search(r"\b(this|these|here|today|yesterday|tomorrow|video|photo|image|footage)\b",claim,re.I):
        return outcome(UNCERTAIN,"The date or identity of the event or media needs direct verification.")
    try:
        hits=_ddgs(claim)
    except Exception:
        hits=[]
    addresses=list(dict.fromkeys([*feed_urls,*(hit["url"] for hit in hits if publisher(hit["url"]))]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        articles=[item for item in pool.map(article,addresses[:5]) if item]
    items=[]
    terms=tokens(claim)
    for item in articles:
        sentences=re.split(r"(?<=[.!?])\s+(?=[A-Z])",item.pop("body"))
        for s in sorted(sentences,key=lambda s:-len(tokens(s)&terms))[:2]:
            if 35<=len(s)<=900 and len(tokens(s)&terms)>=max(2,.35*len(terms)):
                items.append(dict(item,snippet=s))
    if items:
        scores=compare(claim,[i["snippet"] for i in items])
        groups=[[],[]]
        for item,score in zip(items,scores):
            for index, strength in enumerate(score):
                if strength>=.97:
                    groups[index].append(item)
        for index,group in enumerate(groups):
            sources={i["source"].replace("bbc.co.uk","bbc.com") for i in group}
            distinct=any(a["source"]!=b["source"] and len(tokens(a["snippet"])&tokens(b["snippet"]))/
                         max(1,len(tokens(a["snippet"])|tokens(b["snippet"])))<.7 for a in group for b in group)
            if len(sources)>=2 and distinct and not groups[1-index]:
                return outcome(["Real","Fake"][index],"Two recent publishers support the claim" if index==0
                               else "Two recent publishers contradict the claim",group,group[0]["snippet"])
    return outcome(UNCERTAIN,"Insufficient corroboration from dated news articles.",items[:3])
