"""Conservative live-event checks using dated publisher articles."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
import re
from urllib.parse import urlsplit

import requests

from evidence_search import _ddgs
from reviewed_claims import HEADERS

PUBLISHERS = ("bbc.com","bbc.co.uk","theguardian.com","reuters.com","apnews.com",
              "npr.org","indiatoday.in","indianexpress.com","thehindu.com","pib.gov.in")


def publisher(url):
    parts = urlsplit(url)
    if parts.scheme!="https" or parts.username or parts.password or parts.port not in (None,443):
        return None
    host=parts.hostname or ""
    return next((p for p in PUBLISHERS if host==p or host.endswith("."+p)),None)


def article(url):
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
                if not 0<=age<=7:
                    continue
                body=node.get("articleBody") or node.get("description")
                if not isinstance(body,str):
                    continue
                body=" ".join(html.unescape(re.sub("<[^>]+>"," ",body)).split())
                return dict(title=str(node.get("headline","Article")),body=body[:15000],source=domain,url=url,date=date)
    except (ValueError,TypeError,requests.RequestException):
        return None
    return None


def screen_recent(claim):
    from live_screening import tokens, compare, outcome, UNCERTAIN
    if re.search(r"\b(this|these|here|today|yesterday|tomorrow|video|photo|image|footage)\b",claim,re.I):
        return outcome(UNCERTAIN,"The date or identity of the event or media needs direct verification.")
    try:
        hits=_ddgs(claim)
    except Exception:
        hits=[]
    addresses=[hit["url"] for hit in hits if publisher(hit["url"])]
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
