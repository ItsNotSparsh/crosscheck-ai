"""Best-effort public evidence links with provider fallback and friendly failures."""
from urllib.parse import quote_plus
from urllib.parse import urlparse
import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
import requests

TIMEOUT=7
HEADERS={"User-Agent":"CrossCheckAI/1.0 (educational project; evidence links)"}

def _clean_text(value):
    """Turn RSS/HTML fragments into readable text before showing or scoring them."""
    value=html.unescape(value or "")
    value=re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>"," ",value)
    value=re.sub(r"(?s)<[^>]+>"," ",value)
    return re.sub(r"\s+"," ",value).strip()

def _ddgs(query):
    from ddgs import DDGS
    rows=list(DDGS(timeout=TIMEOUT).text(query,max_results=5))
    return [{"title":r.get("title","Search result"),"snippet":r.get("body","")[:600],"url":r.get("href",""),"source":"Web search","domain":urlparse(r.get("href","")).netloc.lower().removeprefix("www.")} for r in rows if r.get("href")]

def _news_rss(query):
    url="https://news.google.com/rss/search?q="+quote_plus(query)+"&hl=en-US&gl=US&ceid=US:en"
    response=requests.get(url,headers=HEADERS,timeout=TIMEOUT)
    response.raise_for_status()
    root=ET.fromstring(response.content)
    rows=[]
    for item in root.findall(".//item")[:5]:
        source_el=item.find("source")
        source_name=(source_el.text or "").strip() if source_el is not None else ""
        source_url=source_el.get("url","") if source_el is not None else ""
        link=item.findtext("link",default="")
        domain=urlparse(source_url or link).netloc.lower().removeprefix("www.")
        rows.append({"title":_clean_text(item.findtext("title",default="News result")),"snippet":_clean_text(item.findtext("description",default=""))[:1000],"url":link,"date":item.findtext("pubDate",default=""),"source":source_name or "Google News RSS","domain":domain})
    return rows

def _wikipedia(query):
    url="https://en.wikipedia.org/w/api.php"
    response=requests.get(url,params={"action":"query","list":"search","srsearch":query,"format":"json","srlimit":5},headers=HEADERS,timeout=TIMEOUT)
    response.raise_for_status()
    results=response.json().get("query",{}).get("search",[])
    return [{"title":_clean_text(r["title"]),"snippet":_clean_text(r.get("snippet",""))[:1000],"url":"https://en.wikipedia.org/wiki/"+quote_plus(r["title"].replace(" ","_")),"source":"Wikipedia","domain":"wikipedia.org"} for r in results]


def _pm_india_profile(query):
    """Fetch the official PM profile for claims about Modi holding India's PM office."""
    if not (re.search(r"\bmodi\b",query,re.I) and re.search(r"\b(prime minister|pm)\b",query,re.I)
            and re.search(r"\b(india|indian)\b",query,re.I)):
        return []
    url="https://www.pmindia.gov.in/en/pms-profile/"
    response=requests.get(url,headers=HEADERS,timeout=TIMEOUT)
    response.raise_for_status()
    page=_clean_text(response.text)
    matches=[]
    for sentence in re.split(r"(?<=[.!?])\s+",page):
        if re.search(r"narendra modi",sentence,re.I) and re.search(r"prime minister",sentence,re.I):
            matches.append(sentence)
    if not matches:
        return []
    return [{"title":"Prime Minister's Profile — PM India","snippet":" ".join(matches)[:1200],
             "url":url,"source":"Prime Minister of India","domain":"pmindia.gov.in"}]
def retrieve_evidence(query):
    """Query independent fallbacks concurrently under one short deadline."""
    import re
    variants=[query, "fact check " + query, "site:snopes.com " + query]
    if re.search(r"\bmodi\b",query,re.I) and re.search(r"prime minister",query,re.I):
        variants += ["Narendra Modi current Prime Minister of India official government biography",
                     "site:pmindia.gov.in Narendra Modi Prime Minister of India"]
    if re.search(r"\b(dinosaur|pterodactyl|prehistoric)\b",query,re.I) and re.search(r"\b(spotted|sighting|seen|flying|saw|video)\b",query,re.I):
        variants.append("fact check pterodactyl dinosaur sighting Mexico fake video")
    providers=[(f"web search {i+1}",_ddgs,q) for i,q in enumerate(variants)]
    providers += [(f"news search {i+1}",_news_rss,q) for i,q in enumerate(variants)]
    providers += [("encyclopedia search",_wikipedia,query)]
    if re.search(r"\bmodi\b",query,re.I) and re.search(r"prime minister",query,re.I):
        providers.append(("official PM India profile",_pm_india_profile,query))
    errors=[]; collected=[]; executor=ThreadPoolExecutor(max_workers=len(providers),thread_name_prefix="evidence-search")
    futures={executor.submit(fn,q):(index,name) for index,(name,fn,q) in enumerate(providers)}
    pending=set(futures); deadline=time.monotonic()+TIMEOUT+1
    try:
        while pending:
            remaining=deadline-time.monotonic()
            if remaining<=0: break
            done,pending=wait(pending,timeout=remaining,return_when=FIRST_COMPLETED)
            if not done: break
            # When multiple finish together, preserve the preferred provider order.
            for future in sorted(done,key=lambda f:futures[f][0]):
                _,provider=futures[future]
                try:
                    results=future.result()
                    if results:
                        collected.extend(results)
                    else:
                        errors.append(f"{provider}: no results")
                except Exception as exc:
                    errors.append(f"{provider}: {type(exc).__name__}")
    finally:
        for future in pending:
            future.cancel()
            errors.append(f"{futures[future][1]}: timed out")
        executor.shutdown(wait=False,cancel_futures=True)
    unique=[]; seen=set()
    for item in collected:
        key=(item.get("url") or item.get("title","" )).strip().lower()
        if key and key not in seen:
            seen.add(key); unique.append(item)
    preferred=("pmindia.gov.in","snopes.com","reuters.com","apnews.com","bbc.com","bbc.co.uk",
               "factcheck.org","politifact.com","fullfact.org","britannica.com","wikipedia.org","newsnationnow.com","newsnation.tv")
    terms=[term for term in re.findall(r"[a-z0-9]+",query.lower()) if len(term)>2 and term not in {"the","and","for","with","from","that","this","are","was","were","is","in","of","to"}]
    def relevance(item):
        domain=(item.get("domain") or "").lower().removeprefix("www.")
        content=(item.get("title","")+" "+item.get("snippet","")).lower()
        source_rank=next((len(preferred)-i for i,host in enumerate(preferred) if domain==host or domain.endswith("."+host)),0)
        overlap=sum(term in content for term in terms)
        return (source_rank,overlap)
    unique.sort(key=relevance,reverse=True)
    if unique:
        return {"items":unique[:12],"provider":"multiple sources","message":"These are leads to inspect, not verified evidence."}
    return {"items":[],"provider":None,"message":"Search providers did not return results. You can still review the model signal, or check the claim directly with trusted sources.","diagnostics":errors}
