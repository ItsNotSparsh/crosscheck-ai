"""Check explicit relationships against identified, active Wikidata statements."""
from datetime import datetime, timezone
from functools import lru_cache
import re
import time

import requests

from reviewed_claims import HEADERS

API = "https://www.wikidata.org/w/api.php"
RELATIONS = {"capital": ("P36","capital"), "currency": ("P38","currency"),
             "author": ("P50","author"), "director": ("P57","director"),
             "composer": ("P86","composer")}
VERBS = {"written":"author","directed":"director","composed":"composer"}
NAME = r"[\w][\w .'’,&()\-]{0,85}?"


def parse(claim):
    text = claim.strip().rstrip(".").strip()
    copula = r"(?:is|was)(?P<negative> not)?"
    role = r"(?P<role>capital(?: city)?|currency|author|director|composer)"
    patterns = [
        rf"(?P<value>{NAME}) {copula} (?:the |a |an ){role} of (?P<entity>{NAME})",
        rf"(?:the )?{role} of (?P<entity>{NAME}) {copula} (?P<value>{NAME})",
        rf"(?P<entity>{NAME}) {copula} (?P<verb>written|directed|composed) by (?P<value>{NAME})",
    ]
    for pattern in patterns:
        m = re.fullmatch(pattern,text,re.I)
        if m:
            g = m.groupdict()
            key = VERBS[g["verb"].lower()] if g.get("verb") else g["role"].lower().replace(" city","")
            if key in {"capital","currency"} and re.search(r"\bwas\b",text,re.I):
                return None
            return key, g["entity"],g["value"],bool(g["negative"])
    return None


def fold(name):
    return " ".join(name.casefold().replace("’","'").split()).removeprefix("the ")


@lru_cache(maxsize=256)
def get_data(query, period):
    response = requests.get(API,params=dict(query,format="json"),headers=HEADERS,timeout=(3,6))
    response.raise_for_status()
    return response.json()


def api(period, **params):
    return get_data(tuple(sorted(params.items())),period)


def active_values(statements, now):
    rows = []
    for st in statements:
        quals = st.get("qualifiers",{})
        if st["rank"]=="deprecated" or any(q in quals for q in ("P582","P585","P518")):
            continue
        starts = [q.get("datavalue",{}).get("value",{}).get("time","").lstrip("+")[:10] for q in quals.get("P580",[])]
        if any(start > now for start in starts):
            continue
        item = st["mainsnak"].get("datavalue",{}).get("value",{})
        if isinstance(item,dict) and item.get("id"):
            rows.append((item["id"],st["rank"]))
    preferred = [v for v,rank in rows if rank=="preferred"]
    return list(dict.fromkeys(preferred or [v for v,_ in rows]))


def check(claim):
    from live_screening import outcome
    parsed = parse(claim)
    if not parsed:
        return None
    key,subject,value,negated = parsed
    prop,relation = RELATIONS[key]
    period = int(time.time()//14400)
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        hits = api(period,action="wbsearchentities",search=subject,language="en",limit="6").get("search",[])
        ids = [x["id"] for x in hits if fold(subject) in {fold(x["label"]),fold(x.get("match",{}).get("text",""))}]
        if not ids:
            return None
        entities = api(period,action="wbgetentities",ids="|".join(ids),props="labels|sitelinks",sitefilter="enwiki",languages="en")["entities"]
        matches = list(entities.values())
        if len(matches)>1:
            matches = [e for e in matches if fold(e.get("sitelinks",{}).get("enwiki",{}).get("title",""))==fold(subject)]
        if len(matches)!=1:
            return None
        e = matches[0]
        statements = api(period,action="wbgetclaims",entity=e["id"],property=prop)
        values = active_values(statements.get("claims",{}).get(prop,[]),today)
        if not values or len(values)>50:
            return None
        objects = api(period,action="wbgetentities",ids="|".join(values),props="labels|aliases",languages="en")["entities"]
        for obj in objects.values():
            english = obj.get("labels",{}).get("en",{}).get("value","")
            names = [english,*(a["value"] for a in obj.get("aliases",{}).get("en",[]))]
            if fold(value) in {fold(n) for n in names}:
                statement = f'{e["labels"]["en"]["value"]} — {relation}: {english}.'
                item = dict(title=e["labels"]["en"]["value"],snippet=statement,source="Wikidata",
                            url=f'https://www.wikidata.org/wiki/{e["id"]}#{prop}',date=today)
                return outcome("Fake" if negated else "Real","Matched a Wikidata statement",[item],statement)
    except (requests.RequestException,KeyError,TypeError,ValueError):
        pass
    return None
