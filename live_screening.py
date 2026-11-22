"""Screen public claims against referenced records, and captions against the photo."""
import html
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock
from urllib.parse import quote, urlparse

import requests
import torch

from evidence_verification import _load
from reviewed_claims import HEADERS, page, read_index, refresh_if_stale

UNCERTAIN = "Possibly misleading / uncertain"
INFERENCE_LOCK = RLock()
STOP = set("a an and are as at be by did do does for from has have in is it of on or that the this to was were with not".split())
TIME_OR_MEDIA = re.compile(r"\b(today|yesterday|tomorrow|currently|latest|now|this (?:week|month|year)|breaking|president|prime minister|photo|footage|pictured|video|image|spotted|sighting)\b", re.I)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def normalized(text):
    # Preserve numbers, negation and internal punctuation in quoted claims.
    return " ".join(text.casefold().replace("’", "'").split()).strip(' .“”"')


def outcome(label, basis, items=(), detail=""):
    return {"label": label, "decision_basis": basis, "reasons": [detail or basis],
            "evidence": {"provider": "Source passages", "items": list(items), "message": ""},
            "evidence_check": None, "probabilities": [], "similarity": None}


def reviewed_match(claim):
    refresh_if_stale()
    records = read_index()
    rows = [r for r in records if normalized(r["claim"]) == normalized(claim)]
    exact = bool(rows)
    if not rows:
        # Only transfer a published rating across a close paraphrase if both
        # directions entail each other and numeric qualifications are preserved.
        terms = tokens(claim)
        numbers = re.findall(r"\d+(?:[.,]\d+)*%?",claim)
        ranked = []
        for record in records:
            other = tokens(record["claim"])
            similarity = len(terms & other) / max(1,len(terms | other))
            if similarity >= .30 and len(terms & other)>=3 and sorted(numbers)==sorted(re.findall(r"\d+(?:[.,]\d+)*%?",record["claim"])):
                ranked.append((similarity,record))
        candidates = [r for _,r in sorted(ranked,key=lambda item:-item[0])[:3]]
        if candidates:
            for record, (support, _) in zip(candidates, compare(claim,[r["claim"] for r in candidates])):
                if support >= .97 and compare(record["claim"],[claim])[0][0] >= .97:
                    rows.append(record)
    if not rows:
        return None
    ratings = {r["rating"].lower().strip() for r in rows}
    label = "Real" if ratings == {"true"} else "Fake" if ratings == {"false"} else UNCERTAIN
    if re.search(r"\b(this|these|today|yesterday|now|currently|footage|video|photo)\b",claim,re.I) or (
            TIME_OR_MEDIA.search(claim) and not re.search(r"\b(?:19|20)\d{2}\b", claim)):
        label = UNCERTAIN
    items = [dict(url=r["url"], title=f'Publisher rating: {r["rating"]}', snippet=r["claim"],
                  source=urlparse(r["url"]).hostname, date=r["date"]) for r in rows]
    return outcome(label, "Published review of the same wording" if exact else "Published review of a closely matching claim", items,
                   "The rating applies to the original claim at the linked date. Image identity and later events require separate evidence.")


def official_office(claim):
    match = re.fullmatch(r"([a-z][a-z .'-]{1,45}?) is (not )?(?:currently )?(?:the |a )?(?:current )?(?:prime minister|pm) of india[.!]?", claim.strip(), re.I)
    if not match:
        return None
    url = "https://www.pmindia.gov.in/en/"
    # Use dated announcements identifying the office holder, never an old inauguration.
    doc = page(url)
    for date, body in re.findall(r'<span class="date">([^<]+)</span>\s*<p>(.*?)</p>', doc, re.S):
        try:
            age = (datetime.now(timezone.utc).date() - datetime.strptime(date.strip(), "%d %b, %Y").date()).days
        except ValueError:
            continue
        sentence = " ".join(html.unescape(re.sub("<[^>]+>", " ", body)).split())
        holder = re.match(r"The Prime Minister,\s*(?:Shri |Smt\.? )?([A-Z][\w'-]+(?: [A-Z][\w'-]+){1,3})\s+(?:has|will|said|met|addressed)\b", sentence)
        if holder and 0 <= age <= 14:
            person = match[1].casefold()
            name = holder[1].casefold()
            if person not in (name, name.split()[-1]):
                return None
            item = dict(url=url,title="Prime Minister's Office, India",snippet=sentence,source="PM India",date=date)
            return outcome("Fake" if match[2] else "Real", "Recent official statement", [item], sentence)
    return None


def wiki_passages(claim):
    api = "https://en.wikipedia.org/w/api.php"
    params = dict(action="query", format="json", generator="search", gsrsearch=claim,
                  gsrlimit=4, prop="extracts", exintro=1, explaintext=1, exsentences=6, exlimit=4)
    response = requests.get(api, params=params, headers=HEADERS, timeout=(4, 7))
    response.raise_for_status()
    rows = []
    terms = tokens(claim)
    for obj in response.json().get("query", {}).get("pages", {}).values():
        title = obj["title"]
        if "disambiguation" in title.casefold():
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", obj.get("extract",""))[:7]:
            sentence = " ".join(sentence.split())
            if not 25 <= len(sentence) <= 800:
                continue
            overlap = len(terms & tokens(sentence))
            if overlap < min(2, len(terms)) or overlap == 0:
                continue
            rows.append((overlap, dict(title=title, snippet=sentence,
                source="Wikipedia", url="https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")))))
    rows.sort(key=lambda pair: -pair[0])
    return [item for _, item in rows[:8]]


def local_passages(claim):
    path = Path(__file__).resolve().parent / "data" / "simplewiki.sqlite"
    terms = tokens(claim)
    if not path.exists() or not terms:
        return []
    query = " OR ".join(f'"{t}"' for t in sorted(terms))
    conn = sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
    try:
        pages = conn.execute("select title,body,url from pages where pages match ? order by bm25(pages,8,1,0) limit 7",(query,)).fetchall()
    finally:
        conn.close()
    results=[]
    for order,(title,body,url) in enumerate(pages):
        if not tokens(title) & terms or "may refer to" in body[:100]:
            continue
        for idx, sentence in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+",body)):
            sentence=sentence.strip()
            common=len(terms & tokens(sentence))
            if 25<=len(sentence)<=650 and common>=min(2,len(terms)):
                weight=common/len(terms)+.15*(1-order/7)+.08/(1+idx)
                results.append((weight,dict(title=title,snippet=sentence,url=url,
                    source="Simple English Wikipedia (archived)",date="2023-11-01")))
    return [r for _,r in sorted(results,key=lambda pair:-pair[0])[:7]]


def compare(claim, passages):
    with INFERENCE_LOCK, torch.inference_mode():
        tokenizer, model, ids = _load()
        batch = tokenizer(passages, [claim]*len(passages), padding=True, truncation=True,
                          max_length=384, return_tensors="pt").to(next(model.parameters()).device)
        probs = model(**batch).logits.softmax(-1).cpu()
    return [(float(row[ids["supports"]]), float(row[ids["refutes"]])) for row in probs]


def check_news(claim):
    reviewed = reviewed_match(claim)
    if reviewed:
        return reviewed
    try:
        office = official_office(claim)
        if office:
            return office
    except (requests.RequestException, ValueError):
        pass
    recent = TIME_OR_MEDIA.search(claim) or re.search(r"\b(this|these|here|last night|announc(?:es|ed)|resign(?:s|ed)|elected|minister|CEO|will|just|killed|arrested|explosion|earthquake|outbreak|launched|reported|20(?:2[4-9]|[3-9]\d))\b",claim,re.I)
    if recent:
        from news_sources import screen_recent
        return screen_recent(claim)
    local = local_passages(claim)
    try:
        live = wiki_passages(claim)
    except (requests.RequestException, ValueError):
        live = []
    items = local + live
    if not items:
        return outcome(UNCERTAIN, "No relevant reference passage found.")
    scores = compare(claim, [i["snippet"] for i in items])
    supported, contradicted = [], []
    for item, (support, refute) in zip(items, scores):
        subject = re.split(r"\b(?:is|was|are|were|has|have|had|can|did|does)\b",claim,1,flags=re.I)[0]
        if len(subject)<len(claim) and tokens(subject) and not tokens(subject) & tokens(item["snippet"]):
            continue
        if support >= .95:
            supported.append(item)
        if refute >= .97:
            contradicted.append(item)
    if bool(supported) != bool(contradicted):
        hits = supported or contradicted
        return outcome("Real" if supported else "Fake",
                       "Reference passage supports the claim" if supported else "Reference passage contradicts the claim",
                       hits, hits[0]["snippet"])
    return outcome(UNCERTAIN, "The reference passages were inconclusive or disagreed.", items[:2])


SUBJECT = r"(?:cat|dog|bird|horse|cow|sheep|elephant|zebra|giraffe|bear|person|man|woman|child|boy|girl|car|bus|truck|train|boat|bicycle|motorcycle|chair|sofa|couch|bench|table|book|cup|bottle|ball|kite|umbrella|airplane)"
COLOR = r"(?:red|blue|green|yellow|black|white|brown|orange|gray|grey|pink|purple)"
NOUN = rf"(?:(?:{COLOR}) )?{SUBJECT}"
PLACE = r"(?:sofa|couch|bed|bench|chair|floor|ground|grass|table|road|street|water|sand|snow|sky|field|beach|tree)"


def photo_questions(claim):
    text = normalized(claim)
    text = re.sub(r"^(?:this (?:photo|image|picture) shows |there is )", "", text)
    m = re.fullmatch(rf"(?:a|an|the) ({NOUN})(?: is (standing up|standing|sitting|lying|sleeping|walking|running|flying|parked))?(?: (on|in|under|beside) (?:a |an |the )?({PLACE}))?", text)
    if not m:
        return []
    subject, action, relation, place = m.groups()
    questions = [f"Is there a {subject} in the picture?"]
    if action or relation:
        phrase = " ".join(x for x in [action, relation, "the" if place else None, place] if x)
        questions.append(f"Is the {subject} {phrase}?")
    return questions


@lru_cache(maxsize=1)
def visual_model():
    from transformers import ViltForQuestionAnswering, ViltProcessor
    folder = Path(__file__).resolve().parent / "artifacts" / "vilt-vqa"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (ViltProcessor.from_pretrained(folder,local_files_only=True),
            ViltForQuestionAnswering.from_pretrained(folder,local_files_only=True).to(device).eval())


@lru_cache(maxsize=1)
def caption_model():
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    folder = Path(__file__).resolve().parent / "artifacts" / "blip-itm"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device=="cuda" else torch.float32
    return (BlipProcessor.from_pretrained(folder,local_files_only=True,use_fast=False),
            BlipForImageTextRetrieval.from_pretrained(folder,local_files_only=True,
                torch_dtype=dtype).to(device).eval())


def check_photo(claim, image):
    questions = photo_questions(claim)
    if not questions:
        # The explicit photo mode accepts longer descriptive captions as well.
        # Preserve the complete predicate; dropping a clause changes the question.
        m = re.fullmatch(r"((?:a|an|the|two|three|four|some) [a-z -]{2,55}?) (is|are) ([a-z -]{2,90})",normalized(claim))
        if not m or TIME_OR_MEDIA.search(claim):
            return outcome(UNCERTAIN, "Use a simple visible description, such as “A cat is sitting on a sofa”.")
        questions=[f"{m[2].capitalize()} {m[1]} {m[3]}?"]
    answers = []
    with INFERENCE_LOCK, torch.inference_mode():
        processor, model = visual_model()
        for question in questions:
            batch = processor(image, question, return_tensors="pt").to(next(model.parameters()).device)
            scores = model(**batch).logits[0].sigmoid().cpu()
            top = model.config.id2label[int(scores.argmax())]
            answers.append(top if top in ("yes","no") and float(scores.max()) >= .80 else "unclear")
        processor, matcher = caption_model()
        inputs=processor(images=image,text=claim,return_tensors="pt",truncation=True,max_length=64)
        inputs=inputs.to(matcher.device)
        inputs["pixel_values"]=inputs["pixel_values"].to(matcher.dtype)
        match_score=float(matcher(**inputs).itm_score.softmax(-1)[0,1])
    label = UNCERTAIN
    if answers[0]=="no" and match_score <= .3:
        label = "Fake"
    elif all(a=="yes" for a in answers) and match_score >= .7:
        label = "Real"
    result = outcome(label, "Visible caption contradicted" if label=="Fake" else
                     "Visible caption supported" if label=="Real" else "Visual answers were inconclusive")
    result["reasons"] = [f"{q} — {a.capitalize()}" for q,a in zip(questions, answers)]
    result["reasons"].append("BLIP judged the complete caption a strong match." if match_score>=.7 else
                             "BLIP judged the complete caption a weak match." if match_score<=.3 else
                             "BLIP's full-caption comparison was inconclusive.")
    return result
