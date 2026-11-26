"""Screen public claims against referenced records, and captions against the photo."""
import html
import json
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from urllib.parse import quote, urlparse
from html.parser import HTMLParser

import requests
import torch

from evidence_verification import _load
from reviewed_claims import HEADERS, page, read_index, refresh_if_stale, search_reviews
from publisher_index import candidates as indexed_reviews

UNCERTAIN = "Possibly misleading / uncertain"
QUALIFIERS = re.compile(r"\b(?:some|all|every|always|never|only|most|prolonged|healthy|sometimes|allegedly|possibly|if|unless|except)\b",re.I)
INFERENCE_LOCK = RLock()
STOP = set("a an and are as at be by did do does for from has have in is it of on or that the this to was were with not".split())
TIME_OR_MEDIA = re.compile(r"\b(today|yesterday|tomorrow|currently|latest|now|this (?:week|month|year)|breaking|president|prime minister|photo|footage|pictured|video|image|spotted|sighting)\b", re.I)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def normalized(text):
    # Preserve numbers, negation and internal punctuation in quoted claims.
    return " ".join(text.casefold().translate(str.maketrans({"’":"'","‘":"'","“":'"',"”":'"'})).split()).strip(' ."')


def outcome(label, basis, items=(), detail=""):
    return {"label": label, "decision_basis": basis, "reasons": [detail or basis],
            "evidence": {"provider": "Source passages", "items": list(items), "message": ""},
            "evidence_check": None, "probabilities": [], "similarity": None}


@lru_cache(maxsize=1)
def historical_reviews():
    path = Path(__file__).resolve().parent / "data" / "averitec_archive.json"
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else []


def reviewed_match(claim):
    refresh_if_stale()
    terms = tokens(claim)
    local = read_index() + historical_reviews() + indexed_reviews(terms)
    queries = [claim]
    if len(terms)>5 and not any(normalized(r["claim"])==normalized(claim) for r in local):
        names = re.findall(r"\b[A-Z][a-zA-Z'-]{2,}\b",claim)
        names = [n for n in names if n.lower() in terms and n.lower() not in {"there","since","while","according"}]
        anchors = sorted(terms-{n.lower() for n in names},key=lambda w:(-len(w),w))
        short = " ".join(dict.fromkeys(names[:3]+anchors[:2]))
        if short and short!=claim:
            queries.append(short)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = local + [r for result in pool.map(search_reviews,queries) for r in result]
    records = list({(r["url"],r["rating"],normalized(r["claim"]),r["date"]):r for r in records}.values())
    rows = [r for r in records if normalized(r["claim"]) == normalized(claim)]
    exact = bool(rows)
    if not rows:
        # Only transfer a published rating across a close paraphrase if both
        # directions entail each other and numeric qualifications are preserved.
        terms = tokens(claim)
        numbers = re.findall(r"\d+(?:[.,]\d+)*%?",claim)
        ranked = []
        for record in records:
            if set(QUALIFIERS.findall(claim.casefold())) != set(QUALIFIERS.findall(record["claim"].casefold())):
                continue
            other = tokens(record["claim"])
            similarity = len(terms & other) / max(1,len(terms | other))
            if similarity >= .30 and len(terms & other)>=3 and sorted(numbers)==sorted(re.findall(r"\d+(?:[.,]\d+)*%?",record["claim"])):
                ranked.append((similarity,record))
        candidates = [r for _,r in sorted(ranked,key=lambda item:-item[0])[:10]]
        if candidates:
            for record, (support, _) in zip(candidates, compare(claim,[r["claim"] for r in candidates])):
                if support >= .97 and compare(record["claim"],[claim])[0][0] >= .97:
                    rows.append(record)
    if not rows:
        return None
    def verdict(rating):
        head = re.split(r"[.:!\n]|\s+[–—]",rating.casefold().strip(),maxsplit=1)[0].strip()
        if head in {"true","correct","accurate","supported","verdadero"}:
            return "Real"
        if head in {"false","incorrect","inaccurate","refuted","falso","pants on fire","pants-fire","fabricated","fake"}:
            return "Fake"
        return UNCERTAIN
    labels = set()
    for r in rows:
        rating = r["rating"]
        mapped = verdict(rating)
        # Some publishers write a short explanation in their rating field.
        # A qualified verdict or an absence of evidence must retain its meaning.
        if (mapped==UNCERTAIN and len(tokens(rating))>=6 and not re.search(
            r"\b(mostly|partly|partially|mixed|half|cherry|misleading|unproven|unclear|uncertain|unverified|evidence|exaggerated|context)\b",
            rating,re.I)):
            if relevant_passages(claim,[{"snippet":rating}]):
                support,refute = compare(claim,[rating])[0]
                if support>=.97 or refute>=.98:
                    mapped = "Real" if support>refute else "Fake"
        labels.add(mapped)
    label = next(iter(labels)) if len(labels)==1 else UNCERTAIN
    relative = re.search(r"\b(today|yesterday|tomorrow|now|currently|recently|last (?:week|month|year)|this (?:week|month|year)|ago|these|this (?:image|photo|clip|video)|footage|video|photo)\b",claim,re.I)
    current_role = re.search(r"\b(?:is|are|serves? as)\b.{0,60}\b(?:president|director|head|officer|minister|governor|mayor|CEO|leader)\b",claim,re.I)
    if relative or (current_role and not re.search(r"\b(?:19|20)\d{2}\b",claim)):
        label = UNCERTAIN
    items = [dict(url=r["url"], title=f'{r.get("archive","Publisher")} rating: {r["rating"]}', snippet=r["claim"],
                  source=urlparse(r["url"]).hostname, date=r["date"]) for r in rows]
    return outcome(label, "Published review of the same wording" if exact else "Published review of a closely matching claim", items,
                   "The rating applies to the original claim at the linked date. Image identity and later events require separate evidence.")


@lru_cache(maxsize=1)
def passage_ranker():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = Path(__file__).resolve().parent/"artifacts/passage-ranker"
    model = AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True)
    return AutoTokenizer.from_pretrained(path,local_files_only=True), model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def relevant_passages(claim, items):
    if not items:
        return []
    with INFERENCE_LOCK, torch.inference_mode():
        tokenizer, model = passage_ranker()
        batch = tokenizer([claim]*len(items), [i["snippet"] for i in items],
                          padding=True,truncation=True,max_length=384,return_tensors="pt").to(model.device)
        logits = model(**batch).logits.flatten().tolist()
    return [i for score,i in sorted(zip(logits,items),key=lambda pair:-pair[0]) if score>0][:6]


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


class WikiParagraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current = None
        self.rows = []

    def handle_starttag(self,tag,attrs):
        if tag=="p":
            self.current=[]

    def handle_data(self,data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self,tag):
        if tag=="p" and self.current is not None:
            self.rows.append(re.sub(r"\[\d+\]","","".join(self.current)))
            self.current=None


def wiki_passages(claim):
    subject = re.split(r"\b(?:is|was|are|were|has|had|starred|features|died)\b",claim,1,flags=re.I)[0].strip()
    response = requests.get("https://api.wikimedia.org/core/v1/wikipedia/en/search/page",
                            params={"q":subject or claim,"limit":4},headers=HEADERS,timeout=(3,5))
    response.raise_for_status()
    def passages(candidate):
        try:
            url = "https://api.wikimedia.org/core/v1/wikipedia/en/page/" + quote(candidate["key"],safe="") + "/html"
            with requests.get(url,headers=HEADERS,timeout=(3,5),stream=True) as r:
                r.raise_for_status()
                buf = bytearray()
                for chunk in r.iter_content(65536):
                    buf.extend(chunk)
                    if len(buf)>=2_000_000:
                        break
            parser=WikiParagraphs()
            parser.feed(buf.decode("utf8","replace"))
            return dict(title=candidate["title"],extract="\n".join(parser.rows))
        except (requests.RequestException, ValueError):
            return {}
    rows = []
    terms = tokens(claim)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(passages,response.json().get("pages",[])[:3]))
    for obj in filter(None,pages):
        title = obj["title"]
        if "disambiguation" in title.casefold():
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", obj.get("extract","")):
            sentence = " ".join(sentence.split())
            if not 25 <= len(sentence) <= 800:
                continue
            overlap = len(terms & tokens(sentence))
            if overlap < min(2, len(terms)) or overlap == 0:
                continue
            rows.append((overlap, dict(title=title, snippet=sentence,
                source="Wikipedia", url="https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")))))
    rows.sort(key=lambda pair: -pair[0])
    return [item for _, item in rows[:14]]


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
    try:
        office = official_office(claim)
        if office:
            return office
    except (requests.RequestException, ValueError):
        pass
    from structured_facts import check as check_record
    structured = check_record(claim)
    if structured:
        return structured
    reviewed = reviewed_match(claim)
    if reviewed:
        return reviewed
    recent = TIME_OR_MEDIA.search(claim) or re.search(r"\b(this|these|here|last night|announc(?:es|ed)|resign(?:s|ed)|elected|minister|CEO|will|just|killed|arrested|explosion|earthquake|outbreak|launched|reported|20(?:2[4-9]|[3-9]\d))\b",claim,re.I)
    from news_sources import relevant_headlines, screen_recent
    feed_urls = relevant_headlines(claim)
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b",claim)]
    historical = years and max(years)<datetime.now(timezone.utc).year-1 and not re.search(
        r"\b(today|now|currently|latest|this|these|video|photo|footage|since|still)\b",claim,re.I)
    if (recent and not historical) or feed_urls:
        return screen_recent(claim, feed_urls)
    local = local_passages(claim)
    try:
        live = wiki_passages(claim)
    except (requests.RequestException, ValueError):
        live = []
    items = local + live
    if not items:
        return outcome(UNCERTAIN, "No relevant reference passage found.")
    passages = relevant_passages(claim,items)
    if not passages:
        return outcome(UNCERTAIN,"No passage directly addresses the claim.",items[:2])
    scores = compare(claim, [i["snippet"] for i in passages])
    supported, contradicted = [], []
    for item, (support, refute) in zip(passages, scores):
        if re.search(r"\b(allegedly|unconfirmed|rumou?rs?|speculat(?:ed|ion)|unsubstantiated|hoax|falsely claimed)\b",item["snippet"],re.I):
            continue
        subject = re.split(r"\b(?:is|was|are|were|has|have|had|can|did|does)\b",claim,1,flags=re.I)[0]
        subject_terms = tokens(subject)
        available = tokens(item["title"]) | tokens(item["snippet"])
        if len(subject)<len(claim) and subject_terms and len(subject_terms & available)/len(subject_terms)<.75:
            continue
        if support >= .95:
            supported.append(item)
        if refute >= .97 and len(tokens(claim) & tokens(item["snippet"])) >= .4*len(tokens(claim)):
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


@lru_cache(maxsize=1)
def object_detector():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    path = Path(__file__).resolve().parent / "artifacts" / "fasterrcnn_v2.pth"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    return model.eval().to("cuda" if torch.cuda.is_available() else "cpu"), weights


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
        simple = re.fullmatch(rf"Is there a ({SUBJECT}) in the picture\?", questions[0])
        if simple and len(questions) == 1:
            detector, weights = object_detector()
            name = {"sofa":"couch"}.get(simple[1], simple[1])
            detection = detector([weights.transforms()(image).to(next(detector.parameters()).device)])[0]
            categories = weights.meta["categories"]
            if any(categories[idx] == name and score >= .75 for idx, score in
                   zip(detection["labels"].tolist(), detection["scores"].tolist())):
                return outcome("Real", "The named object was detected in the photo.",
                               detail=f"The detector located a {name}. This check covers its visible presence.")
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
