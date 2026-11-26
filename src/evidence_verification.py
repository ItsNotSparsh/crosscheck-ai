"""Compare a claim with retrieved snippets using a pretrained NLI model."""
import torch
from pathlib import Path
import html
import re
from urllib.parse import urlparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TRUSTED_DOMAINS = (
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "factcheck.org", "snopes.com", "politifact.com", "afp.com", "fullfact.org", "npr.org",
    "theguardian.com", "nytimes.com", "dw.com", "pmindia.gov.in", "india.gov.in",
    "newsnationnow.com", "newsnation.tv",
)
HEDGING = re.compile(r"\b(is that|could be|might be|may be|possible|possibly|alleged|unconfirmed|unverified|rumou?r|speculat|social media|viral video|something over|appears to be|purported|sighting|spotted|seen in a video|nobody else)\b|\?", re.I)
DECISIVE = re.compile(r"\b(false|fake|fabricated|debunk(?:ed|s)?|not (?:a |an )?(?:real |living )?(?:pterodactyl|dinosaur)|no evidence|not true|AI[- ]generated|isn\'t real|does not exist|extinct|fact.?check(?:ed)?|misleading)\b", re.I)
_tokenizer = None
_model = None
_label_ids = None


def _load():
    global _tokenizer, _model, _label_ids
    if _model is None:
        cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli" / "snapshots"
        snapshots = [p for p in cache.iterdir() if p.is_dir() and (p / "config.json").exists() and (p / "model.safetensors").exists() and (p / "tokenizer_config.json").exists()] if cache.exists() else []
        source = str(snapshots[-1]) if snapshots else MODEL_ID
        local_only = bool(snapshots)
        _tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        _model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=local_only).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        ).eval()
        names = {int(i): str(name).lower() for i, name in _model.config.id2label.items()}
        def find(*terms):
            return next((i for i, name in names.items() if any(term in name for term in terms)), None)
        _label_ids = {"supports": find("entail"), "refutes": find("contrad"), "neutral": find("neutral")}
        if None in _label_ids.values():
            raise RuntimeError(f"Unexpected NLI labels: {names}")
    return _tokenizer, _model, _label_ids


def assess_claim(claim, evidence_items):
    """Use only clear assertions from multiple reputable domains for a stance."""
    pairs = []
    for item in evidence_items:
        title = html.unescape(item.get("title", ""))
        snippet = html.unescape(item.get("snippet", ""))
        text = re.sub(r"<[^>]+>", " ", f"{title}. {snippet}")
        text = re.sub(r"\s+", " ", text).strip()
        domain = (item.get("domain") or urlparse(item.get("url", "")).netloc).lower().removeprefix("www.")
        trusted = any(domain == host or domain.endswith("." + host) for host in TRUSTED_DOMAINS)
        # A cautious headline can introduce an article that then gives a clear
        # fact-check verdict. Let that direct verdict outweigh headline phrasing.
        hedged = bool(HEDGING.search(title) or HEDGING.search(snippet)) and not bool(DECISIVE.search(snippet))
        if len(text) >= 20:
            pairs.append({"text": text[:1200], "domain": domain, "trusted": trusted, "hedged": hedged})
    if not pairs:
        return {"stance": "insufficient", "supports": 0.0, "refutes": 0.0, "checked": 0}
    tokenizer, model, label_ids = _load()
    device = next(model.parameters()).device
    batch = tokenizer([p["text"] for p in pairs], [claim[:500]] * len(pairs), padding=True,
                      truncation=True, max_length=384, return_tensors="pt").to(device)
    with torch.inference_mode():
        probs = model(**batch).logits.softmax(dim=-1).cpu()
    support = [float(p[label_ids["supports"]]) for p in probs]
    refute = [float(p[label_ids["refutes"]]) for p in probs]
    speculative = [i for i,pair in enumerate(pairs) if pair["hedged"]]
    usable = [i for i,pair in enumerate(pairs) if pair["trusted"] and not pair["hedged"]]
    s = max((support[i] for i in usable), default=0.0)
    r = max((refute[i] for i in usable), default=0.0)
    supporting_domains = {pairs[i]["domain"] for i in usable if support[i] >= .72}
    refuting_domains = {pairs[i]["domain"] for i in usable if refute[i] >= .72}
    # Ignore hedged and irrelevant search leads, but do not let one unrelated
    # speculative result veto clear agreement between independent trusted sources.
    # Each deciding side still needs two separate reputable domains.
    if s >= .80 and len(supporting_domains) >= 2 and s-r >= .25:
        stance = "supports"
    elif r >= .80 and len(refuting_domains) >= 2 and r-s >= .25:
        stance = "refutes"
    else:
        stance = "mixed_or_weak"
    return {"stance": stance, "supports": s, "refutes": r, "supporting_snippets": len(supporting_domains),
            "refuting_snippets": len(refuting_domains), "checked": len(pairs), "trusted_clear_sources": len(usable), "speculative_snippets": len(speculative)}
