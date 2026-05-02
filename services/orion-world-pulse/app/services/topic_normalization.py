from __future__ import annotations

import re
from dataclasses import dataclass

from orion.schemas.world_pulse import ArticleRecordV1

_WEAK_TOKENS = {
    "update",
    "release",
    "news",
    "statement",
    "announces",
    "announcement",
    "latest",
    "daily",
    "weekly",
    "press",
    "article",
    "blog",
    "report",
    "today",
}

_SOURCE_NOISE = {"npr", "bbc", "nasa", "usgs", "who", "cisa", "hugging", "face"}

_EQUIVALENTS = {
    "artificial": "ai",
    "intelligence": "ai",
    "cybersecurity": "security",
    "vulnerability": "security",
    "vulnerabilities": "security",
    "quakes": "earthquake",
    "seismic": "earthquake",
    "fire": "wildfire",
    "voters": "election",
    "vote": "election",
    "senate": "congress",
    "house": "congress",
}

_ACRONYMS = {"ai", "who", "cisa", "kev", "nasa", "usgs", "gpu", "hhs", "cdc"}


@dataclass(frozen=True)
class TopicNormalizationResult:
    normalized_title_tokens: list[str]
    topic_terms: list[str]
    canonical_topic_key: str
    entities_hint: list[str]
    section: str
    region_scope: str
    topic_bucket: str


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm|utc|mdt|mst)?\b", " ", text.lower())
    cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", cleaned)
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    out: list[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in _WEAK_TOKENS or token in _SOURCE_NOISE:
            continue
        token = _EQUIVALENTS.get(token, token)
        if token in _WEAK_TOKENS or token in _SOURCE_NOISE:
            continue
        out.append(token)
    return out


def normalize_topic(article: ArticleRecordV1, section: str) -> TopicNormalizationResult:
    tokens = _tokenize(article.title)
    excerpt_tokens = _tokenize(article.text_excerpt or "")
    merged = tokens + [t for t in excerpt_tokens if t not in tokens]
    strong_terms = [t for t in merged if len(t) >= 3 or t in _ACRONYMS][:8]
    if len(strong_terms) < 2:
        fallback = [section, article.source_id]
        strong_terms = [t for t in fallback if t][:2]
    entities_hint = [t.upper() for t in strong_terms if t in _ACRONYMS][:4]
    top_terms = sorted(set(strong_terms))[:5]
    bucket = top_terms[0] if top_terms else section
    key = f"{section}|{article.region_scope}|{','.join(top_terms[:3])}"
    return TopicNormalizationResult(
        normalized_title_tokens=tokens[:10],
        topic_terms=top_terms,
        canonical_topic_key=key,
        entities_hint=entities_hint,
        section=section,
        region_scope=str(article.region_scope),
        topic_bucket=bucket,
    )

