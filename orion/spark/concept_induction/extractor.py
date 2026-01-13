from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import spacy
from spacy.language import Language


@dataclass
class ExtractionResult:
    candidates: List[str]
    debug: dict


_WORD_RE = re.compile(r"[A-Za-z][\w'-]+")


class SpacyConceptExtractor:
    """spaCy-first concept candidate extractor with graceful fallback."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self.nlp: Language = self._load_model(model_name)

    @staticmethod
    def _load_model(name: str) -> Language:
        try:
            return spacy.load(name)
        except Exception:
            return spacy.blank("en")

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _chunk_candidates(self, doc) -> List[str]:
        labels: List[str] = []
        for ent in getattr(doc, "ents", []):
            if ent.label_:
                labels.append(ent.text)
        if getattr(doc, "noun_chunks", None):
            labels.extend(chunk.text for chunk in doc.noun_chunks)
        return labels

    def _regex_candidates(self, text: str) -> List[str]:
        return _WORD_RE.findall(text)

    def extract(self, texts: Iterable[str], *, top_k: int = 50) -> ExtractionResult:
        freq: dict[str, float] = {}
        debug_meta: dict[str, dict[str, float]] = {}
        for idx, text in enumerate(texts):
            if not text:
                continue
            doc = self.nlp(text)
            raw_candidates = self._chunk_candidates(doc)
            if not raw_candidates:
                raw_candidates = self._regex_candidates(text)
            normed = [self._normalize(c) for c in raw_candidates if c]
            weight = 1.0 + (idx * 0.01)
            for cand in normed:
                freq[cand] = freq.get(cand, 0.0) + weight
                meta = debug_meta.setdefault(cand, {"count": 0.0})
                meta["count"] += weight

        sorted_items: List[Tuple[str, float]] = sorted(
            freq.items(), key=lambda kv: kv[1], reverse=True
        )
        top_candidates = [c for c, _ in sorted_items[:top_k]]
        return ExtractionResult(candidates=top_candidates, debug={"freq": debug_meta})
