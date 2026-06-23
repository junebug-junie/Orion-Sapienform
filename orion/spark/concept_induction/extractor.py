from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import spacy
from spacy.language import Language


@dataclass
class ExtractionResult:
    candidates: List[str]
    debug: dict
    entity_types: Dict[str, str] = field(default_factory=dict)


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

    def _chunk_candidates(self, doc) -> List[Tuple[str, Optional[str]]]:
        """Return (text, ner_label_or_None) pairs; NER entities first, then noun chunks."""
        results: List[Tuple[str, Optional[str]]] = []
        seen: set[str] = set()
        for ent in getattr(doc, "ents", []):
            if ent.label_:
                results.append((ent.text, ent.label_.lower()))
                seen.add(ent.text)
        try:
            for chunk in doc.noun_chunks:
                if chunk.text not in seen:
                    results.append((chunk.text, None))
        except Exception:
            # Blank models or parser-less pipelines cannot provide noun chunks.
            pass
        return results

    def _regex_candidates(self, text: str) -> List[Tuple[str, Optional[str]]]:
        return [(_w, None) for _w in _WORD_RE.findall(text)]

    def extract(self, texts: Iterable[str], *, top_k: int = 50) -> ExtractionResult:
        freq: dict[str, float] = {}
        debug_meta: dict[str, dict[str, float]] = {}
        entity_types: dict[str, str] = {}
        for idx, text in enumerate(texts):
            if not text:
                continue
            doc = self.nlp(text)
            raw_candidates = self._chunk_candidates(doc)
            if not raw_candidates:
                raw_candidates = self._regex_candidates(text)
            weight = 1.0 + (idx * 0.01)
            for raw_text, ner_label in raw_candidates:
                if not raw_text:
                    continue
                cand = self._normalize(raw_text)
                freq[cand] = freq.get(cand, 0.0) + weight
                meta = debug_meta.setdefault(cand, {"count": 0.0})
                meta["count"] += weight
                if ner_label and cand not in entity_types:
                    entity_types[cand] = ner_label

        sorted_items: List[Tuple[str, float]] = sorted(
            freq.items(), key=lambda kv: kv[1], reverse=True
        )
        top_candidates = [c for c, _ in sorted_items[:top_k]]
        return ExtractionResult(
            candidates=top_candidates,
            debug={"freq": debug_meta},
            entity_types=entity_types,
        )
