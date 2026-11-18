"""Pinned-model mention extractor (spec section 4).

Deterministic by construction: the model is pinned by revision hash, the label
list and threshold are part of ``extractor_version``, inference runs on CPU with
deterministic algorithms, and ``mention_id`` is uuid5 over (source, span,
version), so a replay is a lookup and a version bump is an explicit re-extraction.

``gliner`` and ``torch`` are imported lazily: the linkage reducer and its tests
do not need them, and nothing in a service image imports this module yet (P2 is
a dry run driven by ``scripts/run_person_linkage.py``).

The pin below is the bake-off winner (orion/person_model/evals/run_extractor_bakeoff.py,
results in docs/superpowers/pr-reports/2026-10-14-person-model-p0-p2-pr.md).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from orion.schemas.person_model import PersonMentionV1

_NS = uuid.UUID("6f1d0c52-9a63-4f0e-9a51-6a2f4e1c7b10")

# "person name", not "person": with the bare label "person" GLiNER tags pronouns
# ("I", "you", "me") as people. Bake-off results:
# docs/superpowers/pr-reports/2026-10-14-person-model-p0-p2-pr.md.
PERSON_LABEL = "person name"
RELATION_LABEL = "relationship phrase"


@dataclass(frozen=True)
class ExtractorSpec:
    model_id: str
    revision: str
    labels: Tuple[str, ...]
    threshold: float
    library: str = "gliner==0.2.29"

    @property
    def version(self) -> str:
        label_hash = hashlib.sha256("|".join(self.labels).encode("utf-8")).hexdigest()[:8]
        return f"{self.model_id}@{self.revision[:10]}:labels={label_hash}:t={self.threshold}:{self.library}"


# Bake-off winner (55 labeled prompts, 34 person spans): best F1 with precision
# >= 0.95 is 0.871 (P 0.964, R 0.794) at threshold 0.85, tied with
# dslim/bert-base-NER at the gate and ahead of it on best F1 (0.912 vs 0.875).
# Revision checked on the Hugging Face API 2026-10-14.
PINNED = ExtractorSpec(
    model_id="knowledgator/gliner-multitask-large-v0.5",
    revision="7a95e168036db9ec6f914c0cc6b218edbd87f310",
    labels=(PERSON_LABEL, RELATION_LABEL, "place", "organization"),
    threshold=0.85,
)


def mention_id(source_kind: str, source_id: str, start: int, end: int, version: str) -> str:
    return str(uuid.uuid5(_NS, f"{source_kind}|{source_id}|{start}|{end}|{version}"))


def spans_to_mentions(
    spans: Iterable[dict], *, text: str, source_kind: str, source_id: str, version: str
) -> List[PersonMentionV1]:
    """Model-agnostic: turn ``{start, end, label, score}`` dicts into sorted mentions."""
    out: List[PersonMentionV1] = []
    for sp in spans:
        start, end = int(sp["start"]), int(sp["end"])
        out.append(
            PersonMentionV1(
                mention_id=mention_id(source_kind, source_id, start, end, version),
                source_kind=source_kind,
                source_id=source_id,
                span_start=start,
                span_end=end,
                text=text[start:end],
                label=str(sp["label"]),
                # Rounded so float noise in the last bits cannot change a stored row.
                score=round(float(sp["score"]), 4),
                extractor_version=version,
            )
        )
    out.sort(key=lambda m: (m.span_start, m.span_end, m.label))
    return out


@dataclass
class GlinerExtractor:
    spec: ExtractorSpec = PINNED
    _model: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._model is None:
            import torch  # lazy: heavy, only for extraction runs
            from gliner import GLiNER

            torch.use_deterministic_algorithms(True)
            torch.manual_seed(0)
            self._model = GLiNER.from_pretrained(self.spec.model_id, revision=self.spec.revision, map_location="cpu")
            self._model.eval()
        return self._model

    def raw_spans(self, text: str, labels: Optional[Sequence[str]] = None) -> List[dict]:
        if not text.strip():
            return []
        model = self._load()
        import torch

        with torch.no_grad():
            ents = model.predict_entities(text, list(labels or self.spec.labels), threshold=self.spec.threshold)
        return [{"start": e["start"], "end": e["end"], "label": e["label"], "score": e["score"]} for e in ents]

    def extract(self, text: str, *, source_kind: str, source_id: str) -> List[PersonMentionV1]:
        return spans_to_mentions(
            self.raw_spans(text), text=text, source_kind=source_kind, source_id=source_id, version=self.spec.version
        )

    def whole_label_is_person(self, label: str) -> bool:
        """True when the extractor reads the entire label as one person name.

        Labels in the graph are often lowercased; title-casing is a fixed,
        deterministic transform so "amy smith" is read as a name would be
        written. Used as one linkage feature, never as a decider.
        """
        from orion.person_model.normalize import strip_accents

        text = " ".join(w[:1].upper() + w[1:] for w in strip_accents(label).split())
        for sp in self.raw_spans(text, labels=(PERSON_LABEL,)):
            if sp["start"] == 0 and sp["end"] >= len(text.rstrip()):
                return True
        return False
