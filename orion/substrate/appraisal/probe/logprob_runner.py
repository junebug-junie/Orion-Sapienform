from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

from orion.substrate.appraisal.models import EvidenceKind, RepairEvidenceV1


DETECTOR_NAME = "logprob_probe_v2"
_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*([a-z_]+)\s*:\s*(YES|NO)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _new_evidence_id() -> str:
    return f"ev_{uuid.uuid4().hex[:16]}"


def parse_yes_no_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _LINE_RE.finditer(text or ""):
        out[m.group(1).lower()] = m.group(2).upper()
    return out


def score_binary_logprob(*, logprob_yes: float, logprob_no: float) -> float:
    """Spec: score = sigmoid(logprob_YES - logprob_NO)."""
    delta = float(logprob_yes) - float(logprob_no)
    return 1.0 / (1.0 + math.exp(-delta))


def _top1_margin(entry: dict[str, Any]) -> float | None:
    tops = entry.get("top_logprobs")
    if not isinstance(tops, list) or len(tops) < 2:
        return None
    lps = [float(t["logprob"]) for t in tops if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))]
    if len(lps) < 2:
        return None
    lps.sort(reverse=True)
    return lps[0] - lps[1]


def _logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def _yes_no_logprobs(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    """log P(YES) and log P(NO) at the answer position, summed over spellings.

    A real alternatives list holds several spellings of one answer (live, Strata 2026-11-12:
    ``' NO' -0.54, ' YES' -0.88, ' no' -8.08``). Before this, the last spelling seen won, so the
    rare ``' no'`` replaced the real ``' NO'`` and a near-even answer read as a confident YES. The
    spellings are disjoint tokens for the same answer, so their probabilities add (log-sum-exp)."""
    yes: list[float] = []
    no: list[float] = []
    tops = entry.get("top_logprobs")
    if isinstance(tops, list):
        for t in tops:
            if not isinstance(t, dict):
                continue
            tok = str(t.get("token") or "").strip().upper()
            lp = t.get("logprob")
            if not isinstance(lp, (int, float)):
                continue
            if tok == "YES":
                yes.append(float(lp))
            elif tok == "NO":
                no.append(float(lp))
    return (_logsumexp(yes) if yes else None), (_logsumexp(no) if no else None)


@dataclass(frozen=True)
class KindProbeScore:
    evidence_kind: EvidenceKind
    score: float
    confidence: float
    features: dict[str, float]


def score_kind_from_answer_token(kind: EvidenceKind, entry: dict[str, Any]) -> KindProbeScore | None:
    yes_lp, no_lp = _yes_no_logprobs(entry)
    if yes_lp is None or no_lp is None:
        return None
    margin = _top1_margin(entry)
    if margin is None:
        return None
    score = score_binary_logprob(logprob_yes=yes_lp, logprob_no=no_lp)
    return KindProbeScore(
        evidence_kind=kind,
        score=max(0.0, min(1.0, score)),
        confidence=margin,
        features={"logprob_yes": yes_lp, "logprob_no": no_lp, "margin": margin},
    )


def kind_probe_to_evidence(scored: KindProbeScore, *, source_molecule_id: str = "turn_window") -> RepairEvidenceV1:
    margin = float(scored.features.get("margin", scored.confidence))
    evidence_confidence = min(1.0, max(0.0, margin / 2.0))
    return RepairEvidenceV1(
        evidence_id=_new_evidence_id(),
        source_molecule_id=source_molecule_id,
        evidence_kind=scored.evidence_kind,
        detector=DETECTOR_NAME,
        score=scored.score,
        confidence=evidence_confidence,
        features=scored.features,
    )
