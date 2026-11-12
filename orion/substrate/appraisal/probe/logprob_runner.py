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
    bounded = 0.0
    if (yes_lp is None) != (no_lp is None):
        # The other answer fell out of the top-k list: the model was so sure that it is below every
        # listed alternative. Its logprob is at most the lowest one listed, so that is a safe bound
        # (it under-states, never over-states, the margin). Live 2026-11-12, fast lane: 6 of 25 of
        # Juniper's probes had a kind whose ' YES' was outside the top 8, and the whole reply fell
        # back to fixed text scores.
        listed = [float(t["logprob"]) for t in (entry.get("top_logprobs") or [])
                  if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))]
        if not listed:
            return None
        if yes_lp is None:
            yes_lp = min(listed)
        else:
            no_lp = min(listed)
        bounded = 1.0
    if yes_lp is None or no_lp is None:
        return None
    # Margin between the two ANSWERS (each summed over its spellings), not between the raw top-1
    # and top-2 tokens: ' YES' -0.69 / 'YES' -0.72 / ' NO' -5 is a confident YES, while the raw
    # token margin (0.03) read it as a coin flip and min() then dragged the whole slice to ~0.
    # Identical to the old value whenever each answer has one spelling.
    margin = abs(yes_lp - no_lp)
    score = score_binary_logprob(logprob_yes=yes_lp, logprob_no=no_lp)
    return KindProbeScore(
        evidence_kind=kind,
        score=max(0.0, min(1.0, score)),
        confidence=margin,
        features={"logprob_yes": yes_lp, "logprob_no": no_lp, "margin": margin, "bounded": bounded},
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
