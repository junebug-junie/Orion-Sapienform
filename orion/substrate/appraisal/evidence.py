"""Deterministic phrase-match detector for repair evidence.

No LLM. No embeddings. Every signal is explainable as a phrase hit with a
clear feature vector.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from orion.substrate.molecules import SubstrateMoleculeV1

from .models import EvidenceKind, RepairEvidenceV1


DETECTOR_NAME = "phrase_match_v1"


@dataclass(frozen=True)
class _Phrase:
    pattern: str  # case-insensitive plain substring OR regex if regex=True
    weight: float  # phrase_match_strength contribution
    regex: bool = False


# Phrase tables — one per evidence kind. Source: spec §7.2.
_PHRASES: dict[EvidenceKind, tuple[_Phrase, ...]] = {
    "specificity_demand": (
        _Phrase("nuts and bolts", 0.95),
        _Phrase("operationalize", 0.85),
        _Phrase("concrete", 0.70),
        _Phrase("exactly how", 0.80),
        _Phrase("from a to z", 0.85),
        _Phrase("not hand wavy", 0.90),
        _Phrase("design spec", 0.85),
        _Phrase("implementation", 0.65),
        _Phrase("what files", 0.80),
        _Phrase("how the hell", 0.75),
        _Phrase("stop being vague", 0.90),
    ),
    "trust_rupture": (
        _Phrase("you gave me garbage", 1.0),
        _Phrase("garbage directions", 0.97),
        _Phrase("checked out", 0.85),
        _Phrase("bullshit", 0.85),
        _Phrase("fuckery", 0.85),
        _Phrase("bad directions", 0.85),
        _Phrase("not useful", 0.70),
        _Phrase("stop fucking me around", 0.95),
    ),
    "coherence_gap": (
        _Phrase("swamp", 0.92),
        _Phrase("hand wavy", 0.80),
        _Phrase("doesn't converge", 0.85),
        _Phrase("bolting on", 0.80),
        _Phrase(r"\bugly\b", 0.65, regex=True),
        _Phrase("drift", 0.70),
        _Phrase("making shit up", 0.85),
        _Phrase("not the paradigm", 0.85),
    ),
    "repetition_failure": (
        _Phrase(r"\bagain\b", 0.70, regex=True),
        _Phrase(r"\bstill\b", 0.55, regex=True),
        _Phrase("you keep", 0.85),
        _Phrase("y'all been", 0.75),
        _Phrase("previous", 0.55),
        _Phrase("same thing", 0.75),
        _Phrase("more garbage", 0.85),
    ),
    "operational_block": (
        _Phrase("i need to figure out", 0.80),
        _Phrase("i need this to plug in", 0.85),
        _Phrase("i need a design", 0.85),
        _Phrase("so i can build", 0.80),
        _Phrase("pass off to claude", 0.85),
        _Phrase("what is needed", 0.70),
        _Phrase("design spec for claude", 0.90),
        _Phrase("build me a design spec", 0.90),
    ),
    "explicit_repair_command": (
        _Phrase(r"\bstop\b", 0.75, regex=True),
        _Phrase(r"\bfocus\b", 0.70, regex=True),
        _Phrase("try again", 0.75),
        _Phrase("back up", 0.70),
        _Phrase(r"\bdo not\b", 0.70, regex=True),
        _Phrase("arsonist pov only", 0.95),
        _Phrase("arsonist pov", 0.85),
        _Phrase("no hand waving", 0.85),
    ),
    "assistant_accountability_demand": (
        _Phrase("you are", 0.55),
        _Phrase("you gave", 0.75),
        _Phrase("your work", 0.70),
        _Phrase("your directions", 0.80),
        _Phrase("you keep", 0.80),
        _Phrase("checked out", 0.80),
    ),
}


# Imperative cues raise confidence + score independently of phrase weight.
_IMPERATIVE_PAT = re.compile(
    r"\b(stop|focus|do not|don't|give me|build me|need|must)\b", re.IGNORECASE
)


def _new_evidence_id() -> str:
    return f"ev_{uuid.uuid4().hex[:16]}"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _phrase_hit(text_lower: str, phrase: _Phrase) -> tuple[bool, str | None]:
    """Return (matched, span_text_or_None)."""
    if phrase.regex:
        m = re.search(phrase.pattern, text_lower, re.IGNORECASE)
        if m:
            return True, m.group(0)
        return False, None
    idx = text_lower.find(phrase.pattern.lower())
    if idx >= 0:
        # Return the matched span as-is.
        return True, text_lower[idx : idx + len(phrase.pattern)]
    return False, None


def _extract_text(molecule: SubstrateMoleculeV1) -> str | None:
    payload = molecule.payload or {}
    for key in ("surface_text", "claim_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _score_for_kind(
    text: str,
    kind: EvidenceKind,
    salience: float,
) -> tuple[float, float, str | None, dict[str, float]] | None:
    """Return (score, confidence, span, features) for the strongest hit, or None."""
    text_lower = text.lower()
    best_weight = 0.0
    best_span: str | None = None
    for phrase in _PHRASES[kind]:
        matched, span = _phrase_hit(text_lower, phrase)
        if matched and phrase.weight > best_weight:
            best_weight = phrase.weight
            best_span = span
    if best_weight == 0.0:
        return None

    imperative_strength = 1.0 if _IMPERATIVE_PAT.search(text) else 0.0
    # "Repetition marker" is a coarse signal that the user is re-asserting.
    repetition_marker = 1.0 if re.search(r"\b(again|still|keep)\b", text_lower) else 0.0

    score = _clamp01(
        0.78 * best_weight
        + 0.12 * imperative_strength
        + 0.06 * repetition_marker
        + 0.04 * salience
    )

    # Confidence: high for direct phrase, lower for regex word-boundary cues.
    confidence = _clamp01(0.50 + 0.35 * best_weight + 0.10 * imperative_strength)

    features = {
        "phrase_weight": best_weight,
        "imperative_strength": imperative_strength,
        "repetition_marker": repetition_marker,
        "substrate_salience": salience,
    }
    return score, confidence, best_span, features


def extract_repair_evidence(
    molecules: Iterable[SubstrateMoleculeV1],
) -> list[RepairEvidenceV1]:
    """Return all repair evidence found across a sequence of molecules.

    Each (molecule, evidence_kind) pair yields at most one RepairEvidenceV1 —
    the strongest phrase match for that kind in that molecule.
    """

    out: list[RepairEvidenceV1] = []
    for molecule in molecules:
        text = _extract_text(molecule)
        if not text:
            continue
        salience = float(molecule.gradients.get("salience", 0.0))
        for kind in _PHRASES.keys():
            scored = _score_for_kind(text, kind, salience)
            if scored is None:
                continue
            score, confidence, span, features = scored
            out.append(
                RepairEvidenceV1(
                    evidence_id=_new_evidence_id(),
                    source_molecule_id=molecule.molecule_id,
                    evidence_kind=kind,
                    detector=DETECTOR_NAME,
                    score=score,
                    confidence=confidence,
                    span=span,
                    features=features,
                )
            )
    return out
