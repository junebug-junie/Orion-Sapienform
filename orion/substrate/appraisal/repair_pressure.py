"""Repair pressure reducer — explicit formula v1.

No mystery models, no LLM. Inputs: a window of SubstrateMoleculeV1.
Output: RepairPressureAppraisalV1 with deterministic dimensions.
"""

from __future__ import annotations

import uuid
from typing import Iterable, get_args

from orion.substrate.molecules import SubstrateMoleculeV1

from .evidence import extract_repair_evidence
from .models import EvidenceKind, RepairEvidenceV1, RepairPressureAppraisalV1


_EVIDENCE_KINDS: tuple[EvidenceKind, ...] = tuple(get_args(EvidenceKind))


def _new_appraisal_id() -> str:
    return f"app_{uuid.uuid4().hex[:16]}"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _kind_score(evidence: list[RepairEvidenceV1], kind: EvidenceKind) -> float:
    """Max score across evidence items of this kind. Spec §9.2."""
    return max((e.score for e in evidence if e.evidence_kind == kind), default=0.0)


def _gradient_mean(molecules: list[SubstrateMoleculeV1], key: str) -> float:
    if not molecules:
        return 0.0
    values = [float(m.gradients.get(key, 0.0)) for m in molecules]
    return sum(values) / len(values)


def _molecule_quality(molecules: list[SubstrateMoleculeV1]) -> float:
    if not molecules:
        return 0.0
    valid = 0
    for m in molecules:
        if m.provenance and m.gradients:
            valid += 1
    return valid / len(molecules)


def appraise_repair_pressure(
    molecules: Iterable[SubstrateMoleculeV1],
    *,
    window_id: str,
) -> RepairPressureAppraisalV1:
    """Reduce a window of molecules to a RepairPressureAppraisalV1.

    Layering: this function does NOT touch any chat code, does NOT emit a
    signal, does NOT change any contract. It only computes.
    """

    mol_list = list(molecules)
    evidence = extract_repair_evidence(mol_list)

    # Per-kind aggregation (spec §9.2 — max across detector hits).
    kind_scores: dict[EvidenceKind, float] = {
        kind: _kind_score(evidence, kind) for kind in _EVIDENCE_KINDS
    }

    mean_salience = _gradient_mean(mol_list, "salience")
    mean_contradiction = _gradient_mean(mol_list, "contradiction")
    mean_coherence = _gradient_mean(mol_list, "coherence")
    mean_novelty = _gradient_mean(mol_list, "novelty")

    # Level formula (spec §9.3).
    raw_level = (
        0.22 * kind_scores["specificity_demand"]
        + 0.20 * kind_scores["trust_rupture"]
        + 0.16 * kind_scores["coherence_gap"]
        + 0.12 * kind_scores["repetition_failure"]
        + 0.12 * kind_scores["operational_block"]
        + 0.10 * kind_scores["explicit_repair_command"]
        + 0.08 * kind_scores["assistant_accountability_demand"]
        + 0.10 * mean_salience
        + 0.08 * mean_contradiction
        - 0.10 * mean_coherence
    )
    level = _clamp01(raw_level)

    # Confidence formula (spec §9.4).
    notes: list[str] = []
    if not evidence:
        level = 0.0
        confidence = min(0.25, 0.10 * _molecule_quality(mol_list))
        notes.append("no_repair_evidence")
    else:
        mean_evidence_confidence = sum(e.confidence for e in evidence) / len(evidence)
        kinds_present = {e.evidence_kind for e in evidence}
        evidence_coverage = len(kinds_present) / len(_EVIDENCE_KINDS)
        quality = _molecule_quality(mol_list)
        confidence = _clamp01(
            0.50 * mean_evidence_confidence
            + 0.25 * evidence_coverage
            + 0.15 * min(1.0, len(evidence) / 5.0)
            + 0.10 * quality
        )
        # Single-weak-evidence cap (spec §9.4 fail-closed clause).
        if len(evidence) == 1 and evidence[0].score < 0.65:
            confidence = min(confidence, 0.45)
            notes.append("single_weak_evidence")

    dimensions: dict[str, float] = {
        "level": level,
        "specificity_demand": kind_scores["specificity_demand"],
        "trust_rupture": kind_scores["trust_rupture"],
        "coherence_gap": kind_scores["coherence_gap"],
        "repetition_failure": kind_scores["repetition_failure"],
        "operational_block": kind_scores["operational_block"],
        "explicit_repair_command": kind_scores["explicit_repair_command"],
        "assistant_accountability_demand": kind_scores[
            "assistant_accountability_demand"
        ],
        "confidence": confidence,
        # Optional substrate-derived dimensions (spec §6.2 "Optional dimensions").
        "salience": mean_salience,
        "contradiction": mean_contradiction,
        "coherence": mean_coherence,
        "novelty": mean_novelty,
    }

    causal_ids = [m.molecule_id for m in mol_list]

    return RepairPressureAppraisalV1(
        appraisal_id=_new_appraisal_id(),
        window_id=window_id,
        dimensions=dimensions,
        evidence=evidence,
        causal_molecule_ids=causal_ids,
        confidence=confidence,
        summary=None if not evidence else _summarize(level, kind_scores),
        notes=notes,
    )


def _summarize(level: float, kind_scores: dict[EvidenceKind, float]) -> str:
    top_kinds = sorted(
        ((k, v) for k, v in kind_scores.items() if v > 0.0),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    parts = [f"{k}={v:.2f}" for k, v in top_kinds]
    return f"repair_pressure level={level:.2f}; top: " + ", ".join(parts)
