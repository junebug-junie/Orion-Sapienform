"""Deterministic 5a draft finalize appraisal — payload only, no reducer cursor."""

from __future__ import annotations

from orion.schemas.harness_finalize import HarnessDraftMoleculeV1, SubstrateFinalizeAppraisalV1


class FinalizeDraftAppraisalError(ValueError):
    """Raised when appraisal cannot satisfy fail-closed constraints."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _draft_molecule_id(mol: HarnessDraftMoleculeV1) -> str:
    return f"draft:{mol.correlation_id}:{mol.thought_event_id}:{mol.draft_hash}"


def _collect_learning_refs(mol: HarnessDraftMoleculeV1) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _add(ref: str) -> None:
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    for receipt in mol.grammar_receipts:
        if receipt.grammar_event_id:
            _add(f"grammar:{receipt.grammar_event_id}")
        elif receipt.summary.strip():
            _add(f"grammar_step:{receipt.step_index}")

    for node_id in mol.coalition_snapshot.attended_node_ids:
        _add(f"coalition:{node_id}")

    for ref in mol.thought_event.strain_refs:
        _add(f"strain:{ref}")

    for ref in mol.thought_event.evidence_refs:
        _add(f"evidence:{ref}")

    return refs


def _compute_open_loop_pressure(mol: HarnessDraftMoleculeV1) -> float:
    coalition = mol.coalition_snapshot
    open_loop_count = len(coalition.open_loop_ids)
    if (
        coalition.selected_open_loop_id
        and coalition.selected_open_loop_id not in coalition.open_loop_ids
    ):
        open_loop_count += 1
    return _clamp01(open_loop_count * 0.2)


def _compute_surprise_level(
    mol: HarnessDraftMoleculeV1,
    *,
    open_loop_pressure: float,
) -> float:
    """Surprise from grammar receipt trajectory vs coalition snapshot."""
    coalition = mol.coalition_snapshot
    thought = mol.thought_event
    receipts = mol.grammar_receipts

    attended = set(coalition.attended_node_ids)
    referenced = set(thought.evidence_refs) | set(thought.strain_refs)

    if attended:
        coverage_gap = len(attended - referenced) / len(attended)
    else:
        coverage_gap = 0.5 if receipts else 0.0

    if attended:
        step_ratio = len(receipts) / len(attended)
        step_excess = _clamp01(max(0.0, step_ratio - 1.0))
    else:
        step_excess = _clamp01(len(receipts) * 0.15)

    stale_factor = 0.2 if coalition.broadcast_stale else 0.0

    raw = (
        0.35 * coverage_gap
        + 0.25 * step_excess
        + 0.25 * open_loop_pressure
        + stale_factor
    )
    return _clamp01(raw)


def _compute_prediction_error_refs(mol: HarnessDraftMoleculeV1) -> list[str]:
    coalition = mol.coalition_snapshot
    thought = mol.thought_event
    attended = set(coalition.attended_node_ids)
    referenced = set(thought.evidence_refs) | set(thought.strain_refs)

    refs: list[str] = []
    for node_id in sorted(attended - referenced):
        refs.append(f"prediction_error:unattended_coalition:{node_id}")
    for receipt in mol.grammar_receipts:
        if receipt.grammar_event_id:
            refs.append(f"grammar:{receipt.grammar_event_id}")
    return refs


def _compute_strain_shift_refs(mol: HarnessDraftMoleculeV1) -> list[str]:
    attended = set(mol.coalition_snapshot.attended_node_ids)
    return [ref for ref in mol.thought_event.strain_refs if ref not in attended]


def _compute_alignment_hints(
    mol: HarnessDraftMoleculeV1,
    *,
    surprise_level: float,
    open_loop_pressure: float,
) -> list[str]:
    hints: list[str] = []
    if mol.coalition_snapshot.broadcast_stale:
        hints.append("coalition_broadcast_stale")
    if open_loop_pressure >= 0.2:
        hints.append("open_loops_present")
    if surprise_level >= 0.5:
        hints.append("elevated_surprise")
    return hints


def appraise_draft_molecule(mol: HarnessDraftMoleculeV1) -> SubstrateFinalizeAppraisalV1:
    """Appraise a harness draft molecule from inline payload — no reducer wait."""
    learning_refs = _collect_learning_refs(mol)
    if not learning_refs:
        raise FinalizeDraftAppraisalError(
            "learning_refs required; payload yielded none"
        )

    open_loop_pressure = _compute_open_loop_pressure(mol)
    surprise_level = _compute_surprise_level(
        mol,
        open_loop_pressure=open_loop_pressure,
    )

    return SubstrateFinalizeAppraisalV1(
        correlation_id=mol.correlation_id,
        molecule_id=_draft_molecule_id(mol),
        draft_hash=mol.draft_hash,
        surprise_level=surprise_level,
        strain_shift_refs=_compute_strain_shift_refs(mol),
        open_loop_pressure=open_loop_pressure,
        prediction_error_refs=_compute_prediction_error_refs(mol),
        learning_refs=learning_refs,
        alignment_hints=_compute_alignment_hints(
            mol,
            surprise_level=surprise_level,
            open_loop_pressure=open_loop_pressure,
        ),
    )
