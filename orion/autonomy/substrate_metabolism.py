from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import yaml

from orion.autonomy.models import MetabolismResultV1
from orion.core.schemas.drives import ArtifactProvenance, TensionEventV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.world_pulse import SectionRollupV1, WorldPulseRunResultV1
from orion.signals.models import OrionSignalV1

_TRUTHY = {"1", "true", "yes", "on"}
_GAP_STATUSES = frozenset({"missing", "no_articles", "source_unavailable"})
_PREDICTIVE_DELTA = 0.15
_RECOMMENDED_STRENGTH = 0.65
_DEFAULT_STRENGTH = 0.45


def metabolism_enabled() -> bool:
    return str(os.getenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "false")).strip().lower() in _TRUTHY


def _load_recommended_sections() -> set[str]:
    path = Path(__file__).resolve().parents[2] / "config" / "world_pulse" / "sources.yaml"
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return set(data.get("recommended_sections") or [])


def _gap_strength(section: str, recommended: set[str]) -> float:
    return _RECOMMENDED_STRENGTH if section in recommended else _DEFAULT_STRENGTH


def _tension_from_gap(*, section: str, run_id: str, strength: float) -> TensionEventV1:
    return TensionEventV1(
        artifact_id=f"tension-gap-{section}-{run_id}"[:80],
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="substrate.world_coverage_gap",
        magnitude=strength,
        drive_impacts={"predictive": _PREDICTIVE_DELTA},
        provenance=ArtifactProvenance(
            intake_channel="orion:world_pulse:run:result",
            correlation_id=run_id,
            evidence_summary=f"section {section} digest_item_count=0",
        ),
        related_nodes=[f"world_pulse:section:{section}"],
    )


def _signal_from_gap(*, section: str, run_id: str, strength: float) -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=[f"section:{section}"],
        signal_strength=strength,
        evidence_summary=f"world coverage gap: {section} had zero digest items",
        confidence=strength,
        notes=[f"run_id:{run_id}", "source:substrate_metabolism"],
    )


def _gaps_from_rollups(
    rollups: Sequence[SectionRollupV1],
    *,
    run_id: str,
    recommended: set[str],
) -> tuple[list[TensionEventV1], list[FrontierInvocationSignalV1], dict[str, float]]:
    tensions: list[TensionEventV1] = []
    signals: list[FrontierInvocationSignalV1] = []
    deltas: dict[str, float] = {}
    for rollup in rollups:
        if int(rollup.digest_item_count or 0) != 0:
            continue
        status = str(rollup.status or "missing")
        if status == "covered":
            continue
        if status not in _GAP_STATUSES and status != "missing":
            continue
        section = str(rollup.section or "").strip()
        if not section:
            continue
        strength = _gap_strength(section, recommended)
        tensions.append(_tension_from_gap(section=section, run_id=run_id, strength=strength))
        signals.append(_signal_from_gap(section=section, run_id=run_id, strength=strength))
        deltas["predictive"] = min(1.0, deltas.get("predictive", 0.0) + _PREDICTIVE_DELTA)
    return tensions, signals, deltas


def metabolize_substrate_signals(
    *,
    signals: Sequence[OrionSignalV1] = (),
    molecules: Sequence[object] | None = None,
    world_pulse_result: WorldPulseRunResultV1 | None = None,
) -> MetabolismResultV1:
    if world_pulse_result is None:
        return MetabolismResultV1()

    digest = world_pulse_result.digest
    if digest is None:
        return MetabolismResultV1()

    run_id = str(world_pulse_result.run.run_id)
    recommended = _load_recommended_sections()
    tensions, curiosity_signals, drive_deltas = _gaps_from_rollups(
        digest.section_rollups,
        run_id=run_id,
        recommended=recommended,
    )
    return MetabolismResultV1(
        drive_deltas=drive_deltas,
        tensions=tensions,
        curiosity_signals=curiosity_signals,
    )
