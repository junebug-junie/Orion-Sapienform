from __future__ import annotations

from typing import Any

from orion.schemas.cockpit_sighting import CockpitHopV1, CockpitStageV1

_HUB_PRODUCER = "orion-hub"


def _base_hop(
    *,
    correlation_id: str,
    seq: int,
    stage: CockpitStageV1,
    visor_line: str,
    status: str,
    summary: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> CockpitHopV1:
    return CockpitHopV1(
        correlation_id=correlation_id,
        seq=seq,
        stage=stage,
        visor_line=visor_line,
        status=status,  # type: ignore[arg-type]
        summary=summary or {},
        raw=raw or {},
        producer=_HUB_PRODUCER,
    )


def hop_from_thought(
    *,
    correlation_id: str,
    seq: int,
    thought: dict[str, Any],
) -> CockpitHopV1:
    disposition = str(thought.get("disposition", "unknown"))
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="stance_decision",
        visor_line=f"stance · {disposition}",
        status="ok",
        summary={"disposition": disposition},
        raw=dict(thought),
    )


def hop_from_motor_step(
    *,
    correlation_id: str,
    seq: int,
    step_index: int,
    step: dict[str, Any],
) -> CockpitHopV1:
    step_name = step.get("name") or step.get("type") or "step"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="motor_hop",
        visor_line=f"motor · {step_name}",
        status="ok",
        summary={"step_index": step_index},
        raw={"step_index": step_index, "step": step},
    )


def _has_draft_appraisal(run: dict[str, Any]) -> bool:
    return bool(run.get("draft_text") or run.get("substrate_appraisal"))


def _has_finalize(run: dict[str, Any]) -> bool:
    return bool(
        run.get("reflection")
        or run.get("final_text")
        or run.get("finalize_ran")
    )


def hop_from_run_artifact(
    *,
    correlation_id: str,
    seq: int,
    run: dict[str, Any],
) -> list[CockpitHopV1]:
    hops: list[CockpitHopV1] = []
    current_seq = seq

    if _has_draft_appraisal(run):
        hops.append(
            _base_hop(
                correlation_id=correlation_id,
                seq=current_seq,
                stage="draft_appraisal",
                visor_line="draft · substrate appraisal",
                status="ok",
                summary={
                    "draft_text_len": len(str(run.get("draft_text") or "")),
                },
                raw=dict(run),
            )
        )
        current_seq += 1

    if _has_finalize(run):
        compliance = run.get("compliance_verdict") or "completed"
        hops.append(
            _base_hop(
                correlation_id=correlation_id,
                seq=current_seq,
                stage="finalize",
                visor_line=f"finalize · {compliance}",
                status="ok",
                summary={"compliance_verdict": compliance},
                raw=dict(run),
            )
        )

    return hops


def hop_from_outcome(
    *,
    correlation_id: str,
    seq: int,
    outcome: dict[str, Any],
) -> CockpitHopV1:
    label = outcome.get("status") or outcome.get("kind") or "outcome"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="closure",
        visor_line=f"outcome · {label}",
        status="ok",
        summary={"outcome": label},
        raw=dict(outcome),
    )


def hop_from_closure(
    *,
    correlation_id: str,
    seq: int,
    closure: dict[str, Any],
) -> CockpitHopV1:
    label = closure.get("status") or closure.get("kind") or "closure"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="closure",
        visor_line=f"closure · {label}",
        status="ok",
        summary={"closure": label},
        raw=dict(closure),
    )


def gap_hop(
    *,
    correlation_id: str,
    seq: int,
    stage: CockpitStageV1,
    deferred_to: str = "slice_b",
) -> CockpitHopV1:
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage=stage,
        visor_line=f"gap · {stage} not recorded ({deferred_to})",
        status="gap",
        summary={"deferred_to": deferred_to},
        raw={},
    )
