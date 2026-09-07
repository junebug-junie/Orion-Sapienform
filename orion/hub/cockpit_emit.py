"""Build and publish cockpit hop WS frames for a unified turn."""
from __future__ import annotations

import logging
from typing import Any

from orion.cockpit.builders import (
    extract_mind_quality_fields,
    hop_from_association,
    hop_from_closure,
    hop_from_ingress,
    hop_from_motor_boot,
    hop_from_motor_step,
    hop_from_outcome,
    hop_from_progress,
    hop_from_run_artifact,
    hop_from_situation,
    hop_from_stance_inputs,
    hop_from_thought,
)
from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
from orion.cockpit.publish import publish_cockpit_hop
from orion.cockpit.sequencer import advance_seq, next_seq, reset_seq
from orion.schemas.cockpit_sighting import CockpitHopStatusV1, CockpitHopV1, CockpitStageV1

logger = logging.getLogger("orion.hub.cockpit_emit")


def _hop_frame(correlation_id: str, hop: CockpitHopV1) -> dict[str, Any]:
    return {
        "kind": "cockpit_hop",
        "correlation_id": correlation_id,
        "hop": hop.model_dump(mode="json"),
    }


def timeline_complete_frame(correlation_id: str) -> dict[str, Any]:
    return {"kind": "cockpit_timeline_complete", "correlation_id": correlation_id}


def begin_cockpit_timeline(
    correlation_id: str,
    *,
    ingress: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reset Hub seq ownership and emit the real ingress (user intake) hop once."""
    reset_seq(correlation_id)
    return [
        _hop_frame(
            correlation_id,
            hop_from_ingress(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                ingress=ingress if isinstance(ingress, dict) else {},
            ),
        )
    ]


def emit_progress_hop(
    correlation_id: str,
    *,
    stage: CockpitStageV1,
    status: CockpitHopStatusV1,
    visor_line: str,
    summary: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    producer: str = "orion-hub",
) -> dict[str, Any]:
    hop = hop_from_progress(
        correlation_id=correlation_id,
        seq=next_seq(correlation_id),
        stage=stage,
        status=status,
        visor_line=visor_line,
        summary=summary,
        raw=raw,
        producer=producer,
    )
    return _hop_frame(correlation_id, hop)


def emit_association_hop(
    correlation_id: str,
    association: dict[str, Any],
) -> dict[str, Any]:
    hop = hop_from_association(
        correlation_id=correlation_id,
        seq=next_seq(correlation_id),
        association=association if isinstance(association, dict) else {},
    )
    return _hop_frame(correlation_id, hop)


def emit_stance_hops(
    correlation_id: str,
    thought: dict[str, Any],
    *,
    stance_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Stance input + decision hops. Does not reset seq (progress hops already ran)."""
    frames: list[dict[str, Any]] = []
    frames.append(
        _hop_frame(
            correlation_id,
            hop_from_stance_inputs(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                stance_inputs=stance_inputs if isinstance(stance_inputs, dict) else {},
            ),
        )
    )
    frames.append(
        _hop_frame(
            correlation_id,
            hop_from_thought(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                thought=thought,
            ),
        )
    )
    return frames


def emit_mind_enrichment_hop(
    correlation_id: str,
    thought_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Emit Mind quality when present on the Thought reply; else honest unavailable."""
    mind = extract_mind_quality_fields(thought_payload)
    if mind is None:
        return emit_progress_hop(
            correlation_id,
            stage="mind_enrichment",
            status="skipped",
            visor_line="mind · details_unavailable",
            summary={"mind_details_unavailable": True},
            raw={"mind_details_unavailable": True},
        )
    bits: list[str] = []
    quality = mind.get("mind_quality")
    if quality:
        bits.append(str(quality))
    if mind.get("fallback_contract_only"):
        bits.append("fallback_contract_only")
    if mind.get("coloring_skipped"):
        bits.append("coloring skipped")
    if mind.get("authorized_for_stance_use") is False:
        bits.append("not authorized for stance")
    label = " · ".join(bits) if bits else "present"
    status: CockpitHopStatusV1 = "ok"
    return emit_progress_hop(
        correlation_id,
        stage="mind_enrichment",
        status=status,
        visor_line=f"mind · {label}",
        summary={k: mind[k] for k in mind},
        raw=dict(mind),
    )


def emit_situation_hop(
    correlation_id: str,
    *,
    compact_text: str | None,
    status: CockpitHopStatusV1 = "ok",
    provider_status: dict[str, Any] | None = None,
    source_summary: dict[str, Any] | None = None,
    perception_enabled: bool | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hop = hop_from_situation(
        correlation_id=correlation_id,
        seq=next_seq(correlation_id),
        compact_text=compact_text,
        status=status,
        provider_status=provider_status,
        source_summary=source_summary,
        perception_enabled=perception_enabled,
        diagnostics=diagnostics,
    )
    return _hop_frame(correlation_id, hop)


def emit_pre_motor_hops(
    correlation_id: str,
    thought: dict[str, Any],
    *,
    association: dict[str, Any],
    stance_inputs: dict[str, Any],
    ingress: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Batch helper for tests / late paths that did not stream progress hops.

    Resets seq and emits one ingress + association + stance_inputs + stance_decision.
    Live unified-turn path prefers begin_cockpit_timeline + incremental emits.
    Do not double-emit ingress: this is the only ingress for the batch path.
    """
    stance = stance_inputs if isinstance(stance_inputs, dict) else {}
    if isinstance(ingress, dict):
        ingress_payload = ingress
    else:
        try:
            attachment_count = int(stance.get("attachment_count") or 0)
        except (TypeError, ValueError):
            attachment_count = 0
        ingress_payload = {
            "user_message": str(stance.get("user_message") or ""),
            "session_id": stance.get("session_id"),
            "attachment_count": attachment_count,
            "observation_published": False,
        }
        mode = stance.get("mode") or stance.get("client_mode")
        if mode is not None:
            ingress_payload["mode"] = mode
    frames = begin_cockpit_timeline(correlation_id, ingress=ingress_payload)
    frames.append(emit_association_hop(correlation_id, association))
    frames.extend(
        emit_stance_hops(
            correlation_id,
            thought,
            stance_inputs=stance_inputs,
        )
    )
    return frames


def emit_motor_hop_from_claude_step(
    correlation_id: str,
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if item.get("kind") != "claude_step":
        return None
    step = item.get("step")
    if not isinstance(step, dict):
        step = {}
    if step.get("_cockpit") == COCKPIT_MOTOR_BOOT_MARKER:
        prompt = step.get("prompt")
        hop = hop_from_motor_boot(
            correlation_id=correlation_id,
            seq=next_seq(correlation_id),
            prompt=prompt if isinstance(prompt, str) else "",
        )
        return _hop_frame(correlation_id, hop)
    hop = hop_from_motor_step(
        correlation_id=correlation_id,
        seq=next_seq(correlation_id),
        step_index=int(item.get("step_index") or 0),
        step=step,
    )
    return _hop_frame(correlation_id, hop)


def _run_has_artifact_hops(run: dict[str, Any]) -> bool:
    return bool(
        run.get("draft_text")
        or run.get("substrate_appraisal")
        or run.get("reflection")
        or run.get("final_text")
        or run.get("finalize_ran")
    )


def emit_slice_a_finalize_hops(
    correlation_id: str,
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if _run_has_artifact_hops(run):
        hops = hop_from_run_artifact(
            correlation_id=correlation_id,
            seq=next_seq(correlation_id),
            run=run,
        )
        if hops:
            advance_seq(correlation_id, len(hops))
            frames.extend(_hop_frame(correlation_id, hop) for hop in hops)

    outcome = run.get("outcome")
    closure = run.get("closure") or run.get("post_turn_closure")
    if isinstance(outcome, dict) and outcome:
        hop = hop_from_outcome(
            correlation_id=correlation_id,
            seq=next_seq(correlation_id),
            outcome=outcome,
        )
        frames.append(_hop_frame(correlation_id, hop))
    elif isinstance(closure, dict) and closure:
        hop = hop_from_closure(
            correlation_id=correlation_id,
            seq=next_seq(correlation_id),
            closure=closure,
        )
        frames.append(_hop_frame(correlation_id, hop))

    frames.append(timeline_complete_frame(correlation_id))
    return frames


async def publish_cockpit_frames(bus: Any, frames: list[dict[str, Any]]) -> None:
    """Publish each cockpit_hop to the bus. Fail-open: never raise into chat."""
    if bus is None:
        return
    for frame in frames:
        if frame.get("kind") != "cockpit_hop":
            continue
        raw_hop = frame.get("hop")
        if not isinstance(raw_hop, dict):
            continue
        try:
            hop = CockpitHopV1.model_validate(raw_hop)
            await publish_cockpit_hop(bus, hop)
        except Exception:
            logger.warning(
                "cockpit hop publish failed corr=%s seq=%s",
                frame.get("correlation_id"),
                raw_hop.get("seq"),
                exc_info=True,
            )
