"""Build and publish Slice-A cockpit hop WS frames for a unified turn."""
from __future__ import annotations

import logging
from typing import Any

from orion.cockpit.builders import (
    gap_hop,
    hop_from_association,
    hop_from_closure,
    hop_from_motor_boot,
    hop_from_motor_step,
    hop_from_outcome,
    hop_from_run_artifact,
    hop_from_stance_inputs,
    hop_from_thought,
)
from orion.cockpit.markers import COCKPIT_MOTOR_BOOT_MARKER
from orion.cockpit.publish import publish_cockpit_hop
from orion.cockpit.sequencer import advance_seq, next_seq, reset_seq
from orion.schemas.cockpit_sighting import CockpitHopV1

logger = logging.getLogger("orion.hub.cockpit_emit")


def _hop_frame(correlation_id: str, hop: CockpitHopV1) -> dict[str, Any]:
    return {
        "kind": "cockpit_hop",
        "correlation_id": correlation_id,
        "hop": hop.model_dump(mode="json"),
    }


def timeline_complete_frame(correlation_id: str) -> dict[str, Any]:
    return {"kind": "cockpit_timeline_complete", "correlation_id": correlation_id}


def emit_pre_motor_hops(
    correlation_id: str,
    thought: dict[str, Any],
    *,
    association: dict[str, Any],
    stance_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    reset_seq(correlation_id)
    frames: list[dict[str, Any]] = []
    frames.append(
        _hop_frame(
            correlation_id,
            gap_hop(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                stage="ingress",
                deferred_to="slice_c",
            ),
        )
    )
    frames.append(
        _hop_frame(
            correlation_id,
            hop_from_association(
                correlation_id=correlation_id,
                seq=next_seq(correlation_id),
                association=association if isinstance(association, dict) else {},
            ),
        )
    )
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
