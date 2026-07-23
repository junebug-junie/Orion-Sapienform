"""chat_turn metacog trigger: correlator + gate.

See docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md's
"`chat_turn` trigger: implementation spec" section for the full design and
gate-condition table.

Accumulates real evidence per correlation_id -- a ThoughtEventV1
(orion:thought:artifact), a HarnessRunV1 (orion:harness:run:artifact), and/or
a timeout GrammarEventV1 (orion:grammar:event) -- in a short-TTL Redis key,
and fires a "chat_turn" MetacogTriggerV1 once the evidence is terminal (no
more evidence will ever arrive for that correlation_id) AND at least one real
gate condition is true. No new Pydantic schema: the accumulator holds plain
dicts of the already-registered ThoughtEventV1/HarnessRunV1 shapes, since it
is ephemeral internal state, not a durable artifact.

Terminal evidence, four cases:
- run_artifact arrived -- the turn ran to completion on one of HarnessRunV1's
  real exit paths (see orion/harness/finalize.py); every gate condition is
  evaluable.
- exec_turn_timeout -- the harness-governor RPC never returned at all
  (orion/hub/turn_orchestrator.py's `if run is None:` branch, Patch B / PR
  #1287, semantic_role == "exec_turn_timeout"); run_artifact will never be
  published for this correlation_id.
- stance_react_timeout -- the *earlier* ThoughtClient.react() RPC itself
  never returned to Hub (orion/hub/turn_orchestrator.py's `if thought is
  None:` branch, ~line 416, semantic_role == "stance_disposition" with
  text_value == "stance_timeout"). Hub gives up and returns without ever
  calling the harness governor, so run_artifact will never arrive either --
  same shape of gap as exec_turn_timeout, one step earlier in the pipeline.
  Note orion-thought's handle_stance_react_request still runs to completion
  and eventually publishes a real ThoughtEventV1 on orion:thought:artifact
  regardless of whether Hub already gave up waiting on it
  (services/orion-thought/app/bus_listener.py) -- if that arrives before this
  correlator has already fired on the timeout, its disposition/boundary_register
  are folded in too; if it arrives after, it's lost (the entry was already
  cleared). Not otherwise recoverable without holding entries open indefinitely.
- thought_event.disposition in ("defer", "refuse") -- orion/hub/
  turn_orchestrator.py short-circuits BEFORE calling the harness governor on
  this path (line ~438), so run_artifact will never arrive either. Confirmed
  by direct code reading, not assumed -- this is why the correlator cannot
  simply wait for run_artifact unconditionally the way the original design-doc
  draft of question 1 assumed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.chat_turn_metacog_gate")

_KEY_PREFIX = "orion:equilibrium:chat_turn_corr"
_TERMINAL_DISPOSITIONS = ("defer", "refuse")


def is_chat_turn_evidence_terminal(
    *,
    thought_event: dict[str, Any] | None,
    run_artifact: dict[str, Any] | None,
    timed_out: bool,
) -> bool:
    """True once no further evidence will ever arrive for this correlation_id."""
    if timed_out or run_artifact is not None:
        return True
    if thought_event is not None and thought_event.get("disposition") in _TERMINAL_DISPOSITIONS:
        return True
    return False


def evaluate_chat_turn_gate_conditions(
    *,
    thought_event: dict[str, Any] | None,
    run_artifact: dict[str, Any] | None,
    timed_out: bool,
    surprise_threshold: float,
    timeout_reason: str | None = None,
) -> list[str]:
    """Read fired condition names directly off real ThoughtEventV1/HarnessRunV1
    fields -- no invented booleans. Returns [] when nothing fired, which is the
    common case for an unremarkable turn.
    """
    fired: list[str] = []

    if timed_out:
        fired.append(f"timeout={timeout_reason}" if timeout_reason else "timeout=unknown")

    if thought_event is not None:
        disposition = thought_event.get("disposition")
        if disposition and disposition != "proceed":
            fired.append(f"disposition={disposition}")
        if thought_event.get("boundary_register") is True:
            fired.append("boundary_register=true")

    if timed_out or run_artifact is None:
        # run_artifact will never arrive once timed_out -- nothing else to evaluate.
        return fired

    reflection = run_artifact.get("reflection")
    if isinstance(reflection, dict):
        alignment_verdict = reflection.get("alignment_verdict")
        if alignment_verdict and alignment_verdict != "aligned":
            fired.append(f"alignment_verdict={alignment_verdict}")
        if reflection.get("strain_unresolved") is True:
            fired.append("strain_unresolved=true")

    substrate_appraisal = run_artifact.get("substrate_appraisal")
    if isinstance(substrate_appraisal, dict):
        surprise_level = substrate_appraisal.get("surprise_level")
        if isinstance(surprise_level, (int, float)) and surprise_level >= surprise_threshold:
            fired.append(f"surprise_level={float(surprise_level):.3f}")

    compliance_verdict = run_artifact.get("compliance_verdict")
    if compliance_verdict and compliance_verdict != "completed":
        fired.append(f"compliance_verdict={compliance_verdict}")

    exit_code = run_artifact.get("exit_code")
    if exit_code not in (0, None):
        fired.append(f"exit_code={exit_code}")

    finalize_degraded_reason = run_artifact.get("finalize_degraded_reason")
    if finalize_degraded_reason:
        fired.append(f"finalize_degraded_reason={finalize_degraded_reason}")

    return fired


def build_chat_turn_metacog_trigger(
    *,
    correlation_id: str,
    thought_event: dict[str, Any] | None,
    run_artifact: dict[str, Any] | None,
    timed_out: bool,
    timeout_reason: str | None = None,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    surprise_threshold: float,
) -> MetacogTriggerV1 | None:
    """Turn accumulated, terminal chat-turn evidence into a "chat_turn" metacog
    trigger. Returns None when no gate condition fired -- most turns are
    unremarkable and should produce zero triggers, same as every other
    metacog gate in this service.
    """
    fired_conditions = evaluate_chat_turn_gate_conditions(
        thought_event=thought_event,
        run_artifact=run_artifact,
        timed_out=timed_out,
        timeout_reason=timeout_reason,
        surprise_threshold=surprise_threshold,
    )
    if not fired_conditions:
        return None

    reflection = run_artifact.get("reflection") if isinstance(run_artifact, dict) else None
    substrate_appraisal = run_artifact.get("substrate_appraisal") if isinstance(run_artifact, dict) else None
    reason = f"chat_turn:{','.join(fired_conditions)}"

    return MetacogTriggerV1(
        trigger_kind="chat_turn",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[correlation_id] if correlation_id else [],
        upstream={
            "fired_conditions": fired_conditions,
            "timed_out": timed_out,
            "timeout_reason": timeout_reason,
            "disposition": (thought_event or {}).get("disposition"),
            "disposition_reasons": (thought_event or {}).get("disposition_reasons"),
            "boundary_register": (thought_event or {}).get("boundary_register"),
            "compliance_verdict": (run_artifact or {}).get("compliance_verdict"),
            "grounding_status": (run_artifact or {}).get("grounding_status"),
            "exit_code": (run_artifact or {}).get("exit_code"),
            "finalize_degraded_reason": (run_artifact or {}).get("finalize_degraded_reason"),
            "alignment_verdict": reflection.get("alignment_verdict") if isinstance(reflection, dict) else None,
            "alignment_notes": reflection.get("alignment_notes") if isinstance(reflection, dict) else None,
            "strain_unresolved": reflection.get("strain_unresolved") if isinstance(reflection, dict) else None,
            "surprise_level": (
                substrate_appraisal.get("surprise_level") if isinstance(substrate_appraisal, dict) else None
            ),
            "grounding_capsule": (thought_event or {}).get("grounding_capsule"),
            "autonomy_slice": (thought_event or {}).get("autonomy_slice"),
        },
    )


class ChatTurnCorrelator:
    """Redis-backed accumulator keyed by correlation_id.

    Thin I/O wrapper around the pure functions above -- kept separate so the
    gate-condition logic is unit-testable without a Redis connection, matching
    this service's existing per-trigger-kind module convention.
    """

    def __init__(self, redis: Any, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    def _key(self, correlation_id: str) -> str:
        return f"{_KEY_PREFIX}:{correlation_id}"

    async def _load(self, correlation_id: str) -> dict[str, Any]:
        try:
            raw = await self._redis.get(self._key(correlation_id))
        except Exception:
            logger.warning("chat_turn_corr_read_failed correlation_id=%s", correlation_id, exc_info=True)
            return {}
        if raw is None:
            return {}
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            parsed = json.loads(raw)
        except Exception:
            logger.warning("chat_turn_corr_malformed correlation_id=%s", correlation_id, exc_info=True)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _save(self, correlation_id: str, state: dict[str, Any]) -> None:
        try:
            await self._redis.setex(self._key(correlation_id), self._ttl_seconds, json.dumps(state))
        except Exception:
            logger.warning("chat_turn_corr_write_failed correlation_id=%s", correlation_id, exc_info=True)

    async def _clear(self, correlation_id: str) -> None:
        try:
            await self._redis.delete(self._key(correlation_id))
        except Exception:
            logger.warning("chat_turn_corr_clear_failed correlation_id=%s", correlation_id, exc_info=True)

    async def accumulate(
        self,
        *,
        correlation_id: str,
        thought_event: dict[str, Any] | None = None,
        run_artifact: dict[str, Any] | None = None,
        timed_out: bool = False,
        timeout_reason: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, str | None]:
        """Merge new evidence into this correlation_id's accumulated state.

        Returns the merged (thought_event, run_artifact, timed_out,
        timeout_reason) tuple. The caller decides whether to evaluate/fire
        based on is_chat_turn_evidence_terminal against that same tuple, and
        is responsible for calling clear() once it does (whether or not a
        trigger actually fired -- terminal evidence is consumed either way).
        """
        if not correlation_id:
            return thought_event, run_artifact, timed_out, timeout_reason

        state = await self._load(correlation_id)
        merged_thought = thought_event if thought_event is not None else state.get("thought_event")
        merged_run = run_artifact if run_artifact is not None else state.get("run_artifact")
        merged_timed_out = bool(timed_out or state.get("timed_out", False))
        merged_timeout_reason = timeout_reason or state.get("timeout_reason")

        if is_chat_turn_evidence_terminal(
            thought_event=merged_thought, run_artifact=merged_run, timed_out=merged_timed_out
        ):
            await self._clear(correlation_id)
        else:
            await self._save(
                correlation_id,
                {
                    "thought_event": merged_thought,
                    "run_artifact": merged_run,
                    "timed_out": merged_timed_out,
                    "timeout_reason": merged_timeout_reason,
                },
            )

        return merged_thought, merged_run, merged_timed_out, merged_timeout_reason
