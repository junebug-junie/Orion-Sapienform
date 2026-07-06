"""Spontaneous-thought mode — reverie inside orion-thought.

Sibling of the evoked `bus_listener` path. A self-driven tick reads the current
rung-3 winning coalition (no user message), runs the `reverie_narrate` verb over
the same cortex exec rail, and emits a `SpontaneousThoughtV1` on
`orion:reverie:thought`.

Discipline (§0A / hard constraints):
  - default-off (`ORION_REVERIE_ENABLED=false`);
  - degrades to None on absent/stale coalition — never raises;
  - every emitted thought passes the `is_hollow()` guard; hollow thoughts are
    stamped and dropped (not published) rather than masqueraded as cognition.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Callable
from uuid import UUID, uuid4

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

from .bus_listener import extract_stance_react_payload
from .cortex_client import CortexExecClient
from .settings import settings

logger = logging.getLogger("orion-thought.reverie")

# Coalition score fields on OpenLoopV1 — all pre-computed 0..1, never invented here.
_OPEN_LOOP_SCORE_FIELDS = (
    "novelty",
    "continuity_relevance",
    "relational_relevance",
    "predictive_value",
    "concept_value",
    "autonomy_value",
    "emotional_charge",
)

BroadcastReader = Callable[[], AttentionBroadcastProjectionV1 | None]


def _source() -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


def _default_broadcast_reader() -> AttentionBroadcastProjectionV1 | None:
    """Read the current coalition via the shared substrate felt-state reader.

    Mirrors orion/hub/association.py. Any failure degrades to None so a
    reverie tick never raises on a missing/unavailable substrate.
    """
    try:
        # Public fail-open entrypoint — resolves enabled/url/max-age internally and
        # never raises. Same coalition source the evoked stance_react path reads.
        from orion.substrate.felt_state_reader import hydrate_felt_state_ctx

        ctx: dict[str, Any] = {}
        hydrate_felt_state_ctx(ctx)
        raw = ctx.get("attention_broadcast")
        if raw is None:
            return None
        if isinstance(raw, AttentionBroadcastProjectionV1):
            return raw
        return AttentionBroadcastProjectionV1.model_validate(raw)
    except Exception as exc:  # never raise out of a reverie tick
        logger.warning("reverie broadcast read failed: %s", exc)
        return None


def build_coalition_snapshot(
    broadcast: AttentionBroadcastProjectionV1 | None,
) -> CoalitionSnapshotV1 | None:
    """Project the live broadcast into the shared grounding vocabulary."""
    if broadcast is None:
        return None
    return CoalitionSnapshotV1(
        attended_node_ids=list(broadcast.attended_node_ids),
        selected_open_loop_id=broadcast.selected_open_loop_id,
        open_loop_ids=[loop.id for loop in broadcast.frame.open_loops],
        generated_at=broadcast.generated_at,
        broadcast_stale=False,
    )


def derive_salience(broadcast: AttentionBroadcastProjectionV1 | None) -> float:
    """Deterministic salience from existing OpenLoopV1 scores — no invented weights.

    Takes the max pre-computed score of the selected open loop; falls back to the
    coalition's own stability score. A max (not a hand-tuned weighted sum) keeps
    this out of keyword-cathedral territory.
    """
    if broadcast is None:
        return 0.0
    fallback = float(broadcast.coalition_stability_score)
    loop = next(
        (l for l in broadcast.frame.open_loops if l.id == broadcast.selected_open_loop_id),
        None,
    )
    if loop is None:
        return fallback
    scores = [float(getattr(loop, field, 0.0)) for field in _OPEN_LOOP_SCORE_FIELDS]
    return max(scores) if scores else fallback


def _open_loops_for_prompt(broadcast: AttentionBroadcastProjectionV1) -> list[dict[str, Any]]:
    return [
        {
            "id": loop.id,
            "description": loop.description,
            "why_it_matters": loop.why_it_matters,
            "target_type": loop.target_type,
        }
        for loop in broadcast.frame.open_loops
    ]


def build_reverie_context(broadcast: AttentionBroadcastProjectionV1) -> dict[str, Any]:
    return {
        "user_message": None,  # the defining difference from stance_react
        "coalition_projection": {
            "projection_id": broadcast.projection_id,
            "generated_at": broadcast.generated_at.isoformat(),
            "attended_node_ids": list(broadcast.attended_node_ids),
            "selected_open_loop_id": broadcast.selected_open_loop_id,
            "selected_description": broadcast.selected_description,
            "open_loop_ids": [loop.id for loop in broadcast.frame.open_loops],
            "coalition_stability_score": broadcast.coalition_stability_score,
            "dwell_ticks": broadcast.dwell_ticks,
        },
        "open_loops": _open_loops_for_prompt(broadcast),
        "metadata": {"mode": "reverie", "llm_profile": "brain"},
    }


def build_reverie_plan_request(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    correlation_id: str,
) -> PlanExecutionRequest:
    plan = build_plan_for_verb("reverie_narrate", mode="brain")
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source=settings.service_name,
            extra={"llm_profile": "brain", "mode": "reverie"},
        ),
        context=build_reverie_context(broadcast),
    )


def parse_reverie_payload(
    raw: dict[str, Any] | str,
    *,
    coalition: CoalitionSnapshotV1,
    correlation_id: str,
    broadcast: AttentionBroadcastProjectionV1,
) -> SpontaneousThoughtV1:
    """Build a SpontaneousThoughtV1 from LLM output, stamping the hollow guard.

    Deterministic/latent split (§4): the LLM owns only the *interpretation* text;
    salience is computed deterministically from coalition scores here, never taken
    from the LLM (same coalition → same salience, every time).
    """
    data: dict[str, Any] = {}
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        import json

        with suppress(Exception):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed

    interpretation = str(data.get("interpretation") or "").strip()
    evidence_refs = data.get("evidence_refs") or []
    if not isinstance(evidence_refs, list):
        evidence_refs = []
    evidence_refs = [str(x) for x in evidence_refs][: 50]

    salience = derive_salience(broadcast)

    thought = SpontaneousThoughtV1(
        thought_id=str(uuid4()),
        correlation_id=correlation_id,
        coalition=coalition,
        interpretation=interpretation,
        salience=salience,
        evidence_refs=evidence_refs,
        model_id=str(data.get("model_id")) if data.get("model_id") else None,
    )
    return thought.marked_hollow()


def _envelope_correlation_id(raw: str | None) -> UUID:
    if raw:
        try:
            return UUID(str(raw))
        except ValueError:
            pass
    return uuid4()


async def run_reverie_once(
    bus: OrionBusAsync,
    *,
    broadcast_reader: BroadcastReader | None = None,
    cortex_client: CortexExecClient | None = None,
) -> SpontaneousThoughtV1 | None:
    """One spontaneous-thought tick. Returns the published thought, or None.

    Returns None (publishes nothing) when there is no coalition to narrate or the
    narration comes back hollow — never empty-shell cognition, never a raise.
    """
    reader = broadcast_reader or _default_broadcast_reader
    broadcast = reader()
    coalition = build_coalition_snapshot(broadcast)
    if broadcast is None or coalition is None:
        logger.info("reverie tick skipped: no current coalition")
        return None

    correlation_id = str(uuid4())
    client = cortex_client or CortexExecClient(bus)
    try:
        plan_request = build_reverie_plan_request(broadcast, correlation_id=correlation_id)
        exec_result = await client.execute_plan(
            source=_source(),
            req=plan_request,
            correlation_id=correlation_id,
            timeout_sec=settings.stance_react_timeout_sec,
        )
        raw_payload = extract_stance_react_payload(exec_result)

        thought = parse_reverie_payload(
            raw_payload,
            coalition=coalition,
            correlation_id=correlation_id,
            broadcast=broadcast,
        )
        if thought.hollow:
            logger.info(
                "reverie tick dropped hollow thought corr=%s reason=%s",
                correlation_id,
                thought.hollow_reason,
            )
            return None
        if thought.salience < settings.reverie_min_salience:
            logger.info(
                "reverie tick below min salience corr=%s salience=%.3f min=%.3f",
                correlation_id,
                thought.salience,
                settings.reverie_min_salience,
            )
            return None

        envelope = BaseEnvelope(
            kind="reverie.thought.v1",
            source=_source(),
            correlation_id=_envelope_correlation_id(correlation_id),
            payload=thought.model_dump(mode="json"),
        )
        await bus.publish(settings.channel_reverie_thought, envelope)
    except Exception as exc:
        # A tick must never raise (hard constraint) — a bus/narration/parse
        # failure degrades to a dropped tick, not a crash.
        logger.warning("reverie tick failed corr=%s err=%s", correlation_id, exc)
        return None

    logger.info(
        "reverie thought published corr=%s salience=%.3f channel=%s",
        correlation_id,
        thought.salience,
        settings.channel_reverie_thought,
    )
    return thought


async def run_reverie_worker(stop_event: asyncio.Event | None = None) -> None:
    """Self-driven reverie loop. Default-off; a no-op unless ORION_REVERIE_ENABLED."""
    if not settings.reverie_enabled:
        logger.info("reverie disabled; worker not started")
        return
    if not settings.orion_bus_enabled:
        logger.info("bus disabled; reverie worker not started")
        return

    bus = OrionBusAsync(url=settings.orion_bus_url)
    await bus.connect()
    logger.info(
        "reverie worker started interval=%ss min_salience=%.3f channel=%s",
        settings.reverie_interval_sec,
        settings.reverie_min_salience,
        settings.channel_reverie_thought,
    )
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                await run_reverie_once(bus)
            except Exception:
                logger.exception("unhandled reverie tick error")
            try:
                if stop_event is not None:
                    await asyncio.wait_for(stop_event.wait(), timeout=settings.reverie_interval_sec)
                    break
                await asyncio.sleep(settings.reverie_interval_sec)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(Exception):
            await bus.close()
