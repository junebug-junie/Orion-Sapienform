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
import json
import logging
import os
from contextlib import suppress
from typing import Any, Callable
from uuid import UUID, uuid4

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.reverie.referent_loader import default_referent_loader
from orion.reverie.semantic_lift import (
    coalition_audit_refs,
    enforce_semantic_quality,
    resolve_concern_cards,
    reverie_semantic_gate,
)
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.attention_salience import AttentionSalienceTraceV1
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.reverie import ConcernCardV1, SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

from .bus_listener import extract_stance_react_payload
from .cortex_client import CortexExecClient
from .settings import settings
from .store import load_recent_loop_outcomes, persist_reverie_thought, persist_salience_trace

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
    """Read the current coalition directly from the broadcast projection table.

    Uses a minimal direct query (`app.broadcast_reader`) rather than the heavy
    `orion.substrate.felt_state_reader`, which would drag the full graph engine
    (`requests` etc.) that this thin bus service does not ship. Fail-open: any
    failure degrades to None so a reverie tick never raises.
    """
    try:
        from .broadcast_reader import read_latest_broadcast

        return read_latest_broadcast()
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


_TRUTHY = {"1", "true", "yes", "on"}


def _salience_v2_enabled() -> bool:
    """Thin-service read of the salience-v2 flag.

    Deliberately does NOT import `orion.substrate.attention.salience`: importing
    any `orion.substrate` submodule runs `orion/substrate/__init__.py`, which
    eagerly loads the graph engine (`requests` etc.) that this thin bus service
    does not install. Mirrors `salience_v2_enabled()` in that module.
    """
    return os.getenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "false").strip().lower() in _TRUTHY


def _bounded(value: float) -> float:
    """Clamp to [0,1] — local copy of `orion.substrate.attention.common.bounded`
    to keep reverie off the `orion.substrate` import graph (no `requests`)."""
    return max(0.0, min(1.0, float(value)))


def _weights_version() -> str:
    """Provenance string mirroring `default_combiner().weights_version` without
    importing the combiner (thin service: no `orion.substrate` at runtime)."""
    raw = os.getenv("ORION_ATTENTION_SALIENCE_WEIGHTS", "").strip()
    if raw:
        try:
            if isinstance(json.loads(raw), dict):
                return "seed-v1+override"
        except (ValueError, TypeError):
            pass
    return "seed-v1"


def derive_salience(broadcast: AttentionBroadcastProjectionV1 | None) -> float:
    """Salience of the selected coalition.

    v2 (`ORION_ATTENTION_SALIENCE_V2_ENABLED`): read the loop's precomputed
    `salience` (computed upstream by the same combiner selection uses — one
    source of salience truth). Legacy: max of the seven constant score fields,
    else stability score.

    Uses only thin local helpers (no `orion.substrate` import): importing that
    package drags the graph engine (`requests`), which this thin bus service must
    not load (see `test_reverie_thin_import_boundary`).
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
    if _salience_v2_enabled():
        return _bounded(float(loop.salience)) if loop.salience else fallback
    scores = [float(getattr(loop, field, 0.0)) for field in _OPEN_LOOP_SCORE_FIELDS]
    return max(scores) if scores else fallback


def build_salience_trace(
    broadcast: AttentionBroadcastProjectionV1 | None,
    *,
    correlation_id: str,
) -> AttentionSalienceTraceV1 | None:
    """Trace the selected loop's feature vector + salience. None if no selection.

    Reads the loop's precomputed salience/features (computed upstream) and packages
    them — reverie does not recompute, so it needs no combiner. Uses thin local
    helpers only (no `orion.substrate` import; that would drag `requests`). Only
    `orion.core.ids` is imported locally, which is light.
    """
    if broadcast is None or not broadcast.selected_open_loop_id:
        return None
    loop = next(
        (l for l in broadcast.frame.open_loops if l.id == broadcast.selected_open_loop_id),
        None,
    )
    if loop is None:
        return None
    from orion.core.ids import stable_hash_id

    return AttentionSalienceTraceV1(
        trace_id=stable_hash_id("saltrace", [correlation_id, loop.id]),
        loop_id=loop.id,
        theme_key=loop.id,
        description=(loop.description or "")[:200],
        correlation_id=correlation_id,
        salience=_bounded(float(loop.salience)),
        weights_version=_weights_version(),
        features=dict(loop.salience_features or {}),
        scope="reverie",
    )


def _open_loops_for_prompt(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    loop_outcomes: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    outcomes = loop_outcomes or {}
    entries = []
    for loop in broadcast.frame.open_loops:
        entry: dict[str, Any] = {
            "id": loop.id,
            "description": loop.description,
            "why_it_matters": loop.why_it_matters,
            "target_type": loop.target_type,
        }
        outcome = outcomes.get(loop.id)
        if outcome:
            entry["outcome"] = outcome
        entries.append(entry)
    return entries


def build_reverie_context(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    concern_cards: list[ConcernCardV1] | None = None,
    loop_outcomes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if concern_cards:
        coalition_refs = coalition_audit_refs(broadcast)
        return {
            "user_message": None,
            "concern_cards": [c.model_dump(mode="json") for c in concern_cards],
            "coalition_refs": coalition_refs,
            "metadata": {"mode": "metacog", "llm_profile": "metacog"},
            "mode": "metacog",
            "llm_route": "metacog",
            "options": {"llm_lane": "background", "allow_chat_fallback": False},
        }
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
        "open_loops": _open_loops_for_prompt(broadcast, loop_outcomes=loop_outcomes),
        "metadata": {"mode": "reverie", "llm_profile": "brain"},
    }


def build_reverie_plan_request(
    broadcast: AttentionBroadcastProjectionV1,
    *,
    correlation_id: str,
    concern_cards: list[ConcernCardV1] | None = None,
    loop_outcomes: dict[str, dict[str, Any]] | None = None,
) -> PlanExecutionRequest:
    use_lift = settings.reverie_semantic_lift_enabled and bool(concern_cards)
    mode = "metacog" if use_lift else "brain"
    plan = build_plan_for_verb("reverie_narrate", mode=mode)
    extra: dict[str, Any] = {
        "llm_profile": mode,
        "mode": "metacog" if use_lift else "reverie",
    }
    if use_lift:
        extra["llm_route"] = "metacog"
        extra["execution_lane"] = "background"
    return PlanExecutionRequest(
        plan=plan,
        args=PlanExecutionArgs(
            request_id=correlation_id,
            trigger_source=settings.service_name,
            extra=extra,
        ),
        context=build_reverie_context(
            broadcast, concern_cards=concern_cards, loop_outcomes=loop_outcomes
        ),
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
    chain_context: tuple[str, int] | None = None,
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
    concern_cards: list[ConcernCardV1] | None = None
    if settings.reverie_semantic_lift_enabled:
        try:
            loader = default_referent_loader(max_age_hours=settings.reverie_referent_max_age_hours)
            concern_cards = resolve_concern_cards(broadcast, referent_loader=loader)
            if reverie_semantic_gate(concern_cards) == "skip":
                logger.info("reverie tick skipped: reverie_skipped_no_semantic_referent")
                return None
        except Exception as exc:
            logger.warning(
                "reverie semantic lift failed corr=%s err=%s; tick skipped",
                correlation_id,
                exc,
            )
            return None

    client = cortex_client or CortexExecClient(
        bus,
        request_channel=(
            settings.channel_reverie_cortex_exec_request
            if settings.reverie_semantic_lift_enabled
            else settings.channel_cortex_exec_request
        ),
    )
    try:
        loop_outcomes: dict[str, dict[str, Any]] | None = None
        if not (settings.reverie_semantic_lift_enabled and concern_cards):
            loop_outcomes = load_recent_loop_outcomes(
                [loop.id for loop in broadcast.frame.open_loops]
            )
        plan_request = build_reverie_plan_request(
            broadcast,
            correlation_id=correlation_id,
            concern_cards=concern_cards,
            loop_outcomes=loop_outcomes,
        )
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
        if settings.reverie_semantic_lift_enabled and concern_cards:
            thought = enforce_semantic_quality(
                thought,
                concern_cards,
                allowed_refs=coalition_audit_refs(broadcast),
            )
            thought = thought.model_copy(update={"llm_profile": "metacog"})
        if chain_context is not None:
            thought = thought.model_copy(
                update={"chain_id": chain_context[0], "thought_index": chain_context[1]}
            )
        if settings.reverie_ground_consolidation:
            from .grounding import (
                collect_grounding,
                default_episode_loader,
                default_motif_loader,
            )

            motif_refs, episode_refs = collect_grounding(
                motif_loader=default_motif_loader,
                episode_loader=default_episode_loader,
            )
            if motif_refs or episode_refs:
                thought = thought.model_copy(
                    update={"motif_refs": motif_refs, "episode_summary_refs": episode_refs}
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
        # Best-effort persistence for the hub panel — never breaks the tick.
        persist_reverie_thought(thought)
        if settings.attention_salience_v2_enabled:
            trace = build_salience_trace(broadcast, correlation_id=correlation_id)
            if trace is not None:
                with suppress(Exception):
                    await bus.publish(
                        settings.channel_attention_salience_trace,
                        BaseEnvelope(
                            kind="attention.salience.trace.v1",
                            source=_source(),
                            correlation_id=_envelope_correlation_id(correlation_id),
                            payload=trace.model_dump(mode="json"),
                        ),
                    )
                persist_salience_trace(trace)
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
    if settings.reverie_chain_enabled:
        # Chain mode already drives run_reverie_once per step; running the
        # standalone tick too would double-emit to the same channel/table.
        logger.info("reverie chain enabled; standalone reverie worker superseded")
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
