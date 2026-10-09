from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory_graph.cortex_suggest_extract import extract_suggest_draft_dict_from_cortex_payload
from orion.memory_graph.draft_repository import insert_pending_draft
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.draft_sanitize import sanitize_suggest_draft_dict
from orion.memory_graph.suggest_runner import suggest_once, suggest_with_escalation
from orion.memory_graph.suggest_token_budget import suggest_token_budget_config_from_mapping

from app.boundary import legacy_view
from app.window_fetch import legacy_close_decision
from app.classify import classify_turn
from app.settings import settings
from app.window_state import WindowStore
from orion.schemas.memory_consolidation import (
    CHAT_HISTORY_SPARK_META_PATCH_KIND,
    ChatHistorySparkMetaPatchV1,
    MemoryTurnPersistedV1,
)
from orion.schemas.memory_episode import MEMORY_EPISODE_CLOSED_KIND, MemoryEpisodeClosedV1

logger = logging.getLogger(__name__)

_MAX_TURNS_FOR_SUGGEST = 3


def _projection_config_from_settings(s: Any) -> "ProjectionConfig":
    from orion.memory.crystallization.projector import ProjectionConfig

    return ProjectionConfig(
        collection=getattr(s, "CRYSTALLIZER_VECTOR_COLLECTION", "orion_memory_crystallizations"),
        embed_host_url=getattr(s, "CRYSTALLIZER_EMBED_HOST_URL", "") or "",
        embed_mode=getattr(s, "CRYSTALLIZER_EMBED_MODE", "http") or "http",
        embed_timeout_ms=int(getattr(s, "CRYSTALLIZER_EMBED_TIMEOUT_MS", 8000) or 8000),
        graphiti_enabled=bool(getattr(s, "GRAPHITI_ENABLED", False)),
        graphiti_url=getattr(s, "GRAPHITI_ADAPTER_URL", "") or "",
        falkordb_uri=getattr(s, "FALKORDB_URI", "") or "",
        service_name=s.SERVICE_NAME,
        service_version=s.SERVICE_VERSION,
        node_name=s.NODE_NAME,
    )
_MAX_TURN_FIELD_CHARS = 800


def enrich_spark_meta_patch(patch_fields: dict[str, Any]) -> dict[str, Any]:
    """Mirror appraisal novelty to top-level spark_meta fields hub/SQL viewers expect."""
    out = dict(patch_fields or {})
    appraisal = out.get("turn_change_appraisal")
    if not isinstance(appraisal, dict) or appraisal.get("turn_change_status") != "ok":
        return out
    novelty = appraisal.get("novelty_score")
    if isinstance(novelty, (int, float)):
        out["novelty"] = float(novelty)
    from orion.schemas.telemetry.turn_effect import turn_effect_from_appraisal

    turn_effect = turn_effect_from_appraisal({"turn_change_appraisal": appraisal})
    if turn_effect:
        out["turn_effect"] = turn_effect
    return out


def _clip(text: str, *, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def build_window_transcript(turns: list[dict]) -> str:
    selected = turns[-_MAX_TURNS_FOR_SUGGEST:] if len(turns) > _MAX_TURNS_FOR_SUGGEST else turns
    lines = []
    for t in selected:
        sig = t.get("memory_significance_score")
        prefix = f"[sig={sig:.2f}] " if isinstance(sig, (int, float)) else ""
        prompt = _clip(str(t.get("prompt") or ""), limit=_MAX_TURN_FIELD_CHARS)
        response = _clip(str(t.get("response") or ""), limit=_MAX_TURN_FIELD_CHARS)
        lines.append(f"{prefix}User: {prompt}\nOrion: {response}\n")
    return "\n".join(lines)


async def publish_spark_meta_patch(
    bus: OrionBusAsync,
    correlation_id: str,
    patch_fields: dict[str, Any],
) -> None:
    patch_fields = enrich_spark_meta_patch(patch_fields)
    svc_ref = ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )
    patch_env = BaseEnvelope(
        kind=CHAT_HISTORY_SPARK_META_PATCH_KIND,
        correlation_id=correlation_id,
        source=svc_ref,
        payload=ChatHistorySparkMetaPatchV1(
            correlation_id=correlation_id,
            spark_meta=patch_fields,
        ).model_dump(mode="json"),
    )
    await bus.publish(settings.CHANNEL_CHAT_HISTORY_SPARK_META_PATCH, patch_env)


class ConsolidationSuggestRunner:
    def __init__(self, pool, window_store: WindowStore, *, grammar_pool=None):
        self._pool = pool
        self._grammar_pool = grammar_pool
        self._window_store = window_store

    async def consolidate_window(self, window: dict[str, Any], *, bus: OrionBusAsync) -> None:
        window_id = window["memory_window_id"]
        turns = window.get("turns") or []
        corr_ids = window.get("turn_correlation_ids") or []
        output_mode = settings.MEMORY_CONSOLIDATION_OUTPUT
        if output_mode in ("crystallization_propose", "skip_only"):
            try:
                from orion.memory.consolidation_gate import consolidation_memory_gate
                from orion.memory.consolidation_grammar import fetch_grammar_evidence_for_window
                from orion.memory.crystallization.intake_consolidation_window import (
                    build_crystallization_from_window,
                )
                from orion.memory.crystallization.intake_pipeline import process_consolidation_crystallization

                grammar_pool = self._grammar_pool or self._pool
                repair, grammar_event_ids = await fetch_grammar_evidence_for_window(
                    grammar_pool,
                    turns=turns,
                    node_id=settings.NODE_NAME,
                    enabled=settings.MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE,
                )
                gate = consolidation_memory_gate(
                    turns=turns,
                    grammar_repair_signal=repair,
                    grammar_event_ids=grammar_event_ids,
                    min_novelty=settings.MEMORY_CONSOLIDATION_MIN_NOVELTY,
                    min_significance=settings.MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE,
                )
                if gate.action == "skip" or output_mode == "skip_only":
                    await self._window_store.mark_consolidated_skipped(
                        window_id,
                        reasons=gate.reasons,
                    )
                    for corr in corr_ids:
                        await publish_spark_meta_patch(
                            bus,
                            corr,
                            {
                                "consolidation_gate": {
                                    "action": "skip",
                                    "reasons": gate.reasons,
                                }
                            },
                        )
                    return

                crystallization = build_crystallization_from_window(
                    memory_window_id=window_id,
                    turns=turns,
                    gate=gate,
                )
                cid, _final_row, outcome = await process_consolidation_crystallization(
                    self._pool,
                    bus,
                    crystallization=crystallization,
                    settings=settings,
                    project_config=_projection_config_from_settings(settings),
                )
                if outcome == "discarded_external_platform":
                    # No crystallization was inserted -- cid is None. Close the
                    # window the same way a gate.action=="skip" window closes
                    # above: mark_consolidated_skipped, not
                    # mark_crystallization_proposed (which requires a real id).
                    await self._window_store.mark_consolidated_skipped(
                        window_id,
                        reasons=[f"formation_outcome:{outcome}"],
                    )
                else:
                    await self._window_store.mark_crystallization_proposed(
                        window_id,
                        crystallization_id=cid,
                    )
                for corr in corr_ids:
                    await publish_spark_meta_patch(
                        bus,
                        corr,
                        {
                            "consolidation_gate": {
                                "action": "discard" if outcome == "discarded_external_platform" else "propose",
                                "crystallization_id": cid,
                                "formation_outcome": outcome,
                            }
                        },
                    )
                return
            except Exception:
                logger.exception("memory_consolidation_gate_failed window_id=%s", window_id)
                await self._window_store.mark_failed(window_id)
                return

        transcript = build_window_transcript(turns)
        try:
            budget_config = suggest_token_budget_config_from_mapping(settings)
            raw = await suggest_with_escalation(
                bus,
                transcript=transcript,
                cortex_request_channel=settings.CHANNEL_CORTEX_REQUEST,
                cortex_result_prefix=settings.CHANNEL_CORTEX_RESULT_PREFIX,
                source=ServiceRef(
                    name=settings.SERVICE_NAME,
                    version=settings.SERVICE_VERSION,
                    node=settings.NODE_NAME,
                ),
                timeout_sec=float(settings.MEMORY_SUGGEST_TIMEOUT_SEC),
                budget_config=budget_config,
            )
            draft_dict = extract_suggest_draft_dict_from_cortex_payload(raw)
            draft_dict = sanitize_suggest_draft_dict(draft_dict)
            corr_ids = window.get("turn_correlation_ids") or []
            turns = window.get("turns") or []
            from orion.memory_graph.consolidation_draft_hydrate import hydrate_draft_utterance_text

            turns_by_correlation = {
                str(t.get("correlation_id")): t
                for t in turns
                if isinstance(t, dict) and str(t.get("correlation_id") or "").strip()
            }
            draft_dict = hydrate_draft_utterance_text(
                draft_dict,
                turn_correlation_ids=corr_ids,
                turns_by_correlation=turns_by_correlation,
            )
            draft = SuggestDraftV1.model_validate(draft_dict)
            draft_id = await insert_pending_draft(
                self._pool,
                memory_window_id=window_id,
                draft=draft,
                turn_correlation_ids=corr_ids,
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            for corr in corr_ids:
                await publish_spark_meta_patch(
                    bus,
                    corr,
                    {
                        "memory_window_id": window_id,
                        "memory_consolidated_at": now_iso,
                    },
                )
            await self._window_store.mark_consolidated(window_id, draft_id=draft_id)
        except Exception:
            logger.exception("memory_consolidation_suggest_failed window_id=%s", window_id)
            await self._window_store.mark_failed(window_id)


async def _maybe_publish_turn_change_signal(
    bus: OrionBusAsync,
    *,
    correlation_id: str,
    appraisal: dict[str, Any],
) -> None:
    from orion.memory.turn_change_signal import build_turn_change_signal

    if appraisal.get("turn_change_status") != "ok":
        return
    novelty = appraisal.get("novelty_score")
    if not isinstance(novelty, (int, float)) or float(novelty) < settings.TURN_CHANGE_SUBSTRATE_THRESHOLD:
        return
    confidence = appraisal.get("confidence")
    if confidence is None or float(confidence) < settings.TURN_CHANGE_CONFIDENCE_MARGIN:
        return
    shift_kind = str(appraisal.get("shift_kind") or "NONE")
    signal = build_turn_change_signal(
        correlation_id=correlation_id,
        shift_kind=shift_kind,
        novelty_score=float(novelty),
        confidence=float(confidence),
    )
    env = BaseEnvelope(
        kind="signal.memory_consolidation.turn_change",
        correlation_id=correlation_id,
        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        payload=signal.model_dump(mode="json"),
    )
    channel = f"{settings.CHANNEL_SIGNALS_PREFIX}:memory_consolidation"
    await bus.publish(channel, env)


async def publish_episode_closed(bus: OrionBusAsync, event: MemoryEpisodeClosedV1) -> None:
    env = BaseEnvelope(
        kind=MEMORY_EPISODE_CLOSED_KIND,
        correlation_id=event.episode_id,
        source=ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.NODE_NAME,
        ),
        payload=event.model_dump(mode="json"),
    )
    await bus.publish(settings.CHANNEL_MEMORY_EPISODE_CLOSED, env)


async def observe_shadow_episode(
    bus: OrionBusAsync,
    episode_store: Any,
    *,
    turn: MemoryTurnPersistedV1,
    scores: dict[str, Any],
    legacy_close_reason: str | None,
) -> MemoryEpisodeClosedV1 | None:
    """Run boundary Rule 3 in shadow on this turn and publish any episode it closes.

    Fail-open by construction: any error here is logged and swallowed, so the
    live window path never depends on the shadow tracker (or on its migration
    having been applied). AI Town and other discard platforms are excluded --
    episodes are Juniper's conversation, the spec's privacy boundary.
    """
    if episode_store is None or not settings.MEMORY_EPISODE_SHADOW_ENABLED:
        return None
    if turn.source_platform and turn.source_platform in settings.discard_platforms:
        return None
    try:
        event = await episode_store.observe_turn(
            turn, scores, legacy_close_reason=legacy_close_reason
        )
    except Exception:
        logger.exception("memory_episode_shadow_observe_failed corr=%s", turn.correlation_id)
        return None
    pending: list[MemoryEpisodeClosedV1] = [event] if event is not None else []
    try:
        for backlog in await episode_store.unpublished_closed(limit=5):
            if not any(p.episode_id == backlog.episode_id for p in pending):
                pending.append(backlog)
    except Exception:
        logger.exception("memory_episode_shadow_backlog_failed")
    for ev in pending:
        try:
            await publish_episode_closed(bus, ev)
            await episode_store.mark_published(ev.episode_id)
            logger.info(
                "memory_episode_closed episode=%s status=%s reason=%s turns=%s close_lag_sec=%s",
                ev.episode_id,
                ev.episode_status,
                ev.close_reason,
                len(ev.turn_ids),
                ev.close_lag_sec,
            )
        except Exception:
            logger.exception("memory_episode_closed_publish_failed episode=%s", ev.episode_id)
    return event


async def handle_memory_turn_persisted(
    env: BaseEnvelope,
    *,
    bus: OrionBusAsync,
    window_store: WindowStore,
    suggest_runner: ConsolidationSuggestRunner,
    episode_store: Any = None,
) -> None:
    turn = MemoryTurnPersistedV1.model_validate(env.payload)
    assert str(env.correlation_id) == turn.correlation_id, "correlation_id mismatch"
    existing_appraisal = turn.spark_meta.get("turn_change_appraisal")
    if (
        isinstance(existing_appraisal, dict)
        and existing_appraisal.get("turn_change_status") == "ok"
    ):
        return
    if turn.initiated_by == "orion":
        # Orion wrote on their own: there is no prompt to classify against the window, and the
        # legacy crystallization windows are being retired (spec 2026-09-30), so this turn goes
        # only to the episode tracker. It can close a stale episode (Rule 3's time-gap branch;
        # it carries no phase stamp) and it becomes part of the next one, so Orion's own
        # messages are remembered with the replies they got.
        await observe_shadow_episode(bus, episode_store, turn=turn, scores={}, legacy_close_reason=None)
        return
    # Boundary Fix 2: classify each turn once. sql-writer publishes this event
    # twice per turn; the second copy used to be re-classified against a
    # window that already held the turn, so it compared the turn with itself
    # and its (low) score overwrote chat_history_log while the first (high)
    # score had already closed the window. See WindowStore.find_windowed_turn.
    try:
        already = await window_store.find_windowed_turn(turn.correlation_id)
    except Exception:
        logger.exception("memory_turn_dedup_lookup_failed corr=%s", turn.correlation_id)
        already = None
    if isinstance(already, dict):
        logger.info(
            "memory_turn_duplicate_skipped corr=%s window=%s",
            turn.correlation_id,
            already.get("memory_window_id"),
        )
        return
    # Prior turns for classification come from this turn's OWN platform window.
    # Classifying a Juniper turn against a backdrop of NPC dialogue (or the
    # reverse) was scoring novelty and topic shift against an unrelated
    # conversation -- the same global-cursor bug, showing up in the classifier
    # rather than in the queue.
    open_row = await window_store._get_open_window(turn.source_platform)
    prior_turns = (
        await window_store.get_window_turns(open_row["memory_window_id"])
        if open_row is not None
        else []
    )
    live_turn = legacy_view(turn, settings)
    patch_fields = await classify_turn(bus, turn=live_turn, prior_turns=prior_turns, settings=settings)
    await publish_spark_meta_patch(bus, turn.correlation_id, patch_fields)
    try:
        await _maybe_publish_turn_change_signal(
            bus,
            correlation_id=turn.correlation_id,
            appraisal=patch_fields.get("turn_change_appraisal") or {},
        )
    except Exception:
        logger.exception("turn_change_signal_publish_failed corr=%s", turn.correlation_id)
    await window_store.append_turn(turn, scores=patch_fields)
    open_row = await window_store._get_open_window(turn.source_platform)
    window_turns = (
        await window_store.get_window_turns(open_row["memory_window_id"])
        if open_row is not None
        else []
    )
    legacy_reason = legacy_close_decision(live_turn, patch_fields, window_turns=window_turns)
    await observe_shadow_episode(
        bus,
        episode_store,
        turn=turn,
        scores=patch_fields,
        legacy_close_reason=legacy_reason,
    )
    if legacy_reason is not None:
        closed = await window_store.close_current_window(
            turn.correlation_id, source_platform=turn.source_platform
        )
        if closed.get("turn_correlation_ids"):
            try:
                score = patch_fields.get("conversation_boundary_score")
                await window_store.record_close_audit(
                    closed["memory_window_id"],
                    close_reason=legacy_reason,
                    boundary_score_at_close=float(score) if isinstance(score, (int, float)) else None,
                )
            except Exception:
                logger.warning(
                    "memory_window_close_audit_failed window=%s (migration applied?)",
                    closed.get("memory_window_id"),
                    exc_info=True,
                )
            await suggest_runner.consolidate_window(closed, bus=bus)
