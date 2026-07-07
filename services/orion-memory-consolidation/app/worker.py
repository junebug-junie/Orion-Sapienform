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

from app.window_fetch import should_close_turn
from app.classify import classify_turn
from app.settings import settings
from app.window_state import WindowStore
from orion.schemas.memory_consolidation import (
    CHAT_HISTORY_SPARK_META_PATCH_KIND,
    ChatHistorySparkMetaPatchV1,
    MemoryTurnPersistedV1,
)

logger = logging.getLogger(__name__)

_MAX_TURNS_FOR_SUGGEST = 3
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
            from orion.memory.consolidation_gate import consolidation_memory_gate
            from orion.memory.consolidation_grammar import fetch_grammar_evidence_for_window
            from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle
            from orion.memory.crystallization.intake_consolidation_window import (
                build_crystallization_from_window,
            )
            from orion.memory.crystallization.repository import insert_crystallization

            try:
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
                cid = await insert_crystallization(self._pool, crystallization)
                await self._window_store.mark_crystallization_proposed(
                    window_id,
                    crystallization_id=cid,
                )
                await emit_crystallization_lifecycle(
                    bus,
                    lifecycle="proposed",
                    crystallization=crystallization,
                    service_name=settings.SERVICE_NAME,
                    service_version=settings.SERVICE_VERSION,
                    node_name=settings.NODE_NAME,
                )
                for corr in corr_ids:
                    await publish_spark_meta_patch(
                        bus,
                        corr,
                        {
                            "consolidation_gate": {
                                "action": "propose",
                                "crystallization_id": cid,
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


async def handle_memory_turn_persisted(
    env: BaseEnvelope,
    *,
    bus: OrionBusAsync,
    window_store: WindowStore,
    suggest_runner: ConsolidationSuggestRunner,
) -> None:
    turn = MemoryTurnPersistedV1.model_validate(env.payload)
    assert str(env.correlation_id) == turn.correlation_id, "correlation_id mismatch"
    existing_appraisal = turn.spark_meta.get("turn_change_appraisal")
    if (
        isinstance(existing_appraisal, dict)
        and existing_appraisal.get("turn_change_status") == "ok"
    ):
        return
    open_row = await window_store._get_open_window()
    prior_turns = (
        await window_store.get_window_turns(open_row["memory_window_id"])
        if open_row is not None
        else []
    )
    patch_fields = await classify_turn(bus, turn=turn, prior_turns=prior_turns, settings=settings)
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
    open_row = await window_store._get_open_window()
    window_turns = (
        await window_store.get_window_turns(open_row["memory_window_id"])
        if open_row is not None
        else []
    )
    if should_close_turn(turn, patch_fields, window_turns=window_turns):
        closed = await window_store.close_current_window(turn.correlation_id)
        if closed.get("turn_correlation_ids"):
            await suggest_runner.consolidate_window(closed, bus=bus)
