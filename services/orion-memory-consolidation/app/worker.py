from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory_graph.draft_repository import insert_pending_draft
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.draft_sanitize import sanitize_suggest_draft_dict
from orion.memory_graph.suggest_runner import suggest_once

from app.boundary import should_close_window
from app.classify import classify_turn
from app.settings import settings
from app.window_state import WindowStore
from orion.schemas.memory_consolidation import (
    CHAT_HISTORY_SPARK_META_PATCH_KIND,
    ChatHistorySparkMetaPatchV1,
    MemoryTurnPersistedV1,
)

logger = logging.getLogger(__name__)


def build_window_transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        sig = t.get("memory_significance_score")
        prefix = f"[sig={sig:.2f}] " if isinstance(sig, (int, float)) else ""
        lines.append(f"{prefix}User: {t.get('prompt', '')}\nOrion: {t.get('response', '')}\n")
    return "\n".join(lines)


async def publish_spark_meta_patch(
    bus: OrionBusAsync,
    correlation_id: str,
    patch_fields: dict[str, Any],
) -> None:
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
    def __init__(self, pool, window_store: WindowStore):
        self._pool = pool
        self._window_store = window_store

    async def consolidate_window(self, window: dict[str, Any], *, bus: OrionBusAsync) -> None:
        window_id = window["memory_window_id"]
        turns = window.get("turns") or []
        transcript = build_window_transcript(turns)
        try:
            raw = await suggest_once(
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
            )
            draft_dict = _extract_draft_dict(raw)
            draft_dict = sanitize_suggest_draft_dict(draft_dict)
            draft = SuggestDraftV1.model_validate(draft_dict)
            corr_ids = window.get("turn_correlation_ids") or []
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


def _extract_draft_dict(raw: dict[str, Any]) -> dict[str, Any]:
    if "draft" in raw and isinstance(raw["draft"], dict):
        return raw["draft"]
    steps = raw.get("steps") or raw.get("step_results") or []
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        detail = step.get("detail") if isinstance(step.get("detail"), dict) else {}
        output = detail.get("output") or detail.get("draft")
        if isinstance(output, dict):
            return output
        text = detail.get("text") or detail.get("content")
        if isinstance(text, str) and text.strip().startswith("{"):
            return json.loads(text)
    if isinstance(raw.get("result"), dict):
        return _extract_draft_dict(raw["result"])
    raise ValueError("memory_graph_suggest_draft_not_found")


async def handle_memory_turn_persisted(
    env: BaseEnvelope,
    *,
    bus: OrionBusAsync,
    window_store: WindowStore,
    suggest_runner: ConsolidationSuggestRunner,
) -> None:
    turn = MemoryTurnPersistedV1.model_validate(env.payload)
    assert str(env.correlation_id) == turn.correlation_id, "correlation_id mismatch"
    patch_fields = await classify_turn(bus, turn=turn, settings=settings)
    await publish_spark_meta_patch(bus, turn.correlation_id, patch_fields)
    await window_store.append_turn(turn, scores=patch_fields)
    if should_close_window(turn, patch_fields, settings):
        closed = await window_store.close_current_window(turn.correlation_id)
        if closed.get("turn_correlation_ids"):
            await suggest_runner.consolidate_window(closed, bus=bus)
