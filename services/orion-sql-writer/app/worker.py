# services/orion-sql-writer/app/worker.py
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import uuid
from collections import deque
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple, Type

from sqlalchemy import func, inspect, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.types import DateTime, String, Text
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel

from app.settings import settings
from app.db import get_session, remove_session, session_factory
from app.models import (
    BiometricsTelemetry,
    BiometricsSummarySQL,
    BiometricsInductionSQL,
    CausalGeometrySnapshotSQL,
    ChatHistoryLogSQL,
    AitownChatHistoryLogSQL,
    ChatGptLogSQL,
    ChatGptMessageSQL,
    ChatGptImportRunSQL,
    ChatGptConversationSQL,
    ChatGptDerivedExampleSQL,
    ChatMessageSQL,
    ChatResponseFeedbackSQL,
    CollapseEnrichment,
    CollapseMirror,
    MetacogEntry,
    RepairPressureAppraisalLog,
    Dream,
    SparkIntrospectionLogSQL,
    SparkTelemetrySQL,
    BusFallbackLog,
    CognitionTraceSQL,
    ThoughtDecisionSQL,
    MetacognitionTickSQL,
    MetacogTriggerSQL,
    MetacognitiveTraceSQL,
    NotificationRequestDB,
    NotificationReceiptDB,
    RecipientProfileDB,
    NotificationPreferenceDB,
    JournalEntrySQL,
    SocialRoomTurnSQL,
    ExternalRoomMessageSQL,
    ExternalRoomParticipantSQL,
    EndogenousRuntimeRecordSQL,
    EndogenousRuntimeAuditSQL,
    CalibrationProfileAuditSQL,
    JournalEntryIndexSQL,
    EvidenceUnitSQL,
    WorldPulseClaimSQL,
    WorldPulseArticleClusterSQL,
    WorldPulseArticleSQL,
    WorldPulseContextCapsuleSQL,
    WorldPulseDigestItemSQL,
    WorldPulseDigestSQL,
    WorldPulseEntitySQL,
    WorldPulseEventSQL,
    WorldPulseHubMessageSQL,
    WorldPulseLearningDeltaSQL,
    WorldPulsePublishStatusSQL,
    WorldPulseRunSQL,
    WorldPulseSituationBriefSQL,
    WorldPulseSituationChangeSQL,
    WorldPulseWorthReadingSQL,
    WorldPulseWorthWatchingSQL,
    MindRunSQL,
    VisionEventSQL,
    ActionOutcomeSQL,
    DominanceStreakTickSQL,
    DevEconomicsLedgerSQL,
    DocSemanticDriftSQL,
    JuniperAffectiveStateSQL,
    PhiRewardSQL,
    GrammarEventSQL,
    EquilibriumServiceTransitionSQL,
    HarnessTurnTraceSQL,
)
from app.harness_turn_trace_persist import upsert_harness_turn_trace
from orion.autonomy.models import ActionOutcomeEmitV1
from orion.evidence_index import build_evidence_units

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.journaler import (
    JOURNAL_CREATED_KIND,
    JOURNAL_WRITE_KIND,
    JournalEntryWriteV1,
    JournalTriggerV1,
    build_created_event_payload,
    build_journal_entry_index_payload,
)
from orion.schemas.chat_stance import ChatStanceBrief
from orion.schemas.field_goal import DominanceStreakTickV1
from orion.schemas.dev_economics import DevEconomicsLedgerV1
from orion.schemas.doc_semantic_drift import DocSemanticDriftV1
from orion.schemas.affective_state import JuniperAffectiveStateV1

from app.route_coverage import log_route_coverage

# Shared schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry, CollapseMirrorStoredV1
from orion.schemas.metacog_entry import MetacogEntryV1
from orion.schemas.repair_pressure_appraisal import RepairPressureAppraisalV1
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.telemetry.biometrics import BiometricsPayload, BiometricsSummaryV1, BiometricsInductionV1
from orion.schemas.causal_geometry import CausalGeometrySnapshotV1
from orion.schemas.telemetry.dream import DreamRequest, DreamResultV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.thought import ThoughtEventV1
from orion.schemas.chat_history import ChatHistoryMessageV1
from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1
from orion.schemas.chat_gpt_log import (
    ChatGptConversationV1,
    ChatGptDerivedExampleV1,
    ChatGptImportRunV1,
    ChatGptLogTurnV1,
    ChatGptMessageV1,
)
from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1
from orion.schemas.memory_consolidation import (
    MEMORY_TURN_PERSISTED_KIND,
    ChatHistorySparkMetaPatchV1,
    MemoryTurnPersistedV1,
)
from orion.schemas.social_chat import SocialRoomTurnStoredV1, SocialRoomTurnV1
from orion.schemas.social_bridge import (
    ExternalRoomMessageV1,
    ExternalRoomParticipantV1,
    ExternalRoomPostResultV1,
    ExternalRoomTurnSkippedV1,
)
from orion.schemas.telemetry.metacognition import MetacognitionTickV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from orion.core.schemas.endogenous_runtime import EndogenousRuntimeExecutionRecordV1, EndogenousRuntimeAuditV1
from orion.core.schemas.calibration_adoption import CalibrationProfileAuditV1
from orion.schemas.evidence_index import EvidenceUnitV1
from orion.schemas.mind.artifact import MindRunArtifactV1
from orion.schemas.vision import VisionEventBundleItem
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.telemetry.phi_encoder import PhiIntrinsicRewardV1
from orion.schemas.telemetry.system_health import EquilibriumServiceTransitionV1
from orion.schemas.world_pulse import (
    ClaimRecordV1,
    DailyWorldPulseItemV1,
    DailyWorldPulseV1,
    EntityRecordV1,
    EventRecordV1,
    HubWorldPulseMessageV1,
    TopicSituationBriefV1,
    WorldContextCapsuleV1,
    WorldLearningDeltaV1,
    WorldPulseRunResultV1,
    WorthReadingItemV1,
    WorthWatchingItemV1,
)

from app.spark_contract_metrics import SparkContractMetrics, LEGACY_KINDS

try:
    from orion.schemas.telemetry.spark import SparkTelemetryPayload
except ImportError:
    SparkTelemetryPayload = None
try:
    from orion.normalizers.spark import normalize_spark_state_snapshot, normalize_spark_telemetry
except ImportError:
    normalize_spark_state_snapshot = None
    normalize_spark_telemetry = None

logger = logging.getLogger("sql-writer")
GrammarWorkItem = tuple[BaseEnvelope, GrammarEventV1, dict[str, Any], str]
_GRAMMAR_EXECUTORS: list[concurrent.futures.ThreadPoolExecutor] | None = None
_GRAMMAR_QUEUES: list[asyncio.Queue[GrammarWorkItem]] | None = None
_GRAMMAR_DEFERRED: list[deque[GrammarWorkItem]] | None = None
_GRAMMAR_WORKER_TASKS: list[asyncio.Task] = []
_GRAMMAR_BACKGROUND_TASKS: set[asyncio.Task] = set()
_WRITE_SEMAPHORE: asyncio.Semaphore | None = None
_SPARK_CONTRACT_METRICS = SparkContractMetrics()
COLLAPSE_STORED_KIND = "collapse.mirror.stored.v1"
SOCIAL_TURN_STORED_KIND = "social.turn.stored.v1"
INSERT_ONLY_MODELS = {
    JournalEntrySQL,
    SocialRoomTurnSQL,
    ChatResponseFeedbackSQL,
    MindRunSQL,
    CausalGeometrySnapshotSQL,
    EquilibriumServiceTransitionSQL,
}


def _get_write_semaphore() -> asyncio.Semaphore:
    global _WRITE_SEMAPHORE
    if _WRITE_SEMAPHORE is None:
        limit = max(1, int(settings.sql_writer_max_inflight))
        _WRITE_SEMAPHORE = asyncio.Semaphore(limit)
    return _WRITE_SEMAPHORE


def _grammar_shard_count() -> int:
    return max(1, int(settings.sql_writer_grammar_workers))


def grammar_queue_snapshot() -> dict[str, Any]:
    """Expose grammar worker queue depths for runtime truth endpoints."""
    shard_count = _grammar_shard_count()
    queues = _GRAMMAR_QUEUES
    if queues is None:
        return {"workers": shard_count, "total_depth": 0, "shards": []}
    shards = [
        {"shard": idx, "depth": queues[idx].qsize(), "maxsize": queues[idx].maxsize}
        for idx in range(len(queues))
    ]
    return {
        "workers": len(queues),
        "total_depth": sum(item["depth"] for item in shards),
        "shards": shards,
    }


def _grammar_shard_index(trace_id: str) -> int:
    return hash(trace_id) % _grammar_shard_count()


def _get_grammar_executors() -> list[concurrent.futures.ThreadPoolExecutor]:
    global _GRAMMAR_EXECUTORS
    if _GRAMMAR_EXECUTORS is None:
        _GRAMMAR_EXECUTORS = [
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"sql-writer-grammar-{idx}",
            )
            for idx in range(_grammar_shard_count())
        ]
    return _GRAMMAR_EXECUTORS


def _ensure_grammar_workers() -> None:
    global _GRAMMAR_QUEUES, _GRAMMAR_WORKER_TASKS
    shard_count = _grammar_shard_count()
    _get_grammar_executors()
    if _GRAMMAR_QUEUES is None:
        _GRAMMAR_QUEUES = [asyncio.Queue(maxsize=512) for _ in range(shard_count)]
    while len(_GRAMMAR_WORKER_TASKS) < shard_count:
        shard = len(_GRAMMAR_WORKER_TASKS)
        task = asyncio.create_task(_grammar_worker_loop(shard))
        _GRAMMAR_WORKER_TASKS.append(task)
        _GRAMMAR_BACKGROUND_TASKS.add(task)

        def _done(done_task: asyncio.Task) -> None:
            _GRAMMAR_BACKGROUND_TASKS.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                logger.error("grammar_worker_crashed error=%s", exc)

        task.add_done_callback(_done)


def _grammar_deferred(shard: int) -> deque[GrammarWorkItem]:
    global _GRAMMAR_DEFERRED
    if _GRAMMAR_DEFERRED is None:
        _GRAMMAR_DEFERRED = [deque() for _ in range(_grammar_shard_count())]
    return _GRAMMAR_DEFERRED[shard]


async def _grammar_worker_loop(shard: int) -> None:
    queues = _GRAMMAR_QUEUES
    if queues is None:
        return
    queue = queues[shard]
    deferred = _grammar_deferred(shard)
    batch_max = max(1, int(settings.sql_writer_grammar_trace_batch_max))

    while True:
        tasks_from_queue = 0

        if deferred:
            seed = deferred.popleft()
        else:
            seed = await queue.get()
            tasks_from_queue += 1

        items: list[GrammarWorkItem] = [seed]
        trace_id = seed[1].trace_id

        while len(items) < batch_max and items[-1][1].event_kind != "trace_ended":
            if deferred and deferred[0][1].trace_id == trace_id:
                items.append(deferred.popleft())
                continue
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.15)
                tasks_from_queue += 1
                if item[1].trace_id == trace_id:
                    items.append(item)
                else:
                    deferred.append(item)
            except asyncio.TimeoutError:
                break

        try:
            if len(items) == 1:
                env, event, payload, corr_id = items[0]
                await _persist_grammar_event_envelope(
                    env,
                    event=event,
                    payload=payload,
                    corr_id=corr_id,
                    shard=shard,
                )
            else:
                await _persist_grammar_trace_batch_envelope(items, shard=shard)
        except Exception:
            logger.exception(
                "grammar_worker_unhandled shard=%s trace_id=%s batch_size=%s",
                shard,
                trace_id,
                len(items),
            )
        finally:
            for _ in range(tasks_from_queue):
                queue.task_done()


def _spawn_grammar_persist(
    env: BaseEnvelope,
    *,
    event: GrammarEventV1,
    payload: dict[str, Any],
    corr_id: str,
) -> None:
    _ensure_grammar_workers()
    queues = _GRAMMAR_QUEUES
    if queues is None:
        return
    shard = _grammar_shard_index(event.trace_id)
    queue = queues[shard]
    try:
        queue.put_nowait((env, event, payload, corr_id))
    except asyncio.QueueFull:
        logger.warning(
            "grammar_queue_full shard=%s event_id=%s trace_id=%s",
            shard,
            event.event_id,
            event.trace_id,
        )
        asyncio.create_task(
            asyncio.to_thread(
                _write_fallback,
                env.kind,
                corr_id,
                payload,
                "grammar queue full",
            )
        )


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


def _legacy_action(kind: str, mode: str, legacy_kinds: set[str]) -> str:
    if kind not in legacy_kinds:
        return "noop"
    if mode == "warn":
        return "warn"
    if mode == "drop":
        return "drop"
    return "accept"

# Map: route_key -> (SQLAlchemy model, Pydantic schema model)
MODEL_MAP: Dict[str, Tuple[Type[Any], Optional[Type[BaseModel]]]] = {
    "CollapseMirror": (CollapseMirror, CollapseMirrorEntry),
    "MetacogEntry": (MetacogEntry, MetacogEntryV1),
    "RepairPressureAppraisalLog": (RepairPressureAppraisalLog, RepairPressureAppraisalV1),
    "CollapseEnrichment": (CollapseEnrichment, MetaTagsPayload),
    "ChatHistoryLogSQL": (ChatHistoryLogSQL, None),
    "ChatGptLogSQL": (ChatGptLogSQL, ChatGptLogTurnV1),
    "ChatGptMessageSQL": (ChatGptMessageSQL, ChatGptMessageV1),
    "ChatGptImportRunSQL": (ChatGptImportRunSQL, ChatGptImportRunV1),
    "ChatGptConversationSQL": (ChatGptConversationSQL, ChatGptConversationV1),
    "ChatGptDerivedExampleSQL": (ChatGptDerivedExampleSQL, ChatGptDerivedExampleV1),
    "ChatMessageSQL": (ChatMessageSQL, ChatHistoryMessageV1),
    "ChatResponseFeedbackSQL": (ChatResponseFeedbackSQL, ChatResponseFeedbackV1),
    "Dream": (Dream, None),
    "BiometricsTelemetry": (BiometricsTelemetry, BiometricsPayload),
    "BiometricsSummarySQL": (BiometricsSummarySQL, BiometricsSummaryV1),
    "BiometricsInductionSQL": (BiometricsInductionSQL, BiometricsInductionV1),
    "CausalGeometrySnapshotSQL": (CausalGeometrySnapshotSQL, CausalGeometrySnapshotV1),
    "CognitionTraceSQL": (CognitionTraceSQL, CognitionTracePayload),
    "ThoughtDecisionSQL": (ThoughtDecisionSQL, ThoughtEventV1),
    # schema_cls=None: harness-governor already validated each of the four
    # source molecules before publishing (orion/harness/finalize.py); the
    # column this write fills is picked from the payload's own
    # schema_version at persist time (upsert_harness_turn_trace), not a
    # single fixed pydantic model here -- re-validating against one schema
    # class would reject the other three kinds that also route here.
    "HarnessTurnTraceSQL": (HarnessTurnTraceSQL, None),
    "SparkIntrospectionLogSQL": (SparkIntrospectionLogSQL, None),
    "SparkTelemetrySQL": (SparkTelemetrySQL, SparkTelemetryPayload),
    "MetacognitionTickSQL": (MetacognitionTickSQL, MetacognitionTickV1),
    "MetacogTriggerSQL": (MetacogTriggerSQL, MetacogTriggerV1),
    "MetacognitiveTraceSQL": (MetacognitiveTraceSQL, MetacognitiveTraceV1),
    "NotificationRequestDB": (NotificationRequestDB, None),
    "NotificationReceiptDB": (NotificationReceiptDB, None),
    "RecipientProfileDB": (RecipientProfileDB, None),
    "NotificationPreferenceDB": (NotificationPreferenceDB, None),
    "JournalEntrySQL": (JournalEntrySQL, JournalEntryWriteV1),
    "SocialRoomTurnSQL": (SocialRoomTurnSQL, SocialRoomTurnV1),
    "ExternalRoomMessageSQL": (ExternalRoomMessageSQL, ExternalRoomMessageV1),
    "ExternalRoomParticipantSQL": (ExternalRoomParticipantSQL, ExternalRoomParticipantV1),
    "EndogenousRuntimeRecordSQL": (EndogenousRuntimeRecordSQL, EndogenousRuntimeExecutionRecordV1),
    "EndogenousRuntimeAuditSQL": (EndogenousRuntimeAuditSQL, EndogenousRuntimeAuditV1),
    "CalibrationProfileAuditSQL": (CalibrationProfileAuditSQL, CalibrationProfileAuditV1),
    "EvidenceUnitSQL": (EvidenceUnitSQL, EvidenceUnitV1),
    "WorldPulseRunSQL": (WorldPulseRunSQL, WorldPulseRunResultV1),
    "WorldPulseDigestSQL": (WorldPulseDigestSQL, DailyWorldPulseV1),
    "WorldPulseDigestItemSQL": (WorldPulseDigestItemSQL, DailyWorldPulseItemV1),
    "WorldPulseArticleSQL": (WorldPulseArticleSQL, None),
    "WorldPulseArticleClusterSQL": (WorldPulseArticleClusterSQL, None),
    "WorldPulseClaimSQL": (WorldPulseClaimSQL, ClaimRecordV1),
    "WorldPulseEventSQL": (WorldPulseEventSQL, EventRecordV1),
    "WorldPulseHubMessageSQL": (WorldPulseHubMessageSQL, HubWorldPulseMessageV1),
    "WorldPulseEntitySQL": (WorldPulseEntitySQL, EntityRecordV1),
    "WorldPulseSituationBriefSQL": (WorldPulseSituationBriefSQL, TopicSituationBriefV1),
    "WorldPulseSituationChangeSQL": (WorldPulseSituationChangeSQL, None),
    "WorldPulseLearningDeltaSQL": (WorldPulseLearningDeltaSQL, WorldLearningDeltaV1),
    "WorldPulseWorthReadingSQL": (WorldPulseWorthReadingSQL, WorthReadingItemV1),
    "WorldPulseWorthWatchingSQL": (WorldPulseWorthWatchingSQL, WorthWatchingItemV1),
    "WorldPulseContextCapsuleSQL": (WorldPulseContextCapsuleSQL, WorldContextCapsuleV1),
    "WorldPulsePublishStatusSQL": (WorldPulsePublishStatusSQL, None),
    "MindRunSQL": (MindRunSQL, MindRunArtifactV1),
    "ChatResponseFeedbackSQL": (ChatResponseFeedbackSQL, ChatResponseFeedbackV1),
    "GrammarEventSQL": (GrammarEventSQL, GrammarEventV1),
    "VisionEventSQL": (VisionEventSQL, VisionEventBundleItem),
    "ActionOutcomeSQL": (ActionOutcomeSQL, ActionOutcomeEmitV1),
    "DominanceStreakTickSQL": (DominanceStreakTickSQL, DominanceStreakTickV1),
    "DevEconomicsLedgerSQL": (DevEconomicsLedgerSQL, DevEconomicsLedgerV1),
    "DocSemanticDriftSQL": (DocSemanticDriftSQL, DocSemanticDriftV1),
    "JuniperAffectiveStateSQL": (JuniperAffectiveStateSQL, JuniperAffectiveStateV1),
    "PhiRewardSQL": (PhiRewardSQL, PhiIntrinsicRewardV1),
    "EquilibriumServiceTransitionSQL": (EquilibriumServiceTransitionSQL, EquilibriumServiceTransitionV1),
}


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=settings.heartbeat_interval_sec,
        health_channel=settings.health_channel,
        error_channel=settings.error_channel,
        shutdown_timeout_sec=settings.shutdown_grace_sec,
    )


def _normalize_dream_envelope_payload(
    kind: str, payload: Any, extra_sql_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Map dream.result.v1 / dream.log payloads to SQL row fields for the dreams table.
    """
    if not isinstance(payload, dict):
        payload = {}
    corr = extra_sql_fields.get("correlation_id")

    if kind == "dream.result.v1":
        data = dict(payload)
        if corr and not data.get("correlation_id"):
            data["correlation_id"] = corr
        dr = DreamResultV1.model_validate(data)
        return {
            "dream_date": dr.dream_date,
            "tldr": dr.tldr,
            "themes": dr.themes or [],
            "symbols": dr.symbols or {},
            "narrative": dr.narrative,
            "fragments": dr.fragments or [],
            "metrics": dr.merged_metrics_for_sql(),
            "created_at": dr.created_at,
        }

    dq = DreamRequest.model_validate(payload)
    text = (dq.context_text or "").strip()
    return {
        "dream_date": date.today(),
        "tldr": text[:500] if text else None,
        "narrative": text or None,
        "themes": [],
        "symbols": {},
        "fragments": [],
        "metrics": {
            "legacy_dream_log": True,
            "metadata": dq.metadata,
            "integration_mode": dq.integration_mode,
            "mood": dq.mood,
            "correlation_id": corr,
        },
    }


def _coerce_payload(model_cls, payload: Any):
    if model_cls is None:
        return payload
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(payload)
    return payload


_ENVELOPE_MARKER_KEYS = {
    "schema_id",
    "schema",
    "schema_version",
    "id",
    "kind",
    "causality_chain",
    "source",
    "ttl_ms",
    "trace",
    "reply_to",
}


def _looks_like_nested_envelope(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("payload"), dict):
        return False
    return bool(_ENVELOPE_MARKER_KEYS.intersection(payload.keys()))


def _payload_for_schema_validation(payload: Any, schema_cls: Type[BaseModel] | None, *, kind: str | None = None) -> tuple[Any, str]:
    if schema_cls is None:
        return payload, "raw"
    if _looks_like_nested_envelope(payload):
        nested_payload = payload.get("payload")
        logger.info(
            "sql_writer_payload_boundary kind=%s schema=%s boundary=envelope_payload",
            kind,
            getattr(schema_cls, "__name__", str(schema_cls)),
        )
        return nested_payload, "envelope_payload"
    return payload, "payload"


def _extract_journal_index_context(payload: Any) -> tuple[JournalTriggerV1 | None, ChatStanceBrief | None, dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, None, None

    container = payload
    if _looks_like_nested_envelope(payload):
        container = payload

    raw_trigger = container.get("journal_trigger") or container.get("trigger")
    raw_stance = container.get("chat_stance_brief") or container.get("stance")
    raw_stance_meta = container.get("stance_metadata")
    raw_unc = container.get("llm_uncertainty")
    if isinstance(raw_unc, dict):
        if not isinstance(raw_stance_meta, dict):
            raw_stance_meta = {}
        raw_stance_meta = {**raw_stance_meta, "llm_uncertainty": raw_unc}

    trigger = None
    if isinstance(raw_trigger, dict):
        try:
            trigger = JournalTriggerV1.model_validate(raw_trigger)
        except Exception:
            logger.warning("Invalid journal trigger metadata for index payload; continuing without trigger.")

    stance = None
    if isinstance(raw_stance, dict):
        try:
            stance = ChatStanceBrief.model_validate(raw_stance)
        except Exception:
            logger.warning("Invalid chat stance metadata for index payload; continuing without stance.")

    stance_meta = raw_stance_meta if isinstance(raw_stance_meta, dict) else None
    return trigger, stance, stance_meta


def _json_sanitize(obj: Any, *, _seen: Optional[set[int]] = None, _depth: int = 0, _max_depth: int = 20) -> Any:
    if _seen is None:
        _seen = set()

    if _depth > _max_depth:
        return "__max_depth__"

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    oid = id(obj)
    if oid in _seen:
        return "__cycle__"
    _seen.add(oid)

    if hasattr(obj, "model_dump"):
        try:
            return _json_sanitize(obj.model_dump(mode="json"), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            return str(obj)

    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            kk = k if isinstance(k, str) else str(k)
            out[kk] = _json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [
            _json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
            for v in list(obj)
        ]

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, uuid.UUID):
        return str(obj)

    return str(obj)


def _spark_meta_minimal(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = row.get("metadata_") or row.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {"raw_metadata": str(meta)}

    meta = dict(meta)
    turn_effect = deepcopy(meta.get("turn_effect"))
    turn_effect_summary = meta.get("turn_effect_summary")
    meta.pop("spark_state_snapshot", None)

    out = {
        "phi": row.get("phi"),
        "novelty": row.get("novelty"),
        "trace_mode": row.get("trace_mode"),
        "trace_verb": row.get("trace_verb"),
        "timestamp": row.get("timestamp"),
        "stimulus_summary": row.get("stimulus_summary") or meta.get("stimulus_summary"),
        "vector_present": meta.get("vector_present"),
        "vector_ref": meta.get("vector_ref"),
        "node": row.get("node"),
        "metadata": meta,
    }
    if turn_effect is not None:
        out["turn_effect"] = turn_effect
    if turn_effect_summary is not None:
        out["turn_effect_summary"] = turn_effect_summary
    return _json_sanitize(out)


def _chat_history_llm_uncertainty_scalars(unc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(unc, dict):
        return {}
    available = unc.get("available")
    return {
        "llm_uncertainty_source": unc.get("source"),
        "llm_mean_logprob": unc.get("mean_logprob"),
        "llm_min_logprob": unc.get("min_logprob"),
        "llm_mean_top1_margin": unc.get("mean_top1_margin"),
        "llm_low_margin_token_count": unc.get("low_margin_token_count"),
        "llm_low_logprob_token_count": unc.get("low_logprob_token_count"),
        "llm_unstable_span_count": unc.get("unstable_span_count"),
        "llm_uncertainty_available": available if isinstance(available, bool) else None,
    }


# Keys written by memory-consolidation classify; introspector telemetry must not clobber these.
_CLASSIFY_SPARK_META_KEYS = frozenset(
    {
        "turn_change_appraisal",
        "turn_effect",
        "novelty",
        "memory_classify_status",
        "memory_classify_ts",
        "turn_change_classify_route",
        "memory_significance_score",
        "conversation_boundary_score",
    }
)


def _has_ok_turn_change_appraisal(meta: dict[str, Any]) -> bool:
    appraisal = meta.get("turn_change_appraisal")
    return isinstance(appraisal, dict) and appraisal.get("turn_change_status") == "ok"


def _merge_spark_meta(
    existing: Any,
    updates: Dict[str, Any],
    *,
    source: str = "patch",
) -> Dict[str, Any]:
    base: Dict[str, Any]
    if isinstance(existing, dict):
        base = deepcopy(existing)
    elif existing is None:
        base = {}
    else:
        base = {"raw_existing": str(existing)}

    incoming = dict(updates or {})
    if source == "telemetry" and _has_ok_turn_change_appraisal(base):
        incoming = {k: v for k, v in incoming.items() if k not in _CLASSIFY_SPARK_META_KEYS}

    for k, v in incoming.items():
        if v is None:
            continue
        base[k] = v

    return _json_sanitize(base)


def _chat_history_conflict_updates(
    cols: Iterable[str], stmt: Any, *, incoming_wins: bool, model_cls: type = ChatHistoryLogSQL
) -> dict[str, Any]:
    """Per-column ON CONFLICT rule for chat_history_log (or its AI Town
    mirror, ``AitownChatHistoryLogSQL`` -- see ``model_cls`` below).

    ``chat_history_log`` is not append-only -- one row is assembled from three
    separate bus events (user message, assistant message, turn), each carrying a
    different subset of the columns. So a conflict is never "we already have
    this, skip"; it is "merge what this event knows into what is already there".

    The two producers deliberately keep DIFFERENT policies, matching what each
    did before the upsert replaced their read-modify-write:

    ``incoming_wins=True`` (turn path) -- ``coalesce(nullif(excluded.c,''), c)``.
        The turn event is authoritative: it carries ``source`` and the enriched
        metadata, and its caller has already folded prior state into
        ``spark_meta``/``thought_process`` before calling, so those merged values
        must land. Equivalent to the old ``sess.merge()`` plus
        ``_coalesce_chat_history_turn_fields``' protection of prompt/response.

    ``incoming_wins=False`` (message path) -- ``coalesce(c, nullif(excluded.c,''))``.
        Fill-only. Reproduces exactly what ``_ensure_chat_history_from_message``
        used to do field by field (``if not existing.prompt: ...``,
        ``if session_id and not existing.session_id: ...``). Without this the
        assistant message -- published AFTER the turn on the WS path
        (``websocket_handler.py`` ~1802 then ~1911) -- would clobber the turn's
        canonical ``client_meta``/``memory_status``/``memory_tier``.

    ``nullif(..., '')`` is required either way because ``ChatHistoryTurnV1``
    declares ``prompt``/``response`` as required ``str``: the turn event always
    carries both keys and either can legitimately be the empty string, which must
    not overwrite real text.

    ``model_cls``: defaults to ``ChatHistoryLogSQL``. ``AitownChatHistoryLogSQL``
    is column-for-column identical (see that model's own docstring), so the
    same merge policy applies unchanged when Track B Phase 1's dual-write
    targets the mirror table instead -- one implementation of this
    concurrency-critical logic, not two hand-kept-in-sync copies.
    """
    table = model_cls.__table__
    updates: dict[str, Any] = {}
    for name in cols:
        if name == "id":
            continue  # the conflict target itself
        column = table.c.get(name)
        if column is None:
            continue
        incoming = getattr(stmt.excluded, name)
        if isinstance(column.type, (String, Text)):
            incoming = func.nullif(incoming, "")
        ordered = (incoming, column) if incoming_wins else (column, incoming)
        updates[name] = func.coalesce(*ordered)
    return updates


def upsert_chat_history_row(
    sess: Any, values: dict[str, Any], *, incoming_wins: bool = True, model_cls: type = ChatHistoryLogSQL
) -> None:
    """Atomically merge one event's contribution into a chat_history_log row
    (or, when ``model_cls=AitownChatHistoryLogSQL``, its AI Town mirror --
    see that model's docstring and ``_maybe_dual_write_aitown_chat_history``
    below, Track B Phase 1).

    Replaces a SELECT-then-INSERT/merge. That pattern was not safe here: the
    three events for a single turn are dispatched as independent chassis tasks
    (``orion/core/bus/bus_service_chassis.py``) and executed in parallel threads
    via ``asyncio.to_thread`` with thread-local sessions, so all three could
    SELECT-miss together, all three would INSERT, and two would take a 23505 that
    the caller discarded as an idempotent duplicate. Roughly one Hub turn in five
    lost its prompt or its response that way.

    A single INSERT ... ON CONFLICT DO UPDATE has no window between the read and
    the write, so concurrent contributors merge instead of racing.

    Conflicts on the primary key only. Every current producer sets ``id`` equal
    to ``correlation_id`` (``scripts/chat_history.py`` passes ``turn_id`` == the
    correlation id at all call sites, and ``_write_row`` back-fills ``id`` from
    ``correlation_id`` when absent), so this is the same row the old
    ``id OR correlation_id`` lookup found. A producer that ever sends an explicit
    ``id`` different from its ``correlation_id`` would insert a second row rather
    than merge -- narrower than the old lookup, and worth keeping in mind if a
    new producer appears.
    """
    if not values.get("id"):
        raise ValueError("chat_history_log upsert requires an id")
    stmt = pg_insert(model_cls).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[model_cls.id],
        set_=_chat_history_conflict_updates(
            values.keys(), stmt, incoming_wins=incoming_wins, model_cls=model_cls
        ),
    )
    sess.execute(stmt)


# AI Town chat-history table split, Phase 1
# (docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md).
_AITOWN_PLATFORM_TAG = "aitown"
# Value must match services/orion-recall/app/chat_source_tagging.py's
# AITOWN_TAG constant exactly -- reimplemented locally rather than
# cross-imported, matching this repo's service-boundary convention
# (CLAUDE.md section 5) and the same choice
# services/orion-hub/scripts/concept_atlas_routes.py's own
# _AITOWN_PLATFORM_TAG already made for the same signal.


def _is_aitown_client_meta(client_meta: Any) -> bool:
    """The canonical AI Town detection signal: ``client_meta.external_room.platform
    == 'aitown'`` -- not the ``source`` column, which is only a coincidentally
    correlated proxy (100% correlated as of 2026-08-18, but not the sanctioned
    signal the rest of the codebase, e.g. the crystallization gate, builds
    around).
    """
    if not isinstance(client_meta, dict):
        return False
    external_room = client_meta.get("external_room")
    if not isinstance(external_room, dict):
        return False
    return external_room.get("platform") == _AITOWN_PLATFORM_TAG


def _maybe_dual_write_aitown_chat_history(
    sess: Any, values: dict[str, Any], *, incoming_wins: bool
) -> None:
    """Additive mirror write into ``aitown_chat_history_log`` for AI Town
    rows -- Track B Phase 1. Zero consumer-visible change: this never
    touches ``chat_history_log`` itself, only ever ADDS a row to the new
    table, and is a complete no-op unless
    ``SQL_WRITER_AITOWN_DUAL_WRITE_ENABLED`` is set.

    Classification is per-call, from whatever ``client_meta`` this specific
    event's ``values`` carries -- not a lookup against the row's full,
    already-merged state. Known, accepted limitation for Phase 1 (nothing
    reads this table yet, so it costs nothing today): if a message-path
    event without ``client_meta`` lands before the turn event (which always
    carries it), that message's fields go into ``chat_history_log`` as
    normal but are not yet mirrored; the turn event's own arrival still
    creates/merges the mirror row correctly once it lands. A field
    contributed only by a client_meta-less event, on a row later confirmed
    AI Town, could in principle be missing from the mirror row -- acceptable
    for Phase 1's job (new table exists, dual-write proven, zero disruption
    to existing readers), revisit only if Phase 2 needs stronger fidelity.

    Never allowed to take the real ``chat_history_log`` write it rides
    alongside down with it -- same isolation principle as
    ``_ensure_chat_history_from_message``'s own docstring, but a different
    mechanism: both writes share the caller's session/transaction here (by
    design, so they commit together), so a plain try/except around the
    mirror write is NOT enough -- Postgres aborts the whole transaction on
    the first failed statement, so a swallowed exception would still leave
    the caller's own subsequent ``commit()`` doomed ("current transaction is
    aborted"). ``sess.begin_nested()`` opens a SAVEPOINT so a mirror-write
    failure rolls back only the mirror write, not the real one riding
    alongside it in the same transaction.
    """
    if not settings.sql_writer_aitown_dual_write_enabled:
        return
    if not _is_aitown_client_meta(values.get("client_meta")):
        return
    try:
        with sess.begin_nested():
            upsert_chat_history_row(
                sess, values, incoming_wins=incoming_wins, model_cls=AitownChatHistoryLogSQL
            )
    except Exception as exc:
        logger.warning(
            "aitown_chat_history_dual_write_failed id=%s corr=%s error=%s",
            values.get("id"),
            values.get("correlation_id"),
            exc,
        )


def _apply_spark_meta_patch(payload: dict) -> bool:
    patch = ChatHistorySparkMetaPatchV1.model_validate(payload)
    corr = str(patch.correlation_id)
    sess = get_session()
    try:
        existing = (
            sess.query(ChatHistoryLogSQL)
            .filter(ChatHistoryLogSQL.correlation_id == corr)
            .first()
        )
        if existing is None:
            logger.error("spark_meta_patch_missing_row correlation_id=%s", corr)
            return False
        merged = _merge_spark_meta(getattr(existing, "spark_meta", None), patch.spark_meta)
        sess.execute(
            update(ChatHistoryLogSQL)
            .where(ChatHistoryLogSQL.correlation_id == corr)
            .values(spark_meta=merged)
        )
        _maybe_dual_patch_aitown_spark_meta(sess, correlation_id=corr, patch_spark_meta=patch.spark_meta)
        sess.commit()
        return True
    finally:
        sess.close()
        remove_session()


def _maybe_dual_patch_aitown_spark_meta(
    sess: Any, *, correlation_id: str, patch_spark_meta: dict
) -> None:
    """Mirror-table half of ``_apply_spark_meta_patch`` -- Track B Phase 1.

    ``ChatHistorySparkMetaPatchV1`` carries no ``client_meta``, so AI-Town-ness
    can't be classified from the patch payload alone the way the two
    dual-write call sites above do. Instead: patch-only, never insert -- if a
    row with this ``correlation_id`` already exists in
    ``aitown_chat_history_log`` (meaning some earlier dual-write already
    classified it AI Town), apply the same merge there too, keeping the
    mirror's ``spark_meta`` from going stale relative to the primary table.
    If no mirror row exists (never dual-written, or dual-write was off at the
    time), this is a correct no-op -- there is nothing to patch, and this
    function must never create a row on its own classification-blind say-so.

    Code review (2026-08-19) caught two real issues in the first version:

    1. A plain SELECT-then-UPDATE here is exactly the lost-update race
       ``upsert_chat_history_row``'s own docstring says the ON CONFLICT
       redesign exists to avoid -- two concurrent patches for the same
       ``correlation_id`` could each read the same starting ``spark_meta``,
       merge their own patch on top in Python, and the second UPDATE would
       clobber the first's merged keys instead of folding both in. Fixed
       with ``.with_for_update()``: the row lock serializes concurrent
       patches within the SAVEPOINT instead of letting them race. (The
       *primary*-table ``_apply_spark_meta_patch`` just above has this same
       pre-existing race and is NOT touched here -- out of scope for this
       diff, a real follow-up in its own right.)
    2. The existence check was done *inside* ``begin_nested()``, paying for
       a SAVEPOINT on every single no-mirror-row no-op call (the common case
       while few rows have been dual-written yet). Reordered so the cheap
       existence check runs first, un-transacted, and the SAVEPOINT only
       wraps the actual locked read + write once there is real work to do.
    """
    if not settings.sql_writer_aitown_dual_write_enabled:
        return
    mirror_exists = (
        sess.query(AitownChatHistoryLogSQL.id)
        .filter(AitownChatHistoryLogSQL.correlation_id == correlation_id)
        .first()
    )
    if mirror_exists is None:
        return
    try:
        with sess.begin_nested():
            mirror_row = (
                sess.query(AitownChatHistoryLogSQL)
                .filter(AitownChatHistoryLogSQL.correlation_id == correlation_id)
                .with_for_update()
                .first()
            )
            if mirror_row is None:
                return
            mirror_merged = _merge_spark_meta(getattr(mirror_row, "spark_meta", None), patch_spark_meta)
            sess.execute(
                update(AitownChatHistoryLogSQL)
                .where(AitownChatHistoryLogSQL.correlation_id == correlation_id)
                .values(spark_meta=mirror_merged)
            )
    except Exception as exc:
        logger.warning(
            "aitown_chat_history_dual_patch_failed corr=%s error=%s", correlation_id, exc
        )


def _coerce_sql_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _normalized_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _is_missing_scalar(value: Any) -> bool:
    return value is None or value == ""


def _normalize_notification_request_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data or {})
    context = normalized.get("context")
    if not isinstance(context, dict):
        context = {}

    event_kind = str(normalized.get("event_kind") or "").strip()
    is_attention = event_kind == "orion.chat.attention"
    is_message = event_kind == "orion.chat.message"

    if not is_attention and not is_message:
        return normalized

    if is_attention:
        if _is_missing_scalar(normalized.get("attention_id")) and context.get("attention_id"):
            normalized["attention_id"] = context.get("attention_id")
        if "attention_require_ack" not in normalized and "require_ack" in context:
            normalized["attention_require_ack"] = bool(context.get("require_ack"))
        if normalized.get("attention_ack_deadline_minutes") is None and context.get("ack_deadline_minutes") is not None:
            normalized["attention_ack_deadline_minutes"] = context.get("ack_deadline_minutes")
        if normalized.get("attention_escalation_channels") is None and context.get("escalation_channels") is not None:
            normalized["attention_escalation_channels"] = context.get("escalation_channels")
        if normalized.get("attention_expires_at") is None and context.get("expires_at"):
            parsed = _coerce_sql_timestamp(context.get("expires_at"))
            if parsed is not None:
                normalized["attention_expires_at"] = parsed
            else:
                logger.warning(
                    "sql_writer_notify_attention_parse_failed field=expires_at value=%r",
                    context.get("expires_at"),
                )

    if is_message:
        if _is_missing_scalar(normalized.get("message_id")) and context.get("message_id"):
            normalized["message_id"] = context.get("message_id")
        if _is_missing_scalar(normalized.get("message_session_id")):
            session_id = normalized.get("session_id") or context.get("session_id")
            if session_id:
                normalized["message_session_id"] = session_id
        if _is_missing_scalar(normalized.get("message_preview_text")) and context.get("preview_text"):
            normalized["message_preview_text"] = context.get("preview_text")
        if _is_missing_scalar(normalized.get("message_full_text")) and context.get("full_text"):
            normalized["message_full_text"] = context.get("full_text")
        if "message_require_read_receipt" not in normalized and "require_read_receipt" in context:
            normalized["message_require_read_receipt"] = bool(context.get("require_read_receipt"))
        if normalized.get("message_expires_at") is None and context.get("expires_at"):
            parsed = _coerce_sql_timestamp(context.get("expires_at"))
            if parsed is not None:
                normalized["message_expires_at"] = parsed
            else:
                logger.warning(
                    "sql_writer_notify_message_parse_failed field=expires_at value=%r",
                    context.get("expires_at"),
                )

    return normalized


def _extract_thought_process(payload: dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    role = str(payload.get("role") or "").strip().lower()
    has_assistant_response = bool(_normalized_text(payload.get("response")))
    if role and role != "assistant" and not has_assistant_response:
        return None

    rt = payload.get("reasoning_trace")
    if isinstance(rt, dict):
        candidate = _normalized_text(rt.get("content"))
        if candidate:
            return candidate
    elif isinstance(rt, str):
        candidate = _normalized_text(rt)
        if candidate:
            return candidate

    candidate = _normalized_text(payload.get("reasoning_content"))
    if candidate:
        return candidate

    traces = payload.get("metacog_traces")
    if isinstance(traces, list):
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            trace_role = str(trace.get("trace_role") or trace.get("role") or "").strip().lower()
            if trace_role and trace_role != "reasoning":
                continue
            candidate = _normalized_text(trace.get("content"))
            if candidate:
                return candidate

    return None


def _thought_candidate_and_reason(payload: dict[str, Any]) -> tuple[Optional[str], str]:
    if not isinstance(payload, dict):
        return None, "payload_not_dict"
    role = str(payload.get("role") or "").strip().lower()
    has_assistant_response = bool(_normalized_text(payload.get("response")))
    if role and role != "assistant" and not has_assistant_response:
        return None, f"role_filtered:{role}"
    rt = payload.get("reasoning_trace")
    thinking_source = _normalized_text(payload.get("thinking_source")) or "none"
    reasoning_content = _normalized_text(payload.get("reasoning_content"))
    inline_think_content = _normalized_text(payload.get("inline_think_content"))
    spark_meta = payload.get("spark_meta") if isinstance(payload.get("spark_meta"), dict) else {}
    trace_verb = _normalized_text((spark_meta or {}).get("trace_verb"))
    thought_capture_step = _normalized_text((spark_meta or {}).get("thought_capture_step"))
    if (
        trace_verb == "chat_general"
        and thought_capture_step == "llm_chat_general"
        and inline_think_content
    ):
        return inline_think_content, "inline_think_content.chat_general_llm_chat_general"
    if thinking_source == "provider_reasoning" and reasoning_content:
        return reasoning_content, "reasoning_content.provider"
    if thinking_source.startswith("inline_think") and inline_think_content:
        return inline_think_content, f"inline_think_content.{thinking_source}"
    if isinstance(rt, dict):
        candidate = _normalized_text(rt.get("content"))
        if candidate:
            return candidate, f"reasoning_trace.content({thinking_source})"
    elif isinstance(rt, str):
        candidate = _normalized_text(rt)
        if candidate:
            return candidate, "reasoning_trace.string"
    if reasoning_content:
        return reasoning_content, "reasoning_content"
    if inline_think_content:
        return inline_think_content, "inline_think_content"
    traces = payload.get("metacog_traces")
    if isinstance(traces, list):
        for idx, trace in enumerate(traces):
            if not isinstance(trace, dict):
                continue
            trace_role = str(trace.get("trace_role") or trace.get("role") or "").strip().lower()
            if trace_role and trace_role != "reasoning":
                continue
            candidate = _normalized_text(trace.get("content"))
            if candidate:
                return candidate, f"metacog_traces[{idx}].content"
    return None, "none"


def _chat_history_thought_for_merge(sess: Any, filtered_data: dict[str, Any], raw_data: dict[str, Any]) -> Optional[str]:
    incoming = _normalized_text(filtered_data.get("thought_process"))
    candidate_id = filtered_data.get("id") or filtered_data.get("correlation_id") or raw_data.get("correlation_id")
    if not candidate_id:
        return incoming
    existing = (
        sess.query(ChatHistoryLogSQL)
        .filter(
            (ChatHistoryLogSQL.id == str(candidate_id))
            | (ChatHistoryLogSQL.correlation_id == str(candidate_id))
        )
        .first()
    )
    existing_thought = _normalized_text(getattr(existing, "thought_process", None)) if existing is not None else None
    if existing_thought:
        return existing_thought
    return incoming


def _map_spark_to_telemetry_row(
    kind: str,
    payload: Any,
    *,
    envelope_correlation_id: Optional[str],
    envelope_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if normalize_spark_state_snapshot is None or normalize_spark_telemetry is None:
        return None

    if kind == "spark.state.snapshot.v1":
        snapshot = normalize_spark_state_snapshot(payload)
        if snapshot is None:
            return None
        phi_components = snapshot.phi if isinstance(snapshot.phi, dict) else {}
        metadata = snapshot.metadata if isinstance(snapshot.metadata, dict) else {}
        snapshot_dict = snapshot.model_dump(mode="json")
        metadata = dict(metadata)
        metadata["spark_state_snapshot"] = snapshot_dict
        correlation_id = snapshot.correlation_id or envelope_correlation_id or envelope_id
        if not correlation_id:
            return None
        return {
            "correlation_id": str(correlation_id),
            "timestamp": snapshot.snapshot_ts,
            "phi": phi_components.get("coherence"),
            "novelty": phi_components.get("novelty"),
            "trace_mode": snapshot.trace_mode,
            "trace_verb": snapshot.trace_verb,
            "metadata_": metadata,
        }

    telemetry = normalize_spark_telemetry(payload)
    if telemetry is None:
        return None

    correlation_id = telemetry.correlation_id or envelope_correlation_id
    if not correlation_id:
        return None

    metadata = telemetry.metadata if isinstance(telemetry.metadata, dict) else {"raw_metadata": str(telemetry.metadata)}
    metadata = dict(metadata)

    snapshot = telemetry.state_snapshot
    if snapshot is None and isinstance(metadata.get("spark_state_snapshot"), dict):
        snapshot = normalize_spark_state_snapshot(metadata.get("spark_state_snapshot"))

    if snapshot is not None:
        metadata["spark_state_snapshot"] = snapshot.model_dump(mode="json")

    phi_components = snapshot.phi if snapshot is not None and isinstance(snapshot.phi, dict) else {}
    phi_value = telemetry.phi if telemetry.phi is not None else phi_components.get("coherence")
    novelty_value = telemetry.novelty if telemetry.novelty is not None else phi_components.get("novelty")
    ts_value = _coerce_sql_timestamp(telemetry.timestamp) if telemetry.timestamp else None
    if ts_value is None:
        return None

    return {
        "correlation_id": str(correlation_id),
        "timestamp": ts_value,
        "phi": phi_value,
        "novelty": novelty_value,
        "trace_mode": telemetry.trace_mode,
        "trace_verb": telemetry.trace_verb,
        "stimulus_summary": telemetry.stimulus_summary,
        "metadata_": metadata,
    }


def _safe_set(obj: Any, field: str, value: Any) -> None:
    if hasattr(obj, field):
        setattr(obj, field, value)


def _ensure_chat_history_from_message(
    correlation_id: str,
    session_id: str | None,
    role: str,
    content: str,
    memory_status: str | None,
    memory_tier: str | None,
    client_meta: Any,
) -> None:
    """Merge a single chat message into its turn's chat_history_log row.

    Runs in its OWN session, deliberately. This used to share the caller's
    session with the ``chat_message`` write, which meant a conflict raised here
    rolled the whole transaction back and silently destroyed the chat_message
    row too -- while the log line blamed ``chat_message``, a table that never had
    a duplicate. Chat history is a best-effort projection of the message; it must
    never be able to take the message itself down with it.
    """
    if not correlation_id:
        return
    rid = str(correlation_id)

    values: dict[str, Any] = {"id": rid, "correlation_id": rid}
    # Only the half this message actually carries. The other half stays absent
    # rather than being seeded as '', so it is never a candidate for overwrite.
    if role == "user":
        values["prompt"] = content
    elif role == "assistant":
        values["response"] = content
    else:
        return
    if session_id:
        values["session_id"] = str(session_id)
    if memory_status:
        values["memory_status"] = memory_status
    if memory_tier:
        values["memory_tier"] = memory_tier
    if client_meta:
        values["client_meta"] = _json_sanitize(client_meta)

    # session_factory(), not get_session(): the latter is a thread-local
    # scoped_session and would hand back the caller's own session, so the
    # transactions would still be shared and remove_session() would tear the
    # caller's session out from under it.
    own_sess = session_factory()
    try:
        # Fill-only: a message must never overwrite what the turn event already
        # wrote. Reproduces the old per-field `if not existing.<col>` guards.
        upsert_chat_history_row(own_sess, values, incoming_wins=False)
        _maybe_dual_write_aitown_chat_history(own_sess, values, incoming_wins=False)
        own_sess.commit()
    except Exception:
        own_sess.rollback()
        raise
    finally:
        own_sess.close()


def _write_row(sql_model_cls, data: dict) -> bool:
    sess = get_session()
    try:
        mapper = inspect(sql_model_cls)
        write_data = dict(data)
        if sql_model_cls is NotificationRequestDB:
            write_data = _normalize_notification_request_payload(write_data)

        col_key_by_name = {col.name: col.key for col in mapper.columns}
        valid_keys = set(attr.key for attr in mapper.attrs)

        filtered_data: dict = {}
        for k, v in write_data.items():
            kk = col_key_by_name.get(k, k)
            if kk in valid_keys:
                filtered_data[kk] = v

        # Ensure we never insert NULL primary keys when upstream omits them.
        if "telemetry_id" in valid_keys and not filtered_data.get("telemetry_id"):
            filtered_data["telemetry_id"] = str(uuid.uuid4())

        if sql_model_cls in (ChatMessageSQL, ChatGptMessageSQL) and "id" in valid_keys and not filtered_data.get("id"):
            mid = write_data.get("message_id") or write_data.get("id")
            filtered_data["id"] = str(mid) if mid else str(uuid.uuid4())

        if sql_model_cls in (ChatHistoryLogSQL, ChatGptLogSQL) and ("id" in valid_keys) and not filtered_data.get("id"):
            filtered_data["id"] = filtered_data.get("correlation_id") or str(uuid.uuid4())
        if _thought_debug_enabled() and sql_model_cls is ChatHistoryLogSQL:
            logger.info(
                "THOUGHT_DEBUG_SQL_WRITE stage=pre_merge corr=%s id=%s prompt_len=%s response_len=%s thought_process_len=%s thought_process_snippet=%r",
                filtered_data.get("correlation_id") or data.get("correlation_id"),
                filtered_data.get("id"),
                _debug_len(filtered_data.get("prompt")),
                _debug_len(filtered_data.get("response")),
                _debug_len(filtered_data.get("thought_process")),
                _debug_snippet(filtered_data.get("thought_process")),
            )
        if sql_model_cls is ChatHistoryLogSQL:
            preserved_thought = _chat_history_thought_for_merge(sess, filtered_data, data)
            if preserved_thought:
                filtered_data["thought_process"] = preserved_thought
            elif "thought_process" in filtered_data and not _normalized_text(filtered_data.get("thought_process")):
                filtered_data.pop("thought_process", None)
            spark_meta = filtered_data.get("spark_meta")
            top_level_thinking_source = _normalized_text(data.get("thinking_source"))
            if top_level_thinking_source:
                merged_spark_meta = _merge_spark_meta(spark_meta, {"thinking_source": top_level_thinking_source})
                filtered_data["spark_meta"] = merged_spark_meta
            meta_block = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            unc = meta_block.get("llm_uncertainty")
            if not isinstance(unc, dict):
                sm = data.get("spark_meta") if isinstance(data.get("spark_meta"), dict) else {}
                unc = sm.get("llm_uncertainty")
            if isinstance(unc, dict):
                filtered_data["spark_meta"] = _merge_spark_meta(
                    filtered_data.get("spark_meta"), {"llm_uncertainty": unc}
                )
                filtered_data.update(_chat_history_llm_uncertainty_scalars(unc))
            persisted_value = filtered_data.get("thought_process")
            print(
                "===THINK_HOP=== hop=sql_row_ready "
                f"corr={filtered_data.get('correlation_id') or data.get('correlation_id')} "
                f"thought_process_len={len(persisted_value) if persisted_value else 0} "
                f"preview={_preview_text(persisted_value)}",
                flush=True,
            )

        # Standard coercion
        for col in mapper.columns:
            key = col.key
            if key in filtered_data:
                val = filtered_data[key]
                if isinstance(col.type, String) and isinstance(val, uuid.UUID):
                    filtered_data[key] = str(val)
                if isinstance(col.type, DateTime):
                    if isinstance(val, str):
                        try:
                            filtered_data[key] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                        except ValueError:
                            pass
                    elif isinstance(val, (int, float)):
                        try:
                            filtered_data[key] = datetime.fromtimestamp(val)
                        except (ValueError, OSError):
                            pass

        # ---------------------------------------------------------------------
        # 🔄 STRATEGY: Bi-Directional Metadata Sync
        # ---------------------------------------------------------------------

        if sql_model_cls is SparkTelemetrySQL:
            if "metadata_" in valid_keys:
                raw_meta = filtered_data.get("metadata_")
                if raw_meta is None:
                    raw_meta = data.get("metadata") or data.get("metadata_json") or {}
                if not isinstance(raw_meta, dict):
                    raw_meta = {"raw_metadata": str(raw_meta)}
                filtered_data["metadata_"] = _json_sanitize(deepcopy(raw_meta))

            from app.spark_telemetry_persist import upsert_spark_telemetry

            upsert_spark_telemetry(sess, filtered_data)

            corr_id = filtered_data.get("correlation_id")
            if corr_id:
                try:
                    meta_for_chat = _spark_meta_minimal(filtered_data)
                    existing_chat = (
                        sess.query(ChatHistoryLogSQL)
                        .filter(ChatHistoryLogSQL.correlation_id == corr_id)
                        .first()
                    )
                    if existing_chat is not None:
                        merged = _merge_spark_meta(
                            getattr(existing_chat, "spark_meta", None),
                            meta_for_chat,
                            source="telemetry",
                        )
                        stmt = (
                            update(ChatHistoryLogSQL)
                            .where(ChatHistoryLogSQL.correlation_id == corr_id)
                            .values(spark_meta=merged)
                        )
                        sess.execute(stmt)
                        sess.commit()
                except Exception as ex:
                    logger.warning(f"Could not back-populate chat log spark_meta: {ex}")
            return True

        if sql_model_cls is HarnessTurnTraceSQL:
            # Raw write_data, not filtered_data: none of the four source
            # molecules' fields (draft_text, reflection, final_hash, ...) are
            # declared columns on this model -- the whole validated payload
            # is the JSONB value for whichever column its schema_version
            # picks (see app/harness_turn_trace_persist.py).
            return upsert_harness_turn_trace(sess, write_data)

        if sql_model_cls is ChatMessageSQL:
            client_meta = data.get("client_meta")
            if client_meta is not None:
                existing_meta = filtered_data.get("meta") or {}
                if not isinstance(existing_meta, dict):
                    existing_meta = {"raw_meta": str(existing_meta)}
                existing_meta["client_meta"] = _json_sanitize(client_meta)
                filtered_data["meta"] = existing_meta
            try:
                corr_id = data.get("correlation_id")
                session_id = filtered_data.get("session_id") or data.get("session_id")
                role = (filtered_data.get("role") or "").lower()
                content = filtered_data.get("content")
                memory_status = data.get("memory_status")
                memory_tier = data.get("memory_tier")
                if corr_id and role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    _ensure_chat_history_from_message(
                        correlation_id=str(corr_id),
                        session_id=str(session_id) if session_id else None,
                        role=role,
                        content=content,
                        memory_status=memory_status,
                        memory_tier=memory_tier,
                        client_meta=client_meta,
                    )
            except Exception as ex:
                # Its own session now, so this really is contained: the
                # chat_message write below still commits.
                logger.warning(f"Failed to upsert chat_history_log from chat_message: {ex}")

        if sql_model_cls is ChatGptMessageSQL:
            existing_meta = filtered_data.get("meta") or {}
            if not isinstance(existing_meta, dict):
                existing_meta = {"raw_meta": str(existing_meta)}
            export_meta: dict[str, Any] = {}
            for key in (
                "source_message_id",
                "parent_message_id",
                "child_message_ids",
                "content_type",
                "content_blocks",
                "attachments",
                "shared_conversation_id",
            ):
                value = data.get(key)
                if value is not None and value != []:
                    export_meta[key] = _json_sanitize(value)
            message_metadata = data.get("metadata")
            if isinstance(message_metadata, dict):
                export_meta["message_metadata"] = _json_sanitize(message_metadata)
            if export_meta:
                existing_meta.update(export_meta)
                filtered_data["meta"] = existing_meta

        if sql_model_cls in INSERT_ONLY_MODELS:
            try:
                sess.add(sql_model_cls(**filtered_data))
                sess.commit()
                return True
            except IntegrityError as e:
                sess.rollback()
                if hasattr(e.orig, 'pgcode') and e.orig.pgcode == '23505':
                    logger.info(f"Duplicate entry for {sql_model_cls.__tablename__}, skipping (append-only idempotent write).")
                    return False
                if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
                    logger.info(f"Duplicate entry for {sql_model_cls.__tablename__}, skipping (append-only idempotent write).")
                    return False
                raise

        try:
            if sql_model_cls is ChatHistoryLogSQL:
                # Atomic merge, no SELECT first. The previous SELECT-then-merge
                # raced with the two chat.history.message writes for the same
                # turn: all three could miss, all three would INSERT, and two
                # took a 23505 that the handler below discarded as a duplicate.
                # The turn event is the only one carrying `source`, so it was the
                # reliable casualty -- hence every corrupted row having
                # source IS NULL and exactly one of prompt/response.
                upsert_chat_history_row(sess, filtered_data)
                _maybe_dual_write_aitown_chat_history(sess, filtered_data, incoming_wins=True)
                sess.commit()
                return True
            sess.merge(sql_model_cls(**filtered_data))
            sess.commit()
            return True
        except IntegrityError as e:
            sess.rollback()
            duplicate = (
                (hasattr(e.orig, "pgcode") and e.orig.pgcode == "23505")
                or "unique constraint" in str(e).lower()
                or "duplicate key" in str(e).lower()
            )
            # chat_history_log is NOT idempotent-by-duplicate. One row is
            # assembled from three events that each carry a different subset of
            # columns, so "we already have this id" never means "we already have
            # this content" -- treating it that way is what silently dropped the
            # prompt or the response on ~1 Hub turn in 5. The upsert above should
            # make this unreachable; if it fires anyway, that is a real defect
            # and must be loud, not swallowed.
            if duplicate and sql_model_cls is ChatHistoryLogSQL:
                logger.error(
                    "chat_history_log_duplicate_not_idempotent id=%s corr=%s -- "
                    "upsert path should have merged this; data may be lost",
                    filtered_data.get("id"),
                    filtered_data.get("correlation_id"),
                )
                raise
            # Handle unique constraint violations gracefully (idempotency)
            # Postgres unique violation code is 23505
            if duplicate:
                logger.info(f"Duplicate entry for {sql_model_cls.__tablename__}, skipping (idempotent write).")
                return False
            logger.error(f"IntegrityError writing to {sql_model_cls.__tablename__}: {e}")
            raise e

    finally:
        try:
            sess.close()
        finally:
            remove_session()


def _build_collapse_stored_payload(payload: dict[str, Any], *, correlation_id: str | None) -> CollapseMirrorStoredV1:
    mirror_id = (
        payload.get("id")
        or payload.get("event_id")
        or payload.get("correlation_id")
        or correlation_id
    )
    return CollapseMirrorStoredV1(
        mirror_id=str(mirror_id),
        stored_at=datetime.now(timezone.utc).isoformat(),
        correlation_id=correlation_id,
        is_causally_dense=bool(payload.get("is_causally_dense")),
        summary=str(payload.get("summary") or ""),
        trigger=str(payload.get("trigger") or ""),
        what_changed_summary=payload.get("what_changed_summary"),
        mantra=payload.get("mantra"),
    )


def _build_social_turn_stored_payload(payload: dict[str, Any], *, correlation_id: str | None) -> SocialRoomTurnStoredV1:
    base = dict(payload)
    if correlation_id and not base.get("correlation_id"):
        base["correlation_id"] = correlation_id
    if not base.get("text"):
        prompt = str(base.get("prompt") or "").strip()
        response = str(base.get("response") or "").strip()
        base["text"] = f"User: {prompt}\nOrion: {response}".strip()
    return SocialRoomTurnStoredV1.model_validate(base)


def _extract_calibration_profile_id(actions: list[str]) -> str | None:
    for action in actions:
        if isinstance(action, str) and action.startswith("calibration_profile_id:"):
            value = action.split(":", 1)[1].strip()
            if value:
                return value
    return None


def _normalize_endogenous_runtime_record_payload(payload: Any) -> Dict[str, Any]:
    record = EndogenousRuntimeExecutionRecordV1.model_validate(payload)
    return {
        "runtime_record_id": record.runtime_record_id,
        "correlation_id": record.correlation_id,
        "request_id": record.trigger_request.request_id,
        "invocation_surface": record.invocation_surface,
        "trigger_outcome": record.decision.outcome,
        "workflow_type": record.decision.workflow_type,
        "subject_ref": record.subject_ref,
        "anchor_scope": record.trigger_request.anchor_scope,
        "mentor_invoked": record.mentor_invoked,
        "execution_success": record.execution_success,
        "calibration_profile_id": _extract_calibration_profile_id(record.audit_events),
        "materialized_artifact_ids": list(record.materialized_artifact_ids),
        "payload": record.model_dump(mode="json"),
        "created_at": record.created_at,
    }


def _normalize_endogenous_runtime_audit_payload(payload: Any) -> Dict[str, Any]:
    audit = EndogenousRuntimeAuditV1.model_validate(payload)
    return {
        "runtime_record_id": audit.runtime_record_id,
        "invocation_surface": audit.invocation_surface,
        "status": audit.status,
        "workflow_type": audit.workflow_type,
        "decision_outcome": audit.decision_outcome,
        "mentor_invoked": audit.mentor_invoked,
        "cooldown_applied": audit.cooldown_applied,
        "debounce_applied": audit.debounce_applied,
        "error": audit.error,
        "calibration_profile_id": _extract_calibration_profile_id(list(audit.actions)),
        "payload": audit.model_dump(mode="json"),
        "recorded_at": datetime.now(timezone.utc),
    }


def map_world_pulse_situation_brief_row(data: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "topic_id": data.get("topic_id"),
        "run_id": data.get("run_id") or "unknown",
        "title": data.get("title") or "unknown",
        "status": data.get("status") or "developing",
        "tracking_status": data.get("tracking_status") or "candidate",
        "current_assessment": data.get("current_assessment") or "",
        "payload_json": data,
        "schema_version": "v1",
        "created_at": data.get("created_at") or now,
        "updated_at": data.get("updated_at") or now,
    }


def map_world_pulse_situation_change_row(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "change_id": data.get("change_id"),
        "topic_id": data.get("topic_id"),
        "run_id": data.get("run_id") or "unknown",
        "change_type": data.get("change_type") or "new_development",
        "payload_json": data,
        "schema_version": "v1",
        "created_at": data.get("created_at") or datetime.now(timezone.utc),
    }


def _normalize_calibration_profile_audit_payload(payload: Any) -> Dict[str, Any]:
    event = CalibrationProfileAuditV1.model_validate(payload)
    return {
        "audit_id": event.audit_id,
        "profile_id": event.profile_id,
        "previous_profile_id": event.previous_profile_id,
        "event_type": event.event_type,
        "operator_id": event.operator_id,
        "rationale": event.rationale,
        "details": event.details,
        "payload": event.model_dump(mode="json"),
        "recorded_at": event.recorded_at,
    }


def _normalize_phi_reward_payload(payload: Any, *, correlation_id: str | None) -> Dict[str, Any]:
    event = PhiIntrinsicRewardV1.model_validate(payload)
    return {
        "correlation_id": str(correlation_id) if correlation_id else str(uuid.uuid4()),
        "generated_at": event.generated_at,
        "payload": event.model_dump(mode="json"),
    }


def _write_fallback(kind: str, correlation_id: str, payload: Any, error: str = None) -> None:
    sess = get_session()
    try:
        safe_payload = payload
        if not isinstance(safe_payload, (dict, list)):
            safe_payload = {"raw": str(payload)}
        safe_payload = _json_sanitize(safe_payload)

        sess.add(
            BusFallbackLog(
                kind=kind,
                correlation_id=correlation_id,
                payload=safe_payload,
                error=error,
                created_at_ts=datetime.now(timezone.utc),
            )
        )
        sess.commit()
    finally:
        try:
            sess.close()
        finally:
            remove_session()


async def _write(
    sql_model_cls,
    schema_cls,
    payload: Any,
    extra_fields: Dict[str, Any] = None,
    *,
    kind: str | None = None,
) -> bool:
    if schema_cls:
        validation_payload, validation_boundary = _payload_for_schema_validation(payload, schema_cls, kind=kind)
        try:
            obj = _coerce_payload(schema_cls, validation_payload)
        except Exception as e:
            payload_keys = sorted(validation_payload.keys()) if isinstance(validation_payload, dict) else []
            logger.error(
                "sql_writer_validation_failed kind=%s schema=%s boundary=%s payload_type=%s payload_keys=%s error=%s",
                kind,
                getattr(schema_cls, "__name__", str(schema_cls)),
                validation_boundary,
                type(validation_payload).__name__,
                payload_keys,
                e,
            )
            raise
        data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
    else:
        data = payload if isinstance(payload, dict) else {}

    if extra_fields:
        data.update(extra_fields)

    try:
        return await asyncio.to_thread(_write_row, sql_model_cls, data)
    except Exception as e:
        logger.error(f"Failed to write to primary table: {e}")
        raise


async def _persist_grammar_trace_batch_envelope(
    items: list[GrammarWorkItem],
    *,
    shard: int = 0,
) -> None:
    from app.grammar_ledger_handler import cancel_active_grammar_persist, persist_grammar_trace_batch

    events = [item[1] for item in items]
    trace_id = events[0].trace_id if events else "unknown"
    timeout_sec = max(
        float(settings.sql_writer_grammar_persist_timeout_sec),
        float(settings.sql_writer_grammar_trace_batch_timeout_sec),
    )
    loop = asyncio.get_running_loop()
    executor = _get_grammar_executors()[shard]
    fut = loop.run_in_executor(executor, persist_grammar_trace_batch, events, shard)
    try:
        await asyncio.wait_for(fut, timeout=timeout_sec)
    except asyncio.TimeoutError:
        canceled = cancel_active_grammar_persist(shard)
        logger.error(
            "sql_writer_grammar_trace_batch_timeout shard=%s trace_id=%s events=%s timeout_sec=%s canceled=%s",
            shard,
            trace_id,
            len(events),
            timeout_sec,
            canceled,
        )
        for env, _event, payload, corr_id in items:
            try:
                await asyncio.to_thread(
                    _write_fallback,
                    env.kind,
                    corr_id,
                    payload,
                    f"grammar trace batch timeout after {timeout_sec}s",
                )
            except Exception as exc:
                logger.error(
                    "sql_writer_grammar_fallback_failed trace_id=%s error=%s",
                    trace_id,
                    exc,
                )
    except Exception as exc:
        logger.exception(
            "sql_writer_grammar_trace_batch_failed trace_id=%s events=%s error=%s",
            trace_id,
            len(events),
            exc,
        )
        for env, _event, payload, corr_id in items:
            try:
                await asyncio.to_thread(_write_fallback, env.kind, corr_id, payload, str(exc))
            except Exception as fallback_exc:
                logger.error(
                    "sql_writer_grammar_fallback_failed trace_id=%s error=%s",
                    trace_id,
                    fallback_exc,
                )


async def _persist_grammar_event_envelope(
    env: BaseEnvelope,
    *,
    event: GrammarEventV1,
    payload: dict[str, Any],
    corr_id: str,
    shard: int = 0,
) -> None:
    from app.grammar_ledger_handler import cancel_active_grammar_persist, persist_grammar_event

    timeout_sec = float(settings.sql_writer_grammar_persist_timeout_sec)
    loop = asyncio.get_running_loop()
    executor = _get_grammar_executors()[shard]
    fut = loop.run_in_executor(executor, persist_grammar_event, event, shard)
    try:
        await asyncio.wait_for(fut, timeout=timeout_sec)
    except asyncio.TimeoutError:
        canceled = cancel_active_grammar_persist(shard)
        logger.error(
            "sql_writer_grammar_persist_timeout shard=%s event_id=%s trace_id=%s timeout_sec=%s canceled=%s",
            shard,
            event.event_id,
            event.trace_id,
            timeout_sec,
            canceled,
        )
        try:
            await asyncio.to_thread(
                _write_fallback,
                env.kind,
                corr_id,
                payload,
                f"grammar persist timeout after {timeout_sec}s",
            )
        except Exception as exc:
            logger.error(
                "sql_writer_grammar_fallback_failed event_id=%s trace_id=%s error=%s",
                event.event_id,
                event.trace_id,
                exc,
            )
    except Exception as exc:
        logger.exception(
            "sql_writer_grammar_persist_failed event_id=%s trace_id=%s error=%s",
            event.event_id,
            event.trace_id,
            exc,
        )
        try:
            await asyncio.to_thread(_write_fallback, env.kind, corr_id, payload, str(exc))
        except Exception as fallback_exc:
            logger.error(
                "sql_writer_grammar_fallback_failed event_id=%s trace_id=%s error=%s",
                event.event_id,
                event.trace_id,
                fallback_exc,
            )


def _chat_source_platform(client_meta: Any) -> str | None:
    """external_room.platform from a chat.history payload / chat_history_log row.

    Deliberately a thin local reader rather than an import of orion-recall's
    chat_source_tagging.chat_source_platform(): sql-writer does not depend on
    that service's package, and this needs only the platform half (no participant
    name, no text rendering). Handles the dict form (psycopg2 jsonb) and the
    JSON-string form (asyncpg / raw bus payloads) the same way that helper does.
    Returns None for direct hub conversations and for every row predating the
    field -- callers treat None as "not external", the correct default.
    """
    if isinstance(client_meta, str):
        try:
            client_meta = json.loads(client_meta)
        except Exception:
            return None
    if not isinstance(client_meta, dict):
        return None
    external_room = client_meta.get("external_room")
    if not isinstance(external_room, dict):
        return None
    platform = external_room.get("platform")
    return str(platform) if platform else None


def _fetch_chat_turn_for_memory_emit(corr_id: str) -> dict | None:
    if not corr_id:
        return None
    sess = get_session()
    try:
        row = (
            sess.query(ChatHistoryLogSQL)
            .filter(ChatHistoryLogSQL.correlation_id == corr_id)
            .first()
        )
        if row is None:
            return None
        prompt = str(getattr(row, "prompt", "") or "").strip()
        response = str(getattr(row, "response", "") or "").strip()
        if not prompt or not response:
            return None
        spark_meta = getattr(row, "spark_meta", None)
        if not isinstance(spark_meta, dict):
            spark_meta = {}
        appraisal = spark_meta.get("turn_change_appraisal")
        if isinstance(appraisal, dict) and appraisal.get("turn_change_status") == "ok":
            return None
        return {
            "correlation_id": corr_id,
            "prompt": prompt,
            "response": response,
            "spark_meta": spark_meta,
            "session_id": getattr(row, "session_id", None),
            "source_platform": _chat_source_platform(getattr(row, "client_meta", None)),
        }
    finally:
        sess.close()
        remove_session()


async def _publish_post_commit(
    bus: Any,
    channel: str,
    msg: BaseEnvelope,
    *,
    log_label: str,
) -> None:
    await publish_with_reconnect(bus, channel, msg, log_label=log_label)


async def _emit_memory_turn_persisted(
    bus: Any,
    *,
    parent_env: BaseEnvelope,
    turn: dict,
) -> None:
    if bus is None or not settings.sql_writer_emit_memory_turn_persisted:
        return
    corr = str(turn["correlation_id"])
    persisted = MemoryTurnPersistedV1(
        correlation_id=corr,
        prompt=str(turn["prompt"]),
        response=str(turn["response"]),
        spark_meta=turn.get("spark_meta") if isinstance(turn.get("spark_meta"), dict) else {},
        session_id=turn.get("session_id"),
        source_platform=turn.get("source_platform"),
    )
    out_env = parent_env.derive_child(
        kind=MEMORY_TURN_PERSISTED_KIND,
        source=ServiceRef(
            name=settings.service_name,
            version=settings.service_version,
            node=settings.node_name,
        ),
        payload=persisted.model_dump(mode="json"),
        reply_to=None,
    )
    if str(out_env.correlation_id) != corr or str(out_env.payload.get("correlation_id")) != corr:
        logger.error(
            "memory_turn_persisted_corr_mismatch env=%s payload=%s expected=%s",
            out_env.correlation_id,
            out_env.payload.get("correlation_id"),
            corr,
        )
    await _publish_post_commit(
        bus,
        settings.channel_memory_turn_persisted,
        out_env,
        log_label="memory_turn_persisted",
    )


async def _maybe_emit_memory_turn_from_row(
    bus: Any,
    *,
    parent_env: BaseEnvelope,
    corr_id: str,
) -> None:
    turn = await asyncio.to_thread(_fetch_chat_turn_for_memory_emit, corr_id)
    if turn is None:
        return
    try:
        await _emit_memory_turn_persisted(bus, parent_env=parent_env, turn=turn)
    except Exception:
        logger.exception("Failed to emit memory turn persisted event corr=%s", corr_id)


async def handle_envelope(env: BaseEnvelope, *, bus: Any | None = None) -> None:
    if env.kind == "grammar.event.v1":
        payload = env.payload if isinstance(env.payload, dict) else {}
        event = GrammarEventV1.model_validate(payload)
        corr_id = str(env.correlation_id or payload.get("correlation_id") or "")
        _spawn_grammar_persist(env, event=event, payload=payload, corr_id=corr_id)
        return

    async with _get_write_semaphore():
        await _handle_envelope_body(env, bus=bus)


async def _handle_envelope_body(env: BaseEnvelope, *, bus: Any | None = None) -> None:
    route_key = settings.route_map.get(env.kind)
    payload = env.payload if isinstance(env.payload, dict) else {}

    if env.kind == "chat.history.spark_meta.patch.v1":
        await asyncio.to_thread(_apply_spark_meta_patch, payload)
        return

    async def _persist_evidence_units() -> bool:
        try:
            evidence_units = build_evidence_units(
                env.kind,
                env.payload,
                correlation_id=extra_sql_fields.get("correlation_id"),
            )
            if not evidence_units:
                return False
            for unit in evidence_units:
                evidence_row = unit.model_dump(mode="json")
                evidence_row["metadata_json"] = evidence_row.pop("metadata", {})
                await _write(
                    EvidenceUnitSQL,
                    None,
                    evidence_row,
                    {},
                    kind="evidence.unit.v1",
                )
            return True
        except Exception:
            logger.exception(
                "Failed to persist evidence units for kind=%s corr=%s",
                env.kind,
                getattr(env, "correlation_id", None),
            )
            return False
    if env.kind == "metacognitive.trace.v1":
        logger.info(
            "sql_writer_metacog_consumed kind=%s corr=%s route=%s",
            env.kind,
            env.correlation_id,
            route_key,
        )

    # -------------------------------------------------------------------------
    # GLOBAL PRE-PROCESSING: Extract Correlation ID
    # -------------------------------------------------------------------------
    extra_sql_fields: Dict[str, Any] = {}
    if getattr(env, "correlation_id", None):
        extra_sql_fields["correlation_id"] = str(env.correlation_id)
    if env.kind != "chat.history.spark_meta.patch.v1":
        payload = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "chat.history":
        rt = payload.get("reasoning_trace")
        if isinstance(rt, dict):
            trace_payload = dict(rt)
            trace_payload.setdefault("correlation_id", str(env.correlation_id))
            trace_payload.setdefault("session_id", payload.get("session_id"))
            trace_payload.setdefault("message_id", payload.get("id") or payload.get("correlation_id"))
            trace_payload.setdefault("trace_role", "reasoning")
            trace_payload.setdefault("trace_stage", "post_answer")
            trace_payload.setdefault("model", (payload.get("spark_meta") or {}).get("model") or "unknown")
            try:
                await _write(MetacognitiveTraceSQL, MetacognitiveTraceV1, trace_payload, {})
            except Exception as exc:
                logger.warning("Failed to persist reasoning_trace from chat.history corr=%s err=%s", env.correlation_id, exc)
    trace_id = payload.get("trace_id") or payload.get("traceId")
    if trace_id:
        extra_sql_fields["trace_id"] = str(trace_id)
    if env.kind == "substrate.dev_economics_ledger.v1":
        # model_mix is a real dict on the wire schema (DevEconomicsLedgerV1)
        # but DevEconomicsLedgerSQL stores it as a JSON string column
        # (model_mix_json) -- no generic dict->JSON column mapping exists in
        # _write_row, so this is injected explicitly, same pattern as
        # client_meta's own _json_sanitize handling just above.
        extra_sql_fields["model_mix_json"] = json.dumps(payload.get("model_mix") or {})
    memory_status = payload.get("memory_status")
    if memory_status:
        extra_sql_fields["memory_status"] = memory_status
    memory_tier = payload.get("memory_tier")
    if memory_tier:
        extra_sql_fields["memory_tier"] = memory_tier
    memory_reason = payload.get("memory_reason")
    if memory_reason:
        extra_sql_fields["memory_reason"] = memory_reason
    client_meta = payload.get("client_meta")
    if client_meta is not None:
        extra_sql_fields["client_meta"] = _json_sanitize(client_meta)
    thought_process, thought_source = _thought_candidate_and_reason(payload)
    logger.info(
        "sql_writer_thought_selection corr=%s kind=%s reasoning_content_len=%s inline_think_content_len=%s thinking_source=%s selected_source=%s selected_len=%s",
        env.correlation_id,
        env.kind,
        _debug_len(payload.get("reasoning_content")) if isinstance(payload, dict) else 0,
        _debug_len(payload.get("inline_think_content")) if isinstance(payload, dict) else 0,
        payload.get("thinking_source") if isinstance(payload, dict) else None,
        thought_source,
        _debug_len(thought_process),
    )
    if env.kind == "chat.history":
        print(
            "===THINK_HOP=== hop=sql_before_assign "
            f"corr={env.correlation_id} "
            f"source={thought_source or 'none'} "
            f"len={len(thought_process) if thought_process else 0} "
            f"preview={_preview_text(thought_process)} "
            f"payload_keys={sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}",
            flush=True,
        )
    if _thought_debug_enabled() and env.kind == "chat.history":
        logger.info(
            "THOUGHT_DEBUG_SQL_INTAKE stage=chat_history_intake corr=%s payload_keys=%s reasoning_trace_exists=%s reasoning_content_exists=%s inline_think_content_exists=%s thinking_source=%s candidate_source=%s candidate_len=%s candidate_snippet=%r discarded=%s",
            env.correlation_id,
            sorted(list(payload.keys())) if isinstance(payload, dict) else [],
            isinstance(payload.get("reasoning_trace"), dict) or isinstance(payload.get("reasoning_trace"), str),
            bool(str(payload.get("reasoning_content") or "").strip()),
            bool(str(payload.get("inline_think_content") or "").strip()),
            payload.get("thinking_source"),
            thought_source,
            _debug_len(thought_process),
            _debug_snippet(thought_process),
            thought_process is None,
        )
    if thought_process:
        extra_sql_fields["thought_process"] = thought_process
    elif _thought_debug_enabled() and env.kind == "chat.history":
        logger.info(
            "THOUGHT_DEBUG_SQL_INTAKE stage=chat_history_intake_discarded corr=%s reason=%s",
            env.correlation_id,
            thought_source,
        )

    if env.kind.startswith("spark."):
        try:
            _SPARK_CONTRACT_METRICS.observe(env.kind)
            _SPARK_CONTRACT_METRICS.maybe_emit(
                logger,
                node=settings.node_name,
                service=settings.service_name,
            )
            action = _legacy_action(env.kind, settings.spark_legacy_mode_normalized, LEGACY_KINDS)
            if action == "warn":
                logger.warning(
                    "SPARK_LEGACY_DEPRECATED kind=%s mode=warn action=accept_write",
                    env.kind,
                )
            elif action == "drop":
                logger.warning(
                    "SPARK_LEGACY_DEPRECATED kind=%s mode=drop action=skip_write",
                    env.kind,
                )
                return
        except Exception as exc:
            logger.debug("Spark contract metrics emission failed: %s", exc)

    # -------------------------------------------------------------------------
    # 1. SPECIAL CASE: Spark State Snapshot -> SparkTelemetrySQL
    # (Adapts complex snapshot object to flat telemetry row + metadata)
    # -------------------------------------------------------------------------
    if env.kind == "spark.state.snapshot.v1":
        try:
            data_to_process = _map_spark_to_telemetry_row(
                env.kind,
                env.payload,
                envelope_correlation_id=extra_sql_fields.get("correlation_id"),
                envelope_id=str(env.id) if getattr(env, "id", None) else None,
            )
            if not data_to_process:
                logger.warning("Skipping spark.state.snapshot.v1 write: normalization failed.")
                await asyncio.to_thread(
                    _write_fallback,
                    env.kind,
                    extra_sql_fields.get("correlation_id", ""),
                    env.payload,
                    "spark snapshot normalization failed",
                )
                return
            # WRITE: SKIP Pydantic validation
            await _write(SparkTelemetrySQL, None, data_to_process, {})
            logger.info(f"Written {env.kind} -> spark_telemetry (via adapter)")
            return
        except Exception as e:
            logger.exception(f"Error writing {env.kind} via adapter, falling back.")
            await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, str(e))
            return

    # -------------------------------------------------------------------------
    # 2. STANDARD ROUTING
    # -------------------------------------------------------------------------
    if route_key and route_key in MODEL_MAP:
        sql_model, schema_model = MODEL_MAP[route_key]
        try:
            data_to_process = env.payload
            if route_key == "ExternalRoomMessageSQL":
                if env.kind == "external.room.post.result.v1":
                    schema_model = ExternalRoomPostResultV1
                elif env.kind == "external.room.turn.skipped.v1":
                    schema_model = ExternalRoomTurnSkippedV1

            if isinstance(data_to_process, dict):
                data_to_process = data_to_process.copy()
                if _thought_debug_enabled() and sql_model is ChatHistoryLogSQL:
                    pending_thought = extra_sql_fields.get("thought_process")
                    logger.info(
                        "THOUGHT_DEBUG_SQL_INTAKE stage=pre_write_chat_history corr=%s prompt_len=%s response_len=%s thought_process_len=%s thought_process_state=%s",
                        env.correlation_id,
                        _debug_len(data_to_process.get("prompt")),
                        _debug_len(data_to_process.get("response")),
                        _debug_len(pending_thought),
                        "non_null" if pending_thought else "null",
                    )
                if "node" not in data_to_process and env.source and env.source.node:
                    extra_sql_fields.setdefault("node", env.source.node)
                if "source_message_id" not in data_to_process and env.id:
                    extra_sql_fields.setdefault("source_message_id", str(env.id))

            if sql_model is WorldPulseRunSQL and isinstance(data_to_process, dict):
                run_payload = data_to_process.get("run") if isinstance(data_to_process.get("run"), dict) else data_to_process
                data_to_process = {
                    "run_id": run_payload.get("run_id") or extra_sql_fields.get("correlation_id"),
                    "status": run_payload.get("status") or "pending",
                    "date": run_payload.get("date") or date.today().isoformat(),
                    "requested_by": run_payload.get("requested_by"),
                    "dry_run": str(run_payload.get("dry_run")),
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseDigestSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "run_id": data_to_process.get("run_id"),
                    "title": data_to_process.get("title") or "Daily World Pulse",
                    "date": data_to_process.get("date") or date.today().isoformat(),
                    "executive_summary": data_to_process.get("executive_summary") or "",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseDigestItemSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "item_id": data_to_process.get("item_id"),
                    "run_id": data_to_process.get("run_id"),
                    "category": data_to_process.get("category") or "general_world",
                    "title": data_to_process.get("title") or "",
                    "confidence": str(data_to_process.get("confidence") or ""),
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseArticleSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "article_id": data_to_process.get("article_id"),
                    "run_id": data_to_process.get("run_id"),
                    "source_id": data_to_process.get("source_id") or "unknown",
                    "title": data_to_process.get("title") or "",
                    "url": data_to_process.get("url") or "",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseArticleClusterSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "cluster_id": data_to_process.get("cluster_id"),
                    "run_id": data_to_process.get("run_id"),
                    "category": data_to_process.get("category") or "general_world",
                    "title": data_to_process.get("title") or "",
                    "article_count": str(data_to_process.get("article_count") or 0),
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseClaimSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "claim_id": data_to_process.get("claim_id"),
                    "run_id": data_to_process.get("run_id"),
                    "topic_id": data_to_process.get("topic_id"),
                    "promotion_status": data_to_process.get("promotion_status") or "observed",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseEventSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "event_id": data_to_process.get("event_id"),
                    "run_id": data_to_process.get("run_id"),
                    "event_type": data_to_process.get("event_type") or "other",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseEntitySQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "entity_id": data_to_process.get("entity_id"),
                    "canonical_name": data_to_process.get("canonical_name") or "unknown",
                    "entity_type": data_to_process.get("entity_type") or "other",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None
            elif sql_model is WorldPulseSituationBriefSQL and isinstance(data_to_process, dict):
                data_to_process = map_world_pulse_situation_brief_row(data_to_process)
                schema_model = None
            elif sql_model is WorldPulseSituationChangeSQL and isinstance(data_to_process, dict):
                data_to_process = map_world_pulse_situation_change_row(data_to_process)
                schema_model = None
            elif sql_model is WorldPulseLearningDeltaSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "learning_id": data_to_process.get("learning_id"),
                    "run_id": data_to_process.get("run_id"),
                    "topic_id": data_to_process.get("topic_id"),
                    "category": data_to_process.get("category") or "general_world",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                    "created_at": data_to_process.get("created_at") or datetime.now(timezone.utc),
                }
                schema_model = None
            elif sql_model is WorldPulseWorthReadingSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "reading_id": data_to_process.get("reading_id"),
                    "run_id": data_to_process.get("run_id") or "",
                    "category": data_to_process.get("category") or "general_world",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                    "created_at": data_to_process.get("created_at") or datetime.now(timezone.utc),
                }
                schema_model = None
            elif sql_model is WorldPulseWorthWatchingSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "watch_id": data_to_process.get("watch_id"),
                    "run_id": data_to_process.get("run_id") or "",
                    "category": data_to_process.get("category") or "general_world",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                    "created_at": data_to_process.get("created_at") or datetime.now(timezone.utc),
                }
                schema_model = None
            elif sql_model is WorldPulseContextCapsuleSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "capsule_id": data_to_process.get("capsule_id"),
                    "run_id": data_to_process.get("run_id"),
                    "date": data_to_process.get("date") or date.today().isoformat(),
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                    "created_at": data_to_process.get("created_at") or datetime.now(timezone.utc),
                }
                schema_model = None
            elif sql_model is WorldPulseHubMessageSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "message_id": data_to_process.get("message_id"),
                    "run_id": data_to_process.get("run_id") or "",
                    "title": data_to_process.get("title") or "Daily World Pulse",
                    "date": data_to_process.get("date") or date.today().isoformat(),
                    "executive_summary": data_to_process.get("executive_summary") or "",
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                    "created_at": data_to_process.get("created_at") or datetime.now(timezone.utc),
                }
                schema_model = None
            elif sql_model is WorldPulsePublishStatusSQL and isinstance(data_to_process, dict):
                data_to_process = {
                    "status_id": data_to_process.get("message_id") or str(uuid.uuid4()),
                    "run_id": data_to_process.get("run_id") or "",
                    "channel": data_to_process.get("channel") or "hub",
                    "state": data_to_process.get("state") or "published",
                    "detail": data_to_process.get("detail"),
                    "payload_json": data_to_process,
                    "schema_version": "v1",
                }
                schema_model = None

            if sql_model is CollapseMirror and isinstance(data_to_process, dict):
                base_id = (
                    data_to_process.get("id") or data_to_process.get("event_id")
                    or data_to_process.get("correlation_id")
                    or extra_sql_fields.get("correlation_id")
                    or (str(env.id) if getattr(env, "id", None) else None)
                )
                if not base_id: base_id = str(uuid.uuid4())
                if not data_to_process.get("id"): extra_sql_fields["id"] = base_id
                if not data_to_process.get("correlation_id") and not extra_sql_fields.get("correlation_id"):
                    extra_sql_fields["correlation_id"] = base_id

            if sql_model is MetacogEntry and isinstance(data_to_process, dict):
                base_id = (
                    data_to_process.get("event_id")
                    or extra_sql_fields.get("correlation_id")
                    or (str(env.id) if getattr(env, "id", None) else None)
                )
                if not base_id:
                    base_id = str(uuid.uuid4())
                extra_sql_fields["id"] = base_id
                if not data_to_process.get("correlation_id") and not extra_sql_fields.get("correlation_id"):
                    extra_sql_fields["correlation_id"] = base_id

            if sql_model is CollapseEnrichment and isinstance(data_to_process, dict):
                extra_sql_fields["id"] = str(uuid.uuid4())
                target_id = data_to_process.get("id") or data_to_process.get("collapse_id")
                if target_id: extra_sql_fields["collapse_id"] = target_id

            write_ok = True
            if sql_model is SparkTelemetrySQL:
                mapped = _map_spark_to_telemetry_row(
                    env.kind,
                    data_to_process,
                    envelope_correlation_id=extra_sql_fields.get("correlation_id"),
                    envelope_id=str(env.id) if getattr(env, "id", None) else None,
                )
                if not mapped:
                    logger.warning("Skipping spark telemetry write: normalization failed.")
                    await asyncio.to_thread(
                        _write_fallback,
                        env.kind,
                        extra_sql_fields.get("correlation_id", ""),
                        env.payload,
                        "spark telemetry normalization failed",
                    )
                    return
                write_ok = await _write(sql_model, None, mapped, {}, kind=env.kind)
            elif sql_model is Dream:
                normalized = _normalize_dream_envelope_payload(
                    env.kind, data_to_process, extra_sql_fields
                )
                await _write(sql_model, None, normalized, {}, kind=env.kind)
            elif sql_model is EndogenousRuntimeRecordSQL:
                normalized = _normalize_endogenous_runtime_record_payload(data_to_process)
                write_ok = await _write(sql_model, None, normalized, {}, kind=env.kind)
            elif sql_model is EndogenousRuntimeAuditSQL:
                normalized = _normalize_endogenous_runtime_audit_payload(data_to_process)
                write_ok = await _write(sql_model, None, normalized, {}, kind=env.kind)
            elif sql_model is CalibrationProfileAuditSQL:
                normalized = _normalize_calibration_profile_audit_payload(data_to_process)
                write_ok = await _write(sql_model, None, normalized, {}, kind=env.kind)
            elif sql_model is PhiRewardSQL:
                normalized = _normalize_phi_reward_payload(
                    data_to_process,
                    correlation_id=extra_sql_fields.get("correlation_id"),
                )
                write_ok = await _write(sql_model, None, normalized, {}, kind=env.kind)
            elif sql_model is HarnessTurnTraceSQL:
                # extra_fields={}: the generic extra_sql_fields dict (node,
                # source_message_id, envelope-level correlation_id) would
                # otherwise get merged into the JSONB blob by _write()'s
                # data.update(extra_fields) -- contaminating the "this is
                # exactly the payload the harness published" guarantee the
                # column dispatch relies on. data_to_process is already the
                # clean env.payload copy.
                write_ok = await _write(sql_model, None, data_to_process, {}, kind=env.kind)
            else:
                write_ok = await _write(sql_model, schema_model, data_to_process, extra_sql_fields, kind=env.kind)
            if env.kind == JOURNAL_WRITE_KIND and write_ok:
                try:
                    journal_input, _ = _payload_for_schema_validation(env.payload, JournalEntryWriteV1, kind=env.kind)
                    journal_payload = JournalEntryWriteV1.model_validate(journal_input)
                    trigger, chat_stance, stance_meta = _extract_journal_index_context(env.payload)
                    index_payload = build_journal_entry_index_payload(
                        journal_payload,
                        trigger=trigger,
                        chat_stance=chat_stance,
                        stance_metadata=stance_meta,
                    )
                    await _write(JournalEntryIndexSQL, None, index_payload, {}, kind="journal.entry.index.v1")
                except Exception:
                    logger.exception(
                        "Failed to persist journal_entry_index row; preserving append-only write. corr=%s",
                        getattr(env, "correlation_id", None),
                    )
            if write_ok:
                await _persist_evidence_units()

            # Post-commit safety: emit only after _write() has returned success.
            if env.kind == JOURNAL_WRITE_KIND and write_ok and bus is not None and settings.sql_writer_emit_journal_created:
                try:
                    journal_input, _ = _payload_for_schema_validation(env.payload, JournalEntryWriteV1, kind=env.kind)
                    journal_payload = JournalEntryWriteV1.model_validate(journal_input)
                    created_env = env.derive_child(
                        kind=JOURNAL_CREATED_KIND,
                        source=ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name),
                        payload=build_created_event_payload(journal_payload),
                        reply_to=None,
                    )
                    await _publish_post_commit(
                        bus,
                        settings.sql_writer_journal_created_channel,
                        created_env,
                        log_label="journal_created",
                    )
                    logger.info(
                        "journal_created_event_emitted corr=%s entry_id=%s channel=%s",
                        getattr(env, "correlation_id", None),
                        journal_payload.entry_id,
                        settings.sql_writer_journal_created_channel,
                    )
                except Exception:
                    logger.exception("Failed to emit journal created event corr=%s", getattr(env, "correlation_id", None))
            # Semantic stored event is also post-commit only.
            if env.kind == "collapse.mirror" and write_ok and bus is not None:
                try:
                    stored_payload = _build_collapse_stored_payload(
                        env.payload if isinstance(env.payload, dict) else {},
                        correlation_id=extra_sql_fields.get("correlation_id"),
                    )
                    stored_env = env.derive_child(
                        kind=COLLAPSE_STORED_KIND,
                        source=ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name),
                        payload=stored_payload.model_dump(mode="json"),
                        reply_to=None,
                    )
                    await _publish_post_commit(
                        bus,
                        "orion:collapse:stored",
                        stored_env,
                        log_label="collapse_stored",
                    )
                except Exception:
                    logger.exception("Failed to emit collapse stored event corr=%s", getattr(env, "correlation_id", None))
            if (
                env.kind == "social.turn.v1"
                and write_ok
                and bus is not None
                and settings.sql_writer_emit_social_turn_stored
            ):
                try:
                    stored_payload = _build_social_turn_stored_payload(
                        env.payload if isinstance(env.payload, dict) else {},
                        correlation_id=extra_sql_fields.get("correlation_id"),
                    )
                    stored_env = env.derive_child(
                        kind=SOCIAL_TURN_STORED_KIND,
                        source=ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name),
                        payload=stored_payload.model_dump(mode="json"),
                        reply_to=None,
                    )
                    await _publish_post_commit(
                        bus,
                        settings.sql_writer_social_turn_stored_channel,
                        stored_env,
                        log_label="social_turn_stored",
                    )
                except Exception:
                    logger.exception("Failed to emit social turn stored event corr=%s", getattr(env, "correlation_id", None))
            if (
                env.kind == "chat.history"
                and write_ok
                and bus is not None
                and settings.sql_writer_emit_memory_turn_persisted
            ):
                try:
                    corr = str(env.correlation_id or extra_sql_fields.get("correlation_id") or "")
                    prompt = str(data_to_process.get("prompt") or "").strip()
                    response = str(data_to_process.get("response") or "").strip()
                    if corr and prompt and response:
                        await _emit_memory_turn_persisted(
                            bus,
                            parent_env=env,
                            turn={
                                "correlation_id": corr,
                                "prompt": prompt,
                                "response": response,
                                "spark_meta": data_to_process.get("spark_meta")
                                if isinstance(data_to_process.get("spark_meta"), dict)
                                else {},
                                "session_id": data_to_process.get("session_id"),
                                # Read from `payload`, not `data_to_process`: several
                                # routes above rebind data_to_process to a projected
                                # row dict, so it is not a reliable source for a raw
                                # envelope field. `payload` is what
                                # extra_sql_fields["client_meta"] is populated from in
                                # this same scope, so this reads the identical value
                                # that gets written to chat_history_log.client_meta.
                                "source_platform": _chat_source_platform(payload.get("client_meta")),
                            },
                        )
                    elif corr:
                        await _maybe_emit_memory_turn_from_row(bus, parent_env=env, corr_id=corr)
                except Exception:
                    logger.exception(
                        "Failed to emit memory turn persisted event corr=%s",
                        getattr(env, "correlation_id", None),
                    )
            written_label = env.kind
            if schema_model is ChatGptLogTurnV1:
                written_label = "ChatGptLogTurnV1"
            logger.info(f"Written {written_label} -> {sql_model.__tablename__}")
            if (
                write_ok
                and bus is not None
                and env.kind == "chat.history.message.v1"
                and (payload.get("role") or "").lower() == "assistant"
            ):
                corr = str(env.correlation_id or payload.get("correlation_id") or "")
                await _maybe_emit_memory_turn_from_row(bus, parent_env=env, corr_id=corr)

        except Exception as e:
            logger.exception(f"Error writing {env.kind} to {sql_model.__tablename__}, falling back.")
            await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, str(e))
    else:
        if await _persist_evidence_units():
            logger.info("Written %s -> evidence_units (adapter-only path)", env.kind)
            return
        logger.warning(f"Unknown kind {env.kind} (Route: {route_key}), writing to fallback log.")
        await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, "Unknown kind")


def build_hunter() -> Hunter:
    patterns = settings.effective_subscribe_channels
    # Startup gate: a subscribed channel with no route silently sends every
    # event to bus_fallback_log while the service looks perfectly healthy.
    # Happened twice on 2026-08-13/14 -- see app/route_coverage.py for why
    # this is a log line and not a hard failure.
    log_route_coverage(patterns, settings.route_map)
    holder: dict[str, Any] = {}

    async def _handler(env: BaseEnvelope) -> None:
        await handle_envelope(env, bus=holder.get("bus"))

    hunter = Hunter(
        _cfg(),
        patterns=patterns,
        handler=_handler,
        concurrent_handlers=settings.sql_writer_concurrent_handlers,
    )
    holder["bus"] = hunter.bus
    return hunter
