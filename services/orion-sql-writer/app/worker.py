# services/orion-sql-writer/app/worker.py
from __future__ import annotations

import asyncio
import logging
import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Tuple, Type

from sqlalchemy import inspect, update
from sqlalchemy.types import DateTime, String
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    BiometricsTelemetry,
    BiometricsSummarySQL,
    BiometricsInductionSQL,
    ChatHistoryLogSQL,
    ChatGptLogSQL,
    ChatGptMessageSQL,
    ChatMessageSQL,
    CollapseEnrichment,
    CollapseMirror,
    Dream,
    SparkIntrospectionLogSQL,
    SparkTelemetrySQL,
    BusFallbackLog,
    CognitionTraceSQL,
    MetacognitionTickSQL,
    MetacogTriggerSQL,
    NotificationRequestDB,
    NotificationReceiptDB,
    JournalEntrySQL,
    SocialRoomTurnSQL,
    ExternalRoomMessageSQL,
    ExternalRoomParticipantSQL,
)

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import JOURNAL_CREATED_KIND, JOURNAL_WRITE_KIND, JournalEntryWriteV1, build_created_event_payload

# Shared schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry, CollapseMirrorStoredV1
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.telemetry.biometrics import BiometricsPayload, BiometricsSummaryV1, BiometricsInductionV1
from orion.schemas.telemetry.dream import DreamRequest, DreamResultV1
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.chat_history import ChatHistoryMessageV1
from orion.schemas.chat_gpt_log import ChatGptLogTurnV1, ChatGptMessageV1
from orion.schemas.social_chat import SocialRoomTurnStoredV1, SocialRoomTurnV1
from orion.schemas.social_bridge import (
    ExternalRoomMessageV1,
    ExternalRoomParticipantV1,
    ExternalRoomPostResultV1,
    ExternalRoomTurnSkippedV1,
)
from orion.schemas.telemetry.metacognition import MetacognitionTickV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

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
_SPARK_CONTRACT_METRICS = SparkContractMetrics()
COLLAPSE_STORED_KIND = "collapse.mirror.stored.v1"
SOCIAL_TURN_STORED_KIND = "social.turn.stored.v1"
INSERT_ONLY_MODELS = {JournalEntrySQL, SocialRoomTurnSQL}


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
    "CollapseEnrichment": (CollapseEnrichment, MetaTagsPayload),
    "ChatHistoryLogSQL": (ChatHistoryLogSQL, None),
    "ChatGptLogSQL": (ChatGptLogSQL, ChatGptLogTurnV1),
    "ChatGptMessageSQL": (ChatGptMessageSQL, ChatGptMessageV1),
    "ChatMessageSQL": (ChatMessageSQL, ChatHistoryMessageV1),
    "Dream": (Dream, None),
    "BiometricsTelemetry": (BiometricsTelemetry, BiometricsPayload),
    "BiometricsSummarySQL": (BiometricsSummarySQL, BiometricsSummaryV1),
    "BiometricsInductionSQL": (BiometricsInductionSQL, BiometricsInductionV1),
    "CognitionTraceSQL": (CognitionTraceSQL, CognitionTracePayload),
    "SparkIntrospectionLogSQL": (SparkIntrospectionLogSQL, None),
    "SparkTelemetrySQL": (SparkTelemetrySQL, SparkTelemetryPayload),
    "MetacognitionTickSQL": (MetacognitionTickSQL, MetacognitionTickV1),
    "MetacogTriggerSQL": (MetacogTriggerSQL, MetacogTriggerV1),
    "NotificationRequestDB": (NotificationRequestDB, None),
    "NotificationReceiptDB": (NotificationReceiptDB, None),
    "JournalEntrySQL": (JournalEntrySQL, JournalEntryWriteV1),
    "SocialRoomTurnSQL": (SocialRoomTurnSQL, SocialRoomTurnV1),
    "ExternalRoomMessageSQL": (ExternalRoomMessageSQL, ExternalRoomMessageV1),
    "ExternalRoomParticipantSQL": (ExternalRoomParticipantSQL, ExternalRoomParticipantV1),
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


def _merge_spark_meta(existing: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Any]
    if isinstance(existing, dict):
        base = deepcopy(existing)
    elif existing is None:
        base = {}
    else:
        base = {"raw_existing": str(existing)}

    for k, v in (updates or {}).items():
        if v is None:
            continue
        base[k] = v

    return _json_sanitize(base)


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
    sess,
    correlation_id: str,
    session_id: str | None,
    role: str,
    content: str,
    memory_status: str | None,
    memory_tier: str | None,
    client_meta: Any,
) -> None:
    if not correlation_id:
        return
    rid = str(correlation_id)

    existing = (
        sess.query(ChatHistoryLogSQL)
        .filter(ChatHistoryLogSQL.id == rid)
        .first()
    )

    if existing is None:
        existing = ChatHistoryLogSQL(
            id=rid,
            correlation_id=rid,
        )
        _safe_set(existing, "session_id", session_id)
        _safe_set(existing, "prompt", "")
        _safe_set(existing, "response", "")
        _safe_set(existing, "spark_meta", None)
        sess.add(existing)

    if role == "user":
        if hasattr(existing, "prompt") and not (existing.prompt or "").strip():
            existing.prompt = content
    elif role == "assistant":
        if hasattr(existing, "response") and not (existing.response or "").strip():
            existing.response = content

    if session_id and hasattr(existing, "session_id") and not getattr(existing, "session_id", None):
        existing.session_id = session_id
    if memory_status and hasattr(existing, "memory_status") and not getattr(existing, "memory_status", None):
        existing.memory_status = memory_status
    if memory_tier and hasattr(existing, "memory_tier") and not getattr(existing, "memory_tier", None):
        existing.memory_tier = memory_tier
    if client_meta and hasattr(existing, "client_meta") and not getattr(existing, "client_meta", None):
        existing.client_meta = _json_sanitize(client_meta)


def _write_row(sql_model_cls, data: dict) -> bool:
    sess = get_session()
    try:
        mapper = inspect(sql_model_cls)

        col_key_by_name = {col.name: col.key for col in mapper.columns}
        valid_keys = set(attr.key for attr in mapper.attrs)

        filtered_data: dict = {}
        for k, v in data.items():
            kk = col_key_by_name.get(k, k)
            if kk in valid_keys:
                filtered_data[kk] = v

        # Ensure we never insert NULL primary keys when upstream omits them.
        if "telemetry_id" in valid_keys and not filtered_data.get("telemetry_id"):
            filtered_data["telemetry_id"] = str(uuid.uuid4())

        if sql_model_cls in (ChatMessageSQL, ChatGptMessageSQL) and "id" in valid_keys and not filtered_data.get("id"):
            mid = data.get("message_id") or data.get("id")
            filtered_data["id"] = str(mid) if mid else str(uuid.uuid4())

        if sql_model_cls in (ChatHistoryLogSQL, ChatGptLogSQL) and ("id" in valid_keys) and not filtered_data.get("id"):
            filtered_data["id"] = filtered_data.get("correlation_id") or str(uuid.uuid4())

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

            corr_id = filtered_data.get("correlation_id")
            if corr_id:
                existing = (
                    sess.query(SparkTelemetrySQL)
                    .filter(SparkTelemetrySQL.correlation_id == corr_id)
                    .first()
                )

                if existing:
                    for k in (
                        "phi",
                        "novelty",
                        "trace_mode",
                        "trace_verb",
                        "stimulus_summary",
                        "timestamp",
                        "source_service",
                        "source_node",
                        "node",
                    ):
                        if k in filtered_data and filtered_data.get(k) is not None:
                            setattr(existing, k, filtered_data.get(k))

                    try:
                        ex_meta = getattr(existing, "metadata_", None)
                    except Exception:
                        ex_meta = None
                    new_meta = filtered_data.get("metadata_")
                    if isinstance(ex_meta, dict) and isinstance(new_meta, dict):
                        ex_meta.update(new_meta)
                        existing.metadata_ = _json_sanitize(ex_meta)
                    elif isinstance(new_meta, dict):
                        existing.metadata_ = _json_sanitize(new_meta)

                    sess.commit()
                else:
                    sess.add(SparkTelemetrySQL(**filtered_data))
                    sess.commit()

                try:
                    meta_for_chat = _spark_meta_minimal(filtered_data)
                    existing_chat = (
                        sess.query(ChatHistoryLogSQL)
                        .filter(ChatHistoryLogSQL.correlation_id == corr_id)
                        .first()
                    )
                    if existing_chat is not None:
                        merged = _merge_spark_meta(getattr(existing_chat, "spark_meta", None), meta_for_chat)
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
                        sess=sess,
                        correlation_id=str(corr_id),
                        session_id=str(session_id) if session_id else None,
                        role=role,
                        content=content,
                        memory_status=memory_status,
                        memory_tier=memory_tier,
                        client_meta=client_meta,
                    )
            except Exception as ex:
                logger.warning(f"Failed to upsert chat_history_log from chat_message: {ex}")

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
            sess.merge(sql_model_cls(**filtered_data))
            sess.commit()
            return True
        except IntegrityError as e:
            sess.rollback()
            # Handle unique constraint violations gracefully (idempotency)
            # Postgres unique violation code is 23505
            if hasattr(e.orig, 'pgcode') and e.orig.pgcode == '23505':
                logger.info(f"Duplicate entry for {sql_model_cls.__tablename__}, skipping (idempotent write).")
                return False
            # Also catch generic duplicates if pgcode not available or different driver
            if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
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
            )
        )
        sess.commit()
    finally:
        try:
            sess.close()
        finally:
            remove_session()


async def _write(sql_model_cls, schema_cls, payload: Any, extra_fields: Dict[str, Any] = None) -> bool:
    if schema_cls:
        obj = _coerce_payload(schema_cls, payload)
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


async def handle_envelope(env: BaseEnvelope, *, bus: Any | None = None) -> None:
    route_key = settings.route_map.get(env.kind)

    # -------------------------------------------------------------------------
    # GLOBAL PRE-PROCESSING: Extract Correlation ID
    # -------------------------------------------------------------------------
    extra_sql_fields: Dict[str, Any] = {}
    if getattr(env, "correlation_id", None):
        extra_sql_fields["correlation_id"] = str(env.correlation_id)
    payload = env.payload if isinstance(env.payload, dict) else {}
    trace_id = payload.get("trace_id") or payload.get("traceId")
    if trace_id:
        extra_sql_fields["trace_id"] = str(trace_id)
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
                if "node" not in data_to_process and env.source and env.source.node:
                    extra_sql_fields.setdefault("node", env.source.node)
                if "source_message_id" not in data_to_process and env.id:
                    extra_sql_fields.setdefault("source_message_id", str(env.id))

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
                write_ok = await _write(sql_model, None, mapped, {})
                await _write(sql_model, None, mapped, {})
            elif sql_model is Dream:
                normalized = _normalize_dream_envelope_payload(
                    env.kind, data_to_process, extra_sql_fields
                )
                await _write(sql_model, None, normalized, {})
            else:
                write_ok = await _write(sql_model, schema_model, data_to_process, extra_sql_fields)
            # Post-commit safety: emit only after _write() has returned success.
            if env.kind == JOURNAL_WRITE_KIND and write_ok and bus is not None and settings.sql_writer_emit_journal_created:
                try:
                    journal_payload = JournalEntryWriteV1.model_validate(env.payload)
                    created_env = env.derive_child(
                        kind=JOURNAL_CREATED_KIND,
                        source=ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name),
                        payload=build_created_event_payload(journal_payload),
                        reply_to=None,
                    )
                    await bus.publish(settings.sql_writer_journal_created_channel, created_env)
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
                    await bus.publish("orion:collapse:stored", stored_env)
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
                    await bus.publish(settings.sql_writer_social_turn_stored_channel, stored_env)
                except Exception:
                    logger.exception("Failed to emit social turn stored event corr=%s", getattr(env, "correlation_id", None))
            written_label = env.kind
            if schema_model is ChatGptLogTurnV1:
                written_label = "ChatGptLogTurnV1"
            logger.info(f"Written {written_label} -> {sql_model.__tablename__}")

        except Exception as e:
            logger.exception(f"Error writing {env.kind} to {sql_model.__tablename__}, falling back.")
            await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, str(e))
    else:
        logger.warning(f"Unknown kind {env.kind} (Route: {route_key}), writing to fallback log.")
        await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, "Unknown kind")


def build_hunter() -> Hunter:
    patterns = settings.effective_subscribe_channels
    holder: dict[str, Any] = {}

    async def _handler(env: BaseEnvelope) -> None:
        await handle_envelope(env, bus=holder.get("bus"))

    hunter = Hunter(_cfg(), patterns=patterns, handler=_handler)
    holder["bus"] = hunter.bus
    return hunter
