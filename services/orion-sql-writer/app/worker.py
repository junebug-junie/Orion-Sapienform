# services/orion-sql-writer/app/worker.py
from __future__ import annotations

import asyncio
import logging
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Type

from sqlalchemy import inspect, update
from sqlalchemy.types import DateTime, String
from pydantic import BaseModel

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    BiometricsTelemetry,
    ChatHistoryLogSQL,
    ChatMessageSQL,
    CollapseEnrichment,
    CollapseMirror,
    Dream,
    SparkIntrospectionLogSQL,
    SparkTelemetrySQL,
    BusFallbackLog,
    CognitionTraceSQL,
    MetacognitionTickSQL,
    MetacogTriggerSQL
)

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope

# Shared schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.telemetry.biometrics import BiometricsPayload
from orion.schemas.telemetry.dream import DreamRequest
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.chat_history import ChatHistoryMessageV1
from orion.schemas.telemetry.metacognition import MetacognitionTickV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

try:
    from orion.schemas.telemetry.spark import SparkTelemetryPayload
except ImportError:
    SparkTelemetryPayload = None

logger = logging.getLogger("sql-writer")

# Map: route_key -> (SQLAlchemy model, Pydantic schema model)
MODEL_MAP: Dict[str, Tuple[Type[Any], Optional[Type[BaseModel]]]] = {
    "CollapseMirror": (CollapseMirror, CollapseMirrorEntry),
    "CollapseEnrichment": (CollapseEnrichment, MetaTagsPayload),
    "ChatHistoryLogSQL": (ChatHistoryLogSQL, None),
    "ChatMessageSQL": (ChatMessageSQL, ChatHistoryMessageV1),
    "Dream": (Dream, DreamRequest),
    "BiometricsTelemetry": (BiometricsTelemetry, BiometricsPayload),
    "CognitionTraceSQL": (CognitionTraceSQL, CognitionTracePayload),
    "SparkIntrospectionLogSQL": (SparkIntrospectionLogSQL, None),
    "SparkTelemetrySQL": (SparkTelemetrySQL, SparkTelemetryPayload),
    "MetacognitionTickSQL": (MetacognitionTickSQL, MetacognitionTickV1),
    "MetacogTriggerSQL": (MetacogTriggerSQL, MetacogTriggerV1),
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


def _safe_set(obj: Any, field: str, value: Any) -> None:
    if hasattr(obj, field):
        setattr(obj, field, value)


def _ensure_chat_history_from_message(
    sess,
    correlation_id: str,
    session_id: str | None,
    role: str,
    content: str,
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


def _write_row(sql_model_cls, data: dict) -> None:
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

        if sql_model_cls is ChatMessageSQL and "id" in valid_keys and not filtered_data.get("id"):
            mid = data.get("message_id") or data.get("id")
            filtered_data["id"] = str(mid) if mid else str(uuid.uuid4())

        if sql_model_cls is ChatHistoryLogSQL and ("id" in valid_keys) and not filtered_data.get("id"):
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
                return

        if sql_model_cls is ChatMessageSQL:
            try:
                corr_id = data.get("correlation_id")
                session_id = filtered_data.get("session_id") or data.get("session_id")
                role = (filtered_data.get("role") or "").lower()
                content = filtered_data.get("content")
                if corr_id and role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    _ensure_chat_history_from_message(
                        sess=sess,
                        correlation_id=str(corr_id),
                        session_id=str(session_id) if session_id else None,
                        role=role,
                        content=content,
                    )
            except Exception as ex:
                logger.warning(f"Failed to upsert chat_history_log from chat_message: {ex}")

        sess.merge(sql_model_cls(**filtered_data))
        sess.commit()

    finally:
        try:
            sess.close()
        finally:
            remove_session()


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


async def _write(sql_model_cls, schema_cls, payload: Any, extra_fields: Dict[str, Any] = None) -> None:
    if schema_cls:
        obj = _coerce_payload(schema_cls, payload)
        data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
    else:
        data = payload if isinstance(payload, dict) else {}

    if extra_fields:
        data.update(extra_fields)

    try:
        await asyncio.to_thread(_write_row, sql_model_cls, data)
    except Exception as e:
        logger.error(f"Failed to write to primary table: {e}")
        raise


async def handle_envelope(env: BaseEnvelope) -> None:
    route_key = settings.route_map.get(env.kind)

    # -------------------------------------------------------------------------
    # GLOBAL PRE-PROCESSING: Extract Correlation ID
    # -------------------------------------------------------------------------
    extra_sql_fields: Dict[str, Any] = {}
    if getattr(env, "correlation_id", None):
        extra_sql_fields["correlation_id"] = str(env.correlation_id)

    # -------------------------------------------------------------------------
    # 1. SPECIAL CASE: Spark State Snapshot -> SparkTelemetrySQL
    # (Adapts complex snapshot object to flat telemetry row + metadata)
    # -------------------------------------------------------------------------
    if env.kind == "spark.state.snapshot.v1":
        try:
            snapshot = env.payload if isinstance(env.payload, dict) else {}
            data_to_process = {
                # Ensure correlation_id is present (prefer global extraction, fallback to env.id)
                "correlation_id": extra_sql_fields.get("correlation_id") or str(env.id),
                "timestamp": snapshot.get("snapshot_ts"),
                "phi": snapshot.get("valence"),
                "trace_mode": snapshot.get("trace_mode"),
                "trace_verb": snapshot.get("trace_verb"),
                "source_service": snapshot.get("source_service"),
                "source_node": snapshot.get("source_node"),
                "metadata_": {"spark_state_snapshot": snapshot}
            }
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

            if isinstance(data_to_process, dict):
                data_to_process = data_to_process.copy()
                if "node" not in data_to_process and env.source and env.source.node:
                    data_to_process["node"] = env.source.node
                if "source_message_id" not in data_to_process and env.id:
                    data_to_process["source_message_id"] = str(env.id)

            if sql_model is CollapseMirror and isinstance(data_to_process, dict):
                base_id = (
                    data_to_process.get("id") or data_to_process.get("event_id")
                    or data_to_process.get("correlation_id")
                    or extra_sql_fields.get("correlation_id")
                    or (str(env.id) if getattr(env, "id", None) else None)
                )
                if not base_id: base_id = str(uuid.uuid4())
                if not data_to_process.get("id"): extra_sql_fields["id"] = base_id
                if not data_to_process.get("correlation_id"): extra_sql_fields["correlation_id"] = base_id

            if sql_model is CollapseEnrichment and isinstance(data_to_process, dict):
                extra_sql_fields["id"] = str(uuid.uuid4())
                target_id = data_to_process.get("id") or data_to_process.get("collapse_id")
                if target_id: extra_sql_fields["collapse_id"] = target_id

            await _write(sql_model, schema_model, data_to_process, extra_sql_fields)
            logger.info(f"Written {env.kind} -> {sql_model.__tablename__}")

        except Exception as e:
            logger.exception(f"Error writing {env.kind} to {sql_model.__tablename__}, falling back.")
            await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, str(e))
    else:
        logger.warning(f"Unknown kind {env.kind} (Route: {route_key}), writing to fallback log.")
        await asyncio.to_thread(_write_fallback, env.kind, extra_sql_fields.get("correlation_id", ""), env.payload, "Unknown kind")


def build_hunter() -> Hunter:
    patterns = settings.effective_subscribe_channels
    return Hunter(_cfg(), patterns=patterns, handler=handle_envelope)
