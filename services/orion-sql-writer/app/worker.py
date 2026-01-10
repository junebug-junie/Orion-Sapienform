# services/orion-sql-writer/app/worker.py
from __future__ import annotations

import asyncio
import logging
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import inspect, update
from sqlalchemy.types import DateTime, String

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    BiometricsTelemetry,
    ChatHistoryLogSQL,
    ChatMessageSQL,
    CollapseEnrichment,
    CollapseMirror,
    Dream,
    SparkIntrospectionLogSQL,  # Legacy model
    SparkTelemetrySQL,         # New model
    BusFallbackLog,
    CognitionTraceSQL,
)

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope

# Import shared schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.telemetry.biometrics import BiometricsPayload
from orion.schemas.telemetry.dream import DreamRequest
from orion.schemas.chat import RawChat
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload

try:
    from orion.schemas.telemetry.spark import SparkTelemetryPayload
except ImportError:
    SparkTelemetryPayload = None

logger = logging.getLogger("sql-writer")

# Map string keys from settings to actual classes
# We map: (SQL Model, Pydantic Schema)
MODEL_MAP = {
    "CollapseMirror": (CollapseMirror, CollapseMirrorEntry),
    "CollapseEnrichment": (CollapseEnrichment, MetaTagsPayload),
    "ChatHistoryLogSQL": (ChatHistoryLogSQL, None),
    "ChatMessageSQL": (ChatMessageSQL, RawChat),
    "Dream": (Dream, DreamRequest),
    "BiometricsTelemetry": (BiometricsTelemetry, BiometricsPayload),
    "CognitionTraceSQL": (CognitionTraceSQL, CognitionTracePayload),
    "SparkIntrospectionLogSQL": (SparkIntrospectionLogSQL, None),
    "SparkTelemetrySQL": (SparkTelemetrySQL, SparkTelemetryPayload),
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


# -----------------------------------------------------------------------------
# KEEP: JSON safety helpers
# -----------------------------------------------------------------------------
def _json_sanitize(obj: Any, *, _seen: Optional[set[int]] = None, _depth: int = 0, _max_depth: int = 20) -> Any:
    """
    KEEP / IMPORTANT:
    This prevents (builtins.ValueError) Circular reference detected when:
      - Spark snapshots embed metadata that embeds the snapshot again
      - Fallback logs try to store raw payloads that include self-references
      - Chat spark_meta back-population accidentally includes cyclic dicts

    It also forces the payload into JSON-serializable primitives.
    """
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

    # pydantic models
    if hasattr(obj, "model_dump"):
        try:
            return _json_sanitize(obj.model_dump(mode="json"), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            return str(obj)

    # dict-like
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            # keys must be strings in JSON
            kk = k if isinstance(k, str) else str(k)
            out[kk] = _json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
        return out

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [
            _json_sanitize(v, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
            for v in list(obj)
        ]

    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # uuid
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # fallback
    return str(obj)


def _spark_meta_minimal(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    KEEP / IMPORTANT:
    ChatHistoryLog.spark_meta is a convenience cache, not a full telemetry mirror.
    Do NOT shove the entire snapshot/metadata in here (that creates cycles + bloat).
    """
    meta = row.get("metadata_") or row.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {"raw_metadata": str(meta)}

    # strip heavyweight/cyclic fields if present
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


def _write_row(sql_model_cls, data: dict) -> None:
    sess = get_session()
    try:
        mapper = inspect(sql_model_cls)

        # Accept either ORM attribute keys (e.g. "metadata_") OR raw DB column names
        # (e.g. "metadata") coming from upstream payloads.
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

        # chat_history_log uses a non-null string PK named "id".
        # If upstream omits it, use correlation_id (for idempotency) or a new uuid.
        if (
            sql_model_cls is ChatHistoryLogSQL
            and ("id" in valid_keys)
            and not filtered_data.get("id")
        ):
            filtered_data["id"] = filtered_data.get("correlation_id") or str(uuid.uuid4())

        # ---------------------------------------------------------------------
        # Standard Column Coercion
        # ---------------------------------------------------------------------
        for col in mapper.columns:
            key = col.key
            if key in filtered_data:
                val = filtered_data[key]

                # Coerce UUID -> String
                if isinstance(col.type, String) and isinstance(val, uuid.UUID):
                    filtered_data[key] = str(val)

                # Coerce DateTime
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
        # 🔄 STRATEGY: Bi-Directional Metadata Sync (Handling Async Races)
        # ---------------------------------------------------------------------

        # Path A: Writing SparkTelemetry -> update ChatLog.spark_meta if present
        if sql_model_cls is SparkTelemetrySQL:
            # Normalize metadata_ input and make it JSON-safe
            if "metadata_" in valid_keys:
                raw_meta = filtered_data.get("metadata_")
                if raw_meta is None:
                    raw_meta = data.get("metadata") or data.get("metadata_json") or {}
                if not isinstance(raw_meta, dict):
                    raw_meta = {"raw_metadata": str(raw_meta)}
                filtered_data["metadata_"] = _json_sanitize(deepcopy(raw_meta))

            corr_id = filtered_data.get("correlation_id")
            if corr_id:
                # IMPORTANT:
                # DB PK is correlation_id (per your screenshot). ORM may not be.
                # Do explicit upsert-by-correlation_id to avoid UniqueViolation.
                existing = (
                    sess.query(SparkTelemetrySQL)
                    .filter(SparkTelemetrySQL.correlation_id == corr_id)
                    .first()
                )

                if existing:
                    # Update “latest known” telemetry for this correlation_id
                    for k in ("phi", "novelty", "trace_mode", "trace_verb", "stimulus_summary", "timestamp", "source_service", "source_node", "node"):
                        if k in filtered_data and filtered_data.get(k) is not None:
                            setattr(existing, k, filtered_data.get(k))

                    # Merge metadata dicts (telemetry wins)
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
                    # Fresh insert
                    sess.add(SparkTelemetrySQL(**filtered_data))
                    sess.commit()

                # Side effect: attempt to populate ChatHistoryLog.spark_meta (minimal + safe)
                try:
                    meta_for_chat = _spark_meta_minimal(filtered_data)
                    stmt = (
                        update(ChatHistoryLogSQL)
                        .where(ChatHistoryLogSQL.correlation_id == corr_id)
                        .values(spark_meta=meta_for_chat)
                    )
                    sess.execute(stmt)
                    sess.commit()
                except Exception as ex:
                    logger.warning(f"Could not back-populate chat log spark_meta: {ex}")

                return  # done (we already committed)

        # Path B: Writing Chat Log -> Check if SparkTelemetry arrived first
        if sql_model_cls is ChatHistoryLogSQL:
            corr_id = filtered_data.get("correlation_id")
            if corr_id:
                try:
                    telem = (
                        sess.query(SparkTelemetrySQL)
                        .filter(SparkTelemetrySQL.correlation_id == corr_id)
                        .first()
                    )
                    if telem:
                        meta_blob = {
                            "phi": getattr(telem, "phi", None),
                            "novelty": getattr(telem, "novelty", None),
                            "trace_mode": getattr(telem, "trace_mode", None),
                            "trace_verb": getattr(telem, "trace_verb", None),
                            "timestamp": getattr(telem, "timestamp", None),
                            "stimulus_summary": getattr(telem, "stimulus_summary", None),
                            "node": getattr(telem, "node", None),
                            "metadata": getattr(telem, "metadata_", None),
                        }
                        filtered_data["spark_meta"] = _json_sanitize(meta_blob)
                        logger.info(f"Merged existing SparkTelemetry into ChatLog for {corr_id}")
                except Exception as ex:
                    logger.warning(f"Failed to merge existing telemetry into chat log: {ex}")

        # Default behavior for all other models
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
    try:
        if schema_cls:
            obj = _coerce_payload(schema_cls, payload)
            data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
        else:
            data = payload if isinstance(payload, dict) else {}

        if extra_fields:
            data.update(extra_fields)

        await asyncio.to_thread(_write_row, sql_model_cls, data)
    except Exception as e:
        logger.error(f"Failed to write to primary table: {e}")
        raise e


async def handle_envelope(env: BaseEnvelope) -> None:
    route_key = settings.route_map.get(env.kind)

    if route_key and route_key in MODEL_MAP:
        sql_model, schema_model = MODEL_MAP[route_key]
        try:
            data_to_process = env.payload
            if isinstance(data_to_process, dict):
                data_to_process = data_to_process.copy()

                if "node" not in data_to_process and env.source and env.source.node:
                    data_to_process["node"] = env.source.node

                # Normalize Spark state snapshots into SparkTelemetryPayload shape for DB writes.
                # Producer publishes spark.state.snapshot.v1 with payload:
                #   { snapshot_ts, phi: {coherence, novelty, ...}, valid_for_ms, ... }
                # SparkTelemetry expects scalar phi/novelty + timestamp + metadata.
                if env.kind == "spark.state.snapshot.v1":
                    try:
                        _phi = data_to_process.get("phi")
                        _snap_ts = data_to_process.get("snapshot_ts") or data_to_process.get("ts")

                        # snapshot-style: phi is a dict and we have some timestamp notion
                        if isinstance(_phi, dict) and _snap_ts:
                            # KEEP / IMPORTANT: Avoid circular reference by NOT embedding "metadata" inside snapshot copy
                            snap = deepcopy(data_to_process)

                            # Timestamp
                            ts = _snap_ts
                            if not ts and getattr(env, "timestamp", None):
                                ts = env.timestamp.isoformat()
                            if not ts:
                                ts = datetime.utcnow().isoformat() + "Z"

                            raw_meta = snap.get("metadata") or {}
                            if not isinstance(raw_meta, dict):
                                raw_meta = {"raw_metadata": raw_meta}

                            meta = deepcopy(raw_meta)

                            # Acyclic snapshot copy: remove metadata before embedding
                            snap_no_meta = dict(snap)
                            snap_no_meta.pop("metadata", None)
                            meta.setdefault("spark_state_snapshot", snap_no_meta)

                            meta = _json_sanitize(meta)

                            # Extract scalar metrics
                            # NOTE: If producer ever renames keys, keep these fallbacks.
                            coherence = _phi.get("coherence")
                            novelty = _phi.get("novelty")
                            if novelty is None and isinstance(snap.get("novelty"), (int, float)):
                                novelty = snap.get("novelty")
                            if coherence is None and isinstance(snap.get("coherence"), (int, float)):
                                coherence = snap.get("coherence")

                            data_to_process = {
                                "source_service": snap.get("source_service") or (env.source.name if env.source else None),
                                "source_node": snap.get("source_node") or (env.source.node if env.source else None),
                                "phi": coherence,
                                "novelty": novelty,
                                "trace_mode": snap.get("trace_mode"),
                                "trace_verb": snap.get("trace_verb"),
                                "stimulus_summary": snap.get("stimulus_summary"),
                                "timestamp": ts,
                                "metadata": meta,
                            }

                            # Preserve correlation_id
                            if "correlation_id" in snap:
                                data_to_process["correlation_id"] = snap.get("correlation_id")
                            elif getattr(env, "correlation_id", None):
                                data_to_process["correlation_id"] = str(env.correlation_id)

                    except Exception as _ex:
                        logger.warning(f"Failed to normalize spark.state.snapshot.v1 payload: {_ex}")

                if "correlation_id" not in data_to_process:
                    is_ad_hoc = len(env.causality_chain) == 0
                    if not is_ad_hoc and env.correlation_id:
                        data_to_process["correlation_id"] = str(env.correlation_id)

                if "source_message_id" not in data_to_process and env.id:
                    data_to_process["source_message_id"] = str(env.id)

            extra_sql_fields = {}

            if route_key == "CollapseMirror":
                if isinstance(data_to_process, dict) and not data_to_process.get("id"):
                    extra_sql_fields["id"] = str(env.id)

            if route_key == "CollapseEnrichment":
                extra_sql_fields["id"] = str(uuid.uuid4())
                if isinstance(data_to_process, dict):
                    target_id = data_to_process.get("id") or data_to_process.get("collapse_id")
                    if target_id:
                        extra_sql_fields["collapse_id"] = target_id

            await _write(sql_model, schema_model, data_to_process, extra_sql_fields)
            logger.info(f"Written {env.kind} -> {sql_model.__tablename__}")

        except Exception as e:
            logger.exception(f"Error writing {env.kind} to {sql_model.__tablename__}, falling back.")
            await asyncio.to_thread(_write_fallback, env.kind, str(env.correlation_id), env.payload, str(e))
    else:
        logger.warning(f"Unknown kind {env.kind}, writing to fallback log.")
        await asyncio.to_thread(_write_fallback, env.kind, str(env.correlation_id), env.payload, "Unknown kind")


def build_hunter() -> Hunter:
    patterns = settings.effective_subscribe_channels
    return Hunter(_cfg(), patterns=patterns, handler=handle_envelope)
