from __future__ import annotations

import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Dict

from sqlalchemy import inspect, update, select
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
    SparkIntrospectionLogSQL, # Legacy model
    SparkTelemetrySQL,         # New model
    BusFallbackLog,
    CognitionTraceSQL
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
            # Prefer correlation_id -> uuid
            filtered_data["id"] = filtered_data.get("correlation_id") or str(uuid.uuid4())

        # -------------------------------------------------------------------------
        # 🔄 STRATEGY: Bi-Directional Metadata Sync (Handling Async Races)
        # -------------------------------------------------------------------------
        
        # Path A: Writing SparkTelemetry -> Try to update existing Chat Log
        if sql_model_cls is SparkTelemetrySQL:
            # Populate 'metadata_' from payload 'metadata' if needed
            if "metadata_" in valid_keys and filtered_data.get("metadata_") is None:
                filtered_data["metadata_"] = data.get("metadata") or data.get("metadata_json") or {}

            # SIDE EFFECT: Update chat log if it exists
            corr_id = filtered_data.get("correlation_id")
            meta = filtered_data.get("metadata_")
            if corr_id and meta:
                 try:
                    stmt = (
                        update(ChatHistoryLogSQL)
                        .where(ChatHistoryLogSQL.correlation_id == corr_id)
                        .values(spark_meta=meta)
                    )
                    sess.execute(stmt)
                 except Exception as ex:
                     logger.warning(f"Could not back-populate chat log spark_meta: {ex}")

        # Path B: Writing Chat Log -> Check if SparkTelemetry arrived first
        if sql_model_cls is ChatHistoryLogSQL:
            corr_id = filtered_data.get("correlation_id")
            if corr_id:
                try:
                    # Look for orphaned telemetry that missed the update
                    telem = sess.query(SparkTelemetrySQL).filter(SparkTelemetrySQL.correlation_id == corr_id).first()
                    if telem and telem.metadata_:
                        current_meta = filtered_data.get("spark_meta") or {}
                        if isinstance(current_meta, dict) and isinstance(telem.metadata_, dict):
                            # Merge telemetry stats INTO the chat meta (priority to telemetry)
                            current_meta.update(telem.metadata_)
                            filtered_data["spark_meta"] = current_meta
                            logger.info(f"Merged existing SparkTelemetry into ChatLog for {corr_id}")
                except Exception as ex:
                    logger.warning(f"Failed to merge existing telemetry into chat log: {ex}")

        # -------------------------------------------------------------------------
        # Standard Column Coercion
        # -------------------------------------------------------------------------
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
                            filtered_data[key] = datetime.fromisoformat(val)
                        except ValueError:
                            pass
                    elif isinstance(val, (int, float)):
                        try:
                            filtered_data[key] = datetime.fromtimestamp(val)
                        except (ValueError, OSError):
                            pass

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
        sess.add(BusFallbackLog(
            kind=kind,
            correlation_id=correlation_id,
            payload=payload if isinstance(payload, (dict, list)) else {"raw": str(payload)},
            error=error
        ))
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
    patterns = settings.sql_writer_subscribe_channels
    return Hunter(_cfg(), patterns=patterns, handler=handle_envelope)
