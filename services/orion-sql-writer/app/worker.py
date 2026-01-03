from __future__ import annotations

import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Any, Optional, Dict

from sqlalchemy import inspect
from sqlalchemy.types import DateTime

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
    BusFallbackLog,
    CognitionTraceSQL
)
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope

# Import shared schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry
# MetaTagsPayload now includes collapse_id to prevent NotNullViolation
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.telemetry.biometrics import BiometricsPayload
from orion.schemas.dream import DreamRequest
from orion.schemas.chat import RawChat
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload

logger = logging.getLogger("sql-writer")

# Map string keys from settings to actual classes
# We map: (SQL Model, Pydantic Schema)
MODEL_MAP = {
    "CollapseMirror": (CollapseMirror, CollapseMirrorEntry),
    "CollapseEnrichment": (CollapseEnrichment, MetaTagsPayload),
    "ChatHistoryLogSQL": (ChatHistoryLogSQL, None), # Legacy log, no shared schema
    "ChatMessageSQL": (ChatMessageSQL, RawChat),
    "Dream": (Dream, DreamRequest),
    "BiometricsTelemetry": (BiometricsTelemetry, BiometricsPayload),
    "CognitionTraceSQL": (CognitionTraceSQL, CognitionTracePayload),
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
    # Pydantic v2
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    # Pydantic v1
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(payload)
    return payload


def _write_row(sql_model_cls, data: dict) -> None:
    sess = get_session()
    try:
        # 1. Introspect valid columns
        # We rely on SQLAlchemy introspection to find columns and types.
        mapper = inspect(sql_model_cls)
        valid_columns = set(c.key for c in mapper.attrs)

        # 2. Filter input data
        # Only keep keys that exist in the model definition
        filtered_data = {k: v for k, v in data.items() if k in valid_columns}

        # 3. Handle Type Coercion (specifically DateTime)
        for col in mapper.columns:
            key = col.key
            if key in filtered_data:
                val = filtered_data[key]
                # If column is DateTime and value is a string, attempt parse
                if isinstance(col.type, DateTime) and isinstance(val, str):
                    try:
                        # Attempt generic ISO parse
                        filtered_data[key] = datetime.fromisoformat(val)
                    except ValueError:
                        # Fallback: Let SQLAlchemy try or fail, but log warning?
                        # Or just leave it as string if the DB adapter handles it (Postgres/psycopg2 often handles ISO strings)
                        pass

        # 4. Merge
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
            # Dump to dict for SQL
            data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
        else:
            data = payload if isinstance(payload, dict) else {}

        # Merge extra fields (e.g. generated IDs) ensuring they override or augment
        if extra_fields:
            data.update(extra_fields)

        await asyncio.to_thread(_write_row, sql_model_cls, data)
    except Exception as e:
        logger.error(f"Failed to write to primary table: {e}")
        raise e


async def handle_envelope(env: BaseEnvelope) -> None:
    """Route by envelope.kind using the configured route map."""

    route_key = settings.route_map.get(env.kind)

    if route_key and route_key in MODEL_MAP:
        sql_model, schema_model = MODEL_MAP[route_key]
        try:
            # Enrich payload with Envelope metadata if the schema expects it
            data_to_process = env.payload
            if isinstance(data_to_process, dict):
                # Copy to avoid mutation issues if payload is reused
                data_to_process = data_to_process.copy()
                
                # Inject standard envelope fields if missing
                if "node" not in data_to_process and env.source and env.source.node:
                    data_to_process["node"] = env.source.node
                
                if "correlation_id" not in data_to_process and env.correlation_id:
                    data_to_process["correlation_id"] = str(env.correlation_id)
                
                if "source_message_id" not in data_to_process and env.id:
                    data_to_process["source_message_id"] = str(env.id)

            # Prepare extra fields for SQL that might be stripped by Pydantic or missing
            extra_sql_fields = {}
            
            # 1. CollapseMirror: Ensure ID (PK) matches Envelope ID (Idempotency)
            if route_key == "CollapseMirror":
                 # Use envelope ID as primary key if missing in payload
                 if isinstance(data_to_process, dict) and not data_to_process.get("id"):
                      extra_sql_fields["id"] = str(env.id)

            # 2. CollapseEnrichment: Handle ID (PK) and collapse_id (FK)
            if route_key == "CollapseEnrichment":
                 # Generate new PK for the enrichment record itself
                 extra_sql_fields["id"] = str(uuid.uuid4())
                 
                 # Map target ID from payload 'id' to SQL 'collapse_id' if needed
                 # (Payload 'id' usually refers to the target collapse event)
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
