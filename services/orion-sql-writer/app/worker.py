from __future__ import annotations

import asyncio
import logging
import json
from typing import Any, Optional, Dict

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    BiometricsTelemetry,
    ChatHistoryLogSQL,
    CollapseEnrichment,
    CollapseMirror,
    Dream,
    SparkIntrospectionLogSQL,
    BusFallbackLog
)
from app.schemas import (
    BiometricsInput,
    ChatHistoryInput,
    EnrichmentInput,
    DreamInput,
    MirrorInput,
    SparkIntrospectionInput,
)

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope

logger = logging.getLogger("sql-writer")

# Map string keys from settings to actual classes
MODEL_MAP = {
    "CollapseMirror": (CollapseMirror, MirrorInput),
    "CollapseEnrichment": (CollapseEnrichment, EnrichmentInput),
    "ChatHistoryLogSQL": (ChatHistoryLogSQL, ChatHistoryInput),
    "Dream": (Dream, DreamInput),
    "BiometricsTelemetry": (BiometricsTelemetry, BiometricsInput),
    "SparkIntrospectionLogSQL": (SparkIntrospectionLogSQL, SparkIntrospectionInput),
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
    # Pydantic v2 compatibility: support dict or model already
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)  # type: ignore[attr-defined]


def _write_row(sql_model_cls, data: dict) -> None:
    sess = get_session()
    try:
        sess.merge(sql_model_cls(**data))
        sess.commit()
    finally:
        try:
            sess.close()
        finally:
            remove_session()

def _write_fallback(kind: str, correlation_id: str, payload: Any, error: str = None) -> None:
    sess = get_session()
    try:
        # Ensure payload is JSON-serializable if it isn't already (e.g. if it's a dict, sqlalchemy JSON type handles it, but verify)
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

async def _write(sql_model_cls, schema_cls, payload: Any) -> None:
    try:
        obj = _coerce_payload(schema_cls, payload)
        data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()  # type: ignore
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
            await _write(sql_model, schema_model, env.payload)
            logger.info(f"Written {env.kind} -> {sql_model.__tablename__}")
        except Exception as e:
             logger.exception(f"Error writing {env.kind} to {sql_model.__tablename__}, falling back.")
             await asyncio.to_thread(_write_fallback, env.kind, str(env.correlation_id), env.payload, str(e))
    else:
        # Fallback for unknown kinds
        logger.warning(f"Unknown kind {env.kind}, writing to fallback log.")
        await asyncio.to_thread(_write_fallback, env.kind, str(env.correlation_id), env.payload, "Unknown kind")


def build_hunter() -> Hunter:
    patterns = settings.sql_writer_subscribe_channels
    return Hunter(_cfg(), patterns=patterns, handler=handle_envelope)
