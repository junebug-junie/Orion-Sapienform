from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    BiometricsTelemetry,
    ChatHistoryLogSQL,
    CollapseEnrichment,
    CollapseMirror,
    Dream,
    SparkIntrospectionLogSQL,
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


async def _write(sql_model_cls, schema_cls, payload: Any) -> None:
    obj = _coerce_payload(schema_cls, payload)
    data = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()  # type: ignore
    await asyncio.to_thread(_write_row, sql_model_cls, data)


async def handle_envelope(env: BaseEnvelope, channel: Optional[str] = None) -> None:
    """Route by Redis channel (preferred) with a fallback on env.kind."""
    payload = env.payload

    # Channel-first routing (keeps backwards compatibility with legacy producers).
    if channel == settings.channel_collapse_publish:
        await _write(CollapseMirror, MirrorInput, payload)
        return
    if channel == settings.channel_tags_enriched:
        await _write(CollapseEnrichment, EnrichmentInput, payload)
        return
    if channel == settings.channel_chat_log:
        await _write(ChatHistoryLogSQL, ChatHistoryInput, payload)
        return
    if channel == settings.channel_dream:
        await _write(Dream, DreamInput, payload)
        return
    if channel == settings.channel_biometrics:
        await _write(BiometricsTelemetry, BiometricsInput, payload)
        return
    if channel == settings.channel_spark_introspection:
        await _write(SparkIntrospectionLogSQL, SparkIntrospectionInput, payload)
        return

    # Kind fallback for newer envelope producers.
    if env.kind == "collapse.mirror":
        await _write(CollapseMirror, MirrorInput, payload)
        return
    if env.kind in {"collapse.enrichment", "tags.enriched"}:
        await _write(CollapseEnrichment, EnrichmentInput, payload)
        return
    if env.kind in {"chat.history", "chat.log"}:
        await _write(ChatHistoryLogSQL, ChatHistoryInput, payload)
        return
    if env.kind in {"dream.log"}:
        await _write(Dream, DreamInput, payload)
        return
    if env.kind in {"biometrics.telemetry"}:
        await _write(BiometricsTelemetry, BiometricsInput, payload)
        return
    if env.kind in {"spark.introspection.log", "spark.introspection"}:
        await _write(SparkIntrospectionLogSQL, SparkIntrospectionInput, payload)
        return

    logger.debug("No route for kind=%s channel=%s; dropping", env.kind, channel)


def build_hunter() -> Hunter:
    patterns = [
        settings.channel_collapse_publish,
        settings.channel_tags_enriched,
        settings.channel_chat_log,
        settings.channel_dream,
        settings.channel_biometrics,
        settings.channel_spark_introspection,
    ]
    return Hunter(_cfg(), patterns=patterns, handler=handle_envelope)
