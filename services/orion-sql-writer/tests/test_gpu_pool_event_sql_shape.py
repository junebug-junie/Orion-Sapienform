"""Compile-time shape checks for the gpu_pool_events write path (no Postgres required).
Same four guarantees as test_durable_run_state_sql_shape.py: subscribed, routed, every
schema field lands on a column, bounded retention."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL, GpuPoolEventV1  # noqa: E402

from app.grammar_retention_loop import retention_days_for  # noqa: E402
from app.grammar_truth import GRAMMAR_RETENTION_TABLES  # noqa: E402
from app.models.gpu_pool_event import GpuPoolEventSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402


def _env_example(key: str):
    raw = next(line.split("=", 1)[1].strip() for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line.startswith(f"{key}="))
    return json.loads(raw)


def test_the_channel_is_actually_subscribed() -> None:
    assert GPU_POOL_EVENT_CHANNEL in _env_example("SQL_WRITER_SUBSCRIBE_CHANNELS")
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert GPU_POOL_EVENT_CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP["gpu_pool.event.v1"] == "GpuPoolEventSQL"
    assert _env_example("SQL_WRITER_ROUTE_MAP_JSON")["gpu_pool.event.v1"] == "GpuPoolEventSQL"
    assert MODEL_MAP["GpuPoolEventSQL"] == (GpuPoolEventSQL, GpuPoolEventV1)
    assert GpuPoolEventSQL in INSERT_ONLY_MODELS


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(GpuPoolEventSQL).columns}
    for name in GpuPoolEventV1.model_fields:
        if name == "schema_version":
            continue
        assert name in columns, name


def test_the_table_has_bounded_retention_by_default() -> None:
    assert "gpu_pool_events" in dict(GRAMMAR_RETENTION_TABLES)
    assert retention_days_for(Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]))["gpu_pool_events"] == 30
