"""Gate: a normal received bus message must not produce a log line by default.

Fleet regression (2026-10-11): PR #2607 moved Hunter's per-message intake line
to ``logger.debug`` -- but the chassis logs through loguru, whose default stderr
sink is DEBUG. Services that only configure stdlib ``logging`` (orion-sql-writer,
orion-biometrics, ...) never touch loguru, so the "DEBUG" line still printed for
every message (sql-writer: ~3,200 lines / 10 min). The per-message line is now
decided in the chassis from the service's own LOG_LEVEL.

The sinks below are added at DEBUG on purpose: that is exactly what an
unconfigured loguru looks like inside a service container.
"""
from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter

REPO_ROOT = Path(__file__).resolve().parents[4]


def _hunter(n_messages: int, handled: list) -> Hunter:
    env = BaseEnvelope(kind="orion.test.v1", source=ServiceRef(name="t", node="n"), payload={})

    async def handler(e: BaseEnvelope) -> None:
        handled.append(e.kind)

    hunter = Hunter(
        ChassisConfig(
            service_name="test-svc",
            service_version="0",
            node_name="n",
            bus_url="redis://localhost:6379/0",
            bus_enabled=True,
        ),
        handler=handler,
        patterns=["orion:test:*"],
    )
    hunter.bus = MagicMock()
    hunter.bus.enabled = True
    hunter.bus.redis = object()
    hunter.bus.codec.decode = MagicMock(return_value=MagicMock(ok=True, envelope=env))

    @asynccontextmanager
    async def subscribe_ctx(*_a, patterns: bool = False):
        yield MagicMock()

    async def iter_messages(_pubsub):
        for _ in range(n_messages):
            yield {"type": "pmessage", "channel": b"orion:test:x", "pattern": b"orion:test:*", "data": b"x"}
        hunter._stop.set()

    hunter.bus.subscribe = subscribe_ctx
    hunter.bus.iter_messages = iter_messages
    hunter._publish_error = AsyncMock()
    return hunter


@pytest.fixture
def debug_sink():
    """Every record at DEBUG and above -- the unconfigured-loguru default."""
    records: list[tuple[str, str]] = []
    sink_id = logger.add(lambda m: records.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    yield records
    logger.remove(sink_id)


def _intake_lines(records):
    return [(lvl, msg) for lvl, msg in records if msg.startswith("Hunter intake channel=")]


@pytest.mark.parametrize("level", [None, "INFO", "info", "WARNING"])
def test_normal_messages_emit_no_per_message_line_by_default(monkeypatch, debug_sink, level) -> None:
    if level is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", level)
    handled: list = []
    asyncio.run(_hunter(40, handled)._run())

    assert len(handled) == 40
    intake = _intake_lines(debug_sink)
    # Only the reconnect-diagnosis line survives, even with a DEBUG sink attached.
    assert len(intake) == 1, intake
    assert intake[0][0] == "INFO"
    assert "first_after_subscribe=true" in intake[0][1]


def test_log_level_debug_opts_back_into_per_message_lines(monkeypatch, debug_sink) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    handled: list = []
    asyncio.run(_hunter(40, handled)._run())

    intake = _intake_lines(debug_sink)
    assert len(intake) == 40
    assert sum(1 for lvl, _ in intake if lvl == "DEBUG") == 39


def _hunter_services() -> list[Path]:
    out = set()
    for py in (REPO_ROOT / "services").glob("*/**/*.py"):
        if "/tests/" in py.as_posix():
            continue
        try:
            if "Hunter(" in py.read_text(errors="ignore"):
                out.add(py.relative_to(REPO_ROOT / "services").parts[0])
        except OSError:
            continue
    return [REPO_ROOT / "services" / s for s in sorted(out)]


def test_no_hunter_service_ships_log_level_debug_by_default() -> None:
    """The opt-in is LOG_LEVEL=DEBUG, so a service template that defaults to it
    would silently turn the per-message firehose back on for that service."""
    services = _hunter_services()
    assert len(services) >= 10, "Hunter service discovery broke"
    pattern = re.compile(r"^\s*-?\s*LOG_LEVEL\s*[=:]\s*\$?\{?(?:LOG_LEVEL:-)?\s*['\"]?DEBUG", re.I | re.M)
    offenders = []
    for svc in services + [REPO_ROOT]:
        for name in (".env_example", "docker-compose.yml"):
            f = svc / name
            if f.exists() and pattern.search(f.read_text(errors="ignore")):
                offenders.append(str(f.relative_to(REPO_ROOT)))
    assert not offenders, f"LOG_LEVEL defaults to DEBUG in: {offenders}"
