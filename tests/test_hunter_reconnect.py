from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.core.bus import bus_service_chassis
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name="test-hunter",
        service_version="0",
        node_name="node-a",
        bus_url="redis://localhost:6379/0",
        bus_enabled=True,
    )


@pytest.mark.asyncio
async def test_hunter_reconnects_after_subscribe_failure() -> None:
  attempts = {"count": 0}
  handled: list[str] = []

  async def handler(env: BaseEnvelope) -> None:
    handled.append(env.kind)

  hunter = Hunter(_cfg(), handler=handler, patterns=["orion:equilibrium:metacog:trigger"])
  hunter.bus = MagicMock()
  hunter.bus.enabled = True
  hunter.bus.redis = object()
  hunter.bus.connect = AsyncMock()
  hunter.bus.codec.decode = MagicMock()

  @asynccontextmanager
  async def subscribe_ctx(*_channels, patterns: bool = False):
    attempts["count"] += 1
    if attempts["count"] == 1:
      raise ConnectionError("pubsub dropped")
    pubsub = MagicMock()
    yield pubsub

  async def iter_messages(_pubsub):
    env = BaseEnvelope(
      kind="orion.metacog.trigger.v1",
      source=ServiceRef(name="equilibrium", node="n1"),
      payload={},
    )
    yield {"type": "message", "channel": b"orion:equilibrium:metacog:trigger", "data": b"x"}
    hunter._stop.set()
    return

  hunter.bus.subscribe = subscribe_ctx
  hunter.bus.iter_messages = iter_messages
  hunter.bus.codec.decode.return_value = MagicMock(ok=True, envelope=BaseEnvelope(
    kind="orion.metacog.trigger.v1",
    source=ServiceRef(name="equilibrium", node="n1"),
    payload={},
  ))
  hunter._publish_error = AsyncMock()

  await hunter._run()

  assert attempts["count"] >= 2
  assert handled == ["orion.metacog.trigger.v1"]


def test_bus_service_chassis_has_no_mixed_style_logger_calls() -> None:
  """Regression for a real production crash-loop-adjacent bug: several
  `logger.info/warning/error(...)` calls in `bus_service_chassis.py` used to pass a
  `{}`/`{:.1f}` (str.format-style) template *plus* extra positional args to a
  `logger` that is `from loguru import logger` when loguru is importable, but
  falls back to plain `logging.getLogger("orion.bus")` (stdlib, `%`-style) when it
  is not (see the try/except import at the top of that file). A template with
  zero `%` specifiers but non-empty positional args crashes stdlib's
  `record.getMessage()` with `TypeError: not all arguments converted during string
  formatting` -- exactly what was observed in production (loguru not installed
  there) even though local dev (loguru installed) never showed the bug, since
  loguru's `.format()`-style substitution tolerated the `{}` template fine.
  `%s`-style templates have the opposite problem: they work under stdlib but
  loguru's `.format()` leaves them unsubstituted.

  The only call shape that is correct under *both* backends is a fully
  pre-formatted message (f-string or plain string) with no extra positional args
  at all -- the pattern every other logger call in this file already uses. This
  test asserts no `logger.<level>(...)` call in this file passes additional
  positional arguments after the message, via AST inspection (not a runtime
  capture, since caplog can't observe loguru's own sinks)."""
  import ast

  source_path = Path(bus_service_chassis.__file__)
  tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

  offending: list[str] = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in {"info", "warning", "error", "debug", "critical"}):
      continue
    if not (isinstance(func.value, ast.Name) and func.value.id == "logger"):
      continue
    if len(node.args) > 1:
      offending.append(f"line {node.lineno}: logger.{func.attr}(...) called with {len(node.args)} positional args")

  assert not offending, "logger.<level>() calls with template+args (backend-ambiguous) found:\n" + "\n".join(offending)
