"""orion-memory-consolidation's confirmation-loop wiring: the bus consumer and the flag.

The SQL itself is tested against Postgres in orion/memory/episode/tests/test_confirmation_pg.py and
end to end through the Hub route in services/orion-hub/tests/test_ask_resolve_pg.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory.episode import confirmation as c
from orion.schemas.attention_salience import AttentionLoopOutcomeV1

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))
for key in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
    del sys.modules[key]
_spec = importlib.util.spec_from_file_location("mc_confirmation_loop", SERVICE_ROOT / "app" / "confirmation_loop.py")
loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop)

MID = "00000000-0000-0000-0000-000000000007"


def _env(loop_id: str, kind: str = "attention.loop.outcome.v1", **feats) -> BaseEnvelope:
    o = AttentionLoopOutcomeV1(outcome_id="o-1", loop_id=loop_id, theme_key=loop_id, verdict="resolved",
                               note="yes", features_at_close={"resolution": "confirmed", "ask_id": "a", **feats})
    return BaseEnvelope(kind=kind, source=ServiceRef(name="hub", version="t", node="n"),
                        payload=o.model_dump(mode="json"))


def test_memory_outcome_parses_and_other_loops_are_ignored():
    got = loop.outcome_from_envelope(_env(c.loop_id_for(MID)))
    assert (got.outcome_id, got.loop_id, got.resolution, got.note, got.ask_id) == (
        "o-1", c.loop_id_for(MID), "confirmed", "yes", "a")
    assert loop.outcome_from_envelope(_env("open-loop-7376a3da4050")) is None
    assert loop.outcome_from_envelope(_env(c.loop_id_for(MID), kind="memory.turn.persisted.v1")) is None
    bad = BaseEnvelope(kind="attention.loop.outcome.v1", source=ServiceRef(name="h", version="t", node="n"),
                       payload={"loop_id": c.loop_id_for(MID)})
    assert loop.outcome_from_envelope(bad) is None


class _Pool:
    def __init__(self):
        self.acquired = 0

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self_inner):
                pool.acquired += 1
                return object()

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()


def test_handler_applies_memory_outcomes_only_when_enabled(monkeypatch):
    applied = []

    async def _apply(conn, outcome, now=None):
        applied.append(outcome.outcome_id)
        return "confirmed"

    monkeypatch.setattr(loop, "apply_outcome", _apply)
    on = SimpleNamespace(MEMORY_CONFIRMATION_LOOP_ENABLED=True)
    off = SimpleNamespace(MEMORY_CONFIRMATION_LOOP_ENABLED=False)
    pool = _Pool()
    assert asyncio.run(loop.handle_loop_outcome(_env(c.loop_id_for(MID)), pool=pool, settings=on)) == "confirmed"
    assert asyncio.run(loop.handle_loop_outcome(_env("open-loop-1"), pool=pool, settings=on)) is None
    assert asyncio.run(loop.handle_loop_outcome(_env(c.loop_id_for(MID)), pool=pool, settings=off)) is None
    assert asyncio.run(loop.handle_loop_outcome(_env(c.loop_id_for(MID)), pool=None, settings=on)) is None
    assert applied == ["o-1"]


def test_flag_ships_on_everywhere():
    """Juniper's rule: flags ship ON (settings default, .env_example, compose default)."""
    from app.settings import Settings

    assert Settings.model_fields["MEMORY_CONFIRMATION_LOOP_ENABLED"].default is True
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    compose = (SERVICE_ROOT / "docker-compose.yml").read_text()
    assert "\nMEMORY_CONFIRMATION_LOOP_ENABLED=true\n" in env_example
    assert "MEMORY_CONFIRMATION_LOOP_ENABLED=${MEMORY_CONFIRMATION_LOOP_ENABLED:-true}" in compose
    assert "CHANNEL_ATTENTION_LOOP_OUTCOME=${CHANNEL_ATTENTION_LOOP_OUTCOME:-orion:attention:loop_outcome}" in compose


def test_main_subscribes_the_outcome_channel_and_starts_the_ticker():
    main = (SERVICE_ROOT / "app" / "main.py").read_text()
    assert "settings.CHANNEL_ATTENTION_LOOP_OUTCOME" in main
    assert "run_confirmation_loop(pg_pool, settings)" in main
    # The outcome branch runs before the consolidation kill switch, so turning consolidation off
    # does not silently stop Juniper's answers from landing.
    assert main.index("env.kind == LOOP_OUTCOME_KIND") < main.index("if not settings.MEMORY_CONSOLIDATION_ENABLED")


@pytest.mark.parametrize("enabled,expected_calls", [(True, 1), (False, 0)])
def test_ticker_runs_only_when_enabled(monkeypatch, enabled, expected_calls):
    calls = []

    async def _tick(pool):
        calls.append(pool)
        return {"expired": 0, "applied": 0, "opened": 0}

    async def _sleep(_s):
        raise asyncio.CancelledError

    monkeypatch.setattr(loop, "run_tick", _tick)
    monkeypatch.setattr(loop.asyncio, "sleep", _sleep)
    settings = SimpleNamespace(MEMORY_CONFIRMATION_LOOP_ENABLED=enabled, MEMORY_CONFIRMATION_TICK_SEC=60)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.run_confirmation_loop("pool", settings))
    assert len(calls) == expected_calls
