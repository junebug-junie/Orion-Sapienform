"""Prose once per self_signal_episode (spec 2026-10-07 section D): the open row is
drafted by the metacog writer; the close row is NOT dispatched (sql-writer already
persists it off the trigger channel). Every other trigger is unchanged."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, APP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import app.orchestrator as orchestrator  # noqa: E402
from app.clients import CortexExecClient  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402

SOURCE = ServiceRef(name="cortex-orch", version="0", node="n")


def _dispatch(monkeypatch, payload: dict) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def _execute_plan(self, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(CortexExecClient, "execute_plan", _execute_plan)
    env = BaseEnvelope(
        kind="orion.metacog.trigger.v1",
        source=ServiceRef(name="equilibrium", version="0", node="n"),
        payload=payload,
    )
    asyncio.run(orchestrator.dispatch_metacog_trigger(object(), source=SOURCE, env=env))
    return calls


def _payload(phase: str | None, kind: str = "self_signal_episode") -> dict:
    up = {"phase": phase, "signal_id": "node:substrate.execution"} if phase else {}
    return {"trigger_kind": kind, "reason": f"{kind}:{phase}", "zen_state": "unknown", "pressure": 0.1, "upstream": up}


def test_open_row_is_drafted(monkeypatch) -> None:
    assert len(_dispatch(monkeypatch, _payload("open"))) == 1


def test_close_row_is_not_drafted(monkeypatch) -> None:
    assert _dispatch(monkeypatch, _payload("close")) == []


def test_other_kinds_with_a_close_phase_still_draft(monkeypatch) -> None:
    """transport's own episode closes keep today's behaviour."""
    assert len(_dispatch(monkeypatch, _payload("close", kind="transport"))) == 1
