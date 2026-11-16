"""Intention ledger D3: the OPEN PROMISES cue for stance_react. Ships OFF until #3531.

Off (the shipped default) the cue never reads and the prompt is exactly as before. On, it renders only
on Juniper's own turns, up to 3 promises oldest first, and fails open to "".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jinja2 import Environment

import app.open_promises_cue as cue_mod
from app.bus_listener import build_stance_react_context
from app.settings import settings
from orion.grammar.promise_view import PromiseView
from orion.schemas.harness_finalize import OpenPromiseRefV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_REPO = Path(__file__).resolve().parents[3]
_STANCE = _REPO / "orion" / "cognition" / "prompts" / "stance_react.j2"
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
JUNIPER = {"user_message": "hi", "utterance_origin": "juniper"}


def _view(n: int = 1) -> PromiseView:
    refs = [OpenPromiseRefV1(id=f"act_{i}", quote=f"I'll bring back finding {i}") for i in range(n)]
    return PromiseView(open_promises=refs, opened_at={f"act_{i}": NOW - timedelta(days=2, hours=i) for i in range(n)})


def test_ships_off():
    assert settings.open_promises_cue_enabled is False
    example = (_REPO / "services/orion-thought/.env_example").read_text()
    assert "THOUGHT_OPEN_PROMISES_CUE_ENABLED=false" in example


def test_render_shows_up_to_three_oldest_first_with_age():
    view = _view(4)
    text = cue_mod.render_open_promises_cue(view.open_promises, view.opened_at, now=NOW)
    lines = text.splitlines()
    assert lines[0] == cue_mod.HEADER and len(lines) == 4
    assert lines[1] == '- [act:act_0] 2 days ago: "I\'ll bring back finding 0"'
    assert "act_3" not in text
    assert cue_mod.render_open_promises_cue([], {}, now=NOW) == ""


@pytest.mark.asyncio
async def test_off_or_not_juniper_never_reads(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("read while off")

    monkeypatch.setattr("orion.grammar.promise_view.fetch_promise_view", _boom)
    assert await cue_mod.build_open_promises_cue("c", JUNIPER) == ""  # shipped default: off
    assert await cue_mod.build_open_promises_cue("c", {"utterance_origin": "orion"}, enabled=True) == ""
    assert await cue_mod.build_open_promises_cue("c", None, enabled=True) == ""


@pytest.mark.asyncio
async def test_on_renders_and_fails_open(monkeypatch):
    monkeypatch.setattr("orion.grammar.promise_view.fetch_promise_view", lambda sid, corr: _view(1))
    assert (await cue_mod.build_open_promises_cue("c", JUNIPER, enabled=True)).startswith(cue_mod.HEADER)
    monkeypatch.setattr("orion.grammar.promise_view.fetch_promise_view", lambda sid, corr: None)
    assert await cue_mod.build_open_promises_cue("c", JUNIPER, enabled=True) == ""

    def _raise(sid, corr):
        raise RuntimeError("db down")

    monkeypatch.setattr("orion.grammar.promise_view.fetch_promise_view", _raise)
    assert await cue_mod.build_open_promises_cue("c", JUNIPER, enabled=True) == ""


def test_stance_prompt_carries_the_block_only_when_given():
    req = StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="hi",
        association=HubAssociationBundleV1(correlation_id="c-1", broadcast=None, broadcast_stale=True,
                                           read_source="hub_sql_fallback"),
        repair_bundle=None, stance_inputs={"user_message": "hi"})
    base = build_stance_react_context(req)
    assert "open_promises" not in base and build_stance_react_context(req, open_promises="") == base
    block = cue_mod.render_open_promises_cue(_view(1).open_promises, _view(1).opened_at, now=NOW)
    ctx_base = {"user_message": "hi", "stance_inputs": {"user_message": "hi"}, "association": {},
                "repair_bundle": None, "coalition_projection": None}
    env = Environment()
    out = env.from_string(_STANCE.read_text()).render(**ctx_base, open_promises=block)
    assert "OPEN PROMISES (mine, not yet kept)" in out and "I am not told to" in out
    assert "OPEN PROMISES" not in env.from_string(_STANCE.read_text()).render(**ctx_base)
