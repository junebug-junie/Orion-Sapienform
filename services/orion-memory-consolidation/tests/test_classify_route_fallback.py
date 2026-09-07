from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

CLASSIFY_PATH = SERVICE_ROOT / "app" / "classify.py"
SPEC = importlib.util.spec_from_file_location("memory_classify_fallback_tests", CLASSIFY_PATH)
assert SPEC and SPEC.loader
classify_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classify_mod)


@pytest.mark.asyncio
async def test_classify_scores_falls_back_to_alternate_route(monkeypatch):
    bus = AsyncMock()
    settings = importlib.import_module("app.settings").settings

    async def _fake_llm(bus, *, prompt, settings, llm_route):
        if llm_route == "metacog":
            raise TimeoutError("metacog busy")
        return {"novelty_score": 0.72, "shift_kind": "TOPIC", "scoring_source": "logprobs"}

    monkeypatch.setattr(classify_mod, "_llm_classify", _fake_llm)
    scores = await classify_mod._classify_scores(
        bus, prompt="test", settings=settings, primary_route="metacog"
    )
    assert scores["novelty_score"] == 0.72
    assert scores["classify_route_used"] == "quick"


def test_resolve_classify_route_accepts_metacog_background(monkeypatch):
    """2026-09-07: TURN_CHANGE_CLASSIFY_ROUTE's default moved to metacog_background
    so this background classifier yields slot slack to Mind's now-live metacog
    traffic instead of competing evenly. Guards the widened _CLASSIFY_ROUTES set.
    """

    class _FakeSettings:
        TURN_CHANGE_CLASSIFY_ROUTE = "metacog_background"

    assert classify_mod._resolve_classify_route(_FakeSettings()) == "metacog_background"


def test_resolve_classify_route_falls_back_to_metacog_background_on_invalid(monkeypatch):
    class _FakeSettings:
        TURN_CHANGE_CLASSIFY_ROUTE = "not_a_real_route"

    assert classify_mod._resolve_classify_route(_FakeSettings()) == "metacog_background"


@pytest.mark.asyncio
async def test_classify_scores_alternate_for_metacog_background_is_quick(monkeypatch):
    bus = AsyncMock()
    settings = importlib.import_module("app.settings").settings

    async def _fake_llm(bus, *, prompt, settings, llm_route):
        if llm_route == "metacog_background":
            raise TimeoutError("metacog_background busy")
        return {"novelty_score": 0.5, "shift_kind": "TOPIC", "scoring_source": "logprobs"}

    monkeypatch.setattr(classify_mod, "_llm_classify", _fake_llm)
    scores = await classify_mod._classify_scores(
        bus, prompt="test", settings=settings, primary_route="metacog_background"
    )
    assert scores["classify_route_used"] == "quick"


@pytest.mark.asyncio
async def test_classify_scores_alternate_for_quick_is_metacog_background(monkeypatch):
    """quick's fallback yields (metacog_background) rather than competing evenly
    (plain metacog) -- 2026-09-07 change alongside the route-set widening above."""
    bus = AsyncMock()
    settings = importlib.import_module("app.settings").settings

    async def _fake_llm(bus, *, prompt, settings, llm_route):
        if llm_route == "quick":
            raise TimeoutError("quick busy")
        return {"novelty_score": 0.5, "shift_kind": "TOPIC", "scoring_source": "logprobs"}

    monkeypatch.setattr(classify_mod, "_llm_classify", _fake_llm)
    scores = await classify_mod._classify_scores(
        bus, prompt="test", settings=settings, primary_route="quick"
    )
    assert scores["classify_route_used"] == "metacog_background"
