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
