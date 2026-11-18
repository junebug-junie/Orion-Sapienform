"""Browser smoke for the "Ask Orion" modal (button above Send in Hub's chat).

Thin pytest wrapper around evals/orion_abilities_browser.py: the REAL index.html, orion-abilities.js,
its CSS, the catalog JSON and app.js in Chromium with every /api/* call intercepted (no Hub process).
Covers the changed interactions: open, focus trap, Esc, search filter, Insert after typed text with
the chat box focused, Enter-to-insert, nothing sent (with a control that a real Send is detected),
and phone width.

Set ORION_ABILITIES_SMOKE_SHOT=<path.png> to keep screenshots (desktop, -phone, -composer).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

EVAL = Path(__file__).resolve().parents[1] / "evals" / "orion_abilities_browser.py"


def test_orion_abilities_modal_in_a_real_browser() -> None:
    spec = importlib.util.spec_from_file_location("orion_abilities_browser", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.run(os.environ.get("ORION_ABILITIES_SMOKE_SHOT") or None) == 0
