"""Browser smoke for "@claude" in the Hub chat box.

Thin pytest wrapper around evals/claude_address_browser.py: the REAL index.html, claude-address.js
and app.js in Chromium with every /api/* call and the chat websocket mocked (no Hub process).
Covers the changed interaction: the amber "To Claude" cue, Enter sends one invite and no Orion turn,
held while Claude replies, bare @claude fallback, refused invite keeps the text, and the control
that ordinary text (including "@claudette" and mid-sentence "@claude") still reaches Orion.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

EVAL = Path(__file__).resolve().parents[1] / "evals" / "claude_address_browser.py"


def test_claude_address_in_a_real_browser() -> None:
    spec = importlib.util.spec_from_file_location("claude_address_browser", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.run() == 0
