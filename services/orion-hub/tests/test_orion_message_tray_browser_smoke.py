"""Browser smoke for Orion's messages in the notification tray.

Thin pytest wrapper around evals/orion_message_tray_browser.py: the REAL
index.html, orion-message-tray.js and app.js in Chromium with every /api/*
call intercepted (no Hub process). Covers the changed interactions: pinned
Orion rows and badge, the self-clearing fold, Reply (session switch, message in
chat, input focus, receipts) and a saved Dismiss.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

EVAL = Path(__file__).resolve().parents[1] / "evals" / "orion_message_tray_browser.py"


def test_orion_message_tray_in_a_real_browser() -> None:
    spec = importlib.util.spec_from_file_location("orion_message_tray_browser", EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.run(None) == 0
