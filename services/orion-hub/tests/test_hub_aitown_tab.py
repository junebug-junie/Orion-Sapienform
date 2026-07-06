from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts.aitown_ui import render_aitown_tab_blocks


class _Settings:
    HUB_AITOWN_ENABLED = False
    HUB_AITOWN_UI_URL = "http://127.0.0.1:5173"


def test_hub_aitown_tab_rendered_when_enabled() -> None:
    class _Enabled(_Settings):
        HUB_AITOWN_ENABLED = True

    nav, panel = render_aitown_tab_blocks(_Enabled())
    assert 'id="aiTownTabButton"' in nav
    assert 'id="ai-town"' in panel
    assert 'id="aitownFrame"' in panel


def test_hub_aitown_tab_hidden_when_disabled() -> None:
    nav, panel = render_aitown_tab_blocks(_Settings())
    assert nav == ""
    assert panel == ""
