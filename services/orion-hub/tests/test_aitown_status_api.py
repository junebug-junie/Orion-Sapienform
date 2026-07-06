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

from scripts.aitown_status import fetch_aitown_status


class _Settings:
    HUB_AITOWN_ENABLED = False
    HUB_FCC_ENV_PATH = "/tmp/nonexistent-fcc.env"


def test_fetch_aitown_status_disabled() -> None:
    out = fetch_aitown_status(_Settings())
    assert out["ok"] is False
    assert out["error"] == "aitown_disabled"


def test_fetch_aitown_status_missing_convex_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Enabled(_Settings):
        HUB_AITOWN_ENABLED = True

    monkeypatch.setattr("scripts.fcc_env_catalog.load_fcc_env", lambda _p: {})
    monkeypatch.setattr("scripts.fcc_env_catalog.expand_env_path", lambda p: p)
    out = fetch_aitown_status(_Enabled())
    assert out["ok"] is False
    assert out["error"] == "aitown_convex_url_missing"
