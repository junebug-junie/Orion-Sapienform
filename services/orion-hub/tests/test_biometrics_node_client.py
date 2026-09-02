"""Tests for scripts/biometrics_node_client.py (per-node orion-biometrics HTTP client)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

# Required Hub Settings fields (no defaults) for import without a live .env.
for _key, _val in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import biometrics_node_client  # noqa: E402


def test_atlas_never_resolves_to_a_url():
    with pytest.raises(biometrics_node_client.BiometricsNodeClientError):
        biometrics_node_client._base_url("atlas")


def test_unknown_node_never_resolves_to_a_url():
    with pytest.raises(biometrics_node_client.BiometricsNodeClientError):
        biometrics_node_client._base_url("not-a-real-node")


def test_athena_resolves_to_local_url():
    assert biometrics_node_client._base_url("athena") == "http://127.0.0.1:8100"


def test_circe_resolves_to_configured_base_url(monkeypatch):
    monkeypatch.setattr(
        biometrics_node_client.settings, "CIRCE_BIOMETRICS_BASE_URL", "http://10.0.0.5:8100"
    )
    assert biometrics_node_client._base_url("circe") == "http://10.0.0.5:8100"


@pytest.mark.asyncio
async def test_fetch_snapshot_wraps_connection_failure(monkeypatch):
    import aiohttp

    class _FailingSession:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, *a, **k):
            raise aiohttp.ClientConnectionError("boom")

    monkeypatch.setattr(biometrics_node_client.aiohttp, "ClientSession", _FailingSession)

    with pytest.raises(biometrics_node_client.BiometricsNodeClientError):
        await biometrics_node_client.fetch_snapshot("athena")
