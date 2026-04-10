from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts import api_routes


def test_control_plane_postgres_resolution_precedence(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "postgresql://cp:pw@10.0.0.8:5432/control")
    monkeypatch.setenv("SUBSTRATE_POLICY_POSTGRES_URL", "postgresql://policy:pw@10.0.0.9:5432/policy")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db:pw@10.0.0.10:5432/default")

    assert api_routes._resolve_control_plane_postgres_url() == "postgresql://cp:pw@10.0.0.8:5432/control"


def test_control_plane_postgres_resolution_falls_back_to_policy_then_database(monkeypatch) -> None:
    monkeypatch.delenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", raising=False)
    monkeypatch.setenv("SUBSTRATE_POLICY_POSTGRES_URL", "postgresql://policy:pw@10.0.0.9:5432/policy")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db:pw@10.0.0.10:5432/default")

    assert api_routes._resolve_control_plane_postgres_url() == "postgresql://policy:pw@10.0.0.9:5432/policy"

    monkeypatch.delenv("SUBSTRATE_POLICY_POSTGRES_URL", raising=False)

    assert api_routes._resolve_control_plane_postgres_url() == "postgresql://db:pw@10.0.0.10:5432/default"


def test_host_network_mode_warns_on_docker_service_hostname(monkeypatch, caplog) -> None:
    monkeypatch.setenv("HUB_DOCKER_NETWORK_MODE", "host")
    monkeypatch.setenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney")

    with caplog.at_level(logging.WARNING, logger="orion-hub.api"):
        resolved = api_routes._resolve_control_plane_postgres_url()

    assert resolved == "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    assert "looks like a Docker service name" in caplog.text
