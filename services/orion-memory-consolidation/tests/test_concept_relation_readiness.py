"""Regression: concept-relation resolution enabled with empty hosts must be loud.

2026-09-07..10-10 the writer wrote zero decisions because CRYSTALLIZER_EMBED_HOST_URL
and CHROMA_HOST were empty while CONCEPT_RELATION_RESOLUTION_ENABLED=true, and nothing
logged or reported it.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from app import concept_relation_readiness as readiness
from orion.memory.crystallization import candidate_retrieval


def _settings(**kw):
    base = dict(
        CONCEPT_RELATION_RESOLUTION_ENABLED=True,
        CRYSTALLIZER_EMBED_HOST_URL="http://orion-athena-vector-host:8320/embedding",
        CHROMA_HOST="orion-athena-vector-db",
        CHROMA_PORT=8000,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_enabled_with_empty_hosts_is_degraded_and_warns(caplog):
    s = _settings(CRYSTALLIZER_EMBED_HOST_URL="", CHROMA_HOST="")
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(readiness.check_concept_relation_readiness(s))
    assert out["status"] == "degraded"
    assert out["problems"] == ["embed_host_url_empty", "chroma_host_empty"]
    assert readiness.READINESS_STATUS["status"] == "degraded"
    assert any("concept_relation_resolution_degraded" in r.getMessage() for r in caplog.records)


def test_disabled_is_not_degraded():
    s = _settings(CONCEPT_RELATION_RESOLUTION_ENABLED=False, CRYSTALLIZER_EMBED_HOST_URL="", CHROMA_HOST="")
    out = asyncio.run(readiness.check_concept_relation_readiness(s))
    assert out["status"] == "disabled"
    assert out["problems"] == []


def test_unreachable_hosts_are_degraded(monkeypatch):
    async def bad_embed(url):
        return "embed_host_unreachable:ConnectError"

    async def bad_chroma(host, port):
        return "chroma_unreachable:ConnectError"

    monkeypatch.setattr(readiness, "_probe_embed", bad_embed)
    monkeypatch.setattr(readiness, "_probe_chroma", bad_chroma)
    out = asyncio.run(readiness.check_concept_relation_readiness(_settings()))
    assert out["status"] == "degraded"
    assert out["problems"] == ["embed_host_unreachable:ConnectError", "chroma_unreachable:ConnectError"]


def test_reachable_hosts_are_ok(monkeypatch):
    async def ok(*a):
        return None

    monkeypatch.setattr(readiness, "_probe_embed", ok)
    monkeypatch.setattr(readiness, "_probe_chroma", ok)
    out = asyncio.run(readiness.check_concept_relation_readiness(_settings()))
    assert out["status"] == "ok"


def test_health_reports_degraded(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "READINESS_STATUS", {"status": "degraded", "problems": ["chroma_host_empty"]})
    body = asyncio.run(main.health())
    assert body["degraded"] is True
    assert body["concept_relation"]["problems"] == ["chroma_host_empty"]


def test_shipped_defaults_are_on_with_real_hosts(monkeypatch):
    for key in ("CONCEPT_RELATION_RESOLUTION_ENABLED", "CRYSTALLIZER_EMBED_HOST_URL", "CHROMA_HOST"):
        monkeypatch.delenv(key, raising=False)
    from app.settings import Settings

    s = Settings(_env_file=None)
    assert s.CONCEPT_RELATION_RESOLUTION_ENABLED is True
    assert s.CRYSTALLIZER_EMBED_HOST_URL == "http://orion-athena-vector-host:8320/embedding"
    assert s.CHROMA_HOST == "orion-athena-vector-db"
    assert readiness.config_problems(s) == []


def test_candidate_retrieval_warns_when_unconfigured(monkeypatch, caplog):
    monkeypatch.setattr(candidate_retrieval, "_warned_missing_config", False)
    cand = SimpleNamespace(subject="s", summary="x", crystallization_id="c1", kind="stance")
    with caplog.at_level(logging.WARNING):
        out = asyncio.run(candidate_retrieval.fetch_similar_candidates(cand, pool=object(), embed_host_url="", chroma_host=""))
    assert out == []
    assert any("candidate_retrieval_unconfigured" in r.getMessage() for r in caplog.records)
