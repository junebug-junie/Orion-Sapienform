"""Tests for Hub's /api/reverie/* routes (reverie tab backend).

Live-verified separately against real Postgres data (both endpoints, the
image-serving route including a real integrity-checked PNG, and the 400/404
error paths) -- these tests cover the merge/shape logic a live check alone
doesn't pin: multi-row joins, chains with no artifacts, thought extraction
from chain_json, and the downstream-flag computation.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

for _key, _val in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
    ("POSTGRES_URI", "postgresql://test:test@localhost/test"),
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import reverie_routes  # noqa: E402

NOW = datetime(2026, 8, 26, 3, 0, 0, tzinfo=timezone.utc)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """Dispatches on a distinctive substring in the SQL text -- matches this
    repo's established fake-engine test pattern (e.g. app/store.py tests
    across several services this session)."""

    def __init__(self, table_rows: dict[str, list[dict]]):
        self.table_rows = table_rows
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.queries.append(sql)
        if "count(*)" in sql.lower() and "reverie_visual_chain" in sql:
            rows = self.table_rows.get("reverie_visual_chain", [])
            return _FakeResult([{"total": len(rows)}])
        if "FROM reverie_visual_chain" in sql:
            return _FakeResult(self.table_rows.get("reverie_visual_chain", []))
        if "FROM reverie_visual_artifact" in sql and "sha256 =" in sql:
            sha = (params or {}).get("sha256")
            rows = [r for r in self.table_rows.get("reverie_visual_artifact", []) if r["sha256"] == sha]
            return _FakeResult(rows)
        if "FROM reverie_visual_artifact" in sql:
            return _FakeResult(self.table_rows.get("reverie_visual_artifact", []))
        if "FROM substrate_reverie_chain" in sql:
            return _FakeResult(self.table_rows.get("substrate_reverie_chain", []))
        if "FROM substrate_reverie_thought" in sql:
            return _FakeResult(self.table_rows.get("substrate_reverie_thought", []))
        if "FROM dream_compaction_request_queue" in sql:
            return _FakeResult(self.table_rows.get("dream_compaction_request_queue", []))
        if "FROM substrate_reverie_resonance_alert" in sql:
            return _FakeResult(self.table_rows.get("substrate_reverie_resonance_alert", []))
        raise AssertionError(f"unexpected query in fake conn: {sql}")


class _FakeEngine:
    def __init__(self, table_rows: dict[str, list[dict]]):
        self.table_rows = table_rows

    def connect(self):
        return _FakeConn(self.table_rows)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(reverie_routes, "_engine_instance", None)
    app = FastAPI()
    app.include_router(reverie_routes.router)
    return TestClient(app)


def _set_tables(monkeypatch, **tables):
    monkeypatch.setattr(reverie_routes, "_engine", lambda: _FakeEngine(tables))


# --- visual/recent ----------------------------------------------------------


def test_visual_recent_merges_chain_and_artifact(client, monkeypatch):
    _set_tables(
        monkeypatch,
        reverie_visual_chain=[
            {
                "chain_id": "c1",
                "created_at": NOW,
                "theme_key": None,
                "terminal_reason": "max_steps",
                "ema_salience": 0.0,
                "prior_description": "a quiet room",
                "chain_json": {
                    "prompt": "a quiet room. Orion is currently thinking: curiosity about the mesh.",
                    "context_text": "curiosity about the mesh",
                    "self_study_text": "vision events dropped 0.36x vs baseline",
                    "memory_text": "Orion and Juniper talked through the mesh work",
                    "continuity_streak": 1,
                    "continuity_reset": False,
                    "description": "a quiet room",
                },
            }
        ],
        reverie_visual_artifact=[
            {
                "sha256": "a" * 64,
                "chain_id": "c1",
                "step_index": 0,
                "mime": "image/png",
                "bytes": 1234,
                "width": 512,
                "height": 512,
                "description": "a quiet room",
                "created_at": NOW,
            }
        ],
    )
    resp = client.get("/api/reverie/visual/recent")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["chains"]) == 1
    chain = body["chains"][0]
    assert chain["chain_id"] == "c1"
    assert chain["prompt"] == "a quiet room. Orion is currently thinking: curiosity about the mesh."
    # Patch 3: surfaced as its own field, not just prose inside `prompt`.
    assert chain["context_text"] == "curiosity about the mesh"
    # Patch 5: same treatment for the self-study context-seed.
    assert chain["self_study_text"] == "vision events dropped 0.36x vs baseline"
    # Patch 6: same treatment for the memory-crystallization context-seed.
    assert chain["memory_text"] == "Orion and Juniper talked through the mesh work"
    # Patch 4: same treatment for the continuity-reset bookkeeping.
    assert chain["continuity_streak"] == 1
    assert chain["continuity_reset"] is False
    assert len(chain["artifacts"]) == 1
    assert chain["artifacts"][0]["sha256"] == "a" * 64
    assert chain["artifacts"][0]["image_url"] == f"/api/reverie/visual/image/{'a' * 64}"


def test_visual_recent_context_text_absent_is_none_not_a_keyerror(client, monkeypatch):
    """A chain_json written before Patch 3 has no context_text key at all --
    `.get()` must degrade to None, never KeyError, on old rows."""
    _set_tables(
        monkeypatch,
        reverie_visual_chain=[
            {
                "chain_id": "c1",
                "created_at": NOW,
                "theme_key": None,
                "terminal_reason": "max_steps",
                "ema_salience": 0.0,
                "prior_description": "a quiet room",
                "chain_json": {"prompt": "a quiet room. Continue...", "description": "a quiet room"},
            }
        ],
        reverie_visual_artifact=[],
    )
    resp = client.get("/api/reverie/visual/recent")
    assert resp.status_code == 200
    chain = resp.json()["chains"][0]
    assert chain["context_text"] is None
    # Patch 5: same discipline for the self-study context-seed key.
    assert chain["self_study_text"] is None
    # Patch 6: same discipline for the memory-crystallization context-seed key.
    assert chain["memory_text"] is None
    # Patch 4: a chain_json written before Patch 4 has neither key either --
    # same .get() degrade-to-None discipline, never KeyError.
    assert chain["continuity_streak"] is None
    assert chain["continuity_reset"] is None


def test_visual_recent_has_more_true_when_extra_row_fetched(client, monkeypatch):
    """_fetch_visual_recent asks for limit+1 rows and reports has_more from
    whether that extra row showed up (the standard cursor-pagination trick,
    avoiding a second COUNT(*) round trip). The fake conn returns all stored
    rows regardless of the real WHERE/LIMIT clause, so what this test
    actually pins is the trim-and-flag *logic* in Python: 3 stored rows,
    limit=2 -> has_more True and exactly 2 chains returned, not 3."""
    _set_tables(
        monkeypatch,
        reverie_visual_chain=[
            {
                "chain_id": f"c{i}",
                "created_at": NOW,
                "theme_key": None,
                "terminal_reason": "max_steps",
                "ema_salience": 0.0,
                "prior_description": None,
                "chain_json": {"prompt": "p"},
            }
            for i in range(3)
        ],
        reverie_visual_artifact=[],
    )
    resp = client.get("/api/reverie/visual/recent?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["chains"]) == 2
    assert body["has_more"] is True
    assert body["limit"] == 2
    assert body["next_before"] == body["chains"][-1]["created_at"]


def test_visual_recent_has_more_false_when_exactly_limit_rows(client, monkeypatch):
    _set_tables(
        monkeypatch,
        reverie_visual_chain=[
            {
                "chain_id": "c0",
                "created_at": NOW,
                "theme_key": None,
                "terminal_reason": "max_steps",
                "ema_salience": 0.0,
                "prior_description": None,
                "chain_json": {"prompt": "p"},
            }
        ],
        reverie_visual_artifact=[],
    )
    resp = client.get("/api/reverie/visual/recent?limit=5")
    body = resp.json()
    assert len(body["chains"]) == 1
    assert body["has_more"] is False


def test_visual_recent_accepts_before_cursor(client, monkeypatch):
    """The `before` query param must parse as a real datetime and reach the
    fetch function without erroring -- regression coverage for the
    OFFSET -> cursor pagination switch. Uses `params=` (proper query-string
    encoding) rather than string-interpolating the ISO timestamp directly --
    an unencoded `+00:00` UTC offset decodes as a literal space in a raw
    query string, which is exactly why the real JS client always wraps this
    value in encodeURIComponent()."""
    _set_tables(monkeypatch, reverie_visual_chain=[], reverie_visual_artifact=[])
    resp = client.get("/api/reverie/visual/recent", params={"before": NOW.isoformat()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["chains"] == []
    assert body["next_before"] is None


def test_visual_recent_rejects_malformed_before(client):
    resp = client.get("/api/reverie/visual/recent?before=not-a-timestamp")
    assert resp.status_code == 422


def test_visual_recent_surfaces_generation_error(client, monkeypatch):
    """A generation_failed chain's chain_json carries "error", not
    "artifact_sha256"/"description" -- the cockpit needs this to explain why
    a run produced no image instead of rendering an unexplained empty card."""
    _set_tables(
        monkeypatch,
        reverie_visual_chain=[
            {
                "chain_id": "c-err",
                "created_at": NOW,
                "theme_key": None,
                "terminal_reason": "generation_failed",
                "ema_salience": 0.0,
                "prior_description": None,
                "chain_json": {"prompt": "x", "error": "diffusion-host /generate returned HTTP 429"},
            }
        ],
        reverie_visual_artifact=[],
    )
    resp = client.get("/api/reverie/visual/recent")
    chain = resp.json()["chains"][0]
    assert chain["error"] == "diffusion-host /generate returned HTTP 429"


def test_visual_recent_chain_with_no_artifact_yet(client, monkeypatch):
    """generation_failed chains have a chain row but never an artifact row --
    must not crash the merge, must return an empty artifacts list."""
    _set_tables(
        monkeypatch,
        reverie_visual_chain=[
            {
                "chain_id": "c-failed",
                "created_at": NOW,
                "theme_key": None,
                "terminal_reason": "generation_failed",
                "ema_salience": 0.0,
                "prior_description": None,
                "chain_json": {"prompt": "x", "error": "boom"},
            }
        ],
        reverie_visual_artifact=[],
    )
    resp = client.get("/api/reverie/visual/recent")
    assert resp.status_code == 200
    chain = resp.json()["chains"][0]
    assert chain["artifacts"] == []
    assert chain["terminal_reason"] == "generation_failed"


def test_visual_image_rejects_invalid_sha256(client):
    resp = client.get("/api/reverie/visual/image/not-a-sha")
    assert resp.status_code == 400


def test_visual_image_404_when_row_missing(client, monkeypatch):
    _set_tables(monkeypatch, reverie_visual_artifact=[])
    resp = client.get(f"/api/reverie/visual/image/{'b' * 64}")
    assert resp.status_code == 404


# --- text/recent --------------------------------------------------------


def test_text_recent_extracts_thoughts_from_chain_json_and_downstream_flags(client, monkeypatch):
    _set_tables(
        monkeypatch,
        substrate_reverie_chain=[
            {
                "chain_id": "tc1",
                "created_at": NOW,
                "theme_key": "ol-1",
                "terminal_reason": "pressure_discharged",
                "ema_salience": 0.7,
                "committed_proposal_id": None,
                "chain_json": {"thought_ids": ["th-1", "th-2"]},
            }
        ],
        substrate_reverie_thought=[
            {"thought_id": "th-1", "created_at": NOW, "salience": 0.6, "interpretation": "first"},
            {"thought_id": "th-2", "created_at": NOW, "salience": 0.8, "interpretation": "second"},
        ],
        dream_compaction_request_queue=[{"origin_chain_id": "tc1"}],
        substrate_reverie_resonance_alert=[{"theme_key": "ol-1", "n": 2}],
    )
    resp = client.get("/api/reverie/text/recent")
    assert resp.status_code == 200
    chain = resp.json()["chains"][0]
    assert [t["thought_id"] for t in chain["thoughts"]] == ["th-1", "th-2"]
    assert chain["downstream"]["compaction_queued"] is True
    assert chain["downstream"]["theme_resonance_alert_count"] == 2


def test_text_recent_no_downstream_effect_when_nothing_queued(client, monkeypatch):
    _set_tables(
        monkeypatch,
        substrate_reverie_chain=[
            {
                "chain_id": "tc2",
                "created_at": NOW,
                "theme_key": "unknown",
                "terminal_reason": "no_coalition",
                "ema_salience": 0.0,
                "committed_proposal_id": None,
                "chain_json": {},
            }
        ],
        substrate_reverie_thought=[],
        dream_compaction_request_queue=[],
        substrate_reverie_resonance_alert=[],
    )
    resp = client.get("/api/reverie/text/recent")
    chain = resp.json()["chains"][0]
    assert chain["thoughts"] == []
    assert chain["downstream"]["compaction_queued"] is False
    assert chain["downstream"]["theme_resonance_alert_count"] == 0


def test_router_registered_on_api_routes():
    _ensure_hub_scripts_import_path()
    from scripts import api_routes

    paths = {r.path for r in api_routes.router.routes}
    assert "/api/reverie/visual/recent" in paths
    assert "/api/reverie/text/recent" in paths
