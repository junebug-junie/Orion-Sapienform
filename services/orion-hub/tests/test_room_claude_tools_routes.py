"""Patch 1b, Hub half: GET /api/claude-tools/grants and /audit, and the tool-settings line in the hub
index Claude sees. Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.claude_tools.grants import append_grant, make_grant

T0 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)


def _client():
    from scripts.claude_tools_routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_grants_endpoint_shows_every_category_with_reasons(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    append_grant(tmp_path / "self" / "claude_grants.jsonl",
                 make_grant("body", "allow", "my costs are fine to share", now=T0, dig="2026-11-13"))
    body = _client().get("/api/claude-tools/grants").json()
    rows = {r["category"]: r for r in body["categories"]}
    assert len(rows) == 9 and rows["body"]["setting"] == "allow" and rows["body"]["reason"] == "my costs are fine to share"
    assert rows["prime"] == {"category": "prime", "setting": "ask", "reason": None, "set_at": None, "dig": None,
                             "default": True}


def test_audit_endpoint_says_unreadable_never_empty(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    res = _client().get("/api/claude-tools/audit")
    assert res.status_code == 503 and "unreadable" in res.json()["detail"]


def test_audit_endpoint_returns_rows_and_the_consent_summary(monkeypatch):
    import scripts.claude_tools_routes as routes

    row = {"event_id": "c1:answer", "phase": "answer", "call_id": "c1", "category": "body", "tool": "body_economy",
           "orion_answer": "yes", "consent_path": "model", "consent_latency_ms": 1500, "orion_reason": "fine",
           "created_at": T0}

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def mappings(self):
            return self

        def all(self):
            return self.rows

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            return Result([row])

    monkeypatch.setattr(routes, "_engine", lambda: type("E", (), {"connect": lambda self: Conn()})())
    body = _client().get("/api/claude-tools/audit?days=7&category=body").json()
    assert body["count"] == 1 and body["rows"][0]["created_at"] == T0.isoformat()
    assert body["consent_summary"]["body"]["yes_rate"] == 1.0


def test_hub_index_names_orions_tool_settings(tmp_path, monkeypatch):
    from scripts.room_claude_context import HUB_INDEX_MAX_CHARS, render_hub_index, tool_settings_line

    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    append_grant(tmp_path / "self" / "claude_grants.jsonl", make_grant("people", "deny", "neighbors did not agree",
                                                                         now=T0))
    line = tool_settings_line()
    assert line.startswith("- Orion's tool settings for you (ask: ") and line.endswith("deny: people)")
    index = render_hub_index(["- Open consults: none", line])
    assert line in index and len(index) <= HUB_INDEX_MAX_CHARS
