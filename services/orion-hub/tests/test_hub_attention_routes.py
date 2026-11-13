"""Hub attention minutes: heartbeat route + page wiring.

Behaviour of the browser tracker itself is covered by
static/js/hub-attention-tracker.test.js (node:test, run in CI by
orion-static-gates.yml and here via test_tracker_js_cases_pass_under_node).
"""

from __future__ import annotations

import os
import shutil
import subprocess
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
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for path in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import hub_attention_routes as har  # noqa: E402

MINUTE = datetime(2026, 11, 10, 17, 4, tzinfo=timezone.utc)


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _FakeConn:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, stmt, params):
        if self.engine.fail:
            raise RuntimeError('relation "hub_attention_minute" does not exist')
        self.engine.calls.append((str(stmt), dict(params)))
        # Emulate the upsert: key = (server minute, client).
        key = (MINUTE, params["client"])
        row = self.engine.rows.get(key)
        if row is None:
            self.engine.rows[key] = {"beats": 1, "tab": params["tab"]}
        else:
            row["beats"] += 1
            if params["tab"] is not None:
                row["tab"] = params["tab"]
        return _FakeResult((MINUTE,))


class _FakeBegin:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return _FakeConn(self.engine)

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []
        self.rows: dict = {}

    def begin(self):
        return _FakeBegin(self)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(har.router)
    return TestClient(app)


@pytest.fixture
def engine(monkeypatch):
    eng = _FakeEngine()
    monkeypatch.setattr(har, "_engine", lambda: eng)
    monkeypatch.setattr(har.settings, "HUB_ATTENTION_TRACKER_ENABLED", True, raising=False)
    monkeypatch.setattr(har, "_last_error_log_at", 0.0)
    return eng


def test_heartbeat_upserts_one_row_per_minute_and_client(client, engine) -> None:
    for _ in range(3):
        r = client.post("/api/hub-attention/heartbeat", json={"client": "desktop", "tab": "#reading"})
        assert r.status_code == 200
        assert r.json() == {
            "ok": True, "enabled": True, "minute": MINUTE.isoformat(), "client": "desktop", "tab": "reading",
        }
    client.post("/api/hub-attention/heartbeat", json={"client": "mobile"})
    assert engine.rows[(MINUTE, "desktop")] == {"beats": 3, "tab": "reading"}
    assert engine.rows[(MINUTE, "mobile")] == {"beats": 1, "tab": None}


def test_minute_is_bucketed_server_side_and_client_time_is_ignored(client, engine) -> None:
    client.post(
        "/api/hub-attention/heartbeat",
        json={"client": "desktop", "tab": "chat", "minute": "1999-01-01T00:00:00Z", "ts": 1},
    )
    sql, params = engine.calls[0]
    assert "date_trunc('minute', now())" in sql
    assert "ON CONFLICT (minute, client)" in sql
    assert "beats = hub_attention_minute.beats + 1" in sql
    assert set(params) == {"client", "tab"}  # nothing time-like from the client reaches SQL


@pytest.mark.parametrize(
    "payload",
    [{}, {"client": "tablet"}, {"client": 7}, {"tab": "chat"}, {"client": "desktop", "tab": "x" * 600}],
)
def test_bad_payload_is_rejected_without_a_write(client, engine, payload) -> None:
    r = client.post("/api/hub-attention/heartbeat", json=payload)
    assert r.status_code == 422
    assert engine.calls == []


def test_tab_is_sanitized_and_capped() -> None:
    assert har.normalize_tab("#reading") == "reading"
    assert har.normalize_tab("#<script>alert(1)</script>") == "scriptalert1/script"
    assert har.normalize_tab("") is None
    assert har.normalize_tab("#") is None
    assert len(har.normalize_tab("#" + "a" * 300)) == har.TAB_MAX_LEN


def test_db_failure_returns_200_with_reason_and_logs_once(client, monkeypatch, caplog) -> None:
    eng = _FakeEngine(fail=True)
    monkeypatch.setattr(har, "_engine", lambda: eng)
    monkeypatch.setattr(har.settings, "HUB_ATTENTION_TRACKER_ENABLED", True, raising=False)
    monkeypatch.setattr(har, "_last_error_log_at", 0.0)
    with caplog.at_level("WARNING", logger="orion-hub.hub_attention"):
        for _ in range(3):
            r = client.post("/api/hub-attention/heartbeat", json={"client": "desktop"})
            assert r.status_code == 200
            assert r.json() == {"ok": False, "enabled": True, "reason": "db_error"}
    warnings = [rec for rec in caplog.records if "hub_attention_write_failed" in rec.getMessage()]
    assert len(warnings) == 1  # rate-limited: a missing table cannot flood the log


def test_missing_postgres_uri_is_a_db_error_not_a_crash(client, monkeypatch) -> None:
    monkeypatch.setattr(har, "_engine_instance", None)
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.setattr(har.settings, "HUB_ATTENTION_TRACKER_ENABLED", True, raising=False)
    r = client.post("/api/hub-attention/heartbeat", json={"client": "mobile"})
    assert r.status_code == 200
    assert r.json()["reason"] == "db_error"


def test_flag_off_drops_beat_and_tells_browser_to_stop(client, engine, monkeypatch) -> None:
    monkeypatch.setattr(har.settings, "HUB_ATTENTION_TRACKER_ENABLED", False, raising=False)
    r = client.post("/api/hub-attention/heartbeat", json={"client": "desktop"})
    assert r.json() == {"ok": False, "enabled": False}
    assert engine.calls == []


def test_route_is_mounted_on_the_hub_router() -> None:
    src = (HUB_ROOT / "scripts" / "api_routes.py").read_text(encoding="utf-8")
    assert "from .hub_attention_routes import router as hub_attention_router" in src
    assert "router.include_router(hub_attention_router)" in src


def test_index_html_loads_the_tracker_and_static_file_exists() -> None:
    html = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    tag = '<script src="/static/js/hub-attention-tracker.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>'
    assert html.count(tag) == 1
    assert (HUB_ROOT / "static" / "js" / "hub-attention-tracker.js").is_file()
    # Only the top-level shell loads it; iframe sub-pages must not (double counting).
    for sub in ("reading.html", "gpu_pool.html", "temporal_self.html", "personality.html"):
        assert "hub-attention-tracker.js" not in (HUB_ROOT / "templates" / sub).read_text(encoding="utf-8")
    js = (HUB_ROOT / "static" / "js" / "hub-attention-tracker.js").read_text(encoding="utf-8")
    assert "/api/hub-attention/heartbeat" in js


def test_tracker_js_cases_pass_under_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH -- hub-attention-tracker.test.js was NOT executed")
    test_js = HUB_ROOT / "static" / "js" / "hub-attention-tracker.test.js"
    result = subprocess.run(
        [node, "--test", "--test-reporter=tap", str(test_js)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# fail 0" in result.stdout, result.stdout
    assert "# pass 9" in result.stdout, result.stdout
