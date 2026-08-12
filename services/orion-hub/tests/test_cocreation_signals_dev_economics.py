"""dev-economics ledger section of the Cocreation Signals tab -- first real
consumer of orion:substrate:dev_economics_ledger (docs/superpowers/pr-
reports/2026-08-12-dev-economics-hub-observability.md).

Reuses the existing cocreation-signals panel (no new tab, no new six-place
app.js wiring) -- just a new mount id, a new render function, and a new
`/api/cocreation-signals/dev-economics` endpoint.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
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

from scripts import api_routes  # noqa: E402
from scripts import cocreation_signals_routes as routes  # noqa: E402
from scripts.cocreation_signals_routes import build_dev_economics_summary  # noqa: E402

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
TAB_JS = (HUB_ROOT / "static" / "js" / "cocreation-signals.js").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# build_dev_economics_summary (pure)
# --------------------------------------------------------------------------


def test_empty_rows_reports_a_real_absent_fact_not_an_error() -> None:
    summary = build_dev_economics_summary([])
    assert summary["present"] is False
    assert summary["recent_ticks"] == []
    assert summary["window_total_cost_usd"] is None
    assert summary["window_tick_count"] == 0


def test_present_rows_shape_latest_and_recent_ticks() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "observed_at": now,
            "session_count": 4,
            "total_tokens": 5_119_791,
            "total_estimated_cost_usd": 2.8432747,
            "unpriced_session_count": 0,
            "model_mix_json": '{"claude-sonnet-5": 4}',
        },
        {
            "observed_at": now - timedelta(minutes=15),
            "session_count": 2,
            "total_tokens": 100_000,
            "total_estimated_cost_usd": None,
            "unpriced_session_count": 2,
            "model_mix_json": '{"xai/grok-4.5": 2}',
        },
    ]
    summary = build_dev_economics_summary(rows, now=now)
    assert summary["present"] is True
    assert summary["latest"]["session_count"] == 4
    assert summary["latest"]["model_mix"] == {"claude-sonnet-5": 4}
    assert len(summary["recent_ticks"]) == 2
    # Only the priced tick contributes to the real partial total -- the
    # unpriced tick's real activity is still counted in window_total_tokens
    # but must not silently manufacture a $0.00 for its own missing cost.
    assert summary["window_total_cost_usd"] == 2.8433
    assert summary["window_total_tokens"] == 5_219_791


def test_all_unpriced_reports_none_not_fabricated_zero() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "observed_at": now,
            "session_count": 1,
            "total_tokens": 500,
            "total_estimated_cost_usd": None,
            "unpriced_session_count": 1,
            "model_mix_json": "{}",
        }
    ]
    summary = build_dev_economics_summary(rows, now=now)
    assert summary["window_total_cost_usd"] is None


def test_model_mix_json_tolerates_dict_and_string_forms() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {
            "observed_at": now,
            "session_count": 1,
            "total_tokens": 10,
            "total_estimated_cost_usd": 0.01,
            "unpriced_session_count": 0,
            "model_mix_json": {"already": "a dict"},
        }
    ]
    summary = build_dev_economics_summary(rows, now=now)
    assert summary["latest"]["model_mix"] == {"already": "a dict"}


# --------------------------------------------------------------------------
# Wiring contract
# --------------------------------------------------------------------------


def test_dev_economics_route_is_registered() -> None:
    route_paths = {route.path for route in api_routes.router.routes}
    assert "/api/cocreation-signals/dev-economics" in route_paths


def test_template_declares_the_dev_economics_mount() -> None:
    assert 'id="cocreationSignalsDevEconomics"' in INDEX_HTML


def test_tab_js_reads_the_dev_economics_endpoint_and_renders_structure() -> None:
    assert '"/api/cocreation-signals/dev-economics"' in TAB_JS
    assert "function renderDevEconomics(" in TAB_JS
    assert "els.devEconomics" in TAB_JS
    # Same "no raw JSON dump" discipline as the rest of this tab.
    assert "JSON.stringify" not in TAB_JS


# --------------------------------------------------------------------------
# Endpoint behavior (fake engine, no real Postgres)
# --------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *_args, **_kwargs):
        rows = self._rows

        class _Result:
            def mappings(self_inner):
                return self_inner

            def all(self_inner):
                return rows

        return _Result()


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return _FakeConn(self._rows)


def test_dev_economics_endpoint_reports_absent_when_no_rows(monkeypatch) -> None:
    monkeypatch.setattr(routes, "_engine", lambda: _FakeEngine([]))
    result = routes.dev_economics()
    assert result["ok"] is True
    assert result["present"] is False


def test_dev_economics_endpoint_degrades_on_engine_error(monkeypatch) -> None:
    def raise_error():
        raise RuntimeError("db down")

    monkeypatch.setattr(routes, "_engine", raise_error)
    result = routes.dev_economics()
    assert result["ok"] is False
    assert "db down" in result["error"]
    assert result["present"] is False
