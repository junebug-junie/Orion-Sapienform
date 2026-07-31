"""Cocreation Signals operator tab: pure-logic unit tests + wiring contract.

Same rationale as test_attention_organ_page.py: this tab is registered in
six separate places in app.js (element binding, missing-panel fallback,
visibility flag, toggle, button styling, hash routing + click handler) plus
a script tag plus a nav anchor. Missing any one of them produces a tab that
looks fine until a specific interaction (deep link, switching away and back,
a fresh reload) silently falls back to Hub -- a failure shape this file
turns into a failing test instead of a manual click path.
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
from scripts.cocreation_signals_routes import (  # noqa: E402
    ALLOWED_HISTORY_HOURS,
    DEFAULT_HISTORY_HOURS,
    KNOWN_DOMAINS,
    build_domain_summary,
    build_history_series,
    normalize_history_hours,
)

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
TAB_JS = (HUB_ROOT / "static" / "js" / "cocreation-signals.js").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# build_domain_summary
# --------------------------------------------------------------------------


def test_build_domain_summary_includes_every_known_domain_even_with_no_data() -> None:
    """A domain with zero events is a different, louder fact than a domain
    with a calm score -- must not be silently omitted from the list."""
    out = build_domain_summary([], [])
    assert {row["domain"] for row in out} == set(KNOWN_DOMAINS)
    for row in out:
        assert row["present"] is False
        assert row["latest_score"] is None
        assert row["window_event_count"] == 0


def test_build_domain_summary_shapes_present_row_with_real_values() -> None:
    now = datetime.now(timezone.utc)
    observed_at = now - timedelta(seconds=42.5)
    latest_rows = [
        {
            "domain": "git",
            "score": 0.314,
            "observed_at": observed_at,
            "event_json": {
                "domain": "git",
                "git": {"lines_added": 520, "commit_count": 3},
            },
        }
    ]
    window_counts = [{"domain": "git", "n": 5, "avg_score": 0.1, "max_score": 0.9}]

    out = build_domain_summary(latest_rows, window_counts, now=now)
    git_row = next(r for r in out if r["domain"] == "git")
    assert git_row["present"] is True
    assert git_row["latest_score"] == 0.314
    assert git_row["latest_age_sec"] == 42.5
    assert git_row["latest_payload"] == {"lines_added": 520, "commit_count": 3}
    assert git_row["window_event_count"] == 5
    assert git_row["window_avg_score"] == 0.1
    assert git_row["window_max_score"] == 0.9


def test_build_domain_summary_matches_pr_lifecycle_wire_field_not_pr() -> None:
    """CodebaseDeltaV1's real wire field is `pr_lifecycle`, not `pr` --
    KNOWN_DOMAINS and this lookup must key off the same name the producer
    actually publishes and save_codebase_delta_log() actually persists
    (orion/schemas/codebase_delta.py), not the shorter `pr` key
    CodebaseMassBaseline.to_json_dict() happens to use for the same concept
    in a different table."""
    now = datetime.now(timezone.utc)
    latest_rows = [
        {
            "domain": "pr_lifecycle",
            "score": 0.0,
            "observed_at": now,
            "event_json": {
                "domain": "pr_lifecycle",
                "pr_lifecycle": {"submitted_count": 2, "merged_count": 1},
            },
        }
    ]
    out = build_domain_summary(latest_rows, [], now=now)
    row = next(r for r in out if r["domain"] == "pr_lifecycle")
    assert row["present"] is True
    assert row["latest_payload"] == {"submitted_count": 2, "merged_count": 1}


def test_build_domain_summary_tolerates_string_encoded_json() -> None:
    """Some drivers return jsonb as a raw string, not a pre-parsed dict."""
    now = datetime.now(timezone.utc)
    latest_rows = [
        {
            "domain": "graph",
            "score": 0.0,
            "observed_at": now,
            "event_json": '{"domain": "graph", "graph": {"node_count_delta": -3}}',
        }
    ]
    out = build_domain_summary(latest_rows, [], now=now)
    graph_row = next(r for r in out if r["domain"] == "graph")
    assert graph_row["latest_payload"] == {"node_count_delta": -3}


def test_build_domain_summary_unknown_domain_in_rows_is_ignored_not_crashed() -> None:
    """A future fourth domain landing in Postgres before this module is
    updated must not 500 the whole snapshot -- it's just absent from the
    fixed KNOWN_DOMAINS list until this module catches up."""
    now = datetime.now(timezone.utc)
    out = build_domain_summary(
        [{"domain": "future_domain", "score": 1.0, "observed_at": now, "event_json": {}}],
        [],
        now=now,
    )
    assert {row["domain"] for row in out} == set(KNOWN_DOMAINS)


# --------------------------------------------------------------------------
# build_history_series
# --------------------------------------------------------------------------


def test_build_history_series_splits_by_domain_without_interpolation() -> None:
    now = datetime.now(timezone.utc)
    rows = [
        {"domain": "git", "score": 0.1, "observed_at": now},
        {"domain": "pr_lifecycle", "score": 0.0, "observed_at": now + timedelta(minutes=1)},
        {"domain": "git", "score": 0.2, "observed_at": now + timedelta(minutes=2)},
    ]
    series = build_history_series(rows)
    assert set(series.keys()) == set(KNOWN_DOMAINS)
    assert len(series["git"]) == 2
    assert len(series["pr_lifecycle"]) == 1
    assert len(series["graph"]) == 0
    assert series["git"][0]["score"] == 0.1
    assert series["git"][1]["score"] == 0.2


def test_build_history_series_handles_empty_input() -> None:
    series = build_history_series([])
    assert set(series.keys()) == set(KNOWN_DOMAINS)
    assert all(v == [] for v in series.values())


# --------------------------------------------------------------------------
# normalize_history_hours
# --------------------------------------------------------------------------


def test_normalize_history_hours_rejects_unlisted_windows() -> None:
    assert normalize_history_hours(24) == 24
    for allowed in ALLOWED_HISTORY_HOURS:
        assert normalize_history_hours(allowed) == allowed
    assert normalize_history_hours(999) == DEFAULT_HISTORY_HOURS
    assert normalize_history_hours(None) == DEFAULT_HISTORY_HOURS


# --------------------------------------------------------------------------
# Wiring contract
# --------------------------------------------------------------------------


def test_routes_are_registered_on_the_hub_api_router() -> None:
    route_paths = {route.path for route in api_routes.router.routes}
    assert "/api/cocreation-signals/snapshot" in route_paths
    assert "/api/cocreation-signals/history" in route_paths


def test_template_declares_nav_button_panel_and_script_tag() -> None:
    assert 'data-hash-target="#cocreation-signals"' in INDEX_HTML
    assert 'id="cocreationSignalsTabButton"' in INDEX_HTML
    assert 'id="cocreation-signals" data-panel="cocreation-signals"' in INDEX_HTML
    assert "/static/js/cocreation-signals.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML
    for mount_id in (
        "cocreationSignalsStatus",
        "cocreationSignalsBaseline",
        "cocreationSignalsDomains",
    ):
        assert f'id="{mount_id}"' in INDEX_HTML, mount_id


def test_app_js_registers_the_panel_in_every_place_setactivetab_needs() -> None:
    assert 'document.getElementById("cocreation-signals")' in APP_JS  # element binding
    assert 'tabKey === "cocreation-signals" && !cocreationSignalsPanel' in APP_JS  # fallback
    assert 'const isCocreationSignals = effectiveTab === "cocreation-signals";' in APP_JS  # flag
    assert 'cocreationSignalsPanel.classList.toggle("hidden", !isCocreationSignals);' in APP_JS  # toggle
    assert "styleTabButton(cocreationSignalsTabButton, isCocreationSignals);" in APP_JS  # styling
    assert 'h === "#cocreation-signals" && cocreationSignalsPanel' in APP_JS  # hash routing
    assert 'history.replaceState(null, "", "#cocreation-signals");' in APP_JS  # click handler


def test_app_js_drives_the_panels_poll_lifecycle_on_tab_switch() -> None:
    assert "window.OrionCocreationSignals.activate()" in APP_JS
    assert "window.OrionCocreationSignals.deactivate()" in APP_JS


def test_tab_js_is_standalone_and_reads_only_its_own_api() -> None:
    assert "window.OrionCocreationSignals" in TAB_JS
    assert "activate:" in TAB_JS and "deactivate:" in TAB_JS
    assert '"/api/cocreation-signals/snapshot"' in TAB_JS
    assert '"/api/cocreation-signals/history"' in TAB_JS
    assert "window.OrionHub" not in TAB_JS
    assert '"POST"' not in TAB_JS and "method: 'POST'" not in TAB_JS


def test_tab_js_renders_structure_not_a_raw_json_dump() -> None:
    assert "JSON.stringify" not in TAB_JS
    assert "createElementNS" in TAB_JS
    for renderer in ("renderBaseline", "renderDomains", "renderSeries"):
        assert f"function {renderer}(" in TAB_JS, renderer


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

            def first(self_inner):
                return rows[0] if rows else None

            def all(self_inner):
                return rows

        return _Result()


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return _FakeConn(self._rows)


def test_snapshot_reports_cold_start_when_no_data_exists(monkeypatch) -> None:
    monkeypatch.setattr(routes, "_engine", lambda: _FakeEngine([]))
    result = routes.snapshot()
    assert result["baseline"]["ok"] is False
    assert result["domains"]["ok"] is True
    assert {row["domain"] for row in result["domains"]["rows"]} == set(KNOWN_DOMAINS)
    assert all(not row["present"] for row in result["domains"]["rows"])


def test_snapshot_degrades_independently_on_engine_error(monkeypatch) -> None:
    def raise_error():
        raise RuntimeError("db down")

    monkeypatch.setattr(routes, "_engine", raise_error)
    result = routes.snapshot()
    assert result["baseline"]["ok"] is False
    assert "db down" in result["baseline"]["error"]
    assert result["domains"]["ok"] is False


def test_history_reports_truncated_when_row_cap_hit(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    rows = [{"domain": "git", "score": 0.1, "observed_at": now} for _ in range(routes.HISTORY_ROW_CAP)]
    monkeypatch.setattr(routes, "_engine", lambda: _FakeEngine(rows))
    result = routes.history(hours=24)
    assert result["truncated"] is True
    assert result["sample_count"] == routes.HISTORY_ROW_CAP
