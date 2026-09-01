"""Attention Organ operator tab: pure-logic unit tests + wiring contract.

The wiring half exists because this tab is the *fourth* panel added to
index.html that has to be registered in five separate places in app.js
(element binding, missing-panel fallback, visibility toggle, button styling,
hash routing) plus a script tag plus a nav anchor. Missing any one of them
produces a tab that looks fine until a specific interaction (deep link,
switching away and back, a fresh reload) silently falls back to Hub -- a
failure shape this file turns into a failing test instead of a manual click
path.
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
from scripts.attention_organ_routes import (  # noqa: E402
    ALLOWED_HISTORY_MINUTES,
    DEFAULT_HISTORY_MINUTES,
    build_domain_rows,
    normalize_history_minutes,
    parse_predicted_shift_domain,
    reconcile_confidence,
    summarize_history,
)

INDEX_HTML = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
ORGAN_JS = (HUB_ROOT / "static" / "js" / "attention-organ.js").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# parse_predicted_shift_domain
# --------------------------------------------------------------------------


def test_parse_predicted_shift_domain_reads_the_reducers_real_phrasing() -> None:
    assert (
        parse_predicted_shift_domain(
            "bus_synaptic prediction-error falling (trend=-0.3628 over recent window)"
        )
        == "bus_synaptic"
    )
    assert (
        parse_predicted_shift_domain(
            "execution prediction-error rising (trend=+0.4856 over recent window)"
        )
        == "execution"
    )


def test_parse_predicted_shift_domain_returns_none_rather_than_guessing() -> None:
    """A tally is only worth showing if a phrasing this parser does not
    understand lands in `null_domain_count` instead of silently contributing
    a garbage domain name -- an invented domain would be indistinguishable
    from a real one in the dominance chart."""
    assert parse_predicted_shift_domain(None) is None
    assert parse_predicted_shift_domain("") is None
    assert parse_predicted_shift_domain("nothing notable this tick") is None
    assert parse_predicted_shift_domain("prediction-error rising") is None
    # Multi-word prefix: not a domain identifier, so refuse rather than guess.
    assert parse_predicted_shift_domain("some domain prediction-error rising") is None
    assert parse_predicted_shift_domain(12345) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# summarize_history
# --------------------------------------------------------------------------


def _row(minutes_ago: int, **payload):
    base = {
        "heartbeat_verdict": "redundant",
        "heartbeat_mean_ratio": 0.81,
        "prediction_error_confidence": 0.58,
        "confidence": 0.9,
        "attention_reason": "bottom_up_salience",
        "predicted_shift": "execution prediction-error rising (trend=+0.1 over recent window)",
    }
    base.update(payload)
    return {
        "generated_at": datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        + timedelta(minutes=minutes_ago),
        "self_model_json": base,
    }


def test_summarize_history_tallies_verdict_and_domain_distributions() -> None:
    rows = [
        _row(0),
        _row(1, predicted_shift="biometrics prediction-error falling (trend=-0.2 over recent window)"),
        _row(2, heartbeat_verdict="mixed"),
    ]
    out = summarize_history(rows)

    assert out["sample_count"] == 3
    assert out["verdict_counts"] == {"redundant": 2, "mixed": 1}
    assert out["verdict_distinct_values"] == 2
    assert out["domain_counts"] == {"execution": 2, "biometrics": 1}
    assert out["series"][0]["predicted_shift_domain"] == "execution"


def test_summarize_history_keeps_absent_signal_distinct_from_a_band_never_firing() -> None:
    """"No heartbeat data on this row" and "the concentrated band never fired"
    are different claims about the organ, and collapsing them would make a
    broken wire look like a working-but-quiet one."""
    rows = [_row(0, heartbeat_verdict=None), _row(1, predicted_shift=None)]
    out = summarize_history(rows)

    assert out["null_verdict_count"] == 1
    assert out["null_domain_count"] == 1
    assert out["verdict_counts"] == {"redundant": 1}
    assert out["series"][0]["heartbeat_verdict"] is None


def test_summarize_history_survives_json_strings_and_junk_rows() -> None:
    rows = [
        {"generated_at": datetime.now(timezone.utc), "self_model_json": '{"heartbeat_verdict": "mixed"}'},
        {"generated_at": datetime.now(timezone.utc), "self_model_json": "not json at all"},
        {"generated_at": datetime.now(timezone.utc), "self_model_json": None},
    ]
    out = summarize_history(rows)
    assert out["sample_count"] == 1
    assert out["verdict_counts"] == {"mixed": 1}


# --------------------------------------------------------------------------
# build_domain_rows
# --------------------------------------------------------------------------


NOW = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)


def test_build_domain_rows_marks_active_inference_membership() -> None:
    rows = build_domain_rows(
        [
            {
                "node_id": "node:substrate.execution",
                "prediction_error": 0.4,
                "observed_at": "2026-07-30T19:59:00+00:00",
                "activation": 0.9,
            },
            {
                "node_id": "node:substrate.transport",
                "prediction_error": 0.5,
                "observed_at": "2026-07-24T21:00:00+00:00",
                "activation": 0.1,
            },
        ],
        now=NOW,
    )
    by_domain = {r["domain"]: r for r in rows}

    assert by_domain["execution"]["active"] is True
    assert by_domain["execution"]["prediction_error"] == 0.4
    assert by_domain["execution"]["observed_age_sec"] == 60.0
    # transport is a confirmed-dead instrument excluded from the aggregate --
    # still returned so it stays visible rather than hiding.
    assert by_domain["transport"]["active"] is False
    assert by_domain["transport"]["present"] is True


def test_build_domain_rows_flags_ceiling_and_floor_pins() -> None:
    """Both directions have real incident history on these exact nodes
    (bus_synaptic frozen at 1.0, PR #1449; route decayed toward 0.0 by a
    generic staleness loop) -- neither may render as an ordinary value."""
    rows = build_domain_rows(
        [
            {"node_id": "node:substrate.bus_synaptic", "prediction_error": 1.0},
            {"node_id": "node:substrate.route", "prediction_error": 0.0},
            {"node_id": "node:substrate.biometrics", "prediction_error": 0.1},
        ],
        now=NOW,
    )
    by_domain = {r["domain"]: r for r in rows}

    assert by_domain["bus_synaptic"]["at_ceiling"] is True
    assert by_domain["bus_synaptic"]["at_floor"] is False
    assert by_domain["route"]["at_floor"] is True
    assert by_domain["biometrics"]["at_ceiling"] is False
    assert by_domain["biometrics"]["at_floor"] is False


def test_build_domain_rows_distinguishes_missing_node_from_zero_reading() -> None:
    rows = build_domain_rows([{"node_id": "node:substrate.chat", "prediction_error": 0.0}], now=NOW)
    by_domain = {r["domain"]: r for r in rows}

    assert by_domain["chat"]["present"] is True
    assert by_domain["chat"]["prediction_error"] == 0.0
    # A known domain whose node does not exist at all is a louder failure than
    # one reading zero, and must not be rendered as "0.0".
    assert by_domain["execution"]["present"] is False
    assert by_domain["execution"]["prediction_error"] is None
    assert by_domain["execution"]["at_floor"] is False


def test_build_domain_rows_keeps_unknown_domains_instead_of_dropping_them() -> None:
    rows = build_domain_rows(
        [{"node_id": "node:substrate.brand_new_domain", "prediction_error": 0.3}], now=NOW
    )
    assert any(r["domain"] == "brand_new_domain" for r in rows)


def test_build_domain_rows_tolerates_unparsable_values_and_timestamps() -> None:
    rows = build_domain_rows(
        [
            {
                "node_id": "node:substrate.execution",
                "prediction_error": "not-a-number",
                "observed_at": "definitely not a timestamp",
                "activation": "nope",
            }
        ],
        now=NOW,
    )
    row = {r["domain"]: r for r in rows}["execution"]
    assert row["prediction_error"] is None
    assert row["observed_age_sec"] is None
    assert row["activation"] is None


# --------------------------------------------------------------------------
# reconcile_confidence
# --------------------------------------------------------------------------


def test_reconcile_confidence_matches_the_reducers_own_formula() -> None:
    """`1 - mean(prediction_error over ACTIVE_INFERENCE_DOMAINS)` is exactly
    what `_unconditional_prediction_error_confidence()` computes; verified
    live 2026-07-30 that the five live domain values reproduce the persisted
    0.5768 exactly. If this ever stops reconciling, the panel is showing two
    numbers that no longer describe the same thing."""
    domain_rows = build_domain_rows(
        [
            {"node_id": "node:substrate.execution", "prediction_error": 1.0},
            {"node_id": "node:substrate.bus_synaptic", "prediction_error": 1.0},
            {"node_id": "node:substrate.biometrics", "prediction_error": 0.102157},
            {"node_id": "node:substrate.chat", "prediction_error": 0.013468},
            {"node_id": "node:substrate.route", "prediction_error": 0.00025},
            # Excluded domains must not be averaged in, or the check would
            # disagree with the reducer for a reason that isn't a real defect.
            {"node_id": "node:substrate.transport", "prediction_error": 0.556078},
            {"node_id": "node:substrate.harness_closure", "prediction_error": 0.65},
        ],
        now=NOW,
    )
    out = reconcile_confidence(domain_rows, 0.5768, row_age_sec=1.0)

    assert out["domains_used"] == 5
    assert out["recomputed"] == 0.5768
    assert out["delta"] == 0.0
    assert out["verdict"] == "exact"


def test_reconcile_confidence_flags_structural_divergence() -> None:
    domain_rows = build_domain_rows(
        [
            {"node_id": "node:substrate.execution", "prediction_error": 1.0},
            {"node_id": "node:substrate.bus_synaptic", "prediction_error": 1.0},
            {"node_id": "node:substrate.biometrics", "prediction_error": 1.0},
            {"node_id": "node:substrate.chat", "prediction_error": 1.0},
            {"node_id": "node:substrate.route", "prediction_error": 1.0},
        ],
        now=NOW,
    )
    out = reconcile_confidence(domain_rows, 0.96, row_age_sec=1.0)
    assert out["verdict"] == "divergent"
    assert out["recomputed"] == 0.0
    assert "more than one domain" in out["basis"]


def test_reconcile_confidence_reports_unknown_rather_than_a_fake_zero() -> None:
    assert reconcile_confidence([], 0.5)["verdict"] == "unknown"
    rows = build_domain_rows([{"node_id": "node:substrate.execution", "prediction_error": 0.2}], now=NOW)
    assert reconcile_confidence(rows, None)["verdict"] == "unknown"


def test_reconcile_confidence_does_not_cry_divergence_on_ordinary_sampling_skew() -> None:
    """The first version of this check used a flat 0.05 tolerance, and live
    data immediately showed it reporting "divergent" purely as a function of
    when the operator happened to poll: the row is written on a ~30s tick
    while the graph read happens now, and two of the five domains genuinely
    swing across their full range between ticks. A check that fires on its own
    sampling schedule is noise."""
    rows = build_domain_rows(
        [
            # One domain has fully flipped since the row was written; the other
            # four are unchanged. Aggregate moves by 1/5 = 0.2 -- entirely
            # explicable by timing, not by a source mismatch.
            {"node_id": "node:substrate.execution", "prediction_error": 1.0},
            {"node_id": "node:substrate.bus_synaptic", "prediction_error": 0.0},
            {"node_id": "node:substrate.biometrics", "prediction_error": 0.0},
            {"node_id": "node:substrate.chat", "prediction_error": 0.0},
            {"node_id": "node:substrate.route", "prediction_error": 0.0},
        ],
        now=NOW,
    )
    out = reconcile_confidence(rows, 1.0, row_age_sec=14.0)
    assert out["delta"] == -0.2
    assert out["verdict"] == "within_drift"


def test_reconcile_confidence_withholds_a_verdict_on_a_stale_row() -> None:
    """Past a point the two readings are too far apart in time for the
    comparison to carry information, and asserting either way would be
    manufacturing a verdict."""
    rows = build_domain_rows(
        [{"node_id": "node:substrate.execution", "prediction_error": 1.0}], now=NOW
    )
    out = reconcile_confidence(rows, 0.1, row_age_sec=3600.0)
    assert out["verdict"] == "unknown"
    assert "too far from this graph read" in out["basis"]


def test_normalize_history_minutes_rejects_unlisted_windows() -> None:
    for allowed in ALLOWED_HISTORY_MINUTES:
        assert normalize_history_minutes(allowed) == allowed
    assert normalize_history_minutes(None) == DEFAULT_HISTORY_MINUTES
    assert normalize_history_minutes(0) == DEFAULT_HISTORY_MINUTES
    assert normalize_history_minutes(999999) == DEFAULT_HISTORY_MINUTES


# --------------------------------------------------------------------------
# Wiring contract
# --------------------------------------------------------------------------


def test_routes_are_registered_on_the_hub_api_router() -> None:
    route_paths = {route.path for route in api_routes.router.routes}
    assert "/api/attention-organ/snapshot" in route_paths
    assert "/api/attention-organ/history" in route_paths


def test_template_declares_nav_button_panel_and_script_tag() -> None:
    assert 'data-hash-target="#attention-organ"' in INDEX_HTML
    assert 'id="attentionOrganTabButton"' in INDEX_HTML
    assert 'id="attention-organ" data-panel="attention-organ"' in INDEX_HTML
    assert "/static/js/attention-organ.js?v={{HUB_UI_ASSET_VERSION}}" in INDEX_HTML
    for mount_id in (
        "attentionOrganStatus",
        "attentionOrganEnsemble",
        "attentionOrganDiscrimination",
        "attentionOrganDomains",
        "attentionOrganDominance",
        "attentionOrganSelfModel",
        "attentionOrganLink",
        "attentionOrganIntake",
        "attentionOrganDissipation",
    ):
        assert f'id="{mount_id}"' in INDEX_HTML, mount_id


def test_app_js_registers_the_panel_in_every_place_setactivetab_needs() -> None:
    """All five registration points, not just the visible one -- see this
    module's docstring for why each omission fails silently and differently."""
    assert 'document.getElementById("attention-organ")' in APP_JS  # element binding
    assert 'tabKey === "attention-organ" && !attentionOrganPanel' in APP_JS  # fallback
    assert 'const isAttentionOrgan = effectiveTab === "attention-organ";' in APP_JS  # flag
    assert 'attentionOrganPanel.classList.toggle("hidden", !isAttentionOrgan);' in APP_JS  # toggle
    assert "styleTabButton(attentionOrganTabButton, isAttentionOrgan);" in APP_JS  # styling
    assert 'h === "#attention-organ" && attentionOrganPanel' in APP_JS  # hash routing
    assert 'history.replaceState(null, "", "#attention-organ");' in APP_JS  # click handler


def test_app_js_drives_the_panels_poll_lifecycle_on_tab_switch() -> None:
    """Polling must stop when the tab is hidden. Concept Atlas' own comment
    records the opposite bug already happening once in this codebase (Organ
    Signals defines deactivate() but app.js never calls it) -- this asserts
    both halves are wired, not just activate()."""
    assert "window.OrionAttentionOrgan.activate()" in APP_JS
    assert "window.OrionAttentionOrgan.deactivate()" in APP_JS


def test_attention_organ_js_is_standalone_and_reads_only_its_own_api() -> None:
    assert "window.OrionAttentionOrgan" in ORGAN_JS
    assert "activate:" in ORGAN_JS and "deactivate:" in ORGAN_JS
    assert '"/api/attention-organ/snapshot"' in ORGAN_JS
    assert '"/api/attention-organ/history"' in ORGAN_JS
    # Must not depend on app.js's globals -- same standalone contract the
    # Causal Geometry bundle test asserts.
    assert "window.OrionHub" not in ORGAN_JS
    # Read-only surface: no writes to any Orion endpoint from this tab.
    assert '"POST"' not in ORGAN_JS and "method: 'POST'" not in ORGAN_JS


def test_attention_organ_js_renders_structure_not_a_raw_json_dump() -> None:
    """The point of this tab is a legible operator surface. A JSON.stringify
    of the snapshot would technically 'visualize' it and would be worthless."""
    assert "JSON.stringify" not in ORGAN_JS
    assert "createElementNS" in ORGAN_JS  # real SVG charts
    for renderer in (
        "renderEnsemble",
        "renderDiscrimination",
        "renderDomains",
        "renderDominance",
        "renderSelfModel",
        "renderLink",
        "renderIntake",
        "renderDissipation",
    ):
        assert f"function {renderer}(" in ORGAN_JS, renderer


def test_attention_organ_js_bounds_its_rolling_client_side_trace() -> None:
    """A tab left open all day must not grow an unbounded array -- the same
    uncapped-accumulation shape already fixed twice in this repo's reducers."""
    assert "liveTraceMax" in ORGAN_JS
    assert "state.liveTrace.splice" in ORGAN_JS


def test_attention_organ_js_does_not_paint_verdict_from_mean_ratio_bands() -> None:
    """Live 2026-09-01: classify_ensemble_verdict returned mixed while the
    Hub gauge colored mean_ratio>=0.6 as redundant. Mean at cut-5 is
    capacity-saturated; std_ratio and bulk_penetration_depth discriminate.
    Painting concentrated/mixed/redundant on the mean bar is a lie."""
    gauge = ORGAN_JS[
        ORGAN_JS.index("function renderBandGauge") : ORGAN_JS.index(
            "function ensembleVerdictWhy"
        )
    ]
    assert 'el("span", "", "concentrated")' not in gauge
    assert 'el("span", "", "mixed")' not in gauge
    assert 'el("span", "", "redundant")' not in gauge
    assert "band(low, high" not in gauge
    assert "band(high, 1" not in gauge
    assert "silence floor" in gauge.lower()


def test_attention_organ_js_draws_std_and_bulk_classifier_thresholds() -> None:
    """verdict_thresholds() already exports std_mixed/bulk_low/etc on
    /health.config. The organ tab must draw those, not only low_ratio/high_ratio."""
    gauge = ORGAN_JS[
        ORGAN_JS.index("function renderBandGauge") : ORGAN_JS.index(
            "function ensembleVerdictWhy"
        )
    ]
    ensemble = ORGAN_JS[
        ORGAN_JS.index("function renderEnsemble") : ORGAN_JS.index(
            "function renderDiscrimination"
        )
    ]
    assert gauge.count("drawLinearGauge(host,") == 3
    assert "h1.bulk_penetration_depth" in gauge
    assert "h1.bulk_penetration_depth" in ensemble
    assert "thresholds.std_mixed" in gauge
    assert "thresholds.std_redundant_max" in gauge
    assert "thresholds.bulk_low" in gauge
    assert "thresholds.bulk_redundant_min" in gauge
    assert "domainMin: bulkMin" in gauge
    assert "renderBandGauge(host, h1, config)" in ensemble


def test_attention_organ_js_fallback_why_names_failing_conjuncts() -> None:
    """Fallback mixed must not always blame bulk — live std can sit between
    std_redundant_max and std_mixed while bulk already clears bulk_redundant_min."""
    why = ORGAN_JS[
        ORGAN_JS.index("function ensembleVerdictWhy") : ORGAN_JS.index(
            "function renderTrajectories"
        )
    ]
    assert "settled-agreement conjuncts missed" in why
    assert "std " in why and "redundant-max" in why
    assert "vs redundant min" not in why


def test_attention_organ_js_config_row_lists_ensemble_verdict_edges() -> None:
    """The dissipation panel's 'verdict bands' row still advertised the
    retired mean-only 0.2/0.6 split after #2015."""
    dissipation = ORGAN_JS[
        ORGAN_JS.index("function renderDissipation") : ORGAN_JS.index(
            "function setStatus"
        )
    ]
    assert '["verdict bands", "≤" + num(config.low_ratio, 2) + " / ≥" + num(config.high_ratio, 2)]' not in dissipation
    assert "config.std_mixed" in dissipation
    assert "config.bulk_low" in dissipation
    assert "config.bulk_redundant_min" in dissipation


def test_attention_organ_routes_doc_does_not_call_mean_ratio_the_verdict_axis() -> None:
    routes_src = (
        Path(__file__).resolve().parents[1] / "scripts" / "attention_organ_routes.py"
    ).read_text(encoding="utf-8")
    assert "std_mixed" in routes_src
    assert "bulk_low" in routes_src


# --------------------------------------------------------------------------
# Endpoint behavior (real handler execution, backends stubbed)
#
# Review finding T2: the wiring assertions above are string matches and would
# pass on syntactically broken code. These execute the handlers so the
# independent-degradation contract in snapshot()'s docstring, the /history SQL,
# and the `truncated` flag are actually covered.
# --------------------------------------------------------------------------

import pytest  # noqa: E402

from scripts import attention_organ_routes as routes  # noqa: E402


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.statement = str(statement)
        self.params = params
        rows = self._rows
        if params and "row_cap" in params:
            rows = rows[: params["row_cap"]]

        class _Result:
            def mappings(self_inner):
                return self_inner

            def first(self_inner):
                return rows[0] if rows else None

            def all(self_inner):
                return rows

        return _Result()


@pytest.fixture
def stub_backends(monkeypatch):
    """Each of the three sources is independently stubbable so a single
    source's failure can be asserted in isolation."""

    calls = {}

    def set_heartbeat(h1_payload, health_payload):
        def fake_get(path):
            calls.setdefault("heartbeat_paths", []).append(path)
            return h1_payload if path == "/h1" else health_payload

        monkeypatch.setattr(routes, "_heartbeat_get", fake_get)

    def set_self_model(rows):
        monkeypatch.setattr(routes, "_engine", lambda: _FakeEngine(rows))

    class _FakeEngine:
        def __init__(self, rows):
            self._rows = rows

        def connect(self):
            return _FakeConn(self._rows)

    def set_domains(rows_or_exc):
        def fake_load():
            if isinstance(rows_or_exc, Exception):
                raise rows_or_exc
            return rows_or_exc

        monkeypatch.setattr(routes, "_load_domain_rows", fake_load)

    return {
        "heartbeat": set_heartbeat,
        "self_model": set_self_model,
        "domains": set_domains,
        "calls": calls,
    }


_GOOD_H1 = {
    "ok": True,
    "payload": {"ok": True, "h1": {"mean_ratio": 0.81, "std_ratio": 0.03, "tick_count": 99}},
}
_GOOD_HEALTH = {"ok": True, "payload": {"config": {"high_ratio": 0.6, "low_ratio": 0.2}}}
_GOOD_ROW = {
    "generated_at": datetime.now(timezone.utc),
    "self_model_json": {
        "prediction_error_confidence": 0.5768,
        "heartbeat_verdict": "redundant",
        "heartbeat_mean_ratio": 0.81,
        "heartbeat_basis": "orion-heartbeat /h1 ensemble mean_ratio (tick_count=90)",
    },
}
_GOOD_DOMAINS = [
    {"node_id": "node:substrate.execution", "prediction_error": 1.0},
    {"node_id": "node:substrate.bus_synaptic", "prediction_error": 1.0},
    {"node_id": "node:substrate.biometrics", "prediction_error": 0.102157},
    {"node_id": "node:substrate.chat", "prediction_error": 0.013468},
    {"node_id": "node:substrate.route", "prediction_error": 0.00025},
]


def test_snapshot_happy_path_reports_all_three_sources_healthy(stub_backends) -> None:
    stub_backends["heartbeat"](_GOOD_H1, _GOOD_HEALTH)
    stub_backends["self_model"]([_GOOD_ROW])
    stub_backends["domains"](_GOOD_DOMAINS)

    out = routes.snapshot()

    assert out["heartbeat"]["ok"] is True
    assert out["self_model"]["ok"] is True
    assert out["domains"]["ok"] is True
    assert out["domains"]["reconciliation"]["verdict"] in ("exact", "within_drift")
    assert out["link"]["self_model_has_heartbeat"] is True
    assert out["link"]["live_tick_count"] == 99


def test_snapshot_degrades_each_source_independently(stub_backends) -> None:
    """snapshot()'s core contract: one backend being down must not blank the
    other two, which are served by entirely different systems and are still
    perfectly valid."""
    stub_backends["heartbeat"]({"ok": False, "error": "ConnectionError: refused"}, {"ok": False, "error": "boom"})
    stub_backends["self_model"]([_GOOD_ROW])
    stub_backends["domains"](_GOOD_DOMAINS)

    out = routes.snapshot()

    assert out["heartbeat"]["ok"] is False
    assert "ConnectionError" in out["heartbeat"]["h1_error"]
    # The other two survive intact.
    assert out["self_model"]["ok"] is True
    assert out["domains"]["ok"] is True
    assert out["domains"]["reconciliation"]["verdict"] in ("exact", "within_drift")
    # And nothing is invented for the missing organ.
    assert out["link"]["live_mean_ratio"] is None
    assert out["link"]["live_tick_count"] is None


def test_snapshot_reports_domain_query_failure_without_losing_the_rest(stub_backends) -> None:
    stub_backends["heartbeat"](_GOOD_H1, _GOOD_HEALTH)
    stub_backends["self_model"]([_GOOD_ROW])
    stub_backends["domains"](RuntimeError("falkordb unreachable"))

    out = routes.snapshot()

    assert out["domains"]["ok"] is False
    assert "falkordb unreachable" in out["domains"]["error"]
    assert out["domains"]["rows"] == []
    assert out["domains"]["reconciliation"]["verdict"] == "unknown"
    assert out["heartbeat"]["ok"] is True
    assert out["self_model"]["ok"] is True


def test_snapshot_reports_missing_self_model_row_honestly(stub_backends) -> None:
    stub_backends["heartbeat"](_GOOD_H1, _GOOD_HEALTH)
    stub_backends["self_model"]([])
    stub_backends["domains"](_GOOD_DOMAINS)

    out = routes.snapshot()

    assert out["self_model"]["ok"] is False
    assert out["self_model"]["model"] is None
    assert out["self_model"]["generated_at"] is None
    assert out["link"]["self_model_has_heartbeat"] is False
    assert out["link"]["self_model_age_sec"] is None


def test_snapshot_is_not_ok_when_heartbeat_is_reachable_but_has_no_h1_yet(
    stub_backends,
) -> None:
    """Review finding S6: /h1 answers HTTP 200 with {"ok": false} before the
    first ensemble tick. Transport success is not a reading, and the status
    line must not call that healthy while the panel shows a red 'no H1' block."""
    stub_backends["heartbeat"](
        {"ok": True, "payload": {"ok": False, "reason": "no_h1_computed_yet"}}, _GOOD_HEALTH
    )
    stub_backends["self_model"]([_GOOD_ROW])
    stub_backends["domains"](_GOOD_DOMAINS)

    out = routes.snapshot()

    assert out["heartbeat"]["ok"] is False
    assert out["heartbeat"]["h1"] is None
    assert out["heartbeat"]["h1_reason"] == "no_h1_computed_yet"
    # Not an error — the organ is up, it just has not produced a reading.
    assert out["heartbeat"]["h1_error"] is None


def test_history_executes_its_sql_and_reports_truncation(stub_backends) -> None:
    rows = [
        {
            "generated_at": datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=30 * i),
            "self_model_json": {
                "heartbeat_verdict": "redundant",
                "heartbeat_mean_ratio": 0.8,
                "predicted_shift": "execution prediction-error rising (trend=+0.1 over recent window)",
            },
        }
        for i in range(routes.HISTORY_ROW_CAP + 10)
    ]
    stub_backends["self_model"](rows)

    out = routes.history(minutes=360)

    assert out["window_minutes"] == 360
    assert out["sample_count"] == routes.HISTORY_ROW_CAP
    assert out["truncated"] is True
    assert out["verdict_counts"] == {"redundant": routes.HISTORY_ROW_CAP}


def test_history_normalizes_an_unlisted_window_before_querying(stub_backends) -> None:
    stub_backends["self_model"]([_GOOD_ROW])
    out = routes.history(minutes=7)
    assert out["window_minutes"] == routes.DEFAULT_HISTORY_MINUTES
    assert out["truncated"] is False


def test_summarize_history_counts_unreadable_rows_rather_than_dropping_them() -> None:
    """Review finding T3: a window where every row failed to parse must not
    look identical to a window with no activity."""
    out = summarize_history(
        [
            {"generated_at": datetime.now(timezone.utc), "self_model_json": "not json"},
            {"generated_at": datetime.now(timezone.utc), "self_model_json": None},
            {"generated_at": datetime.now(timezone.utc), "self_model_json": {"heartbeat_verdict": "mixed"}},
        ]
    )
    assert out["malformed_row_count"] == 2
    assert out["sample_count"] == 1


def test_attention_organ_js_guards_polling_on_real_visibility() -> None:
    """Review finding M1: app.js's setActiveTab is not the only thing that can
    hide this panel — self_observability.js hides every section[data-panel]
    directly and preventDefault()s its click, so deactivate() never fires on
    that path. The timer must check the DOM, not trust the router."""
    assert 'els.panel.classList.contains("hidden")' in ORGAN_JS
    guard_region = ORGAN_JS[ORGAN_JS.index("function startTimer") : ORGAN_JS.index("function activate")]
    assert "contains(\"hidden\")" in guard_region
    assert "deactivate();" in guard_region


def test_attention_organ_js_uses_strict_number_isfinite_for_nullable_readings() -> None:
    """Review finding M2: global isFinite(null) is true, so an offline organ
    rendered a confident negative tick lag — a number no producer produced."""
    assert "Number.isFinite(link.live_tick_count)" in ORGAN_JS
    assert "isFinite(link.live_tick_count)" not in ORGAN_JS.replace(
        "Number.isFinite(link.live_tick_count)", ""
    )


def test_attention_organ_routes_are_sync_handlers() -> None:
    """Review finding M3: these do blocking I/O (requests, SQLAlchemy, redis)
    and the tab auto-polls as fast as every 2s. Declared async, that blocks
    Hub's event loop — chat, websockets, every route — on every poll."""
    import inspect

    assert not inspect.iscoroutinefunction(routes.snapshot)
    assert not inspect.iscoroutinefunction(routes.history)
