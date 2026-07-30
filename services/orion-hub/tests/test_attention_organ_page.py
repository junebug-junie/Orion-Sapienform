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
    out = reconcile_confidence(domain_rows, 0.5768)

    assert out["domains_used"] == 5
    assert out["recomputed"] == 0.5768
    assert out["delta"] == 0.0
    assert out["reconciles"] is True


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
    out = reconcile_confidence(domain_rows, 0.96)
    assert out["reconciles"] is False
    assert out["recomputed"] == 0.0


def test_reconcile_confidence_reports_unknown_rather_than_a_fake_zero() -> None:
    assert reconcile_confidence([], 0.5)["reconciles"] is None
    rows = build_domain_rows([{"node_id": "node:substrate.execution", "prediction_error": 0.2}], now=NOW)
    assert reconcile_confidence(rows, None)["reconciles"] is None


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
