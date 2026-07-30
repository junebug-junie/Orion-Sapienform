from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE_ROOT)]

from orion.metacog.trend_reducer import MetacogTrendStateV1  # noqa: E402

from app.repair_pressure_trend_gate import (  # noqa: E402
    evaluate_repair_pressure_trend,
    state_from_dict,
    state_to_dict,
)
from app.service import EquilibriumService  # noqa: E402
from app.settings import settings  # noqa: E402

_DEFAULT_KWARGS = dict(
    zen_state="not_zen",
    pressure=0.2,
    recall_enabled=None,
    confidence_floor=0.3,
    min_samples=20,
    elevated_zscore=1.0,
    sustained_hits=3,
)


def test_state_roundtrips_through_dict() -> None:
    state = MetacogTrendStateV1(ewma=0.42, variance=0.01, count=7, consecutive_elevated=2)
    restored = state_from_dict(state_to_dict(state))
    assert restored == state


def test_state_from_dict_falls_back_to_cold_start_on_malformed_input() -> None:
    assert state_from_dict(None) == MetacogTrendStateV1()
    assert state_from_dict({}) == MetacogTrendStateV1()
    assert state_from_dict({"ewma": "not_a_number"}) == MetacogTrendStateV1()
    assert state_from_dict("not a dict") == MetacogTrendStateV1()  # type: ignore[arg-type]


def test_malformed_appraisal_leaves_state_unchanged_and_fires_nothing() -> None:
    state = MetacogTrendStateV1(ewma=0.5, variance=0.02, count=25, consecutive_elevated=1)
    new_state, trigger = evaluate_repair_pressure_trend(state, None, **_DEFAULT_KWARGS)
    assert new_state == state
    assert trigger is None

    new_state2, trigger2 = evaluate_repair_pressure_trend(
        state, {"level": "not_a_number", "confidence": 0.5}, **_DEFAULT_KWARGS
    )
    assert new_state2 == state
    assert trigger2 is None


def test_below_confidence_floor_does_not_fold_the_reading() -> None:
    state = MetacogTrendStateV1(ewma=0.5, variance=0.02, count=25, consecutive_elevated=1)
    new_state, trigger = evaluate_repair_pressure_trend(
        state, {"level": 0.9, "confidence": 0.1}, **_DEFAULT_KWARGS
    )
    assert new_state == state  # unchanged -- reading below floor never reaches apply_reading
    assert trigger is None


def test_cold_start_never_fires_even_with_high_level() -> None:
    state = MetacogTrendStateV1()
    for i in range(10):
        state, trigger = evaluate_repair_pressure_trend(
            state, {"level": 0.9, "confidence": 0.8}, **_DEFAULT_KWARGS
        )
        assert trigger is None  # below min_samples=20 the whole way


def test_sustained_elevated_run_fires_a_trigger_with_real_evidence_fields() -> None:
    state = MetacogTrendStateV1()
    # Warm up a flat, calm baseline past min_samples.
    for _ in range(25):
        state, trigger = evaluate_repair_pressure_trend(
            state, {"level": 0.15, "confidence": 0.65}, **_DEFAULT_KWARGS
        )
        assert trigger is None

    # Now a real, sustained elevated run.
    last_trigger = None
    for _ in range(5):
        state, trigger = evaluate_repair_pressure_trend(
            state, {"level": 0.9, "confidence": 0.85}, **_DEFAULT_KWARGS
        )
        if trigger is not None:
            last_trigger = trigger

    assert last_trigger is not None
    assert last_trigger.trigger_kind == "repair_pressure_trend"
    assert last_trigger.zen_state == "not_zen"
    assert last_trigger.pressure == 0.2
    assert last_trigger.upstream["latest_level"] == 0.9
    assert last_trigger.upstream["latest_confidence"] == 0.85
    assert last_trigger.upstream["consecutive_elevated"] >= 3


def test_confidence_gate_reflects_real_text_fallback_evidence_from_the_repair_pressure_v2_fix() -> None:
    # The 2026-07-30 reduce_repair_level() fix means the common "confidently
    # calm" text-fallback appraisal reads confidence=0.65 (not 0.0) -- this
    # gate's default confidence_floor=0.3 must accept it as real evidence to
    # fold, not silently discard it the way the old bug's confidence=0.0
    # would have (if this gate had gated on the pre-fix semantics).
    state = MetacogTrendStateV1()
    new_state, trigger = evaluate_repair_pressure_trend(
        state,
        {"level": 0.08706577244027125, "confidence": 0.65},
        **_DEFAULT_KWARGS,
    )
    assert new_state.count == 1  # folded, not discarded
    assert trigger is None  # cold start, correctly no trigger yet


def test_repair_pressure_trend_has_its_own_cooldown_lane_not_the_shared_global_one() -> None:
    # Code-review regression test (2026-07-30). repair_pressure_trend is
    # evaluated on the same live message, in the same branch, as the
    # pre-existing `relational` trigger -- without its own dedicated
    # cooldown lane, any appraisal that fires both would have relational's
    # fire silently starve this one via the shared global lane every time
    # (the exact chat_turn bug from 2026-07-23, recurring in a new place).
    # `relational` itself deliberately has NO dedicated lane (it predates
    # this fix and isn't in scope to change here) -- what matters is that
    # repair_pressure_trend's lane is distinct from the shared default.
    service = EquilibriumService()
    assert "repair_pressure_trend" in service._PER_KIND_COOLDOWN_SETTINGS_ATTR
    attr = service._PER_KIND_COOLDOWN_SETTINGS_ATTR["repair_pressure_trend"]
    dedicated_cooldown = getattr(settings, attr)
    assert dedicated_cooldown != settings.metacog_cooldown_sec
    assert service._cooldown_sec_for_kind("repair_pressure_trend") == dedicated_cooldown
    # relational is unaffected by this fix -- still shares the global lane.
    assert "relational" not in service._PER_KIND_COOLDOWN_SETTINGS_ATTR
    assert service._cooldown_sec_for_kind("relational") == settings.metacog_cooldown_sec
