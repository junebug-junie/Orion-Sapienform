from __future__ import annotations

from app.autonomy_slice import build_autonomy_slice
from orion.autonomy.models import AutonomyStateDeltaV1, AutonomyStateV2


def _state_v2(**overrides) -> dict:
    base = dict(
        subject="orion",
        model_layer="graph",
        entity_id="orion",
        dominant_drive="coherence",
        active_drives=["coherence", "continuity"],
        drive_pressures={"coherence": 0.72, "continuity": 0.41},
        tension_kinds=["drive_competition.coherence_continuity", "unresolved_thread", "identity_drift"],
        goal_headlines=[],
        source="reducer",
        confidence=0.63,
    )
    base.update(overrides)
    return AutonomyStateV2.model_validate(base).model_dump(mode="json")


def _delta(**overrides) -> dict:
    base = dict(subject="orion", new_tensions=["fresh_tension"], resolved_tensions=[])
    base.update(overrides)
    return AutonomyStateDeltaV1.model_validate(base).model_dump(mode="json")


def test_build_autonomy_slice_from_populated_state() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_state_delta": _delta(),
        "chat_autonomy_movement_debug": {
            "pressures_before": {"coherence": 0.5, "continuity": 0.4},
            "pressures_after": {"coherence": 0.72, "continuity": 0.41},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.schema_version == "autonomy.slice.v1"
    assert slice_.dominant_drive == "coherence"
    # tension_kinds has 3 entries; must be capped, never exceed 3.
    assert slice_.active_tensions == [
        "drive_competition.coherence_continuity",
        "unresolved_thread",
        "identity_drift",
    ]
    assert len(slice_.active_tensions) <= 3
    assert slice_.pressure_trend == "rising"
    assert slice_.confidence == 0.63


def test_active_tensions_hard_capped_at_three_even_with_more_tension_kinds() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(
            tension_kinds=["a", "b", "c", "d", "e"],
        ),
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == ["a", "b", "c"]


def test_falls_back_to_delta_new_tensions_when_tension_kinds_empty() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(tension_kinds=[], dominant_drive=None),
        "chat_autonomy_state_delta": _delta(new_tensions=["just_minted"]),
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == ["just_minted"]


def test_pressure_trend_none_when_before_pressures_missing_first_turn() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_movement_debug": {
            "pressures_before": None,
            "pressures_after": {"coherence": 0.72},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.pressure_trend is None


def test_pressure_trend_stable_when_movement_below_epsilon() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_movement_debug": {
            "pressures_before": {"coherence": 0.72, "continuity": 0.41},
            "pressures_after": {"coherence": 0.725, "continuity": 0.405},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.pressure_trend == "stable"


def test_returns_none_when_state_absent() -> None:
    assert build_autonomy_slice({}) is None
    assert build_autonomy_slice({"chat_autonomy_state_v2": None}) is None
    assert build_autonomy_slice({"chat_autonomy_state_v2": {}}) is None


def test_returns_none_when_state_present_but_no_meaningful_signal() -> None:
    # dominant_drive/tensions empty; confidence explicitly None (not the model's
    # own 0.5 default) to exercise the true all-empty branch.
    ctx = {
        "chat_autonomy_state_v2": {
            "subject": "orion",
            "dominant_drive": None,
            "tension_kinds": [],
            "confidence": None,
        },
    }
    assert build_autonomy_slice(ctx) is None


def test_does_not_raise_on_malformed_ctx() -> None:
    assert build_autonomy_slice({"chat_autonomy_state_v2": "not-a-dict"}) is None
    assert build_autonomy_slice({"chat_autonomy_state_v2": _state_v2(), "chat_autonomy_movement_debug": "garbage"}) is not None


def _dispatch_action(**overrides) -> dict:
    base = dict(kind="inspect", summary="checked substrate graph health", success=True, observed_at="2026-07-14T00:00:00+00:00")
    base.update(overrides)
    return base


def test_recent_actions_populated_and_capped_at_three_with_more_than_three_success_entries() -> None:
    ctx = {
        "chat_recent_dispatch_actions": [
            _dispatch_action(kind="inspect", summary="entry one"),
            _dispatch_action(kind="inspect", summary="entry two"),
            _dispatch_action(kind="inspect", summary="entry three"),
            _dispatch_action(kind="inspect", summary="entry four"),
            _dispatch_action(kind="inspect", summary="entry five"),
        ],
    }
    slice_ = build_autonomy_slice(ctx, max_recent_actions=3)
    assert slice_ is not None
    assert slice_.recent_actions == [
        "inspect: entry one",
        "inspect: entry two",
        "inspect: entry three",
    ]
    assert len(slice_.recent_actions) <= 3


def test_recent_actions_excludes_failed_or_missing_success_entries() -> None:
    ctx = {
        "chat_recent_dispatch_actions": [
            _dispatch_action(kind="inspect", summary="failed attempt", success=False),
            _dispatch_action(kind="inspect", summary="unknown outcome", success=None),
            _dispatch_action(kind="inspect", summary="real success", success=True),
        ],
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.recent_actions == ["inspect: real success"]


def test_recent_actions_truncated_to_line_char_budget() -> None:
    long_summary = "x" * 500
    ctx = {"chat_recent_dispatch_actions": [_dispatch_action(kind="inspect", summary=long_summary)]}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert len(slice_.recent_actions) == 1
    assert len(slice_.recent_actions[0]) <= 160
    assert slice_.recent_actions[0].startswith("inspect: ")


def test_omit_check_emits_slice_when_only_recent_actions_have_signal() -> None:
    """A turn with only recent-action signal (no drive/tension/trend) must
    still emit a real AutonomySliceV1, not None -- this is the exact trap
    the design spec calls out: the pre-existing state-absent early return
    must not silently drop the whole feature."""
    ctx = {
        "chat_recent_dispatch_actions": [_dispatch_action(kind="inspect", summary="real success")],
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive is None
    assert slice_.active_tensions == []
    assert slice_.pressure_trend is None
    assert slice_.recent_actions == ["inspect: real success"]


def test_recent_actions_empty_or_missing_fails_open_to_empty_list() -> None:
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.recent_actions == []

    ctx_with_empty_list = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_recent_dispatch_actions": [],
    }
    slice_2 = build_autonomy_slice(ctx_with_empty_list)
    assert slice_2 is not None
    assert slice_2.recent_actions == []

    ctx_malformed = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_recent_dispatch_actions": "not-a-list",
    }
    slice_3 = build_autonomy_slice(ctx_malformed)
    assert slice_3 is not None
    assert slice_3.recent_actions == []


def test_returns_none_when_everything_including_recent_actions_is_empty() -> None:
    assert build_autonomy_slice({}) is None
    assert build_autonomy_slice({"chat_recent_dispatch_actions": []}) is None
    assert build_autonomy_slice({"chat_recent_dispatch_actions": [_dispatch_action(success=False)]}) is None


def test_recent_actions_respects_zero_or_negative_limit() -> None:
    """Regression: the cap check must run before appending, not after --
    otherwise max_recent_actions=0 still let exactly one entry through."""
    ctx = {"chat_recent_dispatch_actions": [_dispatch_action(summary="entry one")]}
    assert build_autonomy_slice(ctx, max_recent_actions=0) is None
    assert build_autonomy_slice(ctx, max_recent_actions=-1) is None


def _drive_state(**overrides) -> dict:
    base = dict(
        pressures={"coherence": 0.72, "continuity": 0.41},
        activations={"coherence": True, "continuity": False, "capability": True},
        dominant_drive="coherence",
        summary="coherence pressure elevated",
    )
    base.update(overrides)
    return base


def test_build_autonomy_slice_prefers_drive_state_for_dominant_drive_but_keeps_v2_signal() -> None:
    """chat_drive_state present -> dominant_drive is sourced from DriveEngine
    (the live, well-tested signal). But active_tensions/pressure_trend/
    confidence are always sourced from AutonomyStateV2 when it's also
    present in ctx -- these are real, simultaneously-computed values that
    must not be silently dropped just because drive_state supplied a
    dominant_drive. (Previously this branch discarded them entirely and
    fabricated active_tensions from drive *kinds* instead of real tension
    kinds -- both were bugs, fixed here.)"""
    ctx = {
        "chat_drive_state": _drive_state(),
        "chat_autonomy_state_v2": _state_v2(
            dominant_drive="continuity",
            tension_kinds=["drive_competition.coherence_continuity", "unresolved_thread"],
        ),
        "chat_autonomy_movement_debug": {
            "pressures_before": {"coherence": 0.5},
            "pressures_after": {"coherence": 0.72},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.schema_version == "autonomy.slice.v1"
    # dominant_drive: drive_state wins over V2's "continuity".
    assert slice_.dominant_drive == "coherence"
    # active_tensions: real V2 tension kinds, not drive_state's activation keys.
    assert slice_.active_tensions == ["drive_competition.coherence_continuity", "unresolved_thread"]
    # pressure_trend/confidence: real V2 values, no longer dropped.
    assert slice_.pressure_trend == "rising"
    assert slice_.confidence == 0.63


def test_build_autonomy_slice_falls_back_when_drive_state_empty_or_all_inactive() -> None:
    """chat_drive_state present but empty or all-inactive -> falls through to
    the V2 path (or the existing omit rule) rather than emitting a slice
    with no dominant_drive/active_tensions from an empty drive_state."""
    # Empty dict: falls through to V2, which has real signal here.
    ctx_empty = {
        "chat_drive_state": {},
        "chat_autonomy_state_v2": _state_v2(),
    }
    slice_empty = build_autonomy_slice(ctx_empty)
    assert slice_empty is not None
    assert slice_empty.dominant_drive == "coherence"
    assert slice_empty.pressure_trend is None  # no movement_debug in ctx_empty

    # All-inactive + no dominant_drive, no V2 fallback, no recent actions ->
    # existing omit rule applies (None, not an empty-content slice).
    ctx_inactive = {
        "chat_drive_state": _drive_state(
            dominant_drive=None,
            activations={"coherence": False, "continuity": False},
        ),
    }
    assert build_autonomy_slice(ctx_inactive) is None


def test_build_autonomy_slice_uses_v2_path_when_drive_state_absent() -> None:
    """chat_drive_state absent, chat_autonomy_state_v2 present -> existing V2
    behavior is unchanged (regression guard for the pre-existing path)."""
    ctx = {
        "chat_autonomy_state_v2": _state_v2(),
        "chat_autonomy_state_delta": _delta(),
        "chat_autonomy_movement_debug": {
            "pressures_before": {"coherence": 0.5, "continuity": 0.4},
            "pressures_after": {"coherence": 0.72, "continuity": 0.41},
        },
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive == "coherence"
    assert slice_.active_tensions == [
        "drive_competition.coherence_continuity",
        "unresolved_thread",
        "identity_drift",
    ]
    assert slice_.pressure_trend == "rising"
    assert slice_.confidence == 0.63


def test_build_autonomy_slice_returns_none_when_both_drive_state_and_v2_absent() -> None:
    """Both sources absent, no recent actions -> None (unchanged)."""
    assert build_autonomy_slice({}) is None
    assert build_autonomy_slice({"chat_drive_state": None, "chat_autonomy_state_v2": None}) is None
    assert build_autonomy_slice({"chat_drive_state": {}, "chat_autonomy_state_v2": {}}) is None


def test_build_autonomy_slice_drive_state_folds_in_recent_actions() -> None:
    """recent_actions is sourced independently of which drive/tension source
    won -- must still be folded in when drive_state is the active source."""
    ctx = {
        "chat_drive_state": _drive_state(),
        "chat_recent_dispatch_actions": [_dispatch_action(kind="inspect", summary="real success")],
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive == "coherence"
    assert slice_.recent_actions == ["inspect: real success"]


def test_build_autonomy_slice_drive_state_malformed_activations_fails_open() -> None:
    """drive_state's `activations` field is not read by build_autonomy_slice
    at all (only `dominant_drive` is) -- a malformed value there must not
    raise and must not affect dominant_drive extraction."""
    ctx = {"chat_drive_state": _drive_state(activations="not-a-dict")}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive == "coherence"
    assert slice_.active_tensions == []
