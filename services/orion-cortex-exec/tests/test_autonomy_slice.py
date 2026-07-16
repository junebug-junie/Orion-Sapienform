from __future__ import annotations

from app.autonomy_slice import build_autonomy_slice


def _drive_state(**overrides) -> dict:
    base = dict(
        pressures={"coherence": 0.72, "continuity": 0.41},
        activations={"coherence": True, "continuity": False, "capability": True},
        dominant_drive="coherence",
        summary="coherence pressure elevated",
        tension_kinds=["drive_competition.coherence_continuity", "unresolved_thread", "identity_drift"],
    )
    base.update(overrides)
    return base


def test_build_autonomy_slice_from_populated_drive_state() -> None:
    ctx = {"chat_drive_state": _drive_state()}
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
    # DriveEngine evolves on its own async tick cadence -- no honest
    # single-turn before/after or confidence estimate exists; never fabricated.
    assert slice_.pressure_trend is None
    assert slice_.confidence is None


def test_active_tensions_hard_capped_at_three_even_with_more_tension_kinds() -> None:
    ctx = {"chat_drive_state": _drive_state(tension_kinds=["a", "b", "c", "d", "e"])}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == ["a", "b", "c"]


def test_active_tensions_empty_when_drive_state_has_no_tension_kinds() -> None:
    ctx = {"chat_drive_state": _drive_state(tension_kinds=[])}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == []
    # dominant_drive still real signal -> slice still emitted.
    assert slice_.dominant_drive == "coherence"


def test_active_tensions_dedupes_and_strips() -> None:
    ctx = {"chat_drive_state": _drive_state(tension_kinds=["a", " a ", "", "b"])}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.active_tensions == ["a", "b"]


def test_active_tensions_malformed_tension_kinds_fails_open() -> None:
    ctx = {"chat_drive_state": _drive_state(tension_kinds="not-a-list")}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive == "coherence"
    assert slice_.active_tensions == []


def test_returns_none_when_drive_state_absent() -> None:
    assert build_autonomy_slice({}) is None
    assert build_autonomy_slice({"chat_drive_state": None}) is None
    assert build_autonomy_slice({"chat_drive_state": {}}) is None


def test_returns_none_when_drive_state_present_but_no_meaningful_signal() -> None:
    ctx = {"chat_drive_state": {"dominant_drive": None, "tension_kinds": []}}
    assert build_autonomy_slice(ctx) is None


def test_does_not_raise_on_malformed_ctx() -> None:
    assert build_autonomy_slice({"chat_drive_state": "not-a-dict"}) is None


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
    """A turn with only recent-action signal (no drive/tension) must still
    emit a real AutonomySliceV1, not None -- the state-absent early return
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
    ctx = {"chat_drive_state": _drive_state()}
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.recent_actions == []

    ctx_with_empty_list = {"chat_drive_state": _drive_state(), "chat_recent_dispatch_actions": []}
    slice_2 = build_autonomy_slice(ctx_with_empty_list)
    assert slice_2 is not None
    assert slice_2.recent_actions == []

    ctx_malformed = {"chat_drive_state": _drive_state(), "chat_recent_dispatch_actions": "not-a-list"}
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


def test_build_autonomy_slice_all_inactive_and_no_dominant_drive_omits() -> None:
    ctx = {"chat_drive_state": _drive_state(dominant_drive=None, tension_kinds=[])}
    assert build_autonomy_slice(ctx) is None


def test_build_autonomy_slice_folds_in_recent_actions_alongside_drive_state() -> None:
    ctx = {
        "chat_drive_state": _drive_state(),
        "chat_recent_dispatch_actions": [_dispatch_action(kind="inspect", summary="real success")],
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive == "coherence"
    assert slice_.recent_actions == ["inspect: real success"]
