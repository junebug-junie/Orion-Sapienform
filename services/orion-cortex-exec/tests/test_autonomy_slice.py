from __future__ import annotations

from app.autonomy_slice import build_autonomy_slice

# ctx["chat_drive_state"]-based dominant_drive/active_tensions derivation was
# removed 2026-07-30 (chore/delete-orion-drives Wave 2a): chat_stance.py no
# longer populates that ctx key (its source, the Postgres drive_audits table,
# lost its last producer in Wave 1). build_autonomy_slice() now only responds
# to ctx["chat_recent_dispatch_actions"]; dominant_drive/active_tensions are
# always None/[] on the returned AutonomySliceV1. The recent_actions tests
# below are unaffected by this change and still exercise real behavior.


def _dispatch_action(**overrides) -> dict:
    base = dict(kind="inspect", summary="checked substrate graph health", success=True, observed_at="2026-07-14T00:00:00+00:00")
    base.update(overrides)
    return base


def test_returns_none_when_no_recent_actions() -> None:
    assert build_autonomy_slice({}) is None
    assert build_autonomy_slice({"chat_recent_dispatch_actions": []}) is None
    assert build_autonomy_slice({"chat_recent_dispatch_actions": None}) is None


def test_does_not_raise_on_malformed_ctx() -> None:
    assert build_autonomy_slice({"chat_recent_dispatch_actions": "not-a-list"}) is None


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
    """A turn with real recent-action signal must still emit a real
    AutonomySliceV1, not None -- dominant_drive/active_tensions are always
    empty now (see module docstring), so recent_actions is the only thing
    that can make this non-None."""
    ctx = {
        "chat_recent_dispatch_actions": [_dispatch_action(kind="inspect", summary="real success")],
    }
    slice_ = build_autonomy_slice(ctx)
    assert slice_ is not None
    assert slice_.dominant_drive is None
    assert slice_.active_tensions == []
    assert slice_.pressure_trend is None
    assert slice_.confidence is None
    assert slice_.recent_actions == ["inspect: real success"]


def test_recent_actions_empty_or_missing_fails_open_to_empty_list() -> None:
    assert build_autonomy_slice({"chat_recent_dispatch_actions": []}) is None

    ctx_malformed = {"chat_recent_dispatch_actions": "not-a-list"}
    assert build_autonomy_slice(ctx_malformed) is None


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
