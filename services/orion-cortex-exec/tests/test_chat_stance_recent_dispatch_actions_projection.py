from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.chat_stance import _DISPATCH_ACTION_SUBJECT, _project_recent_dispatch_actions
from orion.autonomy.models import ActionOutcomeRefV1

_EXPECTED_KEYS = {"kind", "summary", "success", "observed_at"}
_NOW = datetime(2026, 7, 13, 0, 0, 0, tzinfo=timezone.utc)


def _outcome(**overrides) -> ActionOutcomeRefV1:
    payload = dict(
        action_id="dispatch-1",
        kind="inspect",
        summary="checked the mesh for anomalies",
        success=True,
        surprise=0.0,
        observed_at=_NOW,
        query="mesh anomalies",
        articles=[],
        salience=0.5,
    )
    payload.update(overrides)
    return ActionOutcomeRefV1(**payload)


def _patched(outcomes):
    return patch("app.chat_stance.load_action_outcomes", return_value=outcomes)


def test_queries_the_orion_subject_directly_not_ctx() -> None:
    # The whole point of this fix: it must not depend on whatever subject
    # ctx["chat_autonomy_state_v2"] happened to be reduced for this turn
    # (e.g. "relationship" during contextual fallback) -- Layer 9 dispatch
    # actions always live under subject="orion".
    with _patched([]) as mock_load:
        _project_recent_dispatch_actions({"chat_autonomy_state_v2": {"subject": "relationship"}})
    mock_load.assert_called_once_with(subject=_DISPATCH_ACTION_SUBJECT)
    assert _DISPATCH_ACTION_SUBJECT == "orion"


def test_empty_ctx_still_queries_and_returns_empty() -> None:
    with _patched([]):
        assert _project_recent_dispatch_actions({}) == []


def test_empty_when_load_action_outcomes_returns_empty_list() -> None:
    with _patched([]):
        assert _project_recent_dispatch_actions({}) == []


def test_caps_at_three_newest_first_and_exact_keys() -> None:
    outcomes = [
        _outcome(action_id="a1", kind="inspect", observed_at=_NOW - timedelta(hours=4)),
        _outcome(action_id="a2", kind="summarize", observed_at=_NOW - timedelta(hours=3)),
        _outcome(action_id="a3", kind="observe", observed_at=_NOW - timedelta(hours=2)),
        _outcome(action_id="a4", kind="inspect", observed_at=_NOW - timedelta(hours=1)),
        _outcome(action_id="a5", kind="noop", observed_at=_NOW),
    ]
    with _patched(outcomes):
        result = _project_recent_dispatch_actions({})

    assert len(result) == 3
    # Newest-first: a5 (NOW), a4 (NOW-1h), a3 (NOW-2h).
    assert [r["kind"] for r in result] == ["noop", "inspect", "observe"]
    for r in result:
        assert set(r.keys()) == _EXPECTED_KEYS
        assert "action_id" not in r
        assert "query" not in r
        assert "articles" not in r
        assert "salience" not in r


def test_projected_dict_field_values() -> None:
    outcome = _outcome(kind="inspect", summary="checked the mesh", success=True, observed_at=_NOW)
    with _patched([outcome]):
        result = _project_recent_dispatch_actions({})

    assert len(result) == 1
    entry = result[0]
    assert entry["kind"] == "inspect"
    assert entry["summary"] == "checked the mesh"
    assert entry["success"] is True
    assert entry["observed_at"] == _NOW.isoformat()


def test_entry_with_none_observed_at_does_not_crash_sort() -> None:
    outcomes = [
        _outcome(action_id="a1", kind="inspect", observed_at=None),
        _outcome(action_id="a2", kind="observe", observed_at=_NOW - timedelta(hours=1)),
        _outcome(action_id="a3", kind="summarize", observed_at=_NOW),
    ]
    with _patched(outcomes):
        result = _project_recent_dispatch_actions({})

    # Newest-first, None-timestamped entry sorts last.
    assert [r["kind"] for r in result] == ["summarize", "observe", "inspect"]
    assert result[-1]["observed_at"] is None


def test_never_raises_when_load_action_outcomes_raises() -> None:
    with patch("app.chat_stance.load_action_outcomes", side_effect=RuntimeError("db down")):
        assert _project_recent_dispatch_actions({}) == []


def test_never_raises_on_malformed_returned_value() -> None:
    with _patched("not-a-list"):
        assert _project_recent_dispatch_actions({}) == []


def test_skips_items_with_missing_kind_or_summary_rather_than_emitting_hollow_row() -> None:
    """A schema-drifted item (e.g. a future writer that doesn't set kind/summary)
    must be dropped, not surfaced as a {kind: None, summary: None, ...} row --
    that would be exactly the empty-shell "evidence" this projection exists to
    prevent from reaching the evidence-gated prompt section."""
    outcomes = [
        {"kind": None, "summary": "has no kind", "success": True, "observed_at": _NOW.isoformat()},
        {"kind": "inspect", "summary": "", "success": True, "observed_at": _NOW.isoformat()},
        {"kind": "inspect", "summary": "valid entry", "success": True, "observed_at": _NOW.isoformat()},
    ]
    with _patched(outcomes):
        result = _project_recent_dispatch_actions({})

    assert len(result) == 1
    assert result[0]["summary"] == "valid entry"


def test_success_none_is_preserved_not_coerced_to_false() -> None:
    """success=None (ActionOutcomeRefV1's documented "genuinely unknown" state)
    must pass through as None, not be silently coerced -- the template layer
    decides how to word an unknown outcome, this projection must not lie about
    it by turning "unknown" into "failed"."""
    outcome = _outcome(success=None)
    with _patched([outcome]):
        result = _project_recent_dispatch_actions({})
    assert result[0]["success"] is None
