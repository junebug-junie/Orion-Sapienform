"""Real trigger for endogenous outreach, replacing the coin-flip stub.

Fixtures are hand-computed run sequences, not read back off the
implementation -- each test states the expected run length in its own name
or comment.

Imports the MODULE (not names out of it) and patches via `patch.object`, not
a string target -- this suite's `conftest.py` clears `sys.modules['scripts.*']`
before every test (autouse), so a string-target `patch("scripts.tension_
outreach_trigger._engine", ...)` re-imports a fresh module object separate
from whatever a bare `from scripts.tension_outreach_trigger import X`
already bound, and the patch silently lands on the wrong object. Same
pattern test_field_channel_glossary_routes.py already uses for this exact
reason.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts import tension_outreach_trigger


def _fake_engine_with_rows(rows: list[tuple[str | None, float | None]]):
    """rows: (winner, deviation) oldest-first, matching the real query's
    `ORDER BY generated_at ASC`."""
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchall.return_value = [
        (w, None if d is None else str(d)) for w, d in rows
    ]
    return fake_engine


def test_no_engine_returns_none_not_a_crash(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    # Reset the shared cached engine (scripts.pg_engine.get_engine, imported
    # here as `_engine`) via its own __globals__ rather than reassigning
    # `scripts.pg_engine._engine` directly -- this suite's conftest.py wipes
    # sys.modules['scripts.*'] before every test, and a fresh `import
    # scripts.pg_engine` elsewhere would bind a DIFFERENT module object than
    # the one this already-imported function's closure actually reads/writes
    # (same reason this file imports the module, not names out of it).
    tension_outreach_trigger._engine.__globals__["_engine"] = None
    assert tension_outreach_trigger.current_run() is None


def test_empty_window_returns_none():
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows([])):
        assert tension_outreach_trigger.current_run() is None


def test_latest_tick_with_no_winner_returns_none():
    rows = [("node:athena", 0.3)] * 10 + [(None, 0.0)]
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        assert tension_outreach_trigger.current_run() is None


def test_run_shorter_than_bar_does_not_fire():
    rows = [("node:athena", 0.3)] * (tension_outreach_trigger.MIN_RUN_LENGTH - 1)
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        assert tension_outreach_trigger.current_run() is None


def test_run_meeting_the_bar_fires_with_real_target_and_length():
    n = tension_outreach_trigger.MIN_RUN_LENGTH
    rows = [("node:athena", 0.2 + 0.01 * i) for i in range(n)]
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        result = tension_outreach_trigger.current_run()
    assert result == tension_outreach_trigger.TensionTriggerReason(
        target_id="node:athena",
        run_length=n,
        peak_deviation_pressure=max(d for _, d in rows),
    )


def test_run_is_broken_by_a_different_winner_not_summed_across_it():
    """A stale earlier episode from a different target must not extend the
    CURRENT run -- only the unbroken tail ending at the latest tick counts."""
    n = tension_outreach_trigger.MIN_RUN_LENGTH
    rows = [("node:atlas", 0.9)] * 20 + [("node:athena", 0.3)] * (n - 1)
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        assert tension_outreach_trigger.current_run() is None  # athena's own run is one short


def test_run_is_broken_by_a_quiet_tick_in_the_middle():
    n = tension_outreach_trigger.MIN_RUN_LENGTH
    rows = (
        [("node:athena", 0.3)] * (n + 2)
        + [(None, 0.0)]
        + [("node:athena", 0.3)] * (n - 1)
    )
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        assert tension_outreach_trigger.current_run() is None  # tail run (after the gap) is one short


def test_reported_magnitude_is_the_max_within_the_run_not_the_latest_value():
    n = tension_outreach_trigger.MIN_RUN_LENGTH
    rows = [("node:athena", 0.9)] + [("node:athena", 0.2)] * (n - 1)
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        result = tension_outreach_trigger.current_run()
    assert result is not None
    assert result.peak_deviation_pressure == 0.9


def test_malformed_deviation_value_degrades_to_zero_not_a_crash():
    n = tension_outreach_trigger.MIN_RUN_LENGTH
    rows = [("node:athena", "not-a-number")] * n  # type: ignore[list-item]
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        result = tension_outreach_trigger.current_run()
    assert result is not None
    assert result.peak_deviation_pressure == 0.0


def test_db_failure_degrades_to_none_never_raises():
    broken_engine = MagicMock()
    broken_engine.connect.side_effect = RuntimeError("connection refused")
    with patch.object(tension_outreach_trigger, "_engine", return_value=broken_engine):
        assert tension_outreach_trigger.current_run() is None


def test_min_run_length_is_overridable_per_call():
    """HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH (settings.py) is wired through
    to this kwarg by scripts/main.py's functools.partial -- prove the kwarg
    itself actually changes the bar, independent of the module constant."""
    rows = [("node:athena", 0.3)] * 3
    with patch.object(tension_outreach_trigger, "_engine", return_value=_fake_engine_with_rows(rows)):
        assert tension_outreach_trigger.current_run(min_run_length=4) is None
        result = tension_outreach_trigger.current_run(min_run_length=3)
    assert result is not None
    assert result.run_length == 3


def test_raw_sql_json_keys_match_the_real_schema_fields():
    """`_fetch_recent_winners` reads `field_json` by hand-written JSON-path
    string, not through the schema -- nothing else enforces that those keys
    still name real `FieldStateV1` fields. This is that contract check: it
    fails loudly if either field is ever renamed in the schema without this
    query being updated to match, instead of failing silently at query time
    in production."""
    from orion.schemas.field_state import FieldStateV1

    field_names = set(FieldStateV1.model_fields)
    assert "tension_borda_winner_target_id" in field_names
    assert "tension_deviation_pressure" in field_names


def test_lookback_query_uses_seconds_not_minutes_for_make_interval():
    """Regression guard, 2026-08-18: `make_interval`'s `mins`/`hours` args are
    `integer` in real Postgres -- a float bound to `mins` raises
    `UndefinedFunction`, silently swallowed by this module's own "never
    raise, degrade to None" contract, so `current_run()` had been returning
    `None` on EVERY call since deploy without a single test catching it (every
    existing test here mocks `_engine()`, so none exercised real Postgres
    type coercion). Confirmed live against the real database before this fix:
    `_fetch_recent_winners(10.0)` raised `sqlalchemy.exc.ProgrammingError:
    ... function make_interval(mins => numeric) does not exist`. `secs` is
    `double precision` and takes a float directly -- same convention
    `services/orion-hub/scripts/chat_history_rehydrate.py`'s own comment
    already documents for this exact pitfall. This test can't spin up a real
    Postgres (this suite has no such fixture), so it locks the query TEXT and
    the bound-parameter shape instead -- a fast, deterministic, non-flaky
    gate for the one mistake that actually caused this."""
    engine = _fake_engine_with_rows([])
    conn = engine.connect.return_value.__enter__.return_value
    with patch.object(tension_outreach_trigger, "_engine", return_value=engine):
        tension_outreach_trigger.current_run(limit_minutes=10.0)

    query, bound_params = conn.execute.call_args.args
    query_text = str(query)
    assert "make_interval(secs =>" in query_text
    assert "make_interval(mins =>" not in query_text
    assert "make_interval(hours =>" not in query_text
    assert bound_params == {"secs": pytest.approx(600.0)}
