"""Unit tests for the durable endogenous-outreach decision log."""
from __future__ import annotations

from unittest.mock import patch

from scripts import endogenous_outreach_decisions as decisions


def test_record_decision_never_raises_without_postgres(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    # Must not raise even though there's nowhere to write.
    decisions.record_decision({"outreach": False, "reason": "no_tension_trigger"})


def _run_thread_target_synchronously(fake_thread) -> None:
    """Shared stub for `patch.object(decisions.threading, "Thread")`: runs
    the thread's target inline instead of on a real thread, so a test can
    assert on its effects without a race against a background thread."""
    fake_thread.side_effect = lambda target, kwargs, **kw: type(
        "T", (), {"start": lambda self: target(**kwargs)}
    )()


def test_record_decision_writes_on_a_background_thread(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    calls: list[dict] = []

    def fake_writer(**kwargs):
        calls.append(kwargs)

    with patch.object(decisions, "_write_decision_to_postgres", side_effect=fake_writer):
        with patch.object(decisions.threading, "Thread") as fake_thread:
            _run_thread_target_synchronously(fake_thread)
            decisions.record_decision(
                {"outreach": True, "reason": "sent", "correlation_id": "corr-1", "session_id": "sess-1"},
                tension_reason=_FakeReason(),
                forced=False,
            )

    assert len(calls) == 1
    call = calls[0]
    assert call["result"]["reason"] == "sent"
    assert call["target_id"] == "node:athena"
    assert call["run_length"] == 9
    assert call["forced"] is False
    # 2026-09-07: identity must reach the durable trail too -- this table is
    # the exact one traced by hand to diagnose the "harness_closure"
    # incident, and it had a real column for the scalar but nothing for
    # which channel/node produced it.
    assert call["sustained_load_pressure_channel"] == "disk_capacity_pressure"
    assert call["sustained_load_pressure_node_id"] == "node:athena"


def test_record_decision_defaults_identity_to_none_for_a_reason_without_it(monkeypatch):
    """A `tension_reason` object that predates the identity fields (or any
    object missing them) must degrade to `None`, not raise -- same
    `getattr(..., None)` contract the pre-existing fields already use."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    calls: list[dict] = []

    def fake_writer(**kwargs):
        calls.append(kwargs)

    class _ReasonWithoutIdentity:
        target_id = "node:athena"
        run_length = 9
        peak_deviation_pressure = 0.42
        sustained_load_pressure = 0.0

    with patch.object(decisions, "_write_decision_to_postgres", side_effect=fake_writer):
        with patch.object(decisions.threading, "Thread") as fake_thread:
            _run_thread_target_synchronously(fake_thread)
            decisions.record_decision(
                {"outreach": False, "reason": "no_tension_trigger"},
                tension_reason=_ReasonWithoutIdentity(),
                forced=False,
            )

    assert len(calls) == 1
    assert calls[0]["sustained_load_pressure_channel"] is None
    assert calls[0]["sustained_load_pressure_node_id"] is None


def test_record_decision_respects_the_disable_flag(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("HUB_ENDOGENOUS_OUTREACH_DECISION_LOG_ENABLED", "false")
    with patch.object(decisions.threading, "Thread") as fake_thread:
        decisions.record_decision({"outreach": False, "reason": "orion_passed"})
    fake_thread.assert_not_called()


def test_record_decision_survives_a_broken_writer(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    with patch.object(decisions, "_write_decision_to_postgres", side_effect=RuntimeError("db down")):
        with patch.object(decisions.threading, "Thread") as fake_thread:
            _run_thread_target_synchronously(fake_thread)
            # Exception happens on the (synchronously-run-here-for-the-test)
            # thread target, not in record_decision itself -- record_decision
            # only ever raises from its own try/except, which this asserts
            # by not raising.
            decisions.record_decision({"outreach": False, "reason": "empty_generation"})


class _FakeReason:
    target_id = "node:athena"
    run_length = 9
    peak_deviation_pressure = 0.42
    sustained_load_pressure = 0.0
    sustained_load_pressure_channel = "disk_capacity_pressure"
    sustained_load_pressure_node_id = "node:athena"
