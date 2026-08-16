"""A deferral leaves a trace, with its duration (ROADMAP A4).

The arc's destination is Orion perceiving *"I wanted to think and had to wait, this long, while
that ran instead."* Until now that event left NO trace: `wait_for_slack` returned True the
instant slack appeared, however long it had blocked, and logged only on TIMEOUT.

So the one event Track A exists to observe was computed and thrown away, and A4's proceed gate
-- ">=1 admission-wait event attributable to Orion's own cognition, **with its duration**" --
could never pass regardless of how much traffic ran.

Measured before this: 24 h of gateway logs contained **zero** `[LLM-GW background]` lines.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from app.priority_admission import wait_for_slack, wait_for_slack_sync


class _Target:
    def __init__(self, url="http://atlas:8013", reserved=2):
        self.url = url
        self.reserved_free_slots = reserved


def _parse(caplog):
    """The structured fields out of the admission log line."""
    out = []
    for rec in caplog.records:
        if "admission waited=" not in rec.getMessage():
            continue
        msg = rec.getMessage()
        fields = {}
        for part in msg.split():
            if "=" in part:
                k, _, v = part.partition("=")
                fields[k] = v
        fields["_level"] = rec.levelname
        out.append(fields)
    return out


# ------------------------------------------------------------ a real wait is recorded

def test_a_wait_records_its_duration_and_outcome(monkeypatch, caplog):
    """THE POINT. Two polls blocked, then admitted -- previously silent."""
    calls = {"n": 0}

    async def fake_free(url, **kw):
        calls["n"] += 1
        return 0 if calls["n"] < 3 else 4      # busy, busy, then slack

    monkeypatch.setattr("app.priority_admission._free_slot_count", fake_free)
    with caplog.at_level(logging.INFO):
        ok = asyncio.run(wait_for_slack(_Target(), poll_interval_sec=0.001, max_wait_sec=5.0))

    assert ok is True
    (rec,) = _parse(caplog)
    assert rec["outcome"] == "admitted"
    assert rec["polls"] == "3"
    assert float(rec["waited"].rstrip("s")) >= 0.0
    assert rec["reserved"] == "2"


def test_the_fast_path_is_recorded_too_as_a_zero_wait(monkeypatch, caplog):
    """`waited=0.0s` distinguishes 'admitted immediately' from 'never checked'. Those are
    different facts and were previously the same silence -- which is how a gate that never
    fires looks identical to a gate that always passes."""
    async def fake_free(url, **kw):
        return 4

    monkeypatch.setattr("app.priority_admission._free_slot_count", fake_free)
    with caplog.at_level(logging.INFO):
        assert asyncio.run(wait_for_slack(_Target(), poll_interval_sec=0.001, max_wait_sec=5.0))

    (rec,) = _parse(caplog)
    assert rec["outcome"] == "admitted"
    assert rec["polls"] == "1"


def test_an_unreachable_upstream_is_distinguishable_from_a_wait(monkeypatch, caplog):
    """/slots unreachable means we did not measure anything -- not that admission was instant.
    Same absent-is-not-zero rule the rest of this arc enforces."""
    async def fake_free(url, **kw):
        return None

    monkeypatch.setattr("app.priority_admission._free_slot_count", fake_free)
    with caplog.at_level(logging.INFO):
        assert asyncio.run(wait_for_slack(_Target(), poll_interval_sec=0.001, max_wait_sec=5.0)) is False

    (rec,) = _parse(caplog)
    assert rec["outcome"] == "unchecked"


def test_a_timeout_still_warns_and_now_carries_the_duration(monkeypatch, caplog):
    """The only case that logged before. It keeps WARNING level -- forwarding without slack is
    a real degradation -- and gains the duration everything else now has."""
    async def fake_free(url, **kw):
        return 0

    monkeypatch.setattr("app.priority_admission._free_slot_count", fake_free)
    with caplog.at_level(logging.INFO):
        assert asyncio.run(wait_for_slack(_Target(), poll_interval_sec=0.001, max_wait_sec=0.01)) is False

    (rec,) = _parse(caplog)
    assert rec["outcome"] == "timeout_forwarded"
    assert rec["_level"] == "WARNING"
    assert float(rec["waited"].rstrip("s")) >= 0.0


# ------------------------------------------------------------ the sync path is not forgotten

def test_the_sync_path_is_instrumented_identically(monkeypatch, caplog):
    """llm_backend.py:1414 uses wait_for_slack_sync. Instrumenting only the async path would
    leave whichever one Orion's traffic actually takes invisible -- and it is not obvious from
    the call sites which that is."""
    calls = {"n": 0}

    def fake_free(url, **kw):
        calls["n"] += 1
        return 0 if calls["n"] < 2 else 4

    monkeypatch.setattr("app.priority_admission._free_slot_count_sync", fake_free)
    with caplog.at_level(logging.INFO):
        assert wait_for_slack_sync(_Target(), poll_interval_sec=0.001, max_wait_sec=5.0)

    (rec,) = _parse(caplog)
    assert rec["outcome"] == "admitted"
    assert rec["polls"] == "2"


def test_the_sync_timeout_also_carries_duration(monkeypatch, caplog):
    monkeypatch.setattr("app.priority_admission._free_slot_count_sync", lambda url, **kw: 0)
    with caplog.at_level(logging.INFO):
        assert wait_for_slack_sync(_Target(), poll_interval_sec=0.001, max_wait_sec=0.01) is False

    (rec,) = _parse(caplog)
    assert rec["outcome"] == "timeout_forwarded"


# ------------------------------------------------------------ behaviour is unchanged

@pytest.mark.parametrize("free,expected", [(4, True), (0, False), (None, False)])
def test_instrumentation_does_not_change_what_the_gate_decides(monkeypatch, free, expected):
    """This patch must be observation only. A gate that also changed behaviour would make the
    A4 measurement unattributable to anything."""
    async def fake_free(url, **kw):
        return free

    monkeypatch.setattr("app.priority_admission._free_slot_count", fake_free)
    assert asyncio.run(
        wait_for_slack(_Target(), poll_interval_sec=0.001, max_wait_sec=0.01)
    ) is expected


def test_every_outcome_emits_exactly_one_record(monkeypatch, caplog):
    """One line per admission decision. Duplicates would inflate any rate computed from these,
    and silence on a path would understate deferrals -- both corrupt A4's gate."""
    async def fake_free(url, **kw):
        return 4

    monkeypatch.setattr("app.priority_admission._free_slot_count", fake_free)
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            asyncio.run(wait_for_slack(_Target(), poll_interval_sec=0.001, max_wait_sec=1.0))
    assert len(_parse(caplog)) == 3
