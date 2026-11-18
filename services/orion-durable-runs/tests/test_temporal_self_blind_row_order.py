"""Stored row order must not follow a blinded experiment arm (#3458).

A dream cycle's hypotheses come back dream arm first, then control, and all rows of a cycle share occurred_at.
If the store inserted them in that order, anyone who can read temporal_self_event could take the first half of
each cycle as the dream arm, even with the arm key hidden. The store inserts in event_id order instead."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app.temporal_self_store import window_event_rows  # noqa: E402
from orion.schemas.dream_cycle import HypothesisArm  # noqa: E402
from orion.temporal_self import sources as S  # noqa: E402

AT = datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc)


def _hyp(hid: str, arm: str):
    return S.dream_hypothesis({"hypothesis_id": hid, "cycle_id": "c1", "arm": arm, "created_at": AT,
                               "expires_at": None, "offered_at": None, "offered_run_id": None})


def test_a_cycle_read_dream_arm_first_is_not_stored_dream_arm_first():
    dream, control = get_args(HypothesisArm)
    # Source order: dream arm first. Ids are random hex, so here the control ids sort first.
    events = [_hyp("dh-f00000000001", dream), _hyp("dh-f00000000002", dream),
              _hyp("dh-000000000003", control), _hyp("dh-000000000004", control)]
    w = SimpleNamespace(events=events, late_events=[], salience={})
    rows = window_event_rows(w)
    assert [r["event_id"] for r in rows] == sorted(e.event_id for e in events)
    assert [r["event_id"] for r in rows] != [e.event_id for e in events]


def test_late_events_are_kept_and_flagged():
    dream, control = get_args(HypothesisArm)
    w = SimpleNamespace(events=[_hyp("dh-b00000000001", dream)], late_events=[_hyp("dh-a00000000002", control)],
                        salience={})
    rows = window_event_rows(w)
    assert [(r["event_id"].split(":")[-1], r["late"]) for r in rows] == [("dh-a00000000002", True),
                                                                          ("dh-b00000000001", False)]
