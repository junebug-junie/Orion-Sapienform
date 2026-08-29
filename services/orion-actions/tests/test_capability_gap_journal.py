"""Tests for folding capability-absence episodes into the daily journal seed.

The load-bearing guarantee here is the *anti-spam* one: a day with no outage
must produce a seed byte-identical to the pre-patch one. That is asserted
directly in `test_quiet_day_seed_is_byte_identical_to_pre_patch`, because it is
the property Juniper actually asked for and the one a well-meaning refactor
(defaulting the key to `[]`) would silently break while every other test stayed
green.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from app.capability_gap_journal import (
    LIVENESS_REASON_PREFIXES,
    build_daily_seed_payload,
    RECOVERY_MARKER,
    CapabilityGapEpisode,
    collect_capability_gaps,
    fetch_recent_attention,
    format_capability_gap_block,
    summarize_capability_gaps,
)

W_START = datetime(2026, 8, 29, 6, 0, tzinfo=timezone.utc)
W_END = datetime(2026, 8, 30, 6, 0, tzinfo=timezone.utc)


def _item(reason: str, created: str, message: str = "boom", att_id: str = "a1") -> dict:
    return {
        "reason": reason,
        "created_at": created,
        "message": message,
        "attention_id": att_id,
    }


# --------------------------------------------------------------------------
# episode reconstruction
# --------------------------------------------------------------------------

def test_alert_then_recovery_is_one_resolved_episode() -> None:
    items = [
        _item("node_availability:circe", "2026-08-29T08:00:00", "Node 'circe' has stopped reporting"),
        _item("node_availability:circe", "2026-08-29T08:45:00", f"[x] {RECOVERY_MARKER}node_availability:circe"),
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.node == "circe"
    assert ep.resolved is True
    assert ep.duration_minutes == 45.0
    # The retained message is the ALERT's, not the recovery's: the recovery text
    # says only "recovered", and the alert is what names the lost capabilities.
    assert "stopped reporting" in ep.message


def test_unrecovered_alert_stays_open() -> None:
    items = [_item("node_availability:circe", "2026-08-29T23:00:00")]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.resolved is False
    assert ep.ended_at is None
    assert ep.duration_minutes is None
    assert ep.to_seed_dict()["started_before_window"] is False


def test_recovery_without_in_window_start_is_not_dropped() -> None:
    """An outage that began before the window still gets recorded.

    Without this branch a gap spanning the window boundary -- the 45-minute
    circe outage started at 00:02 UTC, which is before a 06:00 daily window --
    would vanish entirely, which is the exact class of silence this arc exists
    to remove.
    """
    items = [_item("node_availability:circe", "2026-08-29T06:30:00", f"{RECOVERY_MARKER}node_availability:circe")]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.started_at is None
    assert ep.resolved is True
    assert ep.duration_minutes is None
    assert ep.to_seed_dict()["started_before_window"] is True


def test_repeated_alerts_for_one_reason_fold_into_a_single_episode() -> None:
    """A restart re-fires the transition; that is one outage, not three."""
    items = [
        _item("vision_blind", "2026-08-29T20:25:00", "Orion cannot see", att_id="v1"),
        _item("vision_blind", "2026-08-29T21:00:00", "Orion cannot see", att_id="v2"),
        _item("vision_blind", "2026-08-29T21:35:00", "Orion cannot see", att_id="v3"),
    ]
    episodes = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert len(episodes) == 1
    assert episodes[0].started_at == datetime(2026, 8, 29, 20, 25, tzinfo=timezone.utc)
    assert episodes[0].evidence_ids == ["v1", "v2", "v3"]


def test_two_reasons_produce_two_episodes() -> None:
    items = [
        _item("node_availability:circe", "2026-08-29T08:00:00"),
        _item("vision_blind", "2026-08-29T09:00:00"),
    ]
    episodes = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert {e.reason for e in episodes} == {"node_availability:circe", "vision_blind"}
    assert [e.started_at.hour for e in episodes] == [8, 9], "sorted by start time"


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------

def test_non_liveness_reasons_are_excluded() -> None:
    """Real reason text observed live in the attention store on 2026-08-29."""
    items = [
        _item("Workflow schedule needs attention: GitHub Compactor", "2026-08-29T08:00:00"),
        _item("node_availability:circe", "2026-08-29T08:00:00"),
    ]
    episodes = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert [e.reason for e in episodes] == ["node_availability:circe"]


def test_bare_prefix_without_a_node_is_not_an_episode() -> None:
    items = [_item("node_availability:", "2026-08-29T08:00:00")]
    assert summarize_capability_gaps(items, window_start=W_START, window_end=W_END) == []


def test_items_outside_the_window_are_excluded() -> None:
    items = [
        _item("vision_blind", "2026-08-28T23:00:00"),   # before
        _item("vision_blind", "2026-08-31T00:00:00"),   # after
    ]
    assert summarize_capability_gaps(items, window_start=W_START, window_end=W_END) == []


def test_malformed_rows_do_not_raise() -> None:
    items = [
        "not a dict",
        {},
        {"reason": "vision_blind"},                       # no created_at
        {"reason": "vision_blind", "created_at": "nope"},  # unparseable
        None,
    ]
    assert summarize_capability_gaps(items, window_start=W_START, window_end=W_END) == []


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------

def test_naive_timestamps_are_read_as_utc_not_localised(monkeypatch) -> None:
    """`notify_requests.created_at` is `timestamp without time zone` and the API
    returns it with no offset. Treating it as local time would shift every
    episode by the host offset -- enough to move an outage into the wrong day's
    journal, or out of the window entirely.

    **Forces TZ=America/Denver.** The containers run UTC, where `local == UTC`
    makes the naive-vs-local distinction invisible: mutation-testing this file
    confirmed that swapping the correct `replace(tzinfo=utc)` for a wrong
    `astimezone()` changed nothing on a UTC host and the test stayed green.
    Juniper's own machine is MDT, so the bug is real even though the host hides
    it. Pinning the zone is what makes this assertion able to fail.
    """
    import time

    monkeypatch.setenv("TZ", "America/Denver")
    time.tzset()
    try:
        _assert_naive_is_utc()
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()


def _assert_naive_is_utc() -> None:
    items = [
        _item("vision_blind", "2026-08-29T20:25:33.704942"),
        _item("vision_blind", "2026-08-29T20:55:33.704942", f"{RECOVERY_MARKER}vision_blind"),
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.started_at == datetime(2026, 8, 29, 20, 25, 33, 704942, tzinfo=timezone.utc)
    assert ep.duration_minutes == 30.0


def test_offset_aware_timestamps_are_normalised_to_utc() -> None:
    items = [
        _item("vision_blind", "2026-08-29T14:25:00-06:00"),
        _item("vision_blind", "2026-08-29T15:25:00-06:00", f"{RECOVERY_MARKER}vision_blind"),
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.started_at == datetime(2026, 8, 29, 20, 25, tzinfo=timezone.utc)
    assert ep.duration_minutes == 60.0


# --------------------------------------------------------------------------
# producer contract
# --------------------------------------------------------------------------

def test_recovery_marker_matches_health_monitor_format() -> None:
    """Pin RECOVERY_MARKER against its actual producer.

    The attention record carries no severity field, so a message substring is
    the only available signal that a record closes an episode. That makes this a
    real cross-service contract: if health_monitor renames its recovery prefix,
    every episode here silently becomes unresolved-forever. Read the producer's
    source rather than importing it, to avoid coupling this service's tests to
    another service's dependencies.
    """
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    producer = os.path.join(
        repo_root, "services", "orion-substrate-runtime", "app", "health_monitor.py"
    )
    if not os.path.exists(producer):
        pytest.skip("substrate-runtime not present in this checkout")
    src = open(producer, encoding="utf-8").read()
    assert f'recovered: {{check.key}}' in src, (
        "health_monitor._publish no longer formats recoveries as 'recovered: {check.key}'; "
        "RECOVERY_MARKER in capability_gap_journal.py must be updated to match"
    )
    assert RECOVERY_MARKER == "recovered: "


def test_liveness_prefixes_cover_both_known_producers() -> None:
    assert "node_availability:" in LIVENESS_REASON_PREFIXES  # substrate-runtime, PR #1944
    assert "vision_blind" in LIVENESS_REASON_PREFIXES        # vision watchdog, PR #1805


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def test_block_is_empty_when_nothing_was_absent() -> None:
    assert format_capability_gap_block([]) == ""


def test_block_names_subject_time_and_duration() -> None:
    ep = CapabilityGapEpisode(
        reason="node_availability:circe",
        message="Node 'circe' has stopped reporting. Capabilities affected: local_llm_heavy.",
        started_at=datetime(2026, 8, 29, 8, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 8, 29, 8, 45, tzinfo=timezone.utc),
        node="circe",
    )
    block = format_capability_gap_block([ep])
    assert "circe" in block
    assert "08:00 UTC" in block
    assert "45 minutes" in block
    assert "local_llm_heavy" in block


def test_block_marks_an_unresolved_gap() -> None:
    ep = CapabilityGapEpisode(
        reason="vision_blind",
        message="Orion cannot see.",
        started_at=datetime(2026, 8, 29, 21, 0, tzinfo=timezone.utc),
        ended_at=None,
    )
    assert "still unresolved" in format_capability_gap_block([ep])


# --------------------------------------------------------------------------
# I/O never breaks the journal
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_returns_empty_and_does_not_raise_when_notify_is_down() -> None:
    out = await fetch_recent_attention(
        notify_url="http://127.0.0.1:1", notify_api_token=None, timeout_sec=0.25
    )
    assert out == []


@pytest.mark.asyncio
async def test_collect_returns_empty_on_unparseable_window() -> None:
    out = await collect_capability_gaps(
        notify_url="http://127.0.0.1:1",
        notify_api_token=None,
        window_start_utc="not-a-date",
        window_end_utc="also-not",
    )
    assert out == []


# --------------------------------------------------------------------------
# the anti-spam guarantee
# --------------------------------------------------------------------------

def test_quiet_day_seed_is_byte_identical_to_pre_patch() -> None:
    """The whole point: a day with no outage must not change the prompt at all.

    Mirrors the construction in main.py. If someone "helpfully" makes the key
    always present as [], this fails and every other test in this file still
    passes.
    """
    pre_patch = json.dumps(
        {
            "request_date": "2026-08-30",
            "window_start_utc": W_START.isoformat(),
            "window_end_utc": W_END.isoformat(),
        },
        sort_keys=True,
    )
    gaps = [g.to_seed_dict() for g in summarize_capability_gaps([], window_start=W_START, window_end=W_END)]
    seed_payload = build_daily_seed_payload(
        request_date="2026-08-30",
        window_start_utc=W_START.isoformat(),
        window_end_utc=W_END.isoformat(),
        gaps=gaps,
    )
    assert json.dumps(seed_payload, sort_keys=True) == pre_patch
    assert "capability_gaps" not in seed_payload


def test_seed_omits_the_key_for_every_empty_shape() -> None:
    """None and [] must both mean "absent", not "present but empty"."""
    for empty in (None, [], ()):
        payload = build_daily_seed_payload(
            request_date="2026-08-30",
            window_start_utc=W_START.isoformat(),
            window_end_utc=W_END.isoformat(),
            gaps=empty,
        )
        assert "capability_gaps" not in payload, f"empty shape {empty!r} leaked the key"


def test_seed_includes_the_key_when_a_gap_exists() -> None:
    payload = build_daily_seed_payload(
        request_date="2026-08-30",
        window_start_utc=W_START.isoformat(),
        window_end_utc=W_END.isoformat(),
        gaps=[{"reason": "vision_blind"}],
    )
    assert payload["capability_gaps"] == [{"reason": "vision_blind"}]


def test_seed_carries_the_gap_when_there_was_one() -> None:
    items = [
        _item("node_availability:circe", "2026-08-29T08:00:00", "Node 'circe' has stopped reporting"),
        _item("node_availability:circe", "2026-08-29T08:45:00", f"{RECOVERY_MARKER}node_availability:circe"),
    ]
    gaps = [g.to_seed_dict() for g in summarize_capability_gaps(items, window_start=W_START, window_end=W_END)]
    assert len(gaps) == 1
    assert gaps[0]["node"] == "circe"
    assert gaps[0]["duration_minutes"] == 45.0
    assert gaps[0]["resolved"] is True
    json.dumps(gaps, sort_keys=True)  # must be JSON-serialisable for the seed
