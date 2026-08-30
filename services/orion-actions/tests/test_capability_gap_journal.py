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
    MAX_DETAIL_CHARS,
    LIVENESS_REASON_PREFIXES,
    MAX_EPISODES_IN_SEED,
    RECOVERY_REASON_BY_ALERT,
    build_daily_seed_payload,
    RECOVERY_MARKER,
    CapabilityGapEpisode,
    collect_capability_gaps,
    fetch_recent_attention,
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


def test_repeat_alerts_are_separate_episodes_not_one_long_one() -> None:
    """Three real vision_blind rows from 2026-08-29, verbatim timestamps.

    An earlier version folded these into a single unresolved 1h48m span. They are
    three distinct outages -- vision-host re-arms (`_alerting` must clear before it
    can fire again) -- so folding told Orion's journal something false about his
    own day. Folding was there to absorb an edge-triggered restart re-fire, but
    `health_monitor._has_open_alert` already prevents that: the absence sweep fired
    142 times on 2026-08-29 and produced exactly one attention record.
    """
    items = [
        _item("vision_blind", "2026-08-29T20:25:33", "Orion cannot see", att_id="v1"),
        _item("vision_blind", "2026-08-29T21:00:18", "Orion cannot see", att_id="v2"),
        _item("vision_blind", "2026-08-29T22:13:19", "Orion cannot see", att_id="v3"),
    ]
    episodes = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert len(episodes) == 3, "three arms of the watcher are three outages"
    assert [e.started_at.strftime("%H:%M") for e in episodes] == ["20:25", "21:00", "22:13"]


def test_vision_recovery_uses_a_different_reason_and_still_closes() -> None:
    """vision-host announces recovery as `vision_recovered`, not `vision_blind`.

    liveness.py:342 emits that reason with severity "info" and a message with no
    "recovered: " substring, so a reason-filtered, marker-matched implementation
    discarded it before it could close anything -- every vision episode stayed
    open forever.
    """
    assert RECOVERY_REASON_BY_ALERT["vision_blind"] == "vision_recovered"
    items = [
        _item("vision_blind", "2026-08-29T20:25:00", "Orion cannot see"),
        {
            "reason": "vision_recovered",
            "created_at": "2026-08-29T20:55:00",
            "severity": "info",
            "message": "Vision is working again on athena: 98% of recent tasks succeeding.",
            "attention_id": "r1",
        },
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.resolved is True
    assert ep.duration_minutes == 30.0


def test_severity_info_closes_a_same_reason_episode() -> None:
    """health_monitor._publish sets severity="info" on recovery, "critical" on alert.

    Severity is the primary close signal now. An earlier version used a message
    substring alone, justified by a docstring claiming the record had no severity
    field -- it does, and the claim came from a truncated key listing.
    """
    items = [
        {"reason": "node_availability:circe", "created_at": "2026-08-29T08:00:00",
         "severity": "critical", "message": "Node 'circe' has stopped reporting", "attention_id": "a"},
        {"reason": "node_availability:circe", "created_at": "2026-08-29T08:45:00",
         "severity": "info", "message": "[Orion substrate-runtime] recovered: node_availability:circe",
         "attention_id": "b"},
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.resolved is True and ep.duration_minutes == 45.0


def test_a_critical_alert_mentioning_recovered_does_not_close_an_episode() -> None:
    """The alert message interpolates ', '.join(capability_impacts), which this
    module does not control. Under marker-only matching, a capability literally
    named with that substring turned a node going DOWN into a gap that ENDED."""
    items = [
        {"reason": "node_availability:circe", "created_at": "2026-08-29T08:00:00",
         "severity": "critical",
         "message": "Node 'circe' has stopped reporting. Capabilities affected: not-yet-recovered: gpu.",
         "attention_id": "a"},
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.resolved is False, "a critical alert must never read as a recovery"


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


def test_an_alert_before_the_window_is_carried_forward_not_excluded() -> None:
    """Supersedes an earlier `test_items_outside_the_window_are_excluded`, which
    asserted the defect: it filtered *records* by the window, so an alert at
    23:00 the previous night with no recovery since -- an outage still in
    progress -- was dropped. Episodes are now filtered by overlap instead. The
    "after the window" half of that old test survives as
    `test_records_after_the_window_are_ignored`.
    """
    items = [_item("vision_blind", "2026-08-28T23:00:00")]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.resolved is False
    assert ep.started_at == datetime(2026, 8, 28, 23, 0, tzinfo=timezone.utc)


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


# --------------------------------------------------------------------------
# window overlap, truncation, and producer text
# --------------------------------------------------------------------------

def test_outage_spanning_the_entire_window_is_reported() -> None:
    """Day two of a multi-day outage: the alert is before the window and there is
    no recovery inside it. Filtering *records* by the window made this vanish --
    the exact silence this module exists to remove."""
    items = [_item("node_availability:circe", "2026-08-28T23:00:00", "circe stopped reporting")]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert ep.resolved is False
    assert ep.started_at == datetime(2026, 8, 28, 23, 0, tzinfo=timezone.utc)


def test_an_episode_that_closed_before_the_window_is_not_reported() -> None:
    items = [
        _item("node_availability:circe", "2026-08-28T10:00:00", "down"),
        {"reason": "node_availability:circe", "created_at": "2026-08-28T11:00:00",
         "severity": "info", "message": "recovered: node_availability:circe"},
    ]
    assert summarize_capability_gaps(items, window_start=W_START, window_end=W_END) == []


def test_records_after_the_window_are_ignored() -> None:
    items = [_item("vision_blind", "2026-08-31T00:00:00")]
    assert summarize_capability_gaps(items, window_start=W_START, window_end=W_END) == []


def test_truncation_keeps_the_newest_and_the_unresolved() -> None:
    """Slicing a chronologically-sorted list dropped the NEWEST episode -- the one
    most likely to still be happening. Nothing covered the cap at all."""
    items = []
    for i in range(MAX_EPISODES_IN_SEED + 3):
        items.append(_item(f"node_availability:n{i:02d}", f"2026-08-29T{7 + i:02d}:00:00"))
    episodes = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert len(episodes) == MAX_EPISODES_IN_SEED
    kept = {e.node for e in episodes}
    newest = f"n{MAX_EPISODES_IN_SEED + 2:02d}"
    oldest = "n00"
    assert newest in kept, "the newest episode must survive truncation"
    assert oldest not in kept, "the oldest is the one to drop"
    starts = [e.started_at for e in episodes]
    assert starts == sorted(starts), "display order stays chronological"


def test_producer_message_is_truncated_before_it_reaches_the_prompt() -> None:
    """The live vision_blind message is ~330 chars of docker/VRAM runbook text and
    goes verbatim into the journal prompt. Unbounded producer text has no business
    being interpolated into a prompt."""
    long_msg = "x" * 5000
    items = [_item("vision_blind", "2026-08-29T20:00:00", long_msg)]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    detail = ep.to_seed_dict()["detail"]
    assert len(detail) <= MAX_DETAIL_CHARS
    assert detail.endswith("\u2026")


def test_a_later_alert_bounds_an_earlier_gap_instead_of_leaving_it_open() -> None:
    """A later alert means the producer's watcher started over, so the earlier
    gap stops accruing -- as an UPPER bound on an unknown end, not a measured one.

    Corrected 2026-08-30: this test originally justified the bound by claiming
    vision-host must clear `_alerting` before re-arming, so alert N+1 proved gap
    N ended. Root-causing the missing `vision_recovered` records showed the arm
    state was in-memory, so a restart re-armed it with no recovery at all. The
    behaviour is unchanged and still right -- without it all nine vision_blind
    episodes since 2026-08-21 stayed permanently open (`vision_recovered`: 0
    rows, ever) and a 24h window inherited every one -- but the reason was wrong.
    """
    items = [
        _item("vision_blind", "2026-08-29T20:25:00", "blind", att_id="v1"),
        _item("vision_blind", "2026-08-29T21:00:00", "blind", att_id="v2"),
    ]
    first, second = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    d = first.to_seed_dict()
    assert d["ended_by_a_later_alert"] is True
    assert d["duration_upper_bound_minutes"] == 35.0
    assert d["duration_minutes"] is None, "an inferred end must not report a hard duration"
    assert d["resolved"] is False, "we never saw it recover; we only know it stopped"
    assert second.to_seed_dict()["ended_by_a_later_alert"] is False


def test_bounded_gaps_drop_out_of_a_later_window() -> None:
    """Real 2026-08-21 vision_blind timestamps must not surface in today's entry.

    Each is bounded by the alert that follows it. Note the LAST alert in a chain
    legitimately stays open -- with no recovery and no successor, "still absent"
    is the honest reading of that data, not a bug -- so this fixture carries the
    successor that actually exists in the live store.
    """
    items = [
        _item("vision_blind", "2026-08-21T21:13:23"),
        _item("vision_blind", "2026-08-21T21:31:13"),
        _item("vision_blind", "2026-08-26T22:58:32"),   # bounds the two above
    ]
    episodes = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    assert [e.started_at.strftime("%m-%d") for e in episodes] == ["08-26"], (
        "only the still-open tail of the chain reaches the window"
    )


def test_a_real_recovery_still_reports_a_hard_duration() -> None:
    items = [
        _item("vision_blind", "2026-08-29T20:25:00", "blind"),
        {"reason": "vision_recovered", "created_at": "2026-08-29T20:55:00",
         "severity": "info", "message": "Vision is working again on athena: 98% succeeding."},
    ]
    (ep,) = summarize_capability_gaps(items, window_start=W_START, window_end=W_END)
    d = ep.to_seed_dict()
    assert d["resolved"] is True
    assert d["duration_minutes"] == 30.0
    assert d["duration_upper_bound_minutes"] is None
