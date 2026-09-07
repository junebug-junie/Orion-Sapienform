from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.substrate.recent_attention_cue import build_recent_attention_cue

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)


def _row(process: str, narrative: str, age_sec: float) -> dict:
    return {
        "process": process,
        "reason_narrative": narrative,
        "generated_at": NOW - timedelta(seconds=age_sec),
    }


def test_empty_rows_is_stale() -> None:
    cue = build_recent_attention_cue([], now=NOW)
    assert cue["items"] == []
    assert cue["stale"] is True
    assert cue["as_of"] == NOW.isoformat()


def test_fresh_row_is_moments_ago_and_not_stale() -> None:
    rows = [_row("cortex_turn", "Watching the chat turn unfold.", 5.0)]
    cue = build_recent_attention_cue(rows, now=NOW)
    assert cue["stale"] is False
    assert len(cue["items"]) == 1
    item = cue["items"][0]
    assert item["process"] == "cortex_turn"
    assert item["narrative"] == "Watching the chat turn unfold."
    assert item["age_label"] == "moments ago"


def test_only_old_row_is_stale() -> None:
    rows = [_row("substrate_attention", "Idle tick.", 20 * 60)]
    cue = build_recent_attention_cue(rows, now=NOW, stale_after_sec=900.0)
    assert cue["stale"] is True
    assert cue["items"][0]["age_label"] == "about 20 minutes ago"


def test_caps_to_limit_and_keeps_newest_first() -> None:
    rows = [
        _row("cortex_turn", "oldest", 400),
        _row("curiosity", "middle", 200),
        _row("reverie", "newest", 10),
        _row("durable_run", "dropped", 500),
    ]
    cue = build_recent_attention_cue(rows, now=NOW, limit=3)
    assert len(cue["items"]) == 3
    assert [item["process"] for item in cue["items"]] == [
        "reverie",
        "curiosity",
        "cortex_turn",
    ]


def test_malformed_row_is_dropped_not_a_crash() -> None:
    rows = [
        {"process": "cortex_turn", "generated_at": NOW},  # missing reason_narrative
        _row("reverie", "a real one", 5.0),
    ]
    cue = build_recent_attention_cue(rows, now=NOW)
    assert len(cue["items"]) == 1
    assert cue["items"][0]["process"] == "reverie"
    assert cue["stale"] is False


def test_generated_at_wrong_type_is_dropped_not_a_crash() -> None:
    """A stringified timestamp (e.g. a JSON round-trip upstream) must be
    dropped like any other malformed row, not raise while comparing it
    against `now`."""
    rows = [
        {
            "process": "cortex_turn",
            "reason_narrative": "not a real datetime",
            "generated_at": NOW.isoformat(),
        },
        _row("reverie", "a real one", 5.0),
    ]
    cue = build_recent_attention_cue(rows, now=NOW)
    assert len(cue["items"]) == 1
    assert cue["items"][0]["process"] == "reverie"


def test_naive_datetime_is_treated_as_utc() -> None:
    """The live column is DateTime(timezone=True), so this is a defensive
    fallback rather than an expected input -- but it must not crash, and it
    must not silently misjudge the row's age by comparing naive vs aware."""
    naive_now = NOW.replace(tzinfo=None)
    rows = [
        {
            "process": "cortex_turn",
            "reason_narrative": "naive timestamp",
            "generated_at": naive_now - timedelta(seconds=5),
        }
    ]
    cue = build_recent_attention_cue(rows, now=NOW)
    assert len(cue["items"]) == 1
    assert cue["items"][0]["age_label"] == "moments ago"
    assert cue["stale"] is False


def test_hour_and_day_age_buckets() -> None:
    rows = [_row("cortex_turn", "a while back", 2 * 3600)]
    cue = build_recent_attention_cue(rows, now=NOW)
    assert cue["items"][0]["age_label"] == "about 2 hours ago"

    rows = [_row("cortex_turn", "ages ago", 2 * 86400)]
    cue = build_recent_attention_cue(rows, now=NOW, stale_after_sec=10**9)
    assert cue["items"][0]["age_label"] == "more than a day ago"
