"""perception_gaps on the daily journal seed (walkway spec idea 4).

Same load-bearing guarantee as capability_gaps: a day with nothing unnamed --
or a host where the vision_unresolved migration has not been applied -- must
produce a seed byte-identical to the one before this block existed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import app.perception_gap_journal as pgj
from app.capability_gap_journal import build_daily_seed_payload
from app.perception_gap_journal import (
    MAX_GAPS_IN_SEED,
    collect_perception_gaps,
    summarize_perception_gaps,
)

T0 = datetime(2026, 9, 23, 9, 12, tzinfo=timezone.utc)


def _row(uid: str, minutes: int, **over) -> dict:
    row = {
        "unresolved_id": uid,
        "stream_id": "walkway",
        "camera_id": "walkway",
        "observed_at": T0 + timedelta(minutes=minutes),
        "reason": "no_label",
        "description": "a low shape by the fence",
        "what_was_tried": '["yolo", "council"]',
    }
    row.update(over)
    return row


def _seed(**kw) -> dict:
    return build_daily_seed_payload(
        request_date="2026-09-23",
        window_start_utc="2026-09-23T06:00:00Z",
        window_end_utc="2026-09-24T06:00:00Z",
        gaps=None,
        **kw,
    )


def test_quiet_day_seed_is_byte_identical() -> None:
    pre = json.dumps(
        {"request_date": "2026-09-23", "window_start_utc": "2026-09-23T06:00:00Z",
         "window_end_utc": "2026-09-24T06:00:00Z"},
        sort_keys=True,
    )
    for empty in (None, [], ()):
        assert json.dumps(_seed(perception_gaps=empty), sort_keys=True) == pre


def test_block_is_shaped_like_capability_gaps() -> None:
    gaps = summarize_perception_gaps([_row("u2", 5), _row("u1", 0)])
    assert [g["unresolved_id"] for g in gaps] == ["u1", "u2"]  # chronological
    assert gaps[0]["what_was_tried"] == ["yolo", "council"]
    assert gaps[0]["detail"] == "a low shape by the fence"
    payload = _seed(perception_gaps=gaps, perception_gaps_total=2)
    assert payload["perception_gaps"] == gaps
    assert "perception_gaps_omitted" not in payload


def test_cap_keeps_newest_and_discloses_the_rest() -> None:
    rows = [_row(f"u{i}", i) for i in range(MAX_GAPS_IN_SEED + 3)]
    gaps = summarize_perception_gaps(rows)
    assert len(gaps) == MAX_GAPS_IN_SEED
    assert gaps[-1]["unresolved_id"] == f"u{MAX_GAPS_IN_SEED + 2}"
    assert "u0" not in {g["unresolved_id"] for g in gaps}
    payload = _seed(perception_gaps=gaps, perception_gaps_total=len(rows))
    assert payload["perception_gaps_omitted"] == 3


def test_hollow_rows_are_skipped_not_rendered() -> None:
    rows = [_row("", 0), _row("u2", 1, description="  "), _row("u3", 2, observed_at=None), _row("u4", 3)]
    assert [g["unresolved_id"] for g in summarize_perception_gaps(rows)] == ["u4"]


def test_long_detail_is_truncated() -> None:
    (g,) = summarize_perception_gaps([_row("u1", 0, description="x" * 2000)])
    assert len(g["detail"]) <= pgj.MAX_DETAIL_CHARS


def test_unreadable_table_yields_nothing(monkeypatch) -> None:
    async def _none(*a, **k):
        return None

    monkeypatch.setattr(pgj, "fetch_rows", _none)
    got = asyncio.run(collect_perception_gaps(
        dsn="postgresql://x", window_start_utc="2026-09-23T06:00:00Z",
        window_end_utc="2026-09-24T06:00:00Z",
    ))
    assert got == ([], 0)


def test_collect_passes_the_window_and_counts(monkeypatch) -> None:
    seen = {}

    async def _rows(dsn, sql, params, *, label):
        seen.update(params)
        return [_row("u1", 0)]

    monkeypatch.setattr(pgj, "fetch_rows", _rows)
    gaps, total = asyncio.run(collect_perception_gaps(
        dsn="postgresql://x", window_start_utc="2026-09-23T06:00:00Z",
        window_end_utc="2026-09-24T06:00:00Z",
    ))
    assert total == 1 and gaps[0]["unresolved_id"] == "u1"
    assert seen["start"] == datetime(2026, 9, 23, 6, tzinfo=timezone.utc)


def test_no_dsn_reads_nothing() -> None:
    from app.vision_pg import fetch_rows

    assert asyncio.run(fetch_rows("", "SELECT 1", {}, label="t")) is None


def test_bad_dsn_is_none_not_raise() -> None:
    from app.vision_pg import fetch_rows

    got = asyncio.run(fetch_rows("postgresql://nobody@127.0.0.1:1/none", "SELECT 1", {}, label="t"))
    assert got is None


def test_omitted_uses_window_total_and_ignores_hollow_rows(monkeypatch) -> None:
    rows = [dict(_row(f"u{i}", i), window_total=500) for i in range(20)]
    rows.append(dict(_row("h", 30, description=""), window_total=500))

    async def _rows(dsn, sql, params, *, label):
        return rows

    monkeypatch.setattr(pgj, "fetch_rows", _rows)
    gaps, total = asyncio.run(collect_perception_gaps(
        dsn="x", window_start_utc="2026-09-23T06:00:00Z", window_end_utc="2026-09-24T06:00:00Z",
    ))
    assert len(gaps) == MAX_GAPS_IN_SEED
    assert total == 499  # 500 in the window, one of the read rows was hollow
