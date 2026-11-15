"""#3403: an expectation is judged against the room captions of its whole check window.

Before: the scorer asked for one room caption from the last 180 s before the scoring tick. A tick
scores one row and runs a median 674 s after the window closed (live, 8 days to 2026-11-15), so
most rows found none and were written ``unscored`` with no judge call: 2 real verdicts of 332 in
6 days, while 487 of 487 windows held at least one caption.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jinja2 import Environment

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JUDGE_PROMPT = _REPO_ROOT / "orion" / "cognition" / "prompts" / "reverie_expectation_judge.j2"
_NARRATE_PROMPT = _REPO_ROOT / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"


class Room:
    """The room camera's caption table, read both the old way (newest within max_age of now) and
    the new way (between two row times), so the same world runs on either scorer."""

    def __init__(self, captions: list[tuple[datetime, str]]) -> None:
        self.captions = sorted(captions)
        self.window_reads: list[dict] = []

    def recent(self, *, max_age_sec, limit, stream_ids, **_):
        now = datetime.now(timezone.utc)
        rows = [(t, n) for t, n in self.captions if 0 <= (now - t).total_seconds() <= max_age_sec]
        return [{"narrative": n, "age_sec": (now - t).total_seconds()} for t, n in reversed(rows)][:limit]

    def between(self, *, start, end, limit, stream_ids, **_):
        self.window_reads.append({"start": start, "end": end, "limit": limit, "stream_ids": list(stream_ids)})
        return [{"narrative": n, "created_at": t} for t, n in self.captions if start <= t <= end][:limit]


def _wire(monkeypatch, room: Room, pending: dict, verdict: str = "confirmed"):
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_expectation_scoring_enabled", True)
    monkeypatch.setattr(reverie, "load_pending_expectations", lambda limit: [pending])
    monkeypatch.setattr(vision_reader, "read_recent_vision_events", room.recent)
    monkeypatch.setattr(vision_reader, "read_vision_events_between", room.between, raising=False)
    persisted: list[tuple] = []
    monkeypatch.setattr(reverie, "persist_expectation_verdict", lambda tid, v, ts: persisted.append((tid, v, ts)))
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"final_text": json.dumps({"verdict": verdict})})
    return reverie, cortex, persisted


def _pending(created: datetime, **over) -> dict:
    row = {"thought_id": "t1", "expectation": "the door will still be closed", "created_at": created,
           "expectation_checkable_by": created + timedelta(seconds=1800)}
    row.update(over)
    return row


def _late_tick_start() -> datetime:
    """The expectation's creation time for a scoring tick that runs now, 674 s after the window closed."""
    return datetime.now(timezone.utc) - timedelta(seconds=1800 + 674)


@pytest.mark.asyncio
async def test_a_caption_ten_minutes_into_the_window_gets_a_real_verdict(monkeypatch):
    """The ticket's case. The only room caption is at +600 s; the tick scores the row 674 s after
    the window closed, so nothing is within 180 s of now. The judge is still asked, and answers."""
    t0 = _late_tick_start()
    room = Room([(t0 + timedelta(seconds=600), "the door is closed and the chair is empty")])
    reverie, cortex, persisted = _wire(monkeypatch, room, _pending(t0))
    await reverie._maybe_score_pending_expectation(AsyncMock(), cortex_client=cortex)
    cortex.execute_plan.assert_awaited_once()
    assert [(tid, v) for tid, v, _ in persisted] == [("t1", "confirmed")]
    ctx = cortex.execute_plan.call_args.kwargs["req"].context
    assert ctx["percept_narrative"] == "+10:00  the door is closed and the chair is empty"
    assert ctx["window_minutes"] == 30 and ctx["options"]["llm_lane"] == "background"


@pytest.mark.asyncio
async def test_the_window_is_the_rows_own_times_not_the_ticks_clock(monkeypatch):
    """Captions before the claim and after its check time are not evidence, however fresh they
    are when the tick runs. A naive database timestamp is read as UTC."""
    t0 = _late_tick_start()
    room = Room([(t0 - timedelta(seconds=30), "before the claim"),
                 (t0 + timedelta(seconds=1830), "after the window"),
                 (datetime.now(timezone.utc) - timedelta(seconds=5), "five seconds ago")])
    naive = _pending(t0.replace(tzinfo=None), expectation_checkable_by=(t0 + timedelta(seconds=1800)).replace(tzinfo=None))
    reverie, cortex, persisted = _wire(monkeypatch, room, naive)
    await reverie._maybe_score_pending_expectation(AsyncMock(), cortex_client=cortex)
    cortex.execute_plan.assert_not_awaited()
    assert [(tid, v) for tid, v, _ in persisted] == [("t1", "unscored")]
    (read,) = room.window_reads
    assert read["start"] == t0 and read["end"] == t0 + timedelta(seconds=1800)
    assert read["limit"] == reverie.EXPECTATION_JUDGE_MAX_READ


@pytest.mark.asyncio
async def test_the_verdict_time_is_taken_after_the_read(monkeypatch):
    """A slow read must not stamp the verdict before the evidence was read (clock after the read)."""
    t0 = _late_tick_start()
    room = Room([(t0 + timedelta(seconds=60), "the door is closed")])
    seen: list[datetime] = []
    inner = room.between

    def slow(**kw):
        out = inner(**kw)
        seen.append(datetime.now(timezone.utc))
        return out

    room.between = slow
    reverie, cortex, persisted = _wire(monkeypatch, room, _pending(t0))
    await reverie._maybe_score_pending_expectation(AsyncMock(), cortex_client=cortex)
    assert persisted and persisted[0][2] >= seen[0]


@pytest.mark.asyncio
async def test_a_row_without_created_at_opens_its_window_one_check_window_before_the_end(monkeypatch):
    t0 = _late_tick_start()
    room = Room([(t0 + timedelta(seconds=900), "the door is closed")])
    row = _pending(t0)
    del row["created_at"]
    reverie, cortex, persisted = _wire(monkeypatch, room, row, verdict="disconfirmed")
    await reverie._maybe_score_pending_expectation(AsyncMock(), cortex_client=cortex)
    assert room.window_reads[0]["start"] == t0
    assert [(tid, v) for tid, v, _ in persisted] == [("t1", "disconfirmed")]
    assert cortex.execute_plan.call_args.kwargs["req"].context["percept_narrative"].startswith("+15:00  ")


@pytest.mark.asyncio
async def test_a_busy_window_stays_a_short_prompt(monkeypatch):
    """The live worst case in 8 days was 47 captions (4,359 chars) in one window."""
    t0 = _late_tick_start()
    room = Room([(t0 + timedelta(seconds=30 * i), f"person {i} walks past the desk towards the far door of the office")
                 for i in range(1, 48)])
    reverie, cortex, _ = _wire(monkeypatch, room, _pending(t0))
    await reverie._maybe_score_pending_expectation(AsyncMock(), cortex_client=cortex)
    text = cortex.execute_plan.call_args.kwargs["req"].context["percept_narrative"]
    lines = text.splitlines()
    assert len(text) <= reverie.EXPECTATION_JUDGE_MAX_CHARS + 60
    assert len(lines) <= reverie.EXPECTATION_JUDGE_MAX_LINES + 1
    assert lines[0].startswith("+00:30  person 1 ") and lines[-2].startswith("+23:30  person 47 ")
    assert lines[-1] == f"({len(lines) - 1} of 47 lines shown, evenly spaced)"


def test_a_run_of_the_same_caption_is_one_line_and_long_captions_are_clipped():
    from app import reverie

    t0 = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
    caps = [{"narrative": "the  door is\nclosed", "created_at": t0 + timedelta(seconds=10)},
            {"narrative": "the door is closed", "created_at": t0 + timedelta(seconds=400)},
            {"narrative": "the door is closed", "created_at": t0 + timedelta(seconds=760)},
            {"narrative": "x" * 900, "created_at": t0 + timedelta(seconds=1700)},
            {"narrative": "   ", "created_at": t0 + timedelta(seconds=1750)},
            {"narrative": "no time", "created_at": None}]
    out = reverie.render_window_captions(caps, start=t0).splitlines()
    assert out[0] == "+00:10 to +12:40 (3 captions, all the same)  the door is closed"
    assert out[1] == "+28:20  " + "x" * reverie.EXPECTATION_JUDGE_CAPTION_CHARS and len(out) == 2
    assert reverie.render_window_captions([], start=t0) == ""


def test_window_reader_reads_room_cameras_between_the_two_times_oldest_first():
    from app import vision_reader

    t0 = datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc)
    conn = MagicMock()
    conn.execute.return_value.mappings.return_value.all.return_value = [
        {"narrative": " the door is closed ", "created_at": (t0 + timedelta(seconds=600)).replace(tzinfo=None)},
        {"narrative": "  ", "created_at": t0},
    ]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    with patch.object(vision_reader, "_get_engine", return_value=engine):
        out = vision_reader.read_vision_events_between(start=t0, end=t0 + timedelta(seconds=1800), limit=200,
                                                       stream_ids=["carbon", "cam0"])
    assert out == [{"narrative": "the door is closed", "created_at": t0 + timedelta(seconds=600)}]
    sql, params = str(conn.execute.call_args.args[0]), conn.execute.call_args.args[1]
    assert "created_at >= :start AND created_at <= :end" in sql and "ORDER BY created_at ASC" in sql
    assert "stream_id IN" in sql and "now()" not in sql.lower()
    assert params["start"] == t0 and params["end"] == t0 + timedelta(seconds=1800)
    assert params["stream_ids"] == ["carbon", "cam0"] and params["limit"] == 200
    # refuses a backwards window and a failing database without raising
    assert vision_reader.read_vision_events_between(start=t0, end=t0 - timedelta(seconds=1), limit=5, stream_ids=["cam0"]) == []
    with patch.object(vision_reader, "_get_engine", side_effect=RuntimeError("db down")):
        assert vision_reader.read_vision_events_between(start=t0, end=t0, limit=5, stream_ids=["cam0"]) == []


def test_pending_rows_carry_the_time_the_window_opened():
    from app import store

    conn = MagicMock()
    conn.execute.return_value.mappings.return_value.all.return_value = []
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = conn
    with patch.object(store, "_get_expectation_read_engine", return_value=engine):
        store.load_pending_expectations(limit=1)
    assert "created_at" in str(conn.execute.call_args.args[0]).split("FROM")[0]


def test_judge_prompt_shows_the_window_and_says_what_the_room_camera_cannot_show():
    out = Environment(autoescape=False).from_string(_JUDGE_PROMPT.read_text(encoding="utf-8")).render(
        expectation="the door will still be closed", window_minutes=30,
        percept_narrative="+10:00  the door is closed\n+25:10  the door is open")
    assert "in the 30 minutes after the claim" in out
    assert "+10:00  the door is closed\n+25:10  the door is open" in out
    assert "at least one caption clearly contradicts" in out
    assert "sensor reading" in out and "internal state" in out
    assert len(out) < 2500  # the fixed part stays small; the captions add at most ~3,000 chars


def test_narrate_prompt_no_longer_asks_for_claims_the_room_camera_cannot_check():
    """The smaller honest change for body and sensor expectations (#3403): stop asking for them."""
    from app import reverie
    from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1

    loop = OpenLoopV1(id="ol-1", description="d", salience=0.8, novelty=0.5, source_refs=["n-1"])
    broadcast = AttentionBroadcastProjectionV1(
        projection_id="p-1", generated_at=datetime(2026, 11, 13, 18, 0, tzinfo=timezone.utc),
        frame=AttentionFrameV1(open_loops=[loop]), selected_open_loop_id="ol-1", attended_node_ids=["n-1"])
    ctx = reverie.build_reverie_context(broadcast, recent_percepts=[{"narrative": "the door is closed", "age_sec": 40.0}])
    out = Environment(autoescape=False).from_string(_NARRATE_PROMPT.read_text(encoding="utf-8")).render(**ctx)
    assert "room camera will still show over the next 30 minutes" in " ".join(out.split())
    flat = " ".join(out.split())
    assert "not a temperature or other sensor reading" in flat and "leave it null" in flat
