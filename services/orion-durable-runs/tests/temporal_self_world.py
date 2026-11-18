"""In-memory doubles for the Temporal Self chronicle tests: a source world whose rows become
visible at a chosen wall time (so late writers can be simulated), a reader with the same window
contract as ``SourceReader``, and a store with the same transaction semantics as
``ChronicleStore`` (stale-writer check, all-or-nothing commit, gzipped state round trip)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from orion.schemas.temporal_self import TemporalSelfEventV1
from orion.temporal_self.broadcast import tick_from_log_row
from orion.temporal_self.day import as_utc
from orion.temporal_self.sources import ADAPTERS

from app.temporal_self_sources import available_at
from app.temporal_self_store import LoadedState, StaleWriterError, WindowWrite, pack_state, unpack_state

TZ = "America/Denver"


def broadcast_row(i: int, at: datetime, ref: Optional[str]) -> dict:
    loops = [{"id": f"loop-{ref}", "source_refs": [ref], "description": f"about {ref}"}] if ref else []
    return {"log_id": f"bc-{i:05d}", "generated_at": at,
            "projection_json": {"selected_open_loop_id": f"loop-{ref}" if ref else None, "frame": {"open_loops": loops}}}


@dataclass
class Row:
    kind: str
    row: dict
    visible_at: datetime


class World:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.rows: list[Row] = []
        self.body: dict[str, list[dict]] = {"cluster": [], "cabinet": [], "spike": []}

    def add(self, kind: str, row: dict, *, delay: timedelta = timedelta(seconds=1)) -> None:
        """Visible ``delay`` after the row's own available time (the reducer's order key)."""
        if kind == "broadcast":
            at = as_utc(row["generated_at"])
        else:
            e = ADAPTERS[kind](row, TZ)
            at = available_at(e) if e is not None else as_utc(next(v for v in row.values() if isinstance(v, datetime)))
        self.rows.append(Row(kind, row, at + delay))

    def ticks(self, start: datetime, refs: Iterable[Optional[str]], every: float = 37.0, first: int = 0) -> datetime:
        t = start
        for i, ref in enumerate(refs):
            self.add("broadcast", broadcast_row(first + i, t, ref))
            t += timedelta(seconds=every)
        return t


class FakeReader:
    def __init__(self, world: World, tz: str = TZ) -> None:
        self.world = world
        self.tz = tz
        self.reads: list[tuple[datetime, datetime, bool]] = []

    async def read(self, lo, hi, *, ticks=True, kinds=None):
        self.reads.append((lo, hi, ticks))
        visible = [r for r in self.world.rows if r.visible_at <= self.world.now]
        out_ticks = []
        if ticks:
            out_ticks = sorted((tick_from_log_row(r.row) for r in visible if r.kind == "broadcast"
                                and lo <= as_utc(r.row["generated_at"]) < hi), key=lambda t: t.log_id)
        events = []
        for r in visible:
            if r.kind == "broadcast" or (kinds is not None and r.kind not in kinds):
                continue
            e = ADAPTERS[r.kind](r.row, self.tz)
            if e is not None and lo <= available_at(e) < hi:
                events.append(e)
        return out_ticks, events

    async def read_body(self, lo, hi):
        return {k: [dict(r, _t=as_utc(r.get("observed_at") or r.get("timestamp"))) for r in v
                    if lo <= as_utc(r.get("observed_at") or r.get("timestamp")) <= hi]
                for k, v in self.world.body.items()}


@dataclass
class FakeStore:
    state_row: Optional[tuple] = None   # (watermark, origin, gz)
    events: dict = field(default_factory=dict)   # event_id -> (event, late)
    arcs: dict = field(default_factory=dict)     # arc_id -> arc json
    days: dict = field(default_factory=dict)
    frame: Optional[str] = None
    cursors: dict = field(default_factory=dict)
    commits: int = 0
    fail_next: int = 0
    retention_calls: list = field(default_factory=list)

    async def load_state(self) -> LoadedState:
        if self.state_row is None:
            return LoadedState(None, None)
        wm, origin, gz = self.state_row
        try:
            return LoadedState(unpack_state(gz), wm, origin)
        except Exception as exc:  # noqa: BLE001
            return LoadedState(None, wm, origin, error=type(exc).__name__)

    async def stored_available(self, ids):
        return {i: available_at(self.events[i][0]) for i in ids if i in self.events}

    async def deferrals(self, lo, hi):
        return [e for e, _ in self.events.values() if e.source_kind == "visual_deferral" and lo <= e.occurred_at <= hi]

    async def commit_window(self, w: WindowWrite) -> None:
        stored = self.state_row[0] if self.state_row else None
        if stored != w.expected_prev:
            raise StaleWriterError(f"{stored} != {w.expected_prev}")
        if self.fail_next:
            self.fail_next -= 1
            raise ConnectionError("commit failed")
        for e in w.events:   # ON CONFLICT: keep the stored row, only clear a late flag
            prev = self.events.get(e.event_id)
            self.events[e.event_id] = (e, False) if prev is None else (prev[0], False)
        for e in w.late_events:
            self.events.setdefault(e.event_id, (e, True))
        for a in w.arcs:
            self.arcs[a.arc_id] = a.model_dump_json()
        for d in w.days:
            self.days[d.day_id] = d.model_dump_json()
        self.frame = w.frame.model_dump_json()
        self.cursors = dict(w.state.cursors, read_watermark=w.watermark.isoformat())
        self.state_row = (w.watermark, w.origin or w.watermark, pack_state(w.state))
        self.commits += 1

    async def retention(self, now, *, event_days, arc_days, day_days):
        self.retention_calls.append(now)
        cut = now - timedelta(days=event_days)
        gone = [k for k, (e, _) in self.events.items() if e.occurred_at < cut]
        for k in gone:
            del self.events[k]
        return {"event": len(gone), "arc": 0, "day": 0}


def utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)


def event_ids(store: FakeStore, *, late: bool) -> set[str]:
    return {k for k, (_, is_late) in store.events.items() if is_late == late}


def stored_events(store: FakeStore) -> list[TemporalSelfEventV1]:
    return [e for e, _ in store.events.values()]
