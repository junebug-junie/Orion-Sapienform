"""The Temporal Self chronicle: runs the pure chronology reducer (``orion/temporal_self/``) live.

Called by the ``chronicle`` node of the ``temporal_self.update`` thread (after ``regulate``) on
every step. Each call reads forward from the stored watermark to ``now - read_lag`` in windows of
at most an hour, and for each window:

1. reads every bound source (``temporal_self_sources.SourceReader``): broadcast ticks with
   ``generated_at`` in ``[lo, hi)`` and every event whose available time is in ``[lo, hi)``;
2. probes ``[lo - late_probe, lo)`` for rows that were not there when that window was read
   (absent from ``temporal_self_event``): late rows. They go into the same ``fold`` call, where the
   reducer counts them (``skipped_at_or_before_watermark``) and never folds them, and they are
   stored with ``late_unfolded = true`` so each is counted exactly once;
3. ONE ``fold`` with ticks and events together, ``advance_clock(hi)``, ``build_frame(hi)``,
   ``drain_closed_days`` (the reducer's documented contract);
4. reads the body sensors once per newly closed non-reverie arc (``body.body_window``) and sets
   ``arc.body`` (``summarize_body``);
5. commits events, changed arcs, closed days, the frame, cursors and the reducer state in one
   transaction (``ChronicleStore.commit_window``). Only then does the in-memory state advance, so a
   failed commit re-reads the same window next step: no double fold, no dropped row.

No LLM, no bus publish, no consumer (patch 4 adds the stance cue and the metacog grounding).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

from orion.schemas.temporal_self import TemporalSelfArcV1, TemporalSelfEventV1, TemporalSelfFrameV1, TemporalSelfStateV1
from orion.temporal_self import ReducerConfig, advance_clock, build_frame, drain_closed_days, fold, initial_state
from orion.temporal_self.body import NO_BODY_KINDS, body_window, summarize_body
from orion.temporal_self.day import day_id_for, day_window

from app.temporal_self_store import StaleWriterError, WindowWrite

logger = logging.getLogger("orion-durable-runs.temporal_self_chronicle")

MAX_WINDOW_SEC = 3600.0       # one window reads at most an hour of rows
MAX_WINDOWS_PER_STEP = 48     # first-boot backfill (~1.5 days) finishes in one step
LATE_PROBE_SEC = 1800.0       # rows written up to 30 min after their window was read are counted
# episode_memory rows land a median 13 h after occurred_at (live 10-10): probed 3 days back.
LATE_PROBE_OVERRIDES = {"memory_episode": 3 * 86400.0}
BODY_GROUP_SPAN = timedelta(hours=6)  # one body read covers arcs within six hours of each other
RETENTION_EVERY = timedelta(hours=6)


@dataclass(frozen=True)
class ChronicleConfig:
    reducer: ReducerConfig = field(default_factory=ReducerConfig)
    read_lag_sec: float = 300.0
    backfill_days: int = 1
    event_retention_days: int = 30
    arc_retention_days: int = 90
    day_retention_days: int = 365
    max_window_sec: float = MAX_WINDOW_SEC
    max_windows_per_step: int = MAX_WINDOWS_PER_STEP
    late_probe_sec: float = LATE_PROBE_SEC


def _changed(prev: TemporalSelfStateV1, new: TemporalSelfStateV1) -> list[TemporalSelfArcV1]:
    return [a for k, a in new.arcs.items() if prev.arcs.get(k) != a]


def _needs_body(a: TemporalSelfArcV1) -> bool:
    return a.status == "closed" and a.body is None and a.kind not in NO_BODY_KINDS


class Chronicler:
    def __init__(self, *, store: Any, reader: Any, cfg: ChronicleConfig, now: Callable[[], datetime]) -> None:
        self._store = store
        self._reader = reader
        self._cfg = cfg
        self._now = now
        self._loaded = False
        self._state: Optional[TemporalSelfStateV1] = None
        self._watermark: Optional[datetime] = None
        self._origin: Optional[datetime] = None
        self._stored_watermark: Optional[datetime] = None
        self._last_retention: Optional[datetime] = None
        self.frame: Optional[TemporalSelfFrameV1] = None
        self.steps = 0
        self.windows = 0
        self.late_total = 0
        self.days_closed: list[str] = []
        self.reset_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_retention: Optional[dict] = None

    @property
    def state(self) -> Optional[TemporalSelfStateV1]:
        return self._state

    @property
    def watermark(self) -> Optional[datetime]:
        return self._watermark

    # --- load ---------------------------------------------------------------------------------

    def _start_of(self, local: date) -> datetime:
        return day_window(local.isoformat(), self._cfg.reducer.tz_name)[0]

    async def _load(self, now: datetime) -> None:
        loaded = await self._store.load_state()
        self._stored_watermark = loaded.watermark
        if loaded.state is not None and loaded.watermark is not None:
            self._state, self._watermark = loaded.state, loaded.watermark
            self._origin = loaded.origin or loaded.watermark
        else:
            if loaded.watermark is not None:
                # A stored state that no longer validates: re-fold that whole local day from its
                # midnight (arc ids are deterministic, so rows already written are overwritten
                # with the same arcs), and say so.
                self.reset_reason = f"stored state invalid: {loaded.error}"
                start = self._start_of(date.fromisoformat(day_id_for(loaded.watermark, self._cfg.reducer.tz_name)))
            else:
                today = date.fromisoformat(day_id_for(now, self._cfg.reducer.tz_name))
                start = self._start_of(today - timedelta(days=self._cfg.backfill_days))
            self._state, self._watermark, self._origin = initial_state(), start, start
        self._loaded = True

    # --- one step -----------------------------------------------------------------------------

    async def step(self) -> dict:
        now = self._now()
        try:
            if not self._loaded:
                await self._load(now)
            target = now - timedelta(seconds=self._cfg.read_lag_sec)
            n = 0
            while self._watermark < target and n < self._cfg.max_windows_per_step:
                hi = min(target, self._watermark + timedelta(seconds=self._cfg.max_window_sec))
                await self._window(self._watermark, hi, now)
                n += 1
            await self._maybe_retention(now)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - the window is retried next step from the stored state
            self.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning("temporal_self_chronicle_failed error=%s", self.last_error, exc_info=True)
            if isinstance(exc, StaleWriterError):
                self._loaded = False  # another writer moved the watermark: reload before reading again
            n = -1
        self.steps += 1
        return self.summary(now, windows=n)

    async def _late(self, lo: datetime) -> list[TemporalSelfEventV1]:
        assert self._origin is not None
        found: dict[str, TemporalSelfEventV1] = {}
        plo = max(lo - timedelta(seconds=self._cfg.late_probe_sec), self._origin)
        if plo < lo:
            _, cands = await self._reader.read(plo, lo, ticks=False)
            found.update((e.event_id, e) for e in cands)
        for kind, sec in LATE_PROBE_OVERRIDES.items():
            klo = max(lo - timedelta(seconds=sec), self._origin)
            if klo < plo:
                _, cands = await self._reader.read(klo, plo, ticks=False, kinds=[kind])
                found.update((e.event_id, e) for e in cands)
        if not found:
            return []
        unseen = await self._store.unseen(found)
        return [found[i] for i in sorted(unseen)]

    async def _window(self, lo: datetime, hi: datetime, now: datetime) -> None:
        cfg = self._cfg.reducer
        prev = self._state
        assert prev is not None
        ticks, events = await self._reader.read(lo, hi)
        late = await self._late(lo)
        s = fold(prev, ticks, events + late, cfg)
        s = advance_clock(s, hi, cfg)
        frame = build_frame(s, hi, cfg)
        s, days = drain_closed_days(s)
        arcs: dict[str, TemporalSelfArcV1] = {a.arc_id: a for a in _changed(prev, s)}
        for d in days:
            arcs.update((a.arc_id, a) for a in d.arcs)
        await self._attach_bodies([a for a in arcs.values() if _needs_body(a)], events)
        await self._store.commit_window(WindowWrite(
            state=s, watermark=hi, frame=frame, events=events, late_events=late, arcs=list(arcs.values()),
            days=days, now=now, origin=self._origin, expected_prev=self._stored_watermark))
        self._state, self._watermark, self._stored_watermark = s, hi, hi
        self.frame = frame
        self.windows += 1
        self.late_total += len(late)
        for d in days:
            self.days_closed.append(d.day_id)
            logger.info("temporal_self_day_closed day=%s arcs=%d", d.day_id, len(d.arcs))
        if late:
            logger.info("temporal_self_late_rows window_lo=%s count=%d kinds=%s", lo.isoformat(), len(late),
                        ",".join(sorted({e.source_kind for e in late})))

    async def _attach_bodies(self, arcs: list[TemporalSelfArcV1], window_events: list[TemporalSelfEventV1]) -> None:
        """One read per group of nearby arcs; ``body`` is set in place (on the state's arc or the
        closed day's copy, whichever is persisted)."""
        if not arcs:
            return
        spans = sorted(((body_window(a), a) for a in arcs), key=lambda x: (x[0][0], x[1].arc_id))
        groups: list[tuple[list[datetime], list]] = []
        for span, a in spans:
            if groups and span[0] - groups[-1][0][0] <= BODY_GROUP_SPAN:
                groups[-1][0][1] = max(groups[-1][0][1], span[1])
                groups[-1][1].append((span, a))
            else:
                groups.append(([span[0], span[1]], [(span, a)]))
        for (g0, g1), members in groups:
            rows = await self._reader.read_body(g0, g1)
            deferrals = {e.event_id: e for e in await self._store.deferrals(g0, g1)}
            deferrals.update((e.event_id, e) for e in window_events if e.source_kind == "visual_deferral")
            for (b0, b1), a in members:
                a.body = summarize_body(
                    [r for r in rows["cluster"] if b0 <= r["_t"] <= b1],
                    [r for r in rows["cabinet"] if b0 <= r["_t"] <= b1],
                    [r for r in rows["spike"] if b0 <= r["_t"] <= b1],
                    [e for e in deferrals.values() if b0 <= e.occurred_at <= b1],
                )

    async def _maybe_retention(self, now: datetime) -> None:
        if self._last_retention is not None and now - self._last_retention < RETENTION_EVERY:
            return
        self._last_retention = now
        try:
            self.last_retention = await self._store.retention(
                now, event_days=self._cfg.event_retention_days, arc_days=self._cfg.arc_retention_days,
                day_days=self._cfg.day_retention_days)
            if any(self.last_retention.values()):
                logger.info("temporal_self_retention deleted=%s", self.last_retention)
        except Exception:  # noqa: BLE001 - retention retries in RETENTION_EVERY; the chronicle goes on
            logger.warning("temporal_self_retention_failed", exc_info=True)

    # --- reporting ----------------------------------------------------------------------------

    def summary(self, now: datetime, *, windows: int) -> dict:
        wm = self._watermark
        f = self.frame
        return {
            "watermark": wm.isoformat() if wm else None,
            "lag_sec": round((now - wm).total_seconds(), 1) if wm else None,
            "windows": windows,
            "day_id": f.day_id if f else None,
            "active_arc": f.active_arc.arc_id if f and f.active_arc else None,
            "arcs_today": f.arcs_today_total if f else None,
            "skipped_today": f.skipped_at_or_before_watermark if f else None,
            "error": self.last_error,
        }

    def health(self) -> dict:
        now = self._now().astimezone(timezone.utc)
        return dict(self.summary(now, windows=self.windows), steps=self.steps, windows_total=self.windows,
                    late_total=self.late_total, days_closed=self.days_closed[-7:], reset_reason=self.reset_reason,
                    origin=self._origin.isoformat() if self._origin else None, last_retention=self.last_retention)
