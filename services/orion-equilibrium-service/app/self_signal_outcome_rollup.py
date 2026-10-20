"""Hourly self-signal -> outcome rollup (self-calibration spec section E, "what it tends to lead to").

Adapter around the pure reducer ``orion/attention/self_signal_outcomes.py``. Once per completed UTC
hour (plus ``grace_sec`` so the hour's harness turns and transport_baseline_hourly rows have
landed) it:

1. classifies every internal body signal's hour (high / normal / unknown) from the same 7-day
   ``NodePeHistoryCache`` the self_signal_episode gate already keeps;
2. counts that hour's outcomes from rows Orion already records (``harness_turn_trace``,
   ``transport_baseline_hourly``);
3. publishes one ``SelfSignalHourV1`` per hour (sql-writer -> ``self_signal_hour``), which outlives
   the PE history's 7-day prune, and then one ``SelfSignalOutcomeAssociationV1`` per
   (signal, outcome) computed over every persisted hour (sql-writer ->
   ``self_signal_outcome_association``).

Read failures make the cycle do nothing (retried next poll): an unread hour is never written as
calm, and an unread persisted set never triggers a recompute of hours whose PE history has already
been pruned. The latest associations stay in memory for the episode gate's open-row evidence,
with the last ``consecutive_holds`` hourly computations per pair (seeded from
``self_signal_outcome_association`` at start): an open row carries a clause only once its verdict
held that many hours in a row -- the same rule orion-thought's self-readings reader applies
(``orion.attention.self_signal_outcomes.held_latest_by_pair``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping

from orion.attention.self_signal_outcomes import (
    ASSOCIATION_COLUMNS,
    DEFAULT_CONSECUTIVE_HOLDS,
    HOUR,
    MAX_HOURS,
    WINDOW_7D,
    associate_all,
    association_history_limit,
    association_history_sql,
    best_per_signal,
    build_hour_row,
    classify_signal_hours,
    held_latest_by_pair,
    hour_floor,
    rpc_outcome_hours,
    turn_outcome_hours,
)
from orion.schemas.self_signal_outcome import SelfSignalHourV1, SelfSignalOutcomeAssociationV1

from .pe_history_reader import NodePeHistoryReader, NodePeHistoryReadError

logger = logging.getLogger("orion.equilibrium.self_signal_outcome_rollup")

# Hours computed in one cycle (a cold start backfills at most the 7-day PE window).
MAX_HOURS_PER_CYCLE = 24 * 7
_MEASURABLE_KINDS = frozenset({"level", "count", "trigger", "binary"})


def body_signal_ids(node_ids: Iterable[str]) -> list[str]:
    """Internal node:substrate.* signals whose stored value is a measurement (placeholders such as
    harness_closure, and world signals such as perception, are left out)."""
    from orion.attention.world_first import (
        SUBSTRATE_NODE_PREFIX,
        node_prediction_error_semantics,
        source_kind_for_node,
    )

    out = []
    for nid in sorted(set(node_ids)):
        if not nid.startswith(SUBSTRATE_NODE_PREFIX) or source_kind_for_node(nid) != "internal":
            continue
        kind, _ = node_prediction_error_semantics(nid)
        if kind in _MEASURABLE_KINDS:
            out.append(nid)
    return out


class SelfSignalOutcomeReader(NodePeHistoryReader):
    """The PE history reader plus the three read-only queries the rollup needs. Raises
    ``NodePeHistoryReadError`` on any failure (the caller skips the cycle)."""

    def _rows(self, sql: str, params: tuple) -> list[tuple]:
        conn = self._get_conn()
        if conn is None:
            raise NodePeHistoryReadError("postgres unavailable")
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except Exception as exc:
            self._log_failure("self_signal_outcome_reader_query_failed")
            self.close()
            raise NodePeHistoryReadError(type(exc).__name__) from exc
        self._consecutive_failures = 0
        return list(rows)

    def fetch_turn_rows(self, start: datetime, end: datetime) -> list[tuple[datetime, str | None, bool | None]]:
        # harness_turn_trace.created_at is timestamp WITHOUT time zone written in UTC (the DB
        # timezone is Etc/UTC); the bounds are passed naive-UTC to match.
        rows = self._rows(
            """
            SELECT created_at,
                   run_artifact->>'compliance_verdict',
                   CASE WHEN run_artifact ? 'response_repair_ran'
                        THEN (run_artifact->>'response_repair_ran')::boolean END
            FROM harness_turn_trace
            WHERE created_at >= %s AND created_at < %s AND run_artifact IS NOT NULL
            """,
            (_naive_utc(start), _naive_utc(end)),
        )
        return [(_aware(r[0]), r[1], r[2]) for r in rows if r[0] is not None]

    def fetch_rpc_rows(self, start: datetime, end: datetime) -> list[tuple[datetime, str, int, int]]:
        rows = self._rows(
            """
            SELECT hour_start, key, success_count, timeout_count
            FROM transport_baseline_hourly
            WHERE hour_start >= %s AND hour_start < %s
            """,
            (start, end),
        )
        return [(_aware(r[0]), str(r[1] or ""), int(r[2] or 0), int(r[3] or 0)) for r in rows if r[0] is not None]

    def fetch_hour_rows(self, since: datetime) -> list[SelfSignalHourV1]:
        rows = self._rows(
            """
            SELECT hour_id, hour_start, computed_at, signals, outcomes, history_hours
            FROM self_signal_hour
            WHERE hour_start >= %s
            ORDER BY hour_start DESC
            LIMIT %s
            """,
            (since, MAX_HOURS),
        )
        out: list[SelfSignalHourV1] = []
        for hid, hs, ca, signals, outcomes, hh in rows:
            try:
                out.append(SelfSignalHourV1(
                    hour_id=hid, hour_start=_aware(hs), computed_at=_aware(ca),
                    signals=signals or [], outcomes=outcomes or [], history_hours=float(hh or 0.0),
                ))
            except Exception as exc:  # noqa: BLE001 -- one malformed row must not hide the rest
                logger.warning("self_signal_hour_row_invalid hour_id=%s err=%s", hid, exc)
        return sorted(out, key=lambda r: r.hour_start)

    def fetch_association_rows(self, since: datetime, holds: int) -> list[SelfSignalOutcomeAssociationV1]:
        """The newest persisted association per (signal, outcome, hour) since ``since``: seeds the
        holds rule after a restart so a run that was already going is not reset by it."""
        rows = self._rows(
            association_history_sql("%s", "%s"), (since, association_history_limit(holds)),
        )
        out: list[SelfSignalOutcomeAssociationV1] = []
        for r in rows:
            try:
                out.append(SelfSignalOutcomeAssociationV1.model_validate(dict(zip(ASSOCIATION_COLUMNS, r))))
            except Exception as exc:  # noqa: BLE001 -- one malformed row hides only itself
                logger.warning("self_signal_association_row_invalid err=%s", exc)
        return out


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _naive_utc(ts: datetime) -> datetime:
    return _aware(ts).astimezone(timezone.utc).replace(tzinfo=None)


@dataclass
class CycleResult:
    hour_rows: list[SelfSignalHourV1] = field(default_factory=list)
    associations: list[SelfSignalOutcomeAssociationV1] = field(default_factory=list)
    skipped: str = ""


@dataclass
class SelfSignalOutcomeRollup:
    grace_sec: float = 600.0
    min_hours: int = 24
    # Hours known to be persisted (or published this process). Loaded from self_signal_hour.
    known_hours: dict[datetime, SelfSignalHourV1] = field(default_factory=dict)
    loaded: bool = False
    latest: list[SelfSignalOutcomeAssociationV1] = field(default_factory=list)
    computed: bool = False  # associations computed at least once this process
    # association_ids that really reached the bus: an open row only cites a persisted row.
    published_ids: set[str] = field(default_factory=set)
    # A clause renders only after its verdict held on this many consecutive hourly computations.
    consecutive_holds: int = DEFAULT_CONSECUTIVE_HOLDS
    # Earlier hourly computations that were persisted (seeded from the table, then each cycle's
    # published rows); the newest ``consecutive_holds`` + 2 hours are kept.
    history: list[SelfSignalOutcomeAssociationV1] = field(default_factory=list)

    def renderable(self) -> Mapping[str, SelfSignalOutcomeAssociationV1]:
        """signal_id -> its strongest association that may be shown (episode open rows): only
        pairs whose clause held on the last ``consecutive_holds`` hourly computations, and only
        the current computation's published rows, so the cited association_id resolves and an
        unpublished hour never falls back to last hour's row."""
        current = [a for a in self.latest if a.association_id in self.published_ids]
        ids = {a.association_id for a in current}
        held = held_latest_by_pair(self.history + current, holds=self.consecutive_holds)
        return best_per_signal(a for a in held if a.association_id in ids)

    def _history_since(self, now: datetime) -> datetime:
        return hour_floor(now) - HOUR * (max(1, int(self.consecutive_holds)) + 1)

    def needs_cycle(self, now: datetime, history_start: datetime | None) -> bool:
        """Cheap pre-check (no history copies): something is due, or nothing computed yet."""
        return not self.loaded or not self.computed or bool(self.due_hours(now, history_start))

    def due_hours(self, now: datetime, history_start: datetime | None) -> list[datetime]:
        """Completed hours (plus grace) inside the PE window that have no persisted row yet."""
        if history_start is None:
            return []
        now = _aware(now)
        last = hour_floor(now - timedelta(seconds=self.grace_sec)) - HOUR
        # The first full hour after the window's earliest reading; never older than 7 days.
        first = max(hour_floor(history_start) + HOUR, hour_floor(now - WINDOW_7D) + HOUR)
        out: list[datetime] = []
        h = first
        while h <= last:
            if h not in self.known_hours:
                out.append(h)
            h += HOUR
        return out[-MAX_HOURS_PER_CYCLE:]

    def run_cycle(
        self,
        *,
        now: datetime,
        reader: SelfSignalOutcomeReader,
        history: Callable[[str], list[tuple[datetime, float]]],
        node_ids: Iterable[str],
        history_start: datetime | None,
    ) -> CycleResult:
        """Blocking (call from a thread). Never raises for a read failure: returns ``skipped``."""
        now = _aware(now)
        try:
            if not self.loaded:
                known = {hour_floor(r.hour_start): r for r in reader.fetch_hour_rows(now - timedelta(hours=MAX_HOURS))}
                seeded = reader.fetch_association_rows(self._history_since(now), self.consecutive_holds)
                self.known_hours.update(known)
                self.history = seeded
                self.loaded = True
            due = self.due_hours(now, history_start)
            if not due and self.computed:
                return CycleResult(skipped="nothing_due")
            new_rows: list[SelfSignalHourV1] = []
            if due:
                start, end = due[0], due[-1] + HOUR
                turns = turn_outcome_hours(reader.fetch_turn_rows(start, end))
                rpc = rpc_outcome_hours(reader.fetch_rpc_rows(start, end))
                signals = body_signal_ids(node_ids)
                per_signal = {sid: classify_signal_hours(sid, history(sid), due) for sid in signals}
                for h in due:
                    outcomes = list(turns.get(h, [])) + ([rpc[h]] if h in rpc else [])
                    hist_h = 0.0 if history_start is None else max(0.0, (h - _aware(history_start)).total_seconds() / 3600.0)
                    new_rows.append(build_hour_row(
                        h, signals=[per_signal[s][h] for s in signals], outcomes=outcomes,
                        computed_at=now, history_hours=min(hist_h, WINDOW_7D.total_seconds() / 3600.0),
                    ))
        except NodePeHistoryReadError as exc:
            return CycleResult(skipped=f"read_failed:{exc}")
        for row in new_rows:
            self.known_hours[hour_floor(row.hour_start)] = row
        # Bound memory: keep the newest MAX_HOURS hours.
        if len(self.known_hours) > MAX_HOURS:
            for h in sorted(self.known_hours)[: len(self.known_hours) - MAX_HOURS]:
                self.known_hours.pop(h, None)
        hours = [self.known_hours[h] for h in sorted(self.known_hours)]
        # The previous computation joins the history only if it was persisted (published).
        since = self._history_since(now)
        self.history = [
            a for a in self.history + [a for a in self.latest if a.association_id in self.published_ids]
            if _aware(a.computed_at) >= since
        ]
        self.latest = associate_all(hours, computed_at=now, min_hours=self.min_hours)
        self.published_ids = set()
        self.computed = True
        return CycleResult(hour_rows=new_rows, associations=list(self.latest))


def summarize(assocs: Iterable[SelfSignalOutcomeAssociationV1]) -> str:
    parts = []
    for a in assocs:
        parts.append(
            f"{a.signal_id.removeprefix('node:substrate.')}/{a.outcome}:{a.verdict}"
            f"(h={a.events_high}/{a.n_high},n={a.events_normal}/{a.n_normal},rr={a.ratio})"
        )
    return " ".join(parts)


def association_payload(a: SelfSignalOutcomeAssociationV1) -> dict[str, Any]:
    """What an episode open row carries: enough to check the clause against its persisted row."""
    return {
        "association_id": a.association_id,
        "source_table": "self_signal_outcome_association",
        "outcome": a.outcome,
        "verdict": a.verdict,
        "n_high": a.n_high,
        "events_high": a.events_high,
        "n_normal": a.n_normal,
        "events_normal": a.events_normal,
        "ratio": a.ratio,
        "ci_low": a.ci_low,
        "ci_high": a.ci_high,
        "clause": a.clause,
    }
