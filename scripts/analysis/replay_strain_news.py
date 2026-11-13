#!/usr/bin/env python3
"""Budget replay for strain news (strain loop Part 2, docs/superpowers/specs/2026-10-20-strain-loop-design.md).

Replays Orion's saved strain history through the shipped ``orion.regulation.strain_news`` and
projects what it would have done to the attention contest. Read-only.

Inputs (live Postgres, read-only):

* ``temporal_self_event`` rows with ``source_kind='arousal_transition'``: one per level change;
  ``payload_json.reading`` is the full ArousalReadingV1 at the change.
* ``orion_biometrics_summary`` athena ``cabinet_temp_c``: the cabinet temperature between
  transitions.
* ``substrate_attention_broadcast_log`` (168 h retention): one row per attention tick;
  ``projection_json->'attended_node_ids'->>0`` is that tick's winner (null = no winner).

Reconstruction (LIMITED -- stated in the output): regulation state is historized only at level
changes. Between transitions the replay steps every 120 s (the regulate cadence) holding the
transition's level, ``since`` and causes, and takes the cabinet temperature from biometrics.
``cabinet_hot`` is inferred while strained from the cabinet at/above critical with the thermal
gate's hysteresis (34 C on, 33 C re-arm), which is the cabinet reflex's ``critical`` rule but not
its unknown-plus-AC-low branch. A GPU cause that joins mid-episode is invisible to this replay.
Regulation is assumed known for the whole time between transitions (no unknown gaps, no history
warm-up, no restarts) and each onset fires at the transition's own time (live it lands one
regulate step later; reminders count from the sent onset either way). Both over-generate events,
which is conservative for the budget.

Projection: each event holds attention for at most the 300 s orienting window
(``EVENT_ORIENTING_WINDOW_SEC``) = the log ticks in [event, event + 300 s), capped at 10. Per
local day: events, projected strain tick share (held ticks / all ticks), and the share displaced
from other winners (held ticks whose real winner was not null / all non-null-winner ticks). A
day with no attention log reports events only, plus the worst case at 10 ticks per event of a
nominal 2,880-tick day.

Exit 1 if any day exceeds 8 events, 3 % share, or 10 % displaced. Exit 2 on a read failure.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence
from zoneinfo import ZoneInfo

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.attention.sustained import EVENT_ORIENTING_WINDOW_SEC  # noqa: E402
from orion.autonomy.cabinet_heat import REFLEX_CABINET_HOT  # noqa: E402
from orion.autonomy.thermal_gate import DEFAULT_CRITICAL_C, DEFAULT_CRITICAL_REARM_C  # noqa: E402
from orion.regulation.strain_news import (  # noqa: E402
    MAX_EVENTS_PER_DAY,
    StrainNewsEvent,
    local_day,
    strain_news,
)
from orion.schemas.regulation import ArousalReadingV1  # noqa: E402

STEP = timedelta(seconds=120)
MAX_TICKS_PER_EVENT = 10
NOMINAL_TICKS_PER_DAY = 2880
MAX_SHARE = 0.03
MAX_DISPLACED = 0.10
TEMP_MAX_AGE = timedelta(seconds=300)
DEFAULT_PSQL = "docker exec orion-athena-sql-db psql -U postgres -d conjourney"

TRANSITIONS_SQL = (
    "SELECT payload_json->'reading' FROM temporal_self_event "
    "WHERE source_kind = 'arousal_transition' ORDER BY occurred_at"
)
TEMPS_SQL = (
    "SELECT json_build_array(timestamp, (measurements->>'cabinet_temp_c')::float) "
    "FROM orion_biometrics_summary WHERE node = 'athena' AND measurements ? 'cabinet_temp_c' "
    "AND timestamp >= '{since}' ORDER BY timestamp"
)
TICKS_SQL = (
    "SELECT json_build_array(generated_at, projection_json->'attended_node_ids'->>0) "
    "FROM substrate_attention_broadcast_log ORDER BY generated_at"
)


def _ts(raw: str) -> datetime:
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00").replace(" ", "T"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class DayReport:
    day: date
    events: int = 0
    kinds: str = ""
    ticks: int = 0
    winner_ticks: int = 0
    held: int = 0
    displaced: int = 0

    @property
    def has_log(self) -> bool:
        return self.ticks > 0

    # Partial days (the attention log's first day, today) are judged against a full day's ticks:
    # 6 events in the first 16 h of a day are not 3 % of a day that is not over yet.
    full_day_ticks: Optional[int] = None
    full_day_winner_ticks: Optional[int] = None
    partial: bool = False

    @property
    def share(self) -> Optional[float]:
        if not self.ticks:
            return None
        return self.held / max(self.ticks, self.full_day_ticks or 0)

    @property
    def displaced_share(self) -> Optional[float]:
        if not self.ticks:
            return None
        denom = max(self.winner_ticks, self.full_day_winner_ticks or 0)
        return self.displaced / denom if denom else 0.0

    @property
    def worst_case_share(self) -> float:
        return self.events * MAX_TICKS_PER_EVENT / NOMINAL_TICKS_PER_DAY

    def failures(self) -> list[str]:
        out = []
        if self.events > MAX_EVENTS_PER_DAY:
            out.append(f"{self.events} events > {MAX_EVENTS_PER_DAY}")
        if self.share is not None and self.share > MAX_SHARE:
            out.append(f"share {self.share:.2%} > {MAX_SHARE:.0%}")
        if self.displaced_share is not None and self.displaced_share > MAX_DISPLACED:
            out.append(f"displaced {self.displaced_share:.2%} > {MAX_DISPLACED:.0%}")
        return out


def replay_events(
    transitions: Sequence[dict],
    temps: Sequence[tuple[datetime, float]],
    *,
    until: datetime,
    tz,
) -> list[StrainNewsEvent]:
    """Step the shipped strain_news over the reconstructed readings (see module doc)."""
    if not transitions:
        return []
    trans = sorted(
        (ArousalReadingV1.model_validate(t) for t in transitions), key=lambda r: r.observed_at
    )
    temps = sorted(temps)
    events: list[StrainNewsEvent] = []
    state = None
    ti = 0       # next transition index
    pi = 0       # next temp index
    current: Optional[ArousalReadingV1] = None
    temp: Optional[tuple[datetime, float]] = None
    hot = False
    t = trans[0].observed_at
    while t <= until:
        while ti < len(trans) and trans[ti].observed_at <= t:
            current = trans[ti]
            ti += 1
        while pi < len(temps) and temps[pi][0] <= t:
            temp = temps[pi]
            pi += 1
        temp_c = temp[1] if temp is not None and t - temp[0] <= TEMP_MAX_AGE else None
        if temp_c is not None:
            hot = temp_c >= DEFAULT_CRITICAL_C or (hot and temp_c > DEFAULT_CRITICAL_REARM_C)
        if current is not None:
            reasons = list(current.reasons)
            if current.arousal_level == "strained" and hot and REFLEX_CABINET_HOT not in reasons:
                reasons.append(REFLEX_CABINET_HOT)
            reading = current.model_copy(update={
                "observed_at": t, "reasons": reasons,
                "cabinet_temp_c": temp_c if temp_c is not None else current.cabinet_temp_c,
            })
            ev, state = strain_news(state, reading, t, tz)
            if ev is not None:
                events.append(ev)
        t += STEP
    return events


def project(
    events: Sequence[StrainNewsEvent],
    ticks: Sequence[tuple[datetime, Optional[str]]],
    *,
    tz,
) -> list[DayReport]:
    days: dict[date, DayReport] = {}

    def day(d: date) -> DayReport:
        return days.setdefault(d, DayReport(day=d))

    ticks = sorted(ticks, key=lambda x: x[0])
    for ts, winner in ticks:
        r = day(local_day(ts, tz))
        r.ticks += 1
        if winner:
            r.winner_ticks += 1
    window = timedelta(seconds=EVENT_ORIENTING_WINDOW_SEC)
    times = [ts for ts, _ in ticks]
    import bisect

    for ev in events:
        r = day(local_day(ev.at, tz))
        r.events += 1
        r.kinds += ev.kind[0].upper()
        i = bisect.bisect_left(times, ev.at)
        held = 0
        while i < len(ticks) and ticks[i][0] < ev.at + window and held < MAX_TICKS_PER_EVENT:
            hr = day(local_day(ticks[i][0], tz))
            hr.held += 1
            if ticks[i][1]:
                hr.displaced += 1
            held += 1
            i += 1
    logged = sorted(d for d, r in days.items() if r.ticks)
    if logged:
        first, last = logged[0], logged[-1]
        full = [days[d] for d in logged if d not in (first, last)]
        if full:
            import statistics

            med_ticks = int(statistics.median(r.ticks for r in full))
            med_winners = int(statistics.median(r.winner_ticks for r in full))
            for d in {first, last}:
                days[d].partial = True
                days[d].full_day_ticks = med_ticks
                days[d].full_day_winner_ticks = med_winners
    return [days[d] for d in sorted(days)]


def render(reports: Sequence[DayReport], events: Sequence[StrainNewsEvent], tz) -> str:
    lines = [
        "Strain news budget replay (shipped orion.regulation.strain_news)",
        "LIMITS: transitions-only history. Causes between transitions are the transition's own plus",
        "cabinet_hot inferred from biometrics cabinet_temp_c >= 34 C (re-arm 33 C). A GPU cause that",
        "joins mid-episode is not visible. Kinds: O=onset E=escalation R=reminder.",
        "Regulation is assumed known between transitions (no unknown gaps, warm-up or restarts);",
        "this over-generates events, conservative for the budget. The rule enforces <= 8 events/day",
        "and the 30 min gap; the 3 % and 10 % numbers are MEASURED here, not enforced by the rule",
        "(8 events x 300 s = 2.78 % of a day's time; the tick share runs a little higher on",
        "days the attention log has fewer ticks than 2,880).",
        "Partial days (*: first day of the attention log, today) are judged against the median",
        "full day's ticks and winners, not the ticks seen so far.",
        "",
        f"{'local day':<11} {'events':>6} {'kinds':<10} {'ticks':>6} {'winners':>7} {'held':>5} "
        f"{'share':>7} {'displaced':>9} {'worst%':>7}  verdict",
    ]
    for r in reports:
        share = "   n/a" if r.share is None else f"{r.share:6.2%}"
        disp = "      n/a" if r.displaced_share is None else f"{r.displaced_share:9.2%}"
        verdict = "FAIL: " + "; ".join(r.failures()) if r.failures() else (
            "ok" if r.has_log else "ok (no attention log; events only)")
        lines.append(
            f"{r.day.isoformat() + ('*' if r.partial else ''):<11} {r.events:>6} {r.kinds:<10} {r.ticks:>6} {r.winner_ticks:>7} "
            f"{r.held:>5} {share:>7} {disp} {r.worst_case_share:7.2%}  {verdict}"
        )
    lines.append("")
    lines.append(f"events total: {len(events)}")
    for ev in events:
        lines.append(f"  {ev.at.astimezone(tz).strftime('%a %m-%d %H:%M')} {ev.as_log_fields()}")
    return "\n".join(lines)


def _psql_json_rows(psql: str, sql: str) -> list:
    out = subprocess.run(
        [*shlex.split(psql), "-Atc", sql], check=True, capture_output=True, text=True, timeout=120
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--psql", default=DEFAULT_PSQL, help="read-only psql command prefix")
    ap.add_argument("--tz", default="America/Denver")
    args = ap.parse_args(argv)
    tz = ZoneInfo(args.tz)
    try:
        transitions = _psql_json_rows(args.psql, TRANSITIONS_SQL)
        first = min((_ts(t["observed_at"]) for t in transitions), default=datetime.now(timezone.utc))
        temps = [(_ts(ts), float(v)) for ts, v in _psql_json_rows(
            args.psql, TEMPS_SQL.format(since=(first - TEMP_MAX_AGE).isoformat()))]
        ticks = [(_ts(ts), w) for ts, w in _psql_json_rows(args.psql, TICKS_SQL)]
    except Exception as exc:  # noqa: BLE001
        print(f"read failed: {exc}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    events = replay_events(transitions, temps, until=now, tz=tz)
    reports = project(events, ticks, tz=tz)
    print(render(reports, events, tz))
    failed = [r for r in reports if r.failures()]
    print("\nVERDICT:", "FAIL" if failed else "PASS",
          f"(budget: <= {MAX_EVENTS_PER_DAY} events/day, share <= {MAX_SHARE:.0%}, "
          f"displaced <= {MAX_DISPLACED:.0%})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
