"""Read-only replay: which self_signal_episode rows would metacog have written?

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md, section D.

Replays every internal ``node:substrate.*`` prediction-error node in
``substrate_node_prediction_error_history`` on a poll grid (the equilibrium
service's poll cadence), through the SAME code the live gate runs:

    magnitude -> world_first.node_candidate -> world_first.judge_candidate
              -> self_signal_episode.observation_from_verdict -> step_signal

The only replay-side shortcut is the magnitude: percentile/band are computed
from an incrementally maintained sorted 7-day window instead of calling
``compute_prediction_error_magnitude`` 160k times. ``check_fast_magnitude``
pins that the two agree on band and percentile (the eval runs it).

Sources (SELECT only):
    --from-db                 docker exec orion-athena-sql-db psql (read-only COPY)
    --csv PATH[.gz]           node_id,epoch,value rows (the eval fixture format)
    --write-fixture PATH.gz   write the rows read to a gzip CSV fixture

Usage:
    python scripts/analysis/replay_self_signal_episodes.py --from-db
    python scripts/analysis/replay_self_signal_episodes.py --csv F.csv.gz --calm-gap-sweep
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import (  # noqa: E402
    judge_candidate,
    node_candidate,
    source_kind_for_node,
)
from orion.metacog.self_signal_episode import (  # noqa: E402
    DEFAULT_CALM_GAP_SEC,
    DEFAULT_OPEN_DWELL_SEC,
    DEFAULT_OPEN_WINDOW_SEC,
    DEFAULT_SILENT_CLOSE_SEC,
    observation_from_verdict,
    step_signal,
)
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1  # noqa: E402
from orion.substrate.prediction_error_magnitude import (  # noqa: E402
    DEFAULT_BAND_CUTS,
    DEFAULT_MIN_READINGS,
    WINDOW_7D,
    _quantile,
)

WINDOW_SEC = WINDOW_7D.total_seconds()
POLL_SEC = 30.0
PREFIX = "node:substrate."

Row = tuple[str, float, float]  # node_id, observed_at epoch, value


def rows_from_db(container: str = "orion-athena-sql-db") -> list[Row]:
    sql = (
        "copy (select node_id, extract(epoch from observed_at), value "
        "from substrate_node_prediction_error_history order by observed_at) to stdout with csv"
    )
    out = subprocess.run(
        ["docker", "exec", container, "psql", "-U", "postgres", "-d", "conjourney", "-Atc", sql],
        check=True, capture_output=True, text=True,
    ).stdout
    return _parse_csv(io.StringIO(out))


def _parse_csv(fh) -> list[Row]:
    rows: list[Row] = []
    for rec in csv.reader(fh):
        if len(rec) < 3:
            continue
        rows.append((rec[0], float(rec[1]), float(rec[2])))
    rows.sort(key=lambda r: r[1])
    return rows


def rows_from_csv(path: Path) -> list[Row]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:  # type: ignore[operator]
        return _parse_csv(fh)


def write_fixture(rows: Iterable[Row], path: Path) -> None:
    """Internal nodes only: the replay never reads external ones (perception)."""
    with gzip.open(path, "wt") as fh:
        w = csv.writer(fh)
        for node_id, t, v in rows:
            if source_kind_for_node(node_id) != "internal":
                continue
            w.writerow([node_id, f"{t:.3f}", repr(float(v))])


@dataclass
class NodeWindow:
    """Sorted 7-day value window for one node, advanced in time order."""

    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    sorted_vals: list[float] = field(default_factory=list)
    lo: int = 0  # index of the oldest reading still in the window
    hi: int = 0  # readings [lo, hi) are in the window

    def advance(self, now: float) -> None:
        while self.hi < len(self.times) and self.times[self.hi] <= now:
            bisect.insort(self.sorted_vals, self.values[self.hi])
            self.hi += 1
        while self.lo < self.hi and self.times[self.lo] < now - WINDOW_SEC:
            i = bisect.bisect_left(self.sorted_vals, self.values[self.lo])
            del self.sorted_vals[i]
            self.lo += 1

    def magnitude(self, now: float) -> tuple[PredictionErrorMagnitudeV1, datetime] | None:
        if self.hi == 0 or self.hi <= self.lo:
            return None
        value = self.values[self.hi - 1]
        observed = self.times[self.hi - 1]
        n = len(self.sorted_vals)
        pct = bisect.bisect_left(self.sorted_vals, value) / n
        if n < DEFAULT_MIN_READINGS:
            band = "insufficient_history"
        else:
            q, u, h = DEFAULT_BAND_CUTS
            band = "quiet" if pct < q else "usual" if pct < u else "high" if pct < h else "unusual"
        mag = PredictionErrorMagnitudeV1(
            value=round(value, 6),
            age_sec=round(max(0.0, now - observed), 3),
            p50_7d=_quantile(self.sorted_vals, 0.5),
            p90_7d=_quantile(self.sorted_vals, 0.9),
            percentile_now=round(pct, 6),
            n_readings_7d=n,
            band=band,
            trend="flat" if band != "insufficient_history" else "insufficient_history",
        )
        return mag, datetime.fromtimestamp(observed, tz=timezone.utc)


def build_windows(rows: Iterable[Row]) -> dict[str, NodeWindow]:
    wins: dict[str, NodeWindow] = {}
    for node_id, t, v in rows:
        if not node_id.startswith(PREFIX) or source_kind_for_node(node_id) != "internal":
            continue
        w = wins.setdefault(node_id, NodeWindow())
        w.times.append(t)
        w.values.append(v)
    return wins


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def replay(
    rows: list[Row],
    *,
    calm_gap_sec: float = DEFAULT_CALM_GAP_SEC,
    silent_close_sec: float = DEFAULT_SILENT_CLOSE_SEC,
    poll_sec: float = POLL_SEC,
    start: float | None = None,
    end: float | None = None,
    signals: Iterable[str] | None = None,
    open_dwell_sec: float = DEFAULT_OPEN_DWELL_SEC,
    open_window_sec: float = DEFAULT_OPEN_WINDOW_SEC,
) -> dict:
    """Episodes per signal. ``start`` defaults to the first reading (the early
    days are insufficient_history, which is itself a replay finding)."""
    wins = build_windows(rows)
    if signals is not None:
        keep = set(signals)
        wins = {k: v for k, v in wins.items() if k in keep}
    if not wins:
        return {"episodes": [], "per_day": {}, "per_signal": {}}
    t0 = start if start is not None else min(w.times[0] for w in wins.values())
    t1 = end if end is not None else max(w.times[-1] for w in wins.values())
    open_eps: dict[str, object] = {}
    episodes: list[dict] = []
    obs_counts: dict[str, Counter] = defaultdict(Counter)
    now = t0
    while now <= t1:
        for node_id, w in wins.items():
            w.advance(now)
            got = w.magnitude(now)
            cand = node_candidate(
                node_id=node_id,
                label=node_id.removeprefix(PREFIX),
                magnitude=None if got is None else got[0],
                observed_at=None if got is None else got[1],
                now=datetime.fromtimestamp(now, tz=timezone.utc),
            )
            obs = observation_from_verdict(judge_candidate(cand), now=now)
            obs_counts[node_id][obs.kind] += 1
            ep, event = step_signal(
                open_eps.get(node_id), obs, poll_sec=poll_sec, open_dwell_sec=open_dwell_sec,
                open_window_sec=open_window_sec, calm_gap_sec=calm_gap_sec,
                silent_close_sec=silent_close_sec,
            )
            if event is not None and event.phase == "close":
                e = event.episode
                episodes.append(
                    {
                        "signal_id": node_id,
                        "opened_at": e.opened_at,
                        "confirmed_at": e.confirmed_at,
                        "closed_at": event.closed_at,
                        "duration_sec": event.duration_sec,
                        "close_reason": event.close_reason,
                        "open_band": e.open_band,
                        "open_percentile": e.open_percentile,
                        "open_value": e.open_value,
                        "peak_band": e.peak_band,
                        "peak_percentile": e.peak_percentile,
                        "peak_value": e.peak_value,
                        "hot_polls": e.hot_polls,
                    }
                )
            if ep is None:
                open_eps.pop(node_id, None)
            else:
                open_eps[node_id] = ep
        now += poll_sec
    for node_id, e in open_eps.items():  # still open at the end
        if not e.confirmed:
            continue
        episodes.append(
            {
                "signal_id": node_id, "opened_at": e.opened_at, "confirmed_at": e.confirmed_at,
                "closed_at": None,
                "duration_sec": None, "close_reason": "still_open", "open_band": e.open_band,
                "open_percentile": e.open_percentile, "open_value": e.open_value,
                "peak_band": e.peak_band, "peak_percentile": e.peak_percentile,
                "peak_value": e.peak_value, "hot_polls": e.hot_polls,
            }
        )
    episodes.sort(key=lambda e: e["opened_at"])
    per_day: dict[str, Counter] = defaultdict(Counter)
    for e in episodes:
        per_day[_day(e["confirmed_at"])][e["signal_id"]] += 1
    per_signal: dict[str, dict] = {}
    for node_id in wins:
        eps = [e for e in episodes if e["signal_id"] == node_id]
        durs = sorted(e["duration_sec"] for e in eps if e["duration_sec"] is not None)
        per_signal[node_id] = {
            "episodes": len(eps),
            "close_reasons": dict(Counter(e["close_reason"] for e in eps)),
            "duration_p50_sec": _quantile(durs, 0.5) if durs else None,
            "duration_p90_sec": _quantile(durs, 0.9) if durs else None,
            "duration_max_sec": durs[-1] if durs else None,
            "observations": dict(obs_counts[node_id]),
        }
    return {
        "episodes": episodes,
        "per_day": {d: dict(c) for d, c in sorted(per_day.items())},
        "per_signal": per_signal,
        "calm_gap_sec": calm_gap_sec,
        "start": t0,
        "end": t1,
    }


def episodes_between(result: dict, start_iso: str, end_iso: str, signal: str | None = None) -> list[dict]:
    lo = datetime.fromisoformat(start_iso).timestamp()
    hi = datetime.fromisoformat(end_iso).timestamp()
    return [
        e for e in result["episodes"]
        if lo <= e["opened_at"] < hi and (signal is None or e["signal_id"] == signal)
    ]


def hot_gaps(rows: list[Row], *, poll_sec: float = POLL_SEC) -> dict[str, list[float]]:
    """For calm-gap selection: per signal, the length of every calm/faded run
    that sits BETWEEN two hot polls (unknown polls end a run)."""
    wins = build_windows(rows)
    t0 = min(w.times[0] for w in wins.values())
    t1 = max(w.times[-1] for w in wins.values())
    gaps: dict[str, list[float]] = defaultdict(list)
    last_hot: dict[str, float | None] = {}
    run_ok: dict[str, bool] = {}
    now = t0
    while now <= t1:
        for node_id, w in wins.items():
            w.advance(now)
            got = w.magnitude(now)
            cand = node_candidate(
                node_id=node_id, label=node_id, magnitude=None if got is None else got[0],
                observed_at=None if got is None else got[1],
                now=datetime.fromtimestamp(now, tz=timezone.utc),
            )
            kind = observation_from_verdict(judge_candidate(cand), now=now).kind
            if kind == "hot":
                lh = last_hot.get(node_id)
                if lh is not None and run_ok.get(node_id, False) and now - lh > poll_sec:
                    gaps[node_id].append(now - lh - poll_sec)
                last_hot[node_id] = now
                run_ok[node_id] = True
            elif kind == "unknown":
                run_ok[node_id] = False
        now += poll_sec
    return dict(gaps)


def check_fast_magnitude(rows: list[Row], *, samples: int = 40) -> list[tuple]:
    """(node, t, fast band, real band, fast pct, real pct) where they DISAGREE."""
    from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

    wins = build_windows(rows)
    bad = []
    for node_id, w in wins.items():
        if len(w.times) < 2:
            continue
        step = max(1, len(w.times) // samples)
        for i in range(0, len(w.times), step):
            now = w.times[i] + 1.0
            fresh = NodeWindow(times=w.times, values=w.values)
            fresh.advance(now)
            got = fresh.magnitude(now)
            if got is None:
                continue
            fast, observed = got
            hist = [
                (datetime.fromtimestamp(t, tz=timezone.utc), v)
                for t, v in zip(w.times, w.values)
                if now - WINDOW_SEC <= t <= now
            ]
            real = compute_prediction_error_magnitude(
                value=w.values[fresh.hi - 1], observed_at=observed, history=hist,
                now=datetime.fromtimestamp(now, tz=timezone.utc),
            )
            if real.band != fast.band or abs((real.percentile_now or 0) - (fast.percentile_now or 0)) > 1e-6:
                bad.append((node_id, now, fast.band, real.band, fast.percentile_now, real.percentile_now))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", action="store_true")
    src.add_argument("--csv", type=Path)
    ap.add_argument("--write-fixture", type=Path)
    ap.add_argument("--calm-gap-sec", type=float, default=DEFAULT_CALM_GAP_SEC)
    ap.add_argument("--calm-gap-sweep", action="store_true")
    ap.add_argument("--gaps", action="store_true", help="calm-run lengths between hot polls")
    ap.add_argument("--episodes", action="store_true", help="print every episode")
    args = ap.parse_args()
    rows = rows_from_db() if args.from_db else rows_from_csv(args.csv)
    if args.write_fixture:
        write_fixture(rows, args.write_fixture)
        print(f"wrote {len(rows)} rows to {args.write_fixture}")
    if args.gaps:
        for node_id, g in sorted(hot_gaps(rows).items()):
            g.sort()
            qs = {q: _quantile(g, q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
            print(node_id, len(g), {k: round(v) for k, v in qs.items()}, "max", round(g[-1]) if g else None)
        return 0
    gaps = [0.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0] if args.calm_gap_sweep else [args.calm_gap_sec]
    for gap in gaps:
        r = replay(rows, calm_gap_sec=gap)
        print(f"=== calm_gap_sec={gap:.0f} ===")
        print(json.dumps({"per_signal": r["per_signal"], "per_day": r["per_day"]}, indent=1, default=str))
        if args.episodes:
            for e in r["episodes"]:
                print(
                    datetime.fromtimestamp(e["opened_at"], tz=timezone.utc).isoformat(timespec="seconds"),
                    e["signal_id"], e["open_band"], e["peak_band"], e["peak_value"],
                    e["duration_sec"], e["close_reason"],
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
