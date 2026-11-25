"""Read-only replay for #3917: world-first attention before / after the calm-theory band.

Before: every node is banded against its own 7 days, 200-reading floor (``node_id`` not passed).
After: the forecast-surprise nodes (``orion.substrate.calm_band.ATTENTION_BAND_RULE_BY_NODE_ID``) are
banded on their calm theory; every other node is unchanged.

Inputs are the field contest's own reads (``app.store.AttentionRuntimeStore``, SELECT only):
every node's stored readings (prediction-error + signal history), Juniper's chat turns, and accepted
reading links. Each tick replays the field contest: one candidate per non-retired, non-fenced node,
plus world:chat and world:reading, ranked by ``rank_candidates`` with ``yield_reading_to_chat``.
Not modelled (the same in both arms): world:play, camera frame staleness (camera absence).

Reported per source and arm: eligible ticks, won ticks, win-when-eligible (won / eligible), and the
before -> after change of win-when-eligible with a standard error from hour blocks (ticks within an
hour are correlated, so the hour is the unit of resampling). For node:prediction.chat_turn: every
write at or above its calm "high" value, and whether the after arm let it compete in the 300 s after.

    PYTHONDONTWRITEBYTECODE=1 POSTGRES_URI=postgresql://postgres:<pw>@localhost:55432/conjourney \
        python services/orion-attention-runtime/evals/replay_sparse_pe_calm_band.py --step-sec 60
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SERVICE = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(SERVICE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.attention.world_first import (  # noqa: E402
    CHAT_RATE_WINDOW,
    EVENT_ORIENTING_WINDOW_SEC,
    READING_RATE_WINDOW,
    chat_candidate,
    midrank_percentile,
    node_candidate,
    node_prediction_error_continuous,
    node_rank_percentile,
    rank_candidates,
    reading_candidate,
    yield_reading_to_chat,
)
from orion.schemas.attention_candidate import world_first_fenced  # noqa: E402
from orion.substrate.calm_band import CHAT_TURN_NODE_ID, calm_percentiles, has_calm_band  # noqa: E402
from orion.substrate.prediction_error_freshness import is_retired_prediction_error_node  # noqa: E402
from orion.substrate.prediction_error_magnitude import WINDOW_7D, compute_prediction_error_magnitude  # noqa: E402

ARMS = ("before", "after")


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


class Series:
    def __init__(self) -> None:
        self.times: list[datetime] = []
        self.values: list[float] = []
        self._memo: dict[tuple, tuple] = {}

    def magnitude(self, node_id: str, t: datetime, arm: str):
        hi = bisect.bisect_right(self.times, t)
        if hi == 0:
            return None
        lo = bisect.bisect_left(self.times, t - WINDOW_7D)
        key = (arm, lo, hi)
        got = self._memo.get(key)
        observed, value = self.times[hi - 1], self.values[hi - 1]
        if got is None:
            hist = list(zip(self.times[lo:hi], self.values[lo:hi]))
            vals = self.values[lo:hi]
            if arm == "after":
                mag = compute_prediction_error_magnitude(value=value, observed_at=observed, history=hist,
                                                         now=t, node_id=node_id)
                rank = node_rank_percentile(node_id, value, vals)
            else:
                mag = compute_prediction_error_magnitude(value=value, observed_at=observed, history=hist, now=t)
                rank = midrank_percentile(value, vals)
            got = (mag, rank)
            if len(self._memo) > 64:
                self._memo.clear()
            self._memo[key] = got
        mag, rank = got
        age = max(0.0, (t - observed).total_seconds())
        return mag.model_copy(update={"age_sec": round(age, 3)}), rank, observed, lo, hi


def load(dsn: str, start: datetime):
    from app.store import AttentionRuntimeStore

    store = AttentionRuntimeStore(dsn)
    rows = store.fetch_node_prediction_error_history(start - WINDOW_7D)
    turns = store.fetch_chat_turn_times(start - WINDOW_7D - CHAT_RATE_WINDOW)
    try:
        links = store.fetch_accepted_reading_link_events(start - WINDOW_7D - READING_RATE_WINDOW)
    except Exception as exc:  # noqa: BLE001 -- reading absent in both arms, said so
        print(f"reading links unreadable ({type(exc).__name__}); world:reading absent in both arms", file=sys.stderr)
        links = None
    return rows, turns, links


def replay(rows, turns, links, *, start: datetime, end: datetime, step: timedelta) -> dict:
    series: dict[str, Series] = defaultdict(Series)
    for node_id, observed_at, value in sorted(rows, key=lambda r: _aware(r[1])):
        if is_retired_prediction_error_node(node_id) or world_first_fenced(node_id):
            continue
        if not math.isfinite(float(value)):
            continue
        s = series[node_id]
        s.times.append(_aware(observed_at))
        s.values.append(float(value))
    turns = sorted(_aware(t) for t in turns)
    links = None if links is None else sorted(((_aware(a), b, c) for a, b, c in links), key=lambda x: x[0])
    link_times = None if links is None else [x[0] for x in links]
    continuous = {n for n in series if node_prediction_error_continuous(n)}

    # counts[arm][source] = [eligible, won]; by hour for the block standard error
    counts = {arm: defaultdict(lambda: [0, 0]) for arm in ARMS}
    hourly = {arm: defaultdict(lambda: defaultdict(lambda: [0, 0])) for arm in ARMS}
    ticks = 0
    no_winner = {arm: 0 for arm in ARMS}
    chat_eligible_after: list[datetime] = []
    t = start
    while t <= end:
        ticks += 1
        hour = t.replace(minute=0, second=0, microsecond=0)
        tt = turns[: bisect.bisect_right(turns, t)]
        chat = chat_candidate(tt, now=t)
        reading = None
        if links is not None:
            reading = reading_candidate(links[: bisect.bisect_right(link_times, t)], now=t)
        for arm in ARMS:
            cands = [chat] + ([reading] if reading is not None else [])
            for node_id, s in series.items():
                got = s.magnitude(node_id, t, arm)
                if got is None:
                    continue
                mag, rank, observed, lo, hi = got
                pe_hist = list(zip(s.times[lo:hi], s.values[lo:hi])) if node_id in continuous else None
                cands.append(node_candidate(node_id=node_id, label=node_id, magnitude=mag, observed_at=observed,
                                            now=t, rank_percentile=rank, pe_history=pe_hist))
            r = yield_reading_to_chat(rank_candidates(cands))
            winner = r.winner.candidate.source_id if r.winner is not None else None
            if winner is None:
                no_winner[arm] += 1
            for v in r.eligible:
                sid = v.candidate.source_id
                won = int(sid == winner)
                counts[arm][sid][0] += 1
                counts[arm][sid][1] += won
                hourly[arm][hour][sid][0] += 1
                hourly[arm][hour][sid][1] += won
                if arm == "after" and sid == CHAT_TURN_NODE_ID:
                    chat_eligible_after.append(t)
        t += step

    sources = sorted(set(counts["before"]) | set(counts["after"]))
    hours = sorted(set(hourly["before"]) | set(hourly["after"]))
    per_source = {}
    for sid in sources:
        b_el, b_won = counts["before"][sid]
        a_el, a_won = counts["after"][sid]
        wb = b_won / b_el if b_el else None
        wa = a_won / a_el if a_el else None
        delta = se = None
        if wb is not None and wa is not None:
            delta = wa - wb
            # Hour-block jackknife of the difference of ratios.
            reps = []
            for h in hours:
                be = b_el - hourly["before"][h][sid][0]
                bw = b_won - hourly["before"][h][sid][1]
                ae = a_el - hourly["after"][h][sid][0]
                aw = a_won - hourly["after"][h][sid][1]
                if be and ae and (hourly["before"][h][sid][0] or hourly["after"][h][sid][0]):
                    reps.append(aw / ae - bw / be)
            k = len(reps)
            if k > 1:
                mean = sum(reps) / k
                se = math.sqrt((k - 1) / k * sum((x - mean) ** 2 for x in reps))
        per_source[sid] = {
            "calm_band": has_calm_band(sid),
            "before": {"eligible": b_el, "won": b_won, "win_when_eligible": None if wb is None else round(wb, 4)},
            "after": {"eligible": a_el, "won": a_won, "win_when_eligible": None if wa is None else round(wa, 4)},
            "delta_wwe": None if delta is None else round(delta, 4),
            "delta_se_hour_blocks": None if se is None else round(se, 4),
            "beyond_2se": None if (delta is None or se is None) else abs(delta) > 2 * se,
        }

    # Chat-turn writes that read "high" or above on the calm theory, and whether attention saw them.
    chat = series.get(CHAT_TURN_NODE_ID) or Series()
    chat_writes = []
    for ts, v in zip(chat.times, chat.values):
        if not (start <= ts <= end):
            continue
        pct = calm_percentiles(CHAT_TURN_NODE_ID, v)[0]
        if pct < 0.9:
            continue
        lo = bisect.bisect_left(chat_eligible_after, ts)
        hi = bisect.bisect_right(chat_eligible_after, ts + timedelta(seconds=EVENT_ORIENTING_WINDOW_SEC))
        chat_writes.append({"at": ts.isoformat(), "value": round(v, 4), "calm_percentile": round(pct, 4),
                            "eligible_ticks_after": hi - lo})
    n_chat_writes = sum(1 for ts in chat.times if start <= ts <= end)
    return {
        "start": start.isoformat(), "end": end.isoformat(), "step_sec": step.total_seconds(), "ticks": ticks,
        "no_winner_share": {arm: round(no_winner[arm] / ticks, 4) for arm in ARMS} if ticks else None,
        "per_source": per_source,
        "chat_turn": {
            "writes_in_window": n_chat_writes,
            "writes_reading_high_on_calm_theory": len(chat_writes),
            "with_eligible_ticks_after": sum(1 for w in chat_writes if w["eligible_ticks_after"]),
            "writes": chat_writes,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI", ""))
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--warmup-hours", type=float, default=12.0,
                    help="skip the first hours of the kept history so empirical nodes have a week to rank against")
    ap.add_argument("--step-sec", type=float, default=60.0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    if not args.dsn:
        print("set POSTGRES_URI or --dsn", file=sys.stderr)
        return 2
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=args.days)
    rows, turns, links = load(args.dsn, start)
    first = min((_aware(r[1]) for r in rows), default=start)
    start = max(start, first + timedelta(hours=args.warmup_hours))
    out = replay(rows, turns, links, start=start, end=end, step=timedelta(seconds=args.step_sec))
    text = json.dumps(out, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
