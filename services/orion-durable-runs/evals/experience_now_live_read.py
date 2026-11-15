"""Orion's now, live read (integration design 2026-11-15, Part A patch 1). Read-only.

What it answers, from the live database, before (or after) this patch is deployed:

1. **The new sources have rows.** The chronicle's own SQL (``SourceReader``, read-only transaction) reads
   the last ``--days`` of ``reading_turn``, ``prediction_surprise`` and ``prime_thought`` and adapts them.
2. **Salience is not degenerate** (CLAUDE.md metric gate step 4). For every kind the organ registry gives a
   magnitude, each event of the last ``--days`` is scored against its own kind's 30 days before it (the
   same ``salience_of`` the chronicle uses): share NULL, share at the rest point 0, share >= 0.9, distinct
   values. A kind whose scored values are all one number cannot rank anything: flagged.
3. **Prime's now has substance.** ``build_now`` over the last 24 h of experiences (stored kinds plus the
   new ones), as Prime would be shown it.
4. **The integration metric's baseline.** ``use_rate`` over the last 24 h: experiences per organ, and the
   uses recorded in ``experience_use`` (0 when the table does not exist yet: nothing has been cited).

Usage (host): ``ORION_LIVE_DSN=postgresql://...:55432/conjourney python services/orion-durable-runs/evals/
experience_now_live_read.py [--days 7] [--check]``. ``--check`` exits 1 when a new source has no row, Prime's
now is empty, or a declared magnitude is degenerate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
SERVICE = HERE.parents[1]
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO), str(SERVICE)]

from orion.integration.organs import load_organs  # noqa: E402
from orion.temporal_self.now import SOURCE_HISTORY, build_now, magnitude, salience_of, use_rate  # noqa: E402

NEW_KINDS = ("reading_turn", "prediction_surprise", "prime_thought")
# A trend that is the organ changing, not the magnitude aging: reported, not flagged. Each names its evidence.
KNOWN_TRENDS = {
    "dream_cycle": "#3412: every cycle since 2026-11-13 07:02Z formed 0 hypotheses (4 before): a break, not age",
}
TZ = "America/Denver"


def _shape(e) -> dict:
    from app.temporal_self_store import stored_shape

    return dict(stored_shape(e), event_id=e.event_id, source_kind=e.source_kind, label=e.label,
                related_refs=list(e.related_refs))


def _trend(series: list) -> "float | None":
    import math

    if len(series) < 40:
        return None
    xs = [t.timestamp() for t, _ in series]
    ys = [math.log1p(max(m, 0.0)) for _, m in series]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if not sxx or not syy:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy), 3)


async def main(dsn: str, days: int, check: bool) -> int:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from app.temporal_self_sources import SourceReader

    reg = load_organs()
    now = datetime.now(timezone.utc)
    lo = now - timedelta(days=days)
    hist_lo = lo - SOURCE_HISTORY
    async with AsyncConnectionPool(dsn, min_size=1, max_size=2, open=False,
                                   kwargs={"autocommit": True, "row_factory": dict_row}) as pool:
        # 1. the new sources, through the chronicle's own reads (history window included, for salience)
        reader = SourceReader(pool, TZ)
        new_events = []
        t = hist_lo
        while t < now:
            hi = min(now, t + timedelta(days=1))
            _, evs = await reader.read(t, hi, ticks=False, kinds=list(NEW_KINDS))
            new_events.extend(evs)
            t = hi
        rows: dict[str, list[dict]] = defaultdict(list)
        for e in new_events:
            rows[e.source_kind].append(_shape(e))
        # stored kinds with a declared magnitude (their own payloads), and the last day of every kind
        async with pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                stored_kinds = [k for k in reg.magnitude_rule if k not in NEW_KINDS]
                for r in await (await conn.execute(
                        "SELECT event_id, source_kind, occurred_at, label, verdict, related_refs, payload_json "
                        "FROM temporal_self_event WHERE source_kind = ANY(%(k)s) AND occurred_at >= %(lo)s",
                        {"k": stored_kinds, "lo": hist_lo})).fetchall():
                    rows[r["source_kind"]].append({"event_id": r["event_id"], "source_kind": r["source_kind"],
                                                   "occurred_at": r["occurred_at"], "label": r["label"],
                                                   "verdict": r["verdict"], "related_refs": list(r["related_refs"] or []),
                                                   "payload": r["payload_json"] or {},
                                                   "surprise_nats": (r["payload_json"] or {}).get("surprise_nats")})
                # action_outcome's surprisal enters the payload with this patch; read the source column so
                # the check measures the real values, not their absence before deploy.
                ao = [r for r in rows["action_outcome"] if r.get("surprise_nats") is None]
                if ao:
                    sn = {str(x["id"]): x["surprise_nats"] for x in await (await conn.execute(
                        "SELECT id, surprise_nats FROM substrate_action_outcomes WHERE id = ANY(%(ids)s)",
                        {"ids": [int(r["event_id"].split(":", 1)[1]) for r in ao]})).fetchall()}
                    for r in ao:
                        r["surprise_nats"] = sn.get(r["event_id"].split(":", 1)[1])
                day_rows = await (await conn.execute(
                    "SELECT event_id, source_kind, occurred_at, label, verdict, related_refs, payload_json "
                    "FROM temporal_self_event WHERE occurred_at >= %(lo)s AND source_kind <> ALL(%(k)s)",
                    {"lo": now - timedelta(hours=24), "k": list(reg.magnitude_rule)})).fetchall()
                kind_counts = {r["source_kind"]: int(r["n"]) for r in await (await conn.execute(
                    "SELECT source_kind, count(*) AS n FROM temporal_self_event WHERE occurred_at >= %(lo)s "
                    "GROUP BY 1", {"lo": now - timedelta(hours=24)})).fetchall()}
                has_use = (await (await conn.execute(
                    "SELECT to_regclass('experience_use') IS NOT NULL AS p")).fetchone())["p"]
                uses = []
                if has_use:
                    uses = [(u["event_id"], u["source_kind"], u["citing_organ"]) for u in await (await conn.execute(
                        "SELECT DISTINCT u.event_id, e.source_kind, u.citing_organ FROM experience_use u JOIN "
                        "temporal_self_event e USING (event_id) WHERE e.occurred_at >= %(lo)s AND "
                        "u.cited_at < e.occurred_at + interval '24 hours'", {"lo": now - timedelta(hours=24)})).fetchall()]
    # new kinds are not stored live before deploy: count them from the reads
    for k in NEW_KINDS:
        kind_counts[k] = kind_counts.get(k, 0) + sum(1 for r in rows[k] if r["occurred_at"] >= now - timedelta(hours=24))

    # 2. salience per declared kind
    report: dict = {"as_of": now.isoformat(), "days": days, "new_sources": {}, "salience": {}, "flags": []}
    scored_day: list[dict] = []
    for kind, rule in sorted(reg.magnitude_rule.items()):
        hist = sorted(((r["occurred_at"], magnitude(rule, r), r) for r in rows[kind]), key=lambda x: x[0])
        series = [(t, m) for t, m, _ in hist if m is not None]
        vals = []
        for t, m, r in hist:
            if t < lo:
                continue
            s = salience_of(m, t, series)
            vals.append(s)
            if t >= now - timedelta(hours=24):
                scored_day.append(dict(r, salience=s))
        known = [v for v in vals if v is not None]
        stat = {"rule": rule, "events": len(vals), "null_share": round(1 - len(known) / len(vals), 3) if vals else None,
                "zero_share": round(sum(1 for v in known if v == 0) / len(known), 3) if known else None,
                "high_share": round(sum(1 for v in known if v >= 0.9) / len(known), 3) if known else None,
                "distinct": len(set(round(v, 4) for v in known)),
                "magnitude_distinct": len({m for _, m in series})}
        # Drift: a magnitude that trends with time (not with the event) ranks by age against its own
        # 30 days (review 2026-11-15: the act ledger's surprisal fell 40x in a month). Pearson r of time
        # vs log1p(magnitude) over the whole history; |r| > 0.4 flags.
        stat["time_trend_r"] = _trend(series)
        report["salience"][kind] = stat
        if kind in KNOWN_TRENDS:
            stat["known_trend"] = KNOWN_TRENDS[kind]
        elif stat["time_trend_r"] is not None and abs(stat["time_trend_r"]) > 0.4:
            report["flags"].append(f"{kind}: magnitude trends with time (r={stat['time_trend_r']}), ranks by age")
        # Degenerate: enough history to score, yet every score is one value (or none scores at all).
        if len(series) >= 40 and (not known or stat["distinct"] <= 1):
            report["flags"].append(f"{kind}: salience degenerate ({stat})")
    for k in NEW_KINDS:
        recent = [r for r in rows[k] if r["occurred_at"] >= lo]
        report["new_sources"][k] = {"events": len(recent),
                                    "examples": [r["label"][:160] for r in sorted(recent, key=lambda r: r["occurred_at"])[-3:]]}
        if not recent:
            report["flags"].append(f"{k}: no live rows in {days} days")

    # 3. Prime's now over the last 24 h (stored kinds with no magnitude read salience NULL)
    cands = scored_day + [{"event_id": r["event_id"], "source_kind": r["source_kind"], "occurred_at": r["occurred_at"],
                           "label": r["label"], "verdict": r["verdict"], "related_refs": list(r["related_refs"] or []),
                           "payload": r["payload_json"] or {}, "salience": None, "surprise_nats": None}
                          for r in day_rows]
    pnow = build_now(cands, "prime", since=now - timedelta(hours=24), as_of=now, shown_id="live-read", registry=reg)
    report["prime_now"] = [{"organ": i.organ, "event_id": i.event_id, "salience": i.salience, "label": i.label}
                           for i in pnow.items]
    report["prime_now_candidates"] = pnow.candidates
    if not pnow.items:
        report["flags"].append("prime's now is empty")

    # 4. the metric's baseline
    rate = use_rate(kind_counts, uses, window_start=now - timedelta(hours=24), window_end=now, registry=reg)
    report["experience_use_table_present"] = has_use
    report["metric"] = {"experiences": rate.experiences, "used_by_other": rate.used_by_other, "rate": rate.rate,
                        "closed_organs": rate.closed_organs, "registered_organs": rate.registered_organs,
                        "per_organ": {o.organ: o.experiences for o in rate.organs},
                        "unregistered_kinds": rate.unregistered_kinds}
    print(json.dumps(report, indent=2, default=str))
    return 1 if check and report["flags"] else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", default=os.getenv("ORION_LIVE_DSN"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.dsn:
        sys.exit("set ORION_LIVE_DSN or --dsn (read-only use)")
    sys.exit(asyncio.run(main(a.dsn, a.days, a.check)))
