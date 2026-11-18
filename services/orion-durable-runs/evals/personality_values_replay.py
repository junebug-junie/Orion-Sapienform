#!/usr/bin/env python3
"""Personality rev 4 Move 2 replay: what care and source (c) would read on the market's REAL wants. READ-ONLY.

The metric quality gate's live-data sanity step (spec rev 4, "Metric quality gate (care and endorsement)"):

* tendencies: the live active ``origin = 'choice'`` nodes; when the choice reducer has not folded yet
  (before its deploy), the ``personality_choice`` rows are folded here with ``orion.personality.graph.fold``
  (the same reducer), nothing written;
* wants: the first decision per want in ``want_ledger`` over the window, with the object text the market
  would get (``app.run_bids.want_object_text`` on the run's ``durable_admission_runs.request``; any other
  want has none);
* care per want (care arm: ``orion.physiology.personality_values.care_for``), the runaway guard on the last 7
  days of shown options, source (c) per family and its correlation with Juniper's engagement.

    python3 services/orion-durable-runs/evals/personality_values_replay.py \\
        --dsn "$POSTGRES_URI" --embed-url http://localhost:8320/embedding --hours 72

Checks (exit 1 when one fails):
  * care_zero_share_at_least_30pct: care reads exactly 0 for at least 30% of wants (the rest point is reached);
  * not_saturated: under 10% of the wants with text read care >= 0.99;
  * care_not_inert: when tendencies exist, at least one want with text reads care > 0 (else the term is a
    no-op and should not ship as "binding").
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(SERVICE)]

from orion.personality.graph import fold  # noqa: E402
from orion.physiology import personality_values as pv  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.physiology.weights import endorsement_exclusions, endorsement_for  # noqa: E402

from app.personality_driver import all_texts, choice_evidence  # noqa: E402
from app.personality_sources import choice_from_row  # noqa: E402
from app.personality_values import CHOICES_SQL, TENDENCIES_SQL, parse_choice_row, parse_tendency_row  # noqa: E402
from app.physiology_market import ACTS_SQL, act_from_row  # noqa: E402
from app.run_bids import want_object_text  # noqa: E402

WANTS_SQL = """
SELECT DISTINCT ON (want_id) want_id, family, producer, decision->'want'->>'object_ref' AS object_ref
  FROM want_ledger
 WHERE decided_at >= %(since)s
 ORDER BY want_id, decided_at
"""

RUNS_SQL = "SELECT run_id, request FROM durable_admission_runs WHERE run_id = ANY(%(ids)s)"

ALL_CHOICES_SQL = "SELECT * FROM personality_choice ORDER BY occurred_at"


def tendencies_from_fold(choices: Sequence[Any], vectors: dict[str, list[float]], now: datetime) -> list[pv.Tendency]:
    res = fold([], [choice_evidence(c, vectors) for c in choices], now=now)
    out = []
    for n in res.nodes:
        d = pv.unit(n.centroid)
        if n.status == "active" and d is not None:
            out.append(pv.Tendency(n.disposition_id, n.label, d, float(n.weight)))
    return out


def summarize(cares: Sequence[Optional[float]]) -> dict:
    """``cares``: one per want, None for a want with no object text (care 0, ``no_object_text``)."""
    vals = [0.0 if c is None else c for c in cares]
    text = [c for c in cares if c is not None]
    zero = sum(1 for v in vals if v == 0.0)
    pos = sorted(c for c in text if c > 0)
    sat = sum(1 for c in text if c >= 0.99)
    q = (lambda xs, p: xs[int(p * (len(xs) - 1))] if xs else None)
    return {
        "wants": len(vals), "with_text": len(text), "care_zero": zero,
        "care_zero_share": zero / len(vals) if vals else None,
        "care_zero_share_with_text": (sum(1 for c in text if c == 0.0) / len(text)) if text else None,
        "care_positive": len(pos), "care_p50_pos": q(pos, 0.5), "care_p90_pos": q(pos, 0.9),
        "care_max": max(pos) if pos else 0.0, "saturated": sat,
        "care_mean_with_text": statistics.fmean(text) if text else None,
    }


def checks(s: dict, n_tendencies: int) -> dict:
    return {
        "care_zero_share_at_least_30pct": s["care_zero_share"] is not None and s["care_zero_share"] >= 0.30,
        "not_saturated": not s["with_text"] or s["saturated"] / s["with_text"] < 0.10,
        "care_not_inert": n_tendencies == 0 or s["care_positive"] > 0,
    }


def _embedder(url: str) -> Callable[[str], list[float]]:
    import httpx

    client = httpx.Client(timeout=30)
    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in cache:
            doc = "personality-values-replay-" + hashlib.sha1(text.encode()).hexdigest()[:16]
            resp = client.post(url, json={"doc_id": doc, "text": text})
            resp.raise_for_status()
            cache[text] = list(resp.json()["embedding"])
        return cache[text]

    return embed


def main(argv: list[str] | None = None) -> int:
    import psycopg
    from psycopg.rows import dict_row

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--embed-url", default="http://localhost:8320/embedding")
    ap.add_argument("--hours", type=float, default=72.0)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    embed = _embedder(args.embed_url)
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        conn.execute("SET default_transaction_read_only = on")
        live_t = [t for t in (parse_tendency_row(r) for r in conn.execute(TENDENCIES_SQL).fetchall()) if t]
        choice_rows = list(conn.execute(ALL_CHOICES_SQL).fetchall())
        recent_rows = list(conn.execute(CHOICES_SQL, {"since": now - timedelta(days=30)}).fetchall())
        wants = list(conn.execute(WANTS_SQL, {"since": now - timedelta(hours=args.hours)}).fetchall())
        run_ids = [w["object_ref"].split(":", 1)[1] for w in wants
                   if (w["object_ref"] or "").startswith("durable_run:")]
        runs = {r["run_id"]: r["request"] for r in conn.execute(RUNS_SQL, {"ids": run_ids}).fetchall()}
        acts = [act_from_row(r) for r in conn.execute(ACTS_SQL, {"since": now - timedelta(days=30)}).fetchall()]

    source = "live"
    tendencies = live_t
    if not tendencies:
        choices = [c for c in (choice_from_row(r) for r in choice_rows) if c is not None]
        vectors = {t: embed(t) for t in sorted({t for c in choices for t in all_texts(c)})}
        tendencies = tendencies_from_fold(choices, vectors, now)
        source = f"replayed_fold_of_{len(choices)}_choices"
    print(f"tendencies: {len(tendencies)} ({source})")
    for t in sorted(tendencies, key=lambda x: -x.weight)[:10]:
        print(f"  w={t.weight:6.2f} strength={pv.strength(t.weight):.2f} {t.label[:90]}")

    # Runaway guard on the last 7 days of shown options.
    parsed = [p for p in (parse_choice_row(r) for r in recent_rows) if p is not None]
    entries = []
    for _rec, shown in parsed:
        if shown.occurred_at < now - timedelta(days=pv.CAP_DAYS):
            continue
        for text in shown.texts:
            best, _c, score = pv.best_tendency(embed(text), tendencies, with_cap=False)
            entries.append((shown.occurred_at.date(), best.disposition_id if best and score > 0 else None))
    caps = pv.runaway_caps(pv.daily_shown_shares(entries), now.date())
    print(f"runaway guard: shown_options_7d={len(entries)} capped={[(c.disposition_id, c.mean_share) for c in caps.values()]}")
    tendencies = pv.apply_caps(tendencies, caps)

    # Care on the real wants.
    cares: list[Optional[float]] = []
    by_family: dict[str, list[Optional[float]]] = {}
    examples = []
    for w in wants:
        ref = w["object_ref"] or ""
        text = None
        if ref.startswith("durable_run:"):
            req = runs.get(ref.split(":", 1)[1])
            text = want_object_text(req) if isinstance(req, dict) else None
        care = None
        if text:
            res = pv.care_for(embed(text), tendencies)
            care = res.care
            if res.care > 0:
                examples.append((res.care, w["family"], text[:90]))
        cares.append(care)
        by_family.setdefault(w["family"], []).append(care)
    s = summarize(cares)
    print(f"wants (first decision per want, last {args.hours:.0f} h):")
    for k, v in s.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
    print("per family (wants, with text, care>0, mean care with text):")
    for f, cs in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        txt = [c for c in cs if c is not None]
        print(f"  {f:34s} {len(cs):5d} {len(txt):5d} {sum(1 for c in txt if c > 0):5d} "
              f"{(statistics.fmean(txt) if txt else 0.0):.3f}")
    print("top care examples:")
    for c, f, t in sorted(examples, reverse=True)[:8]:
        print(f"  {c:.3f} {f}: {t}")

    # Source (c) per family and its independence from engagement.
    fams, err = load_families(ROOT / "config/physiology/families.yaml")
    records = [p[0] for p in parsed]
    print(f"source (c): {len(records)} choice rows in 30 days, real choice points: "
          f"{sum(1 for r in records if r.real_choice)}")
    if fams is not None:
        for f in sorted(fams.families):
            rate, n, _ = endorsement_for(f, records, since=now - timedelta(days=30), until=now)
            if rate is not None:
                print(f"  {f}: rate={rate:.3f} n={n}")
        for f, chk in endorsement_exclusions(fams, acts, records, now).items():
            print(f"  independence {f}: r={chk.correlation} paired_days={chk.paired_days} excluded={chk.excluded}")
    holdback = sum(1 for w in wants if pv.holdback_arm(w["want_id"]) == pv.ARM_HOLDBACK)
    print(f"holdback arm: {holdback} of {len(wants)} wants ({holdback / max(1, len(wants)):.1%})")
    ck = checks(s, len(tendencies))
    print(f"checks: {ck}")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
