#!/usr/bin/env python3
"""Eval: which walkway people-count estimator is closest to a human count? (read-only)

Spec PR #2763, Item 2 patch B. Scores the estimators in ``orion/vision/walkway_count.py``
against hand-checked hours in ``fixtures/walkway_count_labels.csv``:

- ``clusters``            person individuals in the hour (the rollup range's high)
- ``most_at_once``        most people in one vision window (the range's low; a floor)
- ``stories``             multi-frame motion stories + clear glimpses (decision 6)
- ``stories_in_range``    ``stories`` held inside [most_at_once, clusters]
- ``stories_multi_only``  multi-frame motion stories only (glimpses never count)
- ``run_peaks``           sum over presence runs of the most people in the run
- ``runs``                presence runs

Gate checks (exit 1 when ``stories`` or ``stories_in_range`` fails one):
1. Synthetic: an empty hour reads 0 for every estimator.
2. Synthetic: one person standing 2 minutes (one ``standing`` story, a window every 5 s,
   flicker glimpses beside the track, the matcher splitting them into 5 clusters) stays 1.
3. Live: every camera-up hour with no person in any window and no person cluster reads 0.
4. Live (reported, not gated): over every live ``standing`` story of 2+ minutes, what each
   estimator says for that span.

Scoring needs at least ``--min-labels`` (default 10) labelled hours. With fewer, the verdict is
INSUFFICIENT LABELS and the eval lists the hours most worth counting by hand: camera up,
people seen, motion stories running, and the estimators disagreeing the most.

Reads Postgres in a READ ONLY transaction. Writes nothing.

Usage (repo root):
    PYTHONPATH=. python services/orion-sql-writer/evals/run_walkway_count_estimator_eval.py \\
        [--days 7] [--stream walkway] [--pg postgresql://...] [--labels path.csv] [--offline]

``--offline`` runs only the synthetic gates and the label-file check (no database).
Exit 0 = gates pass, 1 = a gate failed or the labels file is malformed, 2 = database unreadable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))

from orion.vision.walkway_count import (  # noqa: E402
    ESTIMATORS,
    estimate_all,
    presence_runs,
    story_count,
)
from orion.vision.walkway_rollup import person_count_range, window_peak  # noqa: E402

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
DEFAULT_LABELS = HERE.parent / "fixtures" / "walkway_count_labels.csv"
DEFAULT_PROP_IDS = "fefb24d7e4074e9f827ad711874f6edf"  # VISION_MOTION_PROP_INDIVIDUAL_IDS default
TZ = ZoneInfo("America/Denver")
GATED = ("stories", "stories_in_range")
LABEL_COLUMNS = ("hour_start_utc", "stream", "true_people", "note")


# --- labels -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    hour: datetime
    stream: str
    true_people: int
    note: str


def load_labels(path: Path) -> tuple[list[Label], list[str]]:
    """Rows and format errors. ``#`` lines are comments; the header row is required."""
    if not path.exists():
        return [], [f"labels file missing: {path}"]
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return [], [f"labels file has no header row: {path}"]
    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
        return [], [f"labels header must be {','.join(LABEL_COLUMNS)}, got {reader.fieldnames}"]
    out: list[Label] = []
    errs: list[str] = []
    seen: set[tuple[datetime, str]] = set()
    for i, row in enumerate(reader, start=2):
        try:
            hour = datetime.fromisoformat(str(row["hour_start_utc"]).strip().replace("Z", "+00:00"))
            if hour.tzinfo is None:
                raise ValueError("no timezone")
            hour = hour.astimezone(timezone.utc)
            if (hour.minute, hour.second, hour.microsecond) != (0, 0, 0):
                raise ValueError("not on the hour")
            n = int(str(row["true_people"]).strip())
            if n < 0:
                raise ValueError("negative count")
            stream = str(row["stream"]).strip()
            if not stream:
                raise ValueError("empty stream")
        except (TypeError, ValueError, KeyError) as exc:
            errs.append(f"labels row {i}: {exc}")
            continue
        if (hour, stream) in seen:
            errs.append(f"labels row {i}: duplicate hour {hour.isoformat()} {stream}")
            continue
        seen.add((hour, stream))
        out.append(Label(hour, stream, n, str(row.get("note") or "").strip()))
    return out, errs


# --- synthetic gates ----------------------------------------------------------------------------

T0 = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)


def _det(t: datetime, *, score: float = 0.5, h: float = 300.0, zone: str | None = "sidewalk") -> dict[str, Any]:
    return {"t": t.isoformat(), "label": "person", "score": score, "box": [100.0, 800.0, 200.0, 800.0 + h],
            "zone": zone, "clipped": False, "individual_id": None}


def standing_two_minutes() -> dict[str, Any]:
    """One person stands 2 minutes: one standing story, a window every 5 s with 1 person, three
    clear flicker glimpses beside the track, and the matcher split them into 5 clusters."""
    start, end = T0 + timedelta(minutes=10), T0 + timedelta(minutes=12)
    standing = {"story_id": "s", "kind": "person", "drop_reason": "standing", "started_at": start, "ended_at": end,
                "evidence": [_det(start + timedelta(seconds=5 * i)) for i in range(25)]}
    flicker = [{"story_id": f"g{i}", "kind": "person", "drop_reason": "glimpse",
                "started_at": start + timedelta(seconds=s), "ended_at": start + timedelta(seconds=s),
                "evidence": [_det(start + timedelta(seconds=s), score=0.6)]} for i, s in enumerate((20, 65, 110))]
    windows = [(start + timedelta(seconds=5 * i), 1) for i in range(25)]
    return {"clusters": 5, "most_at_once": 1, "stories": [standing, *flicker], "windows": windows}


def synthetic_gates() -> list[tuple[str, str, int, int, bool]]:
    """(gate, estimator, got, want, ok) for every estimator."""
    out: list[tuple[str, str, int, int, bool]] = []
    empty = estimate_all(clusters=0, most_at_once=0, stories=[], windows=[])
    for k in ESTIMATORS:
        out.append(("empty hour reads 0", k, empty[k], 0, empty[k] == 0))
    stand = estimate_all(**standing_two_minutes())
    for k in ESTIMATORS:
        out.append(("standing 2 min stays 1", k, stand[k], 1, stand[k] == 1))
    return out


# --- live read ----------------------------------------------------------------------------------


def _aware(ts: Any) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def _json(v: Any) -> Any:
    if isinstance(v, (str, bytes)):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def read_live(pg: str, stream: str, since: datetime, until: datetime) -> dict[str, Any]:
    from sqlalchemy import create_engine, text

    eng = create_engine(pg.replace("postgresql+asyncpg://", "postgresql://"), pool_pre_ping=True)
    p = {"s": stream, "lo": since, "hi": until}
    with eng.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rollups = conn.execute(text(
            "SELECT period_start, rollup FROM vision_rollup WHERE stream_id = :s AND grain = 'hour' "
            "AND period_start >= :lo AND period_start < :hi"), p).mappings().all()
        stories = conn.execute(text(
            "SELECT story_id, kind, drop_reason, started_at, ended_at, evidence, frame_h, individual_id "
            "FROM vision_motion_story WHERE stream_id = :s AND kind = 'person' "
            "AND started_at >= :lo AND started_at < :hi"), p).mappings().all()
        windows = conn.execute(text(
            "SELECT observed_at, counts FROM vision_scene_inventory WHERE stream_id = :s "
            "AND observed_at >= :lo AND observed_at < :hi AND counts <> '{}'::jsonb"), p).mappings().all()
        first_story = conn.execute(text(
            "SELECT min(started_at) FROM vision_motion_story WHERE stream_id = :s"), p).scalar()
        spans = []
        for s in stories:
            a, b = _aware(s["started_at"]), _aware(s["ended_at"] or s["started_at"])
            if s["drop_reason"] == "standing" and (b - a).total_seconds() >= 120:
                n = conn.execute(text(
                    "SELECT count(DISTINCT s.individual_id) FROM vision_individual_sighting s "
                    "JOIN vision_individual i USING (individual_id) WHERE s.stream_id = :s AND i.kind = 'person' "
                    "AND NOT s.stationary AND s.ended_at >= :a AND s.started_at <= :b"),
                    {"s": stream, "a": a, "b": b}).scalar()
                spans.append({"story_id": s["story_id"], "a": a, "b": b, "clusters": int(n or 0)})
        conn.rollback()
    return {"rollups": rollups, "stories": stories, "windows": windows,
            "first_story": _aware(first_story) if first_story else None, "standing_spans": spans}


def build_hours(live: Mapping[str, Any], prop_ids: Sequence[str]) -> dict[datetime, dict[str, Any]]:
    hours: dict[datetime, dict[str, Any]] = {}
    for r in live["rollups"]:
        roll = _json(r["rollup"]) or {}
        rng = person_count_range(roll) or {"low": 0, "high": 0}
        cam = roll.get("camera") or {}
        hours[_aware(r["period_start"])] = {
            "low": int(rng["low"]), "high": int(rng["high"]), "camera": cam.get("status"),
            "stories": [], "windows": []}
    for s in live["stories"]:
        h = hours.get(_hour(_aware(s["started_at"])))
        if h is not None:
            d = dict(s)
            d["started_at"] = _aware(s["started_at"])
            d["ended_at"] = _aware(s["ended_at"] or s["started_at"])
            d["evidence"] = _json(s["evidence"]) or []
            h["stories"].append(d)
    for w in live["windows"]:
        t = _aware(w["observed_at"])
        h = hours.get(_hour(t))
        if h is not None:
            h["windows"].append((t, window_peak(_json(w["counts"]))))
    for h in hours.values():
        h["est"] = estimate_all(clusters=h["high"], most_at_once=h["low"], stories=h["stories"],
                                windows=h["windows"], prop_ids=prop_ids)
        h["glimpses"] = story_count(h["stories"], prop_ids=prop_ids)
    return hours


def live_empty_gate(hours: Mapping[datetime, Mapping[str, Any]]) -> tuple[int, list[str]]:
    checked, fails = 0, []
    for t, h in sorted(hours.items()):
        if h["camera"] != "ok" or h["high"] or any(n for _, n in h["windows"]):
            continue
        checked += 1
        for k in GATED:
            if h["est"][k]:
                fails.append(f"empty hour {t.isoformat()} reads {k}={h['est'][k]}")
    return checked, fails


def standing_report(live: Mapping[str, Any], prop_ids: Sequence[str]) -> list[dict[str, Any]]:
    """For each live standing story of 2+ minutes: each estimator over that span alone."""
    out = []
    stories = [dict(s, started_at=_aware(s["started_at"]), ended_at=_aware(s["ended_at"] or s["started_at"]),
                    evidence=_json(s["evidence"]) or []) for s in live["stories"]]
    wins = [(_aware(w["observed_at"]), window_peak(_json(w["counts"]))) for w in live["windows"]]
    for sp in live["standing_spans"]:
        a, b = sp["a"], sp["b"]
        inside = [s for s in stories if s["started_at"] <= b and s["ended_at"] >= a]
        ws = [(t, n) for t, n in wins if a <= t <= b]
        runs = presence_runs(ws)
        out.append({"story_id": sp["story_id"], "minutes": round((b - a).total_seconds() / 60, 1),
                    "clusters": sp["clusters"], "stories": story_count(inside, prop_ids=prop_ids).total,
                    "run_peaks": sum(p for _, _, p in runs), "most_at_once": max((n for _, n in ws), default=0)})
    return out


# --- scoring ------------------------------------------------------------------------------------


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def score(labels: Sequence[Label], hours: Mapping[datetime, Mapping[str, Any]]) -> dict[str, Any]:
    rows = [(lb, hours[lb.hour]) for lb in labels if lb.hour in hours]
    res: dict[str, Any] = {"scored_hours": len(rows), "estimators": {}}
    if not rows:
        return res
    for k in ESTIMATORS:
        err = [h["est"][k] - lb.true_people for lb, h in rows]
        res["estimators"][k] = {"mae": sum(abs(e) for e in err) / len(err), "bias": sum(err) / len(err),
                                "exact": sum(1 for e in err if e == 0)}
    res["inside_range"] = sum(1 for lb, h in rows if h["low"] <= lb.true_people <= h["high"])
    ranked = sorted(ESTIMATORS, key=lambda k: res["estimators"][k]["mae"])
    best, second = ranked[0], ranked[1]
    w = sum(1 for lb, h in rows if abs(h["est"][best] - lb.true_people) < abs(h["est"][second] - lb.true_people))
    l_ = sum(1 for lb, h in rows if abs(h["est"][best] - lb.true_people) > abs(h["est"][second] - lb.true_people))
    res.update(best=best, runner_up=second, sign_test_p=sign_test_p(w, l_), wins=w, losses=l_)
    return res


def hours_to_check(hours: Mapping[datetime, Mapping[str, Any]], *, since_stories: datetime | None,
                   labelled: set[datetime], n: int = 10) -> list[tuple[datetime, Mapping[str, Any]]]:
    """Camera up, people seen, motion stories running: most disagreement first, newest first on ties
    (the newest still have their story pictures)."""
    cands = []
    for t, h in hours.items():
        if t in labelled or h["camera"] != "ok" or h["high"] < 2:
            continue
        if since_stories is None or t < since_stories:
            continue
        spread = max(h["est"].values()) - min(h["est"].values())
        cands.append((spread, t, h))
    cands.sort(key=lambda c: (-c[0], -c[1].timestamp()))
    return [(t, h) for _, t, h in cands[:n]]


def _local(t: datetime) -> str:
    return t.astimezone(TZ).strftime("%a %m-%d %H:%M")


def print_check_list(rows: Sequence[tuple[datetime, Mapping[str, Any]]]) -> None:
    print(f"\nHours to count by hand (add a row to {DEFAULT_LABELS.relative_to(HERE.parents[3])}):")
    for t, h in rows:
        e = h["est"]
        print(f"\n- {_local(t)} MDT  (hour_start_utc {t.strftime('%Y-%m-%dT%H:%M:%SZ')})")
        print("  estimates: " + ", ".join(f"{k}={e[k]}" for k in ESTIMATORS))
        runs = presence_runs(h["windows"])
        shown = ", ".join(f"{a.astimezone(TZ):%H:%M:%S}-{b.astimezone(TZ):%H:%M:%S} (peak {p})" for a, b, p in runs[:8])
        print(f"  people on camera (local): {shown}" + (f", +{len(runs) - 8} more" if len(runs) > 8 else ""))
        stories = sorted(h["stories"], key=lambda s: s["started_at"])
        multi = [s for s in stories if s.get("drop_reason") not in ("glimpse", "prop")][:4]
        for s in multi:
            print(f"  story {s['started_at'].astimezone(TZ):%H:%M:%S} {s.get('drop_reason') or 'published'}: "
                  f"Hub /api/vision/motion-stories/{s['story_id']}/strip/frame.jpg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--pg", default=os.getenv("POSTGRES_URI") or DEFAULT_PG)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--min-labels", type=int, default=10)
    ap.add_argument("--offline", action="store_true", help="synthetic gates and label-file check only")
    args = ap.parse_args()

    fails: list[str] = []
    print("== synthetic gates ==")
    for gate, k, got, want, ok in synthetic_gates():
        print(f"  {'PASS' if ok else 'fail'}  {gate:<24} {k:<20} got {got} want {want}")
        if k in GATED and not ok:
            fails.append(f"synthetic gate '{gate}': {k} got {got}, want {want}")

    labels, errs = load_labels(Path(args.labels))
    fails += errs
    labels = [lb for lb in labels if lb.stream == args.stream]
    print(f"\nlabels for {args.stream}: {len(labels)} (need {args.min_labels} to score)")

    if args.offline:
        return _finish(fails, verdict="OFFLINE (no database read)")

    prop_ids = tuple(s.strip() for s in os.getenv("VISION_MOTION_PROP_INDIVIDUAL_IDS", DEFAULT_PROP_IDS).split(",")
                     if s.strip())
    until = datetime.now(timezone.utc)
    since = _hour(until - timedelta(days=args.days))
    for lb in labels:
        since = min(since, lb.hour)
    try:
        live = read_live(args.pg, args.stream, since, until)
    except Exception as exc:  # noqa: BLE001
        print(f"\nDATABASE UNREADABLE: {type(exc).__name__}: {exc}")
        return 2
    hours = build_hours(live, prop_ids)
    print(f"\n== live: {len(hours)} rollup hours since {since.isoformat()} ==")

    checked, empty_fails = live_empty_gate(hours)
    fails += empty_fails
    print(f"  live empty hours checked: {checked}; failures for {', '.join(GATED)}: {len(empty_fails)}")

    spans = standing_report(live, prop_ids)
    print(f"  live standing stories of 2+ min: {len(spans)}")
    for sp in spans:
        print(f"    {sp['story_id'][:8]} {sp['minutes']} min: clusters={sp['clusters']} stories={sp['stories']} "
              f"run_peaks={sp['run_peaks']} most_at_once={sp['most_at_once']}")

    tot = {k: sum(h["est"][k] for h in hours.values()) for k in ESTIMATORS}
    g = [h["glimpses"] for h in hours.values()]
    print("  totals over the window: " + ", ".join(f"{k}={v}" for k, v in tot.items()))
    print(f"  glimpses: clear+counted={sum(x.clear_glimpses for x in g)} clear-but-covered="
          f"{sum(x.covered_glimpses for x in g)} unclear={sum(x.unclear_glimpses for x in g)}")

    missing = [lb for lb in labels if lb.hour not in hours]
    if missing:
        print(f"  labelled hours with no rollup row (not scored): {len(missing)}")
    scored = score(labels, hours)
    if scored["scored_hours"] < args.min_labels:
        print_check_list(hours_to_check(hours, since_stories=live["first_story"],
                                        labelled={lb.hour for lb in labels}))
        return _finish(fails, verdict=f"INSUFFICIENT LABELS ({scored['scored_hours']} of {args.min_labels})")
    print("\n== score against hand counts ==")
    for k in ESTIMATORS:
        s = scored["estimators"][k]
        print(f"  {k:<20} MAE {s['mae']:.2f}  bias {s['bias']:+.2f}  exact {s['exact']}/{scored['scored_hours']}")
    print(f"  true count inside [most_at_once, clusters]: {scored['inside_range']}/{scored['scored_hours']}")
    sig = scored["sign_test_p"] < 0.05
    verdict = (f"BEST {scored['best']} (beats {scored['runner_up']} {scored['wins']}-{scored['losses']}, "
               f"sign test p={scored['sign_test_p']:.3f}{'' if sig else ', NOT SIGNIFICANT: keep the range'})")
    return _finish(fails, verdict=verdict)


def _finish(fails: Sequence[str], *, verdict: str) -> int:
    print(f"\nVERDICT: {verdict}")
    if fails:
        print("FAIL:")
        for f in fails:
            print(" ", f)
        return 1
    print(f"PASS: gates hold for {', '.join(GATED)} (empty hour reads 0; a person standing 2 minutes stays 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
