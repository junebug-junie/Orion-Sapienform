"""Read-only replay of the street handoff loop over the stored history (metric gate).

docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md, "Metric
gate: handoff surprise" step 4 (live sanity) and phase-2 acceptance checks.

Reads ``vision_motion_story`` (both cameras), ``vision_scene_inventory``
census spans, doorbell Ring events from ``vision_events``, crop vectors and
person-model accept guesses through ``docker exec <container> psql`` in a
read-only session with a statement timeout. Nothing is written.

Replays the reducer's rules in time order with the same pure functions the
loop uses (app/vision_handoff.py): open (projection from ``inside``), predict
(prior band until ``min_arrivals`` arrivals, then split conformal), match, grade with
the 80% "was looking" rule, overlap pairs on the central walk, p-values and
the attention gate. Edges are derived from story evidence when the edges
migration is not applied.

Reports: outcome counts per portal; overlap pairs; learned g per portal with
its spread ((p90 - p10) / median) and per-day medians; p-value uniformity on
met handoffs (decile counts, KS distance); attention-gate passes and whether
every quiet hour (no late / missed) produced zero passes; how many windows
were still ahead when the story landed (could Redis steering have helped);
and a VERDICT line. Too little data for a verdict is reported as exactly that.

Usage:
    PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_handoff_replay_eval.py \\
        [--since 2026-10-13T00:00:00Z] [--container orion-athena-sql-db] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

import numpy as np  # noqa: E402

from app import vision_handoff as vh  # noqa: E402
from app.vision_rhythm import coverage_fraction  # noqa: E402
from orion.vision.topology import load_topology  # noqa: E402

DB = ["-U", "postgres", "-d", "conjourney"]
FRAME = (2560.0, 1920.0)


def psql_json(container: str, sql: str, timeout_s: int = 60):
    cmd = ["docker", "exec", container, "psql", *DB, "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
           "-c", f"SET statement_timeout = '{timeout_s}s'", "-c", "SET default_transaction_read_only = on",
           "-c", f"SELECT coalesce(json_agg(q), '[]'::json) FROM ({sql}) AS q"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 30, check=True).stdout.strip()
    return json.loads(out or "[]")


def ts(v) -> datetime:
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def load(container: str, since: datetime, until: datetime, kinds: Sequence[str]):
    have = {r["column_name"] for r in psql_json(container, (
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'vision_motion_story' "
        "AND column_name IN ('exit_edge','entry_edge','exit_foot_x','exit_speed_fw_per_sec','frame_w','frame_h')"))}
    edge_cols = ("entry_edge", "exit_edge", "exit_foot_x", "exit_speed_fw_per_sec", "frame_w", "frame_h")
    extra = ", ".join(c if c in have else f"NULL AS {c}" for c in edge_cols)
    rows = psql_json(container, (
        "SELECT story_id, stream_id, kind, started_at, ended_at, dir_lateral, dir_depth, zone_path, individual_id, "
        f"crop_ids, evidence, speed_class, drop_reason, created_at, {extra} FROM vision_motion_story "
        f"WHERE ended_at >= '{since.isoformat()}' AND started_at <= '{until.isoformat()}' "
        f"AND kind IN ({','.join(repr(k) for k in kinds)})"), timeout_s=90)
    stories = []
    for r in rows:
        for k in ("started_at", "ended_at", "created_at"):
            r[k] = ts(r[k]) if r[k] else None
        stories.append(vh.facts_from_row(r, FRAME))
    census = defaultdict(list)
    for r in psql_json(container, (
            "SELECT stream_id, COALESCE(window_start_ts, extract(epoch FROM observed_at)) AS a, "
            "COALESCE(window_end_ts, extract(epoch FROM observed_at)) AS b FROM vision_scene_inventory "
            f"WHERE frame_count > 0 AND observed_at >= '{since.isoformat()}' AND observed_at <= '{until.isoformat()}' "
            "AND stream_id IN ('walkway','doorbell')"), timeout_s=90):
        census[r["stream_id"]].append((float(r["a"]), float(r["b"])))
    rings = [(ts(r["created_at"]), r["event_id"], r["event_type"]) for r in psql_json(container, (
        "SELECT event_id, event_type, created_at FROM vision_events WHERE stream_id = 'doorbell' "
        f"AND event_type IN ('doorbell_motion','doorbell_ring') AND created_at >= '{since.isoformat()}'"))]
    known = {r["individual_id"]: r["person_label"] for r in psql_json(container, (
        "SELECT individual_id, person_label FROM vision_person_attribution WHERE revoked_at IS NULL "
        "AND band = 'accept'"))} if psql_json(container, "SELECT to_regclass('vision_person_attribution') AS reg")[0]["reg"] \
        else {}
    return stories, census, sorted(rings), known, bool(have)


def load_vectors(container: str, crop_ids: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    ids = sorted(set(crop_ids))
    for i in range(0, len(ids), 200):
        chunk = ",".join("'" + c.replace("'", "''") + "'" for c in ids[i:i + 200])
        for r in psql_json(container, (
                "SELECT crop_id, embedding, osnet_embedding, face_embedding FROM vision_crop_observation "
                f"WHERE crop_id IN ({chunk})"), timeout_s=90):
            vecs = {ch: vh.unit(v) for ch, v in (("body", r["embedding"]), ("osnet", r["osnet_embedding"]),
                                                 ("face", r["face_embedding"])) if v}
            if vecs:
                out[r["crop_id"]] = vecs
    return out


def cov(census, stream, a: datetime, b: datetime, gap: float) -> float:
    return coverage_fraction(census.get(stream, []), a, b, gap_tolerance_sec=gap)


def ks_uniform(ps: Sequence[float]) -> Optional[float]:
    if not ps:
        return None
    s = sorted(ps)
    n = len(s)
    return max(max((i + 1) / n - p, p - i / n) for i, p in enumerate(s))


def replay(stories, census, rings, known, vecs, topology, cfg: vh.HandoffConfig, tz: ZoneInfo):
    by_stream = defaultdict(list)
    for s in stories:
        by_stream[s.stream_id].append(s)
    for v in by_stream.values():
        v.sort(key=lambda s: s.started_at)
    learned: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    missed_n: Dict[str, Counter] = defaultdict(Counter)
    graded: List[dict] = []
    open_: List[dict] = []
    V = lambda s: [vecs[c] for c in s.crop_ids[-12:] if c in vecs]  # noqa: E731

    def grade(h):
        portal = topology.by_id(h["portal"])
        src = h["src"]
        rec = dict(portal=portal.id, kind=src.kind, departed=src.ended_at, landed=src.created_at,
                   window_end=h["we"], band=h["band"].source, edges=src.edges_source)
        early = vh.early_from(src.ended_at, h["ws"], h["we"])
        if portal.kind == "approach":
            hits = sorted([t for t, _i, _k in rings if early <= t <= h["lu"]]
                          + [s.started_at for s in by_stream[portal.to_stream]
                             if s.kind in cfg.look_kinds and early <= s.started_at <= h["lu"]])
            status = ("early" if hits[0] < h["ws"] else "met" if hits[0] <= h["we"] else "late") if hits else "none"
            target_t = hits[0] if hits else None
            score = None
        else:
            cands = []
            for c in by_stream[portal.to_stream]:
                if c.kind != src.kind or not (early <= c.started_at <= h["lu"]) or c.drop_reason in vh.NOT_AN_ARRIVAL:
                    continue
                if not vh.arrival_consistent(c, portal, cfg):
                    continue
                sc, ch = vh.appearance(V(src), V(c))
                cands.append(vh.Candidate(c, sc, ch))
            d = vh.decide_match(cands, window_start=h["ws"], window_end=h["we"], late_until=h["lu"], cfg=cfg,
                                early=early)
            status = d.status
            target_t = d.chosen.story.started_at if d.chosen else None
            score = d.chosen.score if d.chosen else None
        rec["appearance"] = score
        if status in vh.ARRIVED:
            transit = (target_t - src.ended_at).total_seconds()
            g = vh.observed_g(transit, h["dist"], h["speed"], portal.kind)
            calib_kind = "person" if h["band"].source == "learned_person_widened" else src.kind
            band = vh.Band(h["band"].lo, h["band"].hi, h["band"].source, tuple(learned[portal.id][calib_kind]))
            p = vh.p_value_arrival(g, band, prior_floor=vh.prior_p_floor(cfg.min_arrivals))
            rec["log_score"] = vh.arrival_log_score(g, band, h["speed"], portal.kind)
            learned[portal.id][src.kind].append(g)
            rec.update(status=status, transit=transit, g=g, p=p)
        elif status == "ambiguous":
            rec.update(status="ambiguous")
        else:
            c = cov(census, portal.to_stream, h["ws"], h["we"], cfg.census_gap_sec)
            st = vh.grade_unmatched(c, cfg)
            rec.update(status=st, coverage=c)
            if st == "missed":
                rec["p"] = vh.p_value_missed(missed_n[portal.id][src.kind], len(learned[portal.id][src.kind]))
                missed_n[portal.id][src.kind] += 1
        guess = known.get(src.individual_id)
        rec["known"] = bool(guess)
        rec["gate"] = vh.attention_gate(rec.get("p"), known_person=bool(guess), unusual_hour=False, cfg=cfg)
        rec["gate_if_unusual"] = vh.attention_gate(rec.get("p"), known_person=True, unusual_hour=True, cfg=cfg)
        graded.append(rec)

    for s in sorted(stories, key=lambda s: s.ended_at):
        for h in [h for h in open_ if h["lu"] < s.ended_at]:
            grade(h)
            open_.remove(h)
        if s.drop_reason in vh.NOT_A_DEPARTURE:
            continue
        for portal in topology.of_kind("gap") + topology.of_kind("approach"):
            if portal.kind == "gap":
                dep = vh.gap_departure(s, portal, cfg)
                if dep is None:
                    continue
                dist, speed = dep
            elif vh.approach_departure(s, portal, cfg):
                dist, speed = 0.0, 1.0
            else:
                continue
            band = vh.choose_band(s.kind, learned[portal.id], portal, cfg)
            ws_, we_, lu_ = vh.window_for(s.ended_at, dist, speed, band, cfg)
            open_.append(dict(portal=portal.id, src=s, dist=dist, speed=speed, band=band, ws=ws_, we=we_, lu=lu_))
    for h in open_:
        grade(h)

    overlaps = []
    for portal in topology.of_kind("overlap"):
        a_stream, b_stream = portal.streams
        for d in by_stream[a_stream]:
            if d.kind not in cfg.kinds or d.drop_reason in vh.NOT_A_DEPARTURE:
                continue
            others = [o for o in by_stream[b_stream] if o.kind == d.kind and o.drop_reason not in vh.NOT_AN_ARRIVAL
                      and vh.overlapping(d, o, portal.max_skew_sec)]
            others.sort(key=lambda o: -vh.overlap_seconds(d, o))
            cands = []
            for o in others:
                sc, ch = vh.appearance(V(d), V(o))
                cands.append(vh.Candidate(o, sc, ch))
            dec = vh.decide_overlap(cands, cfg)
            overlaps.append(dict(status=dec.status, door=d.story_id, walk=dec.chosen.story.story_id if dec.chosen else None,
                                 score=dec.chosen.score if dec.chosen else None,
                                 channel=dec.chosen.channel if dec.chosen else None,
                                 at=d.started_at.isoformat(), n=len(cands)))
    unannounced = []
    for t, ev, kind in rings:
        if kind != "doorbell_ring":
            continue
        if not any(r["portal"] == "approach_door" and t - timedelta(seconds=120) <= r["departed"] <= t for r in graded):
            unannounced.append(ev)
    return graded, overlaps, unannounced, learned


def summarize(graded, overlaps, unannounced, learned, tz: ZoneInfo, cfg: vh.HandoffConfig, edges_stored: bool):
    out: dict = {"edges_columns_present": edges_stored}
    per = defaultdict(Counter)
    for r in graded:
        per[r["portal"]][r["status"]] += 1
    out["outcomes"] = {k: dict(v) for k, v in per.items()}
    out["edges_source"] = dict(Counter(r["edges"] for r in graded))
    out["band_source"] = dict(Counter(r["band"] for r in graded))
    out["overlap"] = dict(Counter(o["status"] for o in overlaps))
    out["overlap_pairs"] = [o for o in overlaps if o["status"] == "paired"][:20]
    out["unannounced_rings"] = len(unannounced)
    g_rep = {}
    for portal, kinds in learned.items():
        for kind, gs in kinds.items():
            if not gs:
                continue
            med = statistics.median(gs)
            q = np.quantile(gs, [0.1, 0.9]) if len(gs) >= 2 else (gs[0], gs[0])
            days = defaultdict(list)
            for r in graded:
                if r["portal"] == portal and r["kind"] == kind and r.get("g") is not None:
                    days[r["departed"].astimezone(tz).date().isoformat()].append(r["g"])
            g_rep[f"{portal}/{kind}"] = {
                "n": len(gs), "median": round(med, 3), "p10": round(float(q[0]), 3), "p90": round(float(q[1]), 3),
                "spread_over_median": round(float(q[1] - q[0]) / abs(med), 3) if med else None,
                "per_day_median": {d: round(statistics.median(v), 3) for d, v in sorted(days.items())}}
    out["learned_g"] = g_rep
    met_p = [r["p"] for r in graded if r["status"] == "met" and r.get("p") is not None]
    out["met_p"] = {"n": len(met_p), "ks_uniform": None if not met_p else round(ks_uniform(met_p), 3),
                    "deciles": [sum(1 for p in met_p if i / 10 <= p < (i + 1) / 10 or (i == 9 and p == 1.0))
                                for i in range(10)],
                    "below_0_05": sum(1 for p in met_p if p < 0.05)}
    hours = defaultdict(list)
    for r in graded:
        hours[r["departed"].astimezone(tz).strftime("%Y-%m-%d %H")].append(r)
    quiet = {h: rs for h, rs in hours.items() if not any(r["status"] in ("early", "late", "missed") for r in rs)}
    out["attention_gate"] = {
        "passes": sum(r["gate"] for r in graded),
        "passes_if_every_hour_unusual_and_known": sum(r["gate_if_unusual"] for r in graded),
        "hours_with_handoffs": len(hours), "quiet_hours": len(quiet),
        "quiet_hour_passes": sum(r["gate_if_unusual"] for rs in quiet.values() for r in rs)}
    landed = [r for r in graded if r.get("landed")]
    out["steering"] = {
        "opened": len(landed),
        "window_still_ahead_when_story_landed": sum(1 for r in landed if r["landed"] < r["window_end"]),
        "story_landing_lag_sec_median": round(statistics.median(
            [(r["landed"] - r["departed"]).total_seconds() for r in landed]), 1) if landed else None}
    doorbell_missed = [r for r in graded if r["status"] == "missed" and r["portal"] in ("walkway_to_door", "approach_door")]
    out["doorbell_target"] = {
        "graded": sum(1 for r in graded if r["portal"] in ("walkway_to_door", "approach_door")),
        "unscorable": sum(1 for r in graded if r["portal"] in ("walkway_to_door", "approach_door")
                          and r["status"] == "unscorable"),
        "missed": len(doorbell_missed),
        "missed_min_coverage": min((r["coverage"] for r in doorbell_missed), default=None)}
    # The band switches per portal and kind, so "thin" is judged on the best-fed one.
    arrived_most = max((len(gs) for by_kind in learned.values() for gs in by_kind.values()), default=0)
    reasons = []
    if arrived_most < cfg.min_arrivals:
        reasons.append(f"at most {arrived_most} arrivals for any portal and kind (< {cfg.min_arrivals}): "
                       "no learned band, no uniformity test")
    if len(met_p) < 20:
        reasons.append(f"only {len(met_p)} met p-values (need >= 20 for a uniformity read)")
    if out["doorbell_target"]["missed"] and (out["doorbell_target"]["missed_min_coverage"] or 0) < cfg.min_coverage:
        reasons.append("FAIL: a doorbell-target miss with coverage under the 80% rule")
    if out["attention_gate"]["quiet_hour_passes"] > max(1, 0.05 * len(met_p)):
        reasons.append("FAIL: quiet hours produced gate passes above chance")
    spreads = [v["spread_over_median"] for v in g_rep.values() if v["n"] >= cfg.min_arrivals and v["spread_over_median"]]
    if any(s > 0.5 for s in spreads):
        reasons.append("learned gap spread above 50% of the median (acceptance check)")
    if any(r.startswith("FAIL") for r in reasons):
        out["verdict"] = "FAIL: " + "; ".join(reasons)
    elif arrived_most < cfg.min_arrivals or len(met_p) < 20:
        out["verdict"] = "TOO THIN FOR A VERDICT: " + "; ".join(reasons)
    else:
        out["verdict"] = "PASS" + ("" if not reasons else " with notes: " + "; ".join(reasons))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=None, help="ISO time (default: 7 days ago)")
    ap.add_argument("--until", default=None)
    ap.add_argument("--container", default="orion-athena-sql-db")
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    since = ts(args.since) if args.since else now - timedelta(days=7)
    until = ts(args.until) if args.until else now
    cfg = vh.HandoffConfig()
    tz = ZoneInfo(args.tz)
    topology = load_topology()
    stories, census, rings, known, edges_stored = load(args.container, since, until, cfg.kinds)
    door_times = [s for s in stories if s.stream_id == "doorbell"]
    near = [s for s in stories if s.stream_id == "doorbell" or any(
        abs((s.started_at - d.started_at).total_seconds()) < 900 for d in door_times)]
    vecs = load_vectors(args.container, [c for s in near for c in s.crop_ids[-12:]])
    graded, overlaps, unannounced, learned = replay(stories, census, rings, known, vecs, topology, cfg, tz)
    out = summarize(graded, overlaps, unannounced, learned, tz, cfg, edges_stored)
    out["window"] = {"since": since.isoformat(), "until": until.isoformat(),
                     "stories": dict(Counter(f"{s.stream_id}/{s.kind}" for s in stories)),
                     "census_spans": {k: len(v) for k, v in census.items()},
                     "ring_events": dict(Counter(k for _t, _e, k in rings)), "vectors": len(vecs)}
    text = json.dumps(out, indent=2, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 1 if out["verdict"].startswith("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
