#!/usr/bin/env python3
"""Replay eval: what would the split on "someone else" pictures do on live data?

docs/superpowers/specs/2026-10-16-vision-individual-split-proposal.md, "Replay
eval", made permanent. READ ONLY: one READ ONLY transaction on the live DB
(it cannot write), nothing copied anywhere, no pictures printed.

1. Replays the clusterer (``vision_individuals.apply_batch``, the deployed
   code) over every walkway person crop in 60 s landing-time batches, and
   checks fidelity: how many live person individuals it rebuilds exactly
   (same ``centroid_n``, centroid cosine > 0.999). The crop -> individual
   map is the real membership only when that is all of them.
2. For every answered picture card with a "someone else" picture, runs the
   split rule (``vision_split.plan_split``, the deployed code) on the
   replayed membership as of the answer time: crops moved by the answer,
   face crops that follow, body/face cosine of the new piece vs the rest.
   Also prints what the live split does WITHOUT a membership backfill (only
   crops with a stored individual_id can follow).
3. Runs the face-threshold calibration rule on every labelled face pair and
   prints its verdict.

Asserts the proposal's numbers for 2026-10-16 (62cb5fcc: 2 moved + 0 face
followers; f5242d40: 1 + 3; calibration "hold"). Exit 0 = reproduced,
1 = a number moved, 2 = cannot replay (crops pruned after 7 days, or asks gone).

    python evals/run_vision_split_replay_eval.py --live postgresql://postgres:...@localhost:55432/conjourney
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402
from app import vision_split as vs  # noqa: E402

# The proposal's replay table (2026-10-16): source id prefix -> (moved by answer, face followers).
EXPECTED = {"62cb5fcc": (2, 0), "f5242d40": (1, 3)}
EXPECTED_CALIBRATION = "hold"


def _has_split_migration(conn) -> bool:
    from sqlalchemy import text

    return bool(conn.execute(text(
        "SELECT to_regclass('vision_individual_split') IS NOT NULL AND EXISTS (SELECT 1 FROM "
        "information_schema.columns WHERE table_schema = current_schema() AND "
        "table_name='vision_crop_observation' AND column_name='individual_id')")).scalar())


def _crops(conn, stream: str, migrated: bool):
    from sqlalchemy import text

    stored = "individual_id IS NOT NULL" if migrated else "false"
    rows = conn.execute(text(
        "SELECT crop_id, observation_id, stream_id, observed_at, created_at, label, box_xyxy, zone, embedding, "
        "embedding_ref, artifact_id, zone_no_embed, thumb_ref, face_embedding, face_thumb_ref, "
        f"{stored} AS stored "
        "FROM vision_crop_observation WHERE stream_id=:s AND label='person' AND embedding IS NOT NULL "
        "ORDER BY created_at, crop_id"), {"s": stream}).fetchall()
    return rows


def replay(rows, cfg: vi.IndividualsConfig, tz: ZoneInfo):
    """crop_id -> replay individual id, replay individuals, crop_id -> landing batch end."""
    batches = defaultdict(list)
    for r in rows:
        batches[int(r.created_at.timestamp() // 60)].append(r)
    inds = {}
    member = {}
    for key in sorted(batches):
        crops = [vi._row_to_crop(r) for r in batches[key]]
        res = vi.apply_batch(crops, individuals=inds, latest_sightings={}, no_embed_zones=(),
                             match_threshold=cfg.match_threshold, merge_gap_sec=cfg.merge_gap_sec, tz=tz,
                             face_match_threshold=cfg.face_match_threshold or None,
                             stationary_min_sec=cfg.stationary_min_sec, stationary_iou=cfg.stationary_iou,
                             stationary_max_gap_sec=cfg.stationary_max_gap_sec)
        inds = res.individuals
        member.update(res.crop_individual)
    return member, inds


def match_live(live_inds, replay_inds):
    """live id -> replay id, for live person individuals rebuilt exactly."""
    out = {}
    by_n = defaultdict(list)
    for r in replay_inds.values():
        by_n[r.centroid_n].append(r)
    for li in live_inds:
        for r in by_n.get(li.centroid_n, []):
            if vi.cosine(list(li.centroid), r.centroid) > 0.999:
                out[li.individual_id] = r.individual_id
                break
    return out


def calibration_pairs(conn, crop_face, migrated: bool):
    """(same cosines, different cosines): pair verdicts with a face cosine, plus
    picture verdicts expanded into crop pairs by the person each picture is."""
    from sqlalchemy import text

    same, diff = [], []
    # A split's own cannot-link is not a labelled pair: Juniper judged pictures, not those centroids.
    not_split = " AND ask_id NOT IN (SELECT ask_id FROM vision_individual_split)" if migrated else ""
    for v, c in conn.execute(text(
            "SELECT verdict, face_cosine FROM vision_individual_pair_verdict WHERE face_cosine IS NOT NULL"
            + not_split)).fetchall():
        (same if v == "same" else diff).append(float(c))
    pics = conn.execute(text(
        "SELECT p.ask_id, p.picture_index, p.verdict, p.name, p.crop_id, a.answer FROM vision_ask_picture_verdict p "
        "JOIN orion_ask a ON a.ask_id = p.ask_id WHERE p.verdict IN ('same', 'different') "
        "ORDER BY p.ask_id, p.picture_index")).fetchall()
    people = []
    for p in pics:
        face = crop_face.get(p.crop_id)
        if not face:
            continue
        if p.verdict == "same":
            who = vi._clean_name(p.answer).lower() or None
        else:
            who = vi._clean_name(p.name).lower() or None
        people.append((p.ask_id, who or f"unnamed:{p.ask_id}:{p.picture_index}", who is not None, face))
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            if (not a[2] or not b[2]) and a[0] != b[0]:
                continue  # an unnamed picture is only known to differ from its own card
            (same if a[1] == b[1] else diff).append(vi.cosine(a[3], b[3]))
    return same, diff


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True, help="live DSN (read in a READ ONLY transaction)")
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from sqlalchemy import create_engine, text

    # The live individuals this eval rebuilds were clustered by the rule
    # before 2026-10-16 (single body line 0.80, no continuation lane); the
    # body-match patch's own eval covers the new rule.
    cfg = vi.IndividualsConfig(match_threshold=0.80, continuation_enabled=False)
    tz = ZoneInfo(cfg.local_tz)
    engine = create_engine(args.live)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "splits": [], "problems": []}
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        migrated = _has_split_migration(conn)
        rows = _crops(conn, args.stream, migrated)
        live_inds = conn.execute(text(
            "SELECT individual_id, centroid, centroid_n, label FROM vision_individual "
            "WHERE stream_id=:s AND kind='person'"), {"s": args.stream}).fetchall()
        member, rinds = replay(rows, cfg, tz)
        mapping = match_live(live_inds, rinds)
        report["fidelity"] = {"live_person_individuals": len(live_inds), "rebuilt_exactly": len(mapping),
                              "crops": len(rows), "crops_with_stored_membership": sum(1 for r in rows if r.stored)}
        by_crop = {r.crop_id: r for r in rows}
        crop_face = {r.crop_id: list(r.face_embedding) for r in rows if r.face_embedding}
        verdicts = conn.execute(text(
            "SELECT ask_id, individual_id, picture_index, verdict, name, crop_id, answered_at, "
            "lower(coalesce(name, '')) AS gk FROM vision_ask_picture_verdict "
            "WHERE ask_id IN (SELECT ask_id FROM vision_ask_picture_verdict WHERE verdict='different') "
            "ORDER BY answered_at, ask_id, picture_index")).fetchall()
        asks = defaultdict(list)
        for v in verdicts:
            asks[(v.ask_id, v.individual_id)].append(v)
        for (ask_id, src), vs_rows in asks.items():
            pvs = [vs.PictureVerdict(v.picture_index, v.verdict, v.name, v.crop_id,
                                     v.gk if v.verdict == "different" else "") for v in vs_rows]
            keep, groups = vs.verdict_groups(pvs)
            answered = vs_rows[0].answered_at
            rid = mapping.get(src)
            for gk, (name, move_ids) in sorted(groups.items()):
                entry = {"ask": ask_id[:12], "source": src[:8], "group": "named" if gk else "unnamed"}
                if rid is None:
                    entry["problem"] = "source not rebuilt by the replay"
                    report["splits"].append(entry)
                    continue
                at_answer = [by_crop[c] for c, i in member.items() if i == rid and by_crop[c].created_at <= answered]
                seeds = [by_crop[c] for c in move_ids if c in by_crop and member.get(c) == rid]
                if not seeds:
                    entry["problem"] = "no named crop is still in the replay (pruned?)"
                    report["splits"].append(entry)
                    continue

                def sc(r):
                    return vs.SplitCrop(r.crop_id, r.observed_at, tuple(r.embedding),
                                        tuple(r.face_embedding) if r.face_embedding else None)
                keep_seed = [sc(by_crop[c]) for c in keep if c in by_crop]
                plan = vs.plan_split(
                    source_n=len(at_answer), move_seed=[sc(r) for r in seeds], keep_seed=keep_seed,
                    members=[sc(r) for r in at_answer],
                    follow_min_face_cosine=cfg.split_follow_min_face_cosine, follow_margin=cfg.split_follow_margin,
                    max_move_fraction=cfg.split_max_move_fraction)
                live_plan = vs.plan_split(
                    source_n=len(at_answer), move_seed=[sc(r) for r in seeds], keep_seed=keep_seed,
                    members=[sc(r) for r in at_answer if r.stored],
                    follow_min_face_cosine=cfg.split_follow_min_face_cosine, follow_margin=cfg.split_follow_margin,
                    max_move_fraction=cfg.split_max_move_fraction)
                moved = {c for c, _ in plan.moved}
                new_rows = [by_crop[c] for c in moved]
                rest = [r for r in at_answer if r.crop_id not in moved]
                nb, rb = vs.mean_unit(r.embedding for r in new_rows), vs.mean_unit(r.embedding for r in rest)
                nf = vs.mean_unit(r.face_embedding for r in new_rows if r.face_embedding)
                rf = vs.mean_unit(r.face_embedding for r in rest if r.face_embedding)
                entry.update({
                    "members_at_answer": len(at_answer), "moved_by_answer": plan.by_verdict,
                    "face_followers": plan.by_face, "stay": len(rest), "refused": plan.refused,
                    "body_cos_new_vs_rest": round(vi.cosine(nb, rb), 3) if nb and rb else None,
                    "face_cos_new_vs_rest": round(vi.cosine(nf, rf), 3) if nf and rf else None,
                    "live_without_backfill": [live_plan.by_verdict, live_plan.by_face],
                })
                report["splits"].append(entry)
        same, diff = calibration_pairs(conn, crop_face, migrated)
        report["calibration"] = vs.face_threshold_calibration(same, diff)
        conn.rollback()
    engine.dispose()

    fid = report["fidelity"]
    if fid["rebuilt_exactly"] != fid["live_person_individuals"]:
        report["problems"].append(f"replay rebuilt {fid['rebuilt_exactly']}/{fid['live_person_individuals']} "
                                  "live person individuals: membership is not exact")
    status = 0
    for prefix, (want_moved, want_follow) in EXPECTED.items():
        hits = [e for e in report["splits"] if e["source"] == prefix and "moved_by_answer" in e]
        if not hits:
            report["problems"].append(f"{prefix}: cannot replay (crops pruned or ask missing)")
            status = max(status, 2)
            continue
        got = (sum(e["moved_by_answer"] for e in hits), sum(e["face_followers"] for e in hits))
        if got != (want_moved, want_follow):
            report["problems"].append(f"{prefix}: expected {want_moved}+{want_follow}, got {got[0]}+{got[1]}")
            status = max(status, 1)
    if report["calibration"]["verdict"] != EXPECTED_CALIBRATION:
        report["problems"].append(f"calibration: expected {EXPECTED_CALIBRATION}, got "
                                  f"{report['calibration']['verdict']}")
        status = max(status, 1)
    if report["problems"] and status == 0 and fid["rebuilt_exactly"] != fid["live_person_individuals"]:
        status = 1

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"replay fidelity: {fid['rebuilt_exactly']}/{fid['live_person_individuals']} live person "
              f"individuals rebuilt exactly from {fid['crops']} crops "
              f"({fid['crops_with_stored_membership']} with stored membership)")
        for e in report["splits"]:
            if "problem" in e:
                print(f"  ask {e['ask']} source {e['source']} ({e['group']}): {e['problem']}")
                continue
            print(f"  ask {e['ask']} source {e['source']} ({e['group']}): members {e['members_at_answer']}, "
                  f"moved {e['moved_by_answer']} + {e['face_followers']} face followers, stay {e['stay']}, "
                  f"body cos {e['body_cos_new_vs_rest']}, face cos {e['face_cos_new_vs_rest']}"
                  f"{', REFUSED ' + e['refused'] if e['refused'] else ''}; live without backfill: "
                  f"{e['live_without_backfill'][0]} + {e['live_without_backfill'][1]}")
        c = report["calibration"]
        print(f"face threshold calibration: {c['verdict'].upper()} (same={c['same_pairs']}, "
              f"different={c['different_pairs']}, min same={c['min_same']}, max different={c['max_different']}) "
              f"{c.get('reason') or ''}")
        for p in report["problems"]:
            print(f"PROBLEM: {p}")
        print({0: "REPRODUCED", 1: "MISMATCH", 2: "CANNOT REPLAY"}[status])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
