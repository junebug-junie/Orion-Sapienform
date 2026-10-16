#!/usr/bin/env python3
"""Re-anchor named walkway individuals' body centroids on their own face crops.

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "Re-anchor"
(Juniper approved 2026-10-16). The body-match patch only changes how NEW crops
join. A named individual built under the old 0.80 line carries a body
centroid averaged over several people (62cb5fcc "name A" points at a child
in a chair: 96% of her crops still score >= 0.90 against it), so it keeps
pulling the wrong person in. This rebuilds that centroid from the crops whose
FACE says they are this individual.

Targets: person individuals on the stream with a label that was typed for
them (``label_copied`` false), plus copied labels that share a typed name
(case-insensitive) with one of those.

Members (face-bearing crops only -- ``face_embedding`` and ``embedding`` both
present), the rule recorded as ``rule`` in the audit row:

1. stored membership: ``vision_crop_observation.individual_id`` = the target
   (written since manual_migration_vision_split_v1.sql); a crop stored on any
   OTHER individual is never used (that includes crops a split moved out);
2. unknown membership (``individual_id`` NULL, the crops older than the split
   migration): the target is the crop's nearest face centroid among ALL
   person individuals on the stream (what the face lane would pick);
3. and in both cases the crop's face cosine to the target's face centroid is
   >= ``--face-threshold`` (default VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD,
   0.60): a mixed face set does not drag the new centroid to the wrong side.

New centroid = the normalized mean of those crops' unit body vectors;
``centroid_n`` = how many crops. ``face_centroid`` is not touched. A target
with no such crop is left alone and reported.

Dry run by default: prints, per target, old vs new centroid cosine, crops
used (stored / face-nearest), and how many person crops of the last 24 h
score >= ``--line`` (default VISION_INDIVIDUALS_MATCH_THRESHOLD, 0.90)
against the old vs the new centroid. Nothing is written.

``--apply``: one transaction, under the individuals loop's advisory lock (so
a tick cannot overwrite the new centroid with one it read before): recompute
on the locked rows, write a snapshot of the old rows to
``/tmp/vision-reanchor/<ts>/before.json`` (0600; biometric-adjacent -- delete
after 7 days), update the rows, write one ``vision_individual_reanchor`` row
each (manual_migration_vision_reanchor_v1.sql), commit.

``--undo <before.json>``: dry run of putting the snapshot's centroid and
centroid_n back (byte-for-byte); with ``--apply`` it commits, audited as
action 'undo'. Crops that joined after the re-anchor are folded into the
restored centroid only by the matcher's later ticks, not by undo.

DSN: --dsn, else POSTGRES_URI, else DATABASE_URL.

    python3 scripts/reanchor_vision_individuals.py --dsn postgresql://postgres:...@localhost:55432/conjourney
    python3 scripts/reanchor_vision_individuals.py --dsn ... --apply
    python3 scripts/reanchor_vision_individuals.py --dsn ... --undo /tmp/vision-reanchor/<ts>/before.json --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "services" / "orion-sql-writer"
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

SNAPSHOT_ROOT = Path("/tmp/vision-reanchor")
RULE = "face crops: stored membership, else nearest face centroid; face cosine to target >= {t:.2f}"


@dataclass
class Target:
    individual_id: str
    label: str
    label_copied: bool
    centroid: List[float]
    centroid_n: int
    face_centroid: Optional[List[float]]
    updated_at: Optional[datetime] = None


@dataclass
class Plan:
    target: Target
    stored: List[str] = field(default_factory=list)
    face_nearest: List[str] = field(default_factory=list)
    new_centroid: Optional[List[float]] = None
    old_new_cosine: Optional[float] = None
    recent_total: int = 0
    recent_pass_old: int = 0
    recent_pass_new: Optional[int] = None

    @property
    def crop_ids(self) -> List[str]:
        return sorted(self.stored + self.face_nearest)


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def load_targets(conn, stream: str, for_update: bool = False) -> List[Target]:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT individual_id, label, label_copied, centroid, centroid_n, face_centroid, updated_at "
        "FROM vision_individual WHERE stream_id=:s AND kind='person' AND label IS NOT NULL "
        "AND (NOT label_copied OR lower(label) IN (SELECT lower(label) FROM vision_individual "
        "  WHERE stream_id=:s AND kind='person' AND label IS NOT NULL AND NOT label_copied)) "
        "ORDER BY individual_id" + (" FOR UPDATE" if for_update else "")), {"s": stream}).fetchall()
    return [Target(r.individual_id, r.label, bool(r.label_copied), [float(v) for v in r.centroid],
                   int(r.centroid_n), [float(v) for v in r.face_centroid] if r.face_centroid else None,
                   r.updated_at) for r in rows]


def plan_reanchor(conn, stream: str, targets: Sequence[Target], *, face_threshold: float, line: float,
                  now: datetime, recent_hours: float = 24.0) -> List[Plan]:
    from sqlalchemy import text

    by_id = {t.individual_id: t for t in targets}
    plans = {t.individual_id: Plan(t) for t in targets}
    # Every person face centroid on the stream: "nearest face centroid" is over all of them.
    fc_rows = conn.execute(text(
        "SELECT individual_id, face_centroid FROM vision_individual "
        "WHERE stream_id=:s AND kind='person' AND face_centroid IS NOT NULL ORDER BY individual_id"),
        {"s": stream}).fetchall()
    fc_ids = [r.individual_id for r in fc_rows]
    fc_mat = np.vstack([_unit(r.face_centroid) for r in fc_rows]) if fc_rows else None
    crops = conn.execute(text(
        "SELECT crop_id, individual_id, embedding, face_embedding FROM vision_crop_observation "
        "WHERE stream_id=:s AND label='person' AND embedding IS NOT NULL AND face_embedding IS NOT NULL "
        "ORDER BY crop_id"), {"s": stream}).fetchall()
    bodies: Dict[str, List[np.ndarray]] = {t: [] for t in by_id}
    for c in crops:
        face = _unit(c.face_embedding)
        if c.individual_id is not None:
            tid, how = c.individual_id, "stored"
        else:
            if fc_mat is None or fc_mat.shape[1] != face.shape[0]:
                continue
            tid, how = fc_ids[int(np.argmax(fc_mat @ face))], "face_nearest"
        t = by_id.get(tid)
        if t is None or not t.face_centroid or len(t.face_centroid) != face.shape[0]:
            continue
        if float(_unit(t.face_centroid) @ face) < face_threshold:
            continue
        (plans[tid].stored if how == "stored" else plans[tid].face_nearest).append(c.crop_id)
        bodies[tid].append(_unit(c.embedding))
    recent = conn.execute(text(
        "SELECT embedding FROM vision_crop_observation WHERE stream_id=:s AND label='person' "
        "AND embedding IS NOT NULL AND created_at > :since"),
        {"s": stream, "since": now - timedelta(hours=recent_hours)}).fetchall()
    for tid, p in plans.items():
        old = _unit(p.target.centroid)
        rmat = [_unit(r.embedding) for r in recent if len(r.embedding) == old.shape[0]]
        rmat = np.vstack(rmat) if rmat else np.zeros((0, old.shape[0]))
        p.recent_total = int(rmat.shape[0])
        p.recent_pass_old = int(np.sum(rmat @ old >= line)) if rmat.size else 0
        if bodies[tid]:
            new = _unit(np.sum(np.vstack(bodies[tid]), axis=0))
            p.new_centroid = [float(v) for v in new.astype(np.float32)]
            p.old_new_cosine = round(float(old @ new), 4)
            p.recent_pass_new = int(np.sum(rmat @ new >= line)) if rmat.size else 0
    return [plans[t.individual_id] for t in targets]


def report(plans: Sequence[Plan], line: float) -> List[dict]:
    out = []
    for p in plans:
        out.append({
            "individual_id": p.target.individual_id, "label": p.target.label, "label_copied": p.target.label_copied,
            "old_centroid_n": p.target.centroid_n,
            "new_centroid_n": len(p.crop_ids) if p.new_centroid else None,
            "crops_used": {"stored": len(p.stored), "face_nearest": len(p.face_nearest)},
            "old_new_cosine": p.old_new_cosine,
            f"last24h_pass_{line:.2f}": {"total": p.recent_total, "old": p.recent_pass_old, "new": p.recent_pass_new},
            "action": "reanchor" if p.new_centroid else "skip: no face crop passes the rule",
        })
    return out


def _print_table(rows: Sequence[dict], line: float) -> None:
    key = f"last24h_pass_{line:.2f}"
    print(f"{'individual':<10} {'label':<14} {'copied':<6} {'n old':>6} {'n new':>6} {'stored':>6} "
          f"{'face':>5} {'cos':>7}  last 24 h >= {line:.2f}: old -> new (of total)")
    for r in rows:
        k = r[key]
        print(f"{r['individual_id'][:8]:<10} {r['label'][:14]:<14} {str(r['label_copied'])[0]:<6} "
              f"{r['old_centroid_n']:>6} {str(r['new_centroid_n'] or '-'):>6} {r['crops_used']['stored']:>6} "
              f"{r['crops_used']['face_nearest']:>5} {str(r['old_new_cosine'] if r['old_new_cosine'] is not None else '-'):>7}"
              f"  {k['old']} -> {k['new'] if k['new'] is not None else '-'} (of {k['total']})"
              + ("" if r["action"] == "reanchor" else f"   {r['action']}"))


def _has_audit_table(conn) -> bool:
    from sqlalchemy import text

    return bool(conn.execute(text("SELECT to_regclass('vision_individual_reanchor') IS NOT NULL")).scalar())


def _lock(conn, lock_timeout: str = "60s") -> None:
    """The individuals loop's advisory lock, for this transaction."""
    from sqlalchemy import text

    from app.vision_individuals import INDIVIDUALS_LOCK_KEY

    conn.execute(text("SELECT set_config('lock_timeout', :t, true)"), {"t": lock_timeout})
    conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": INDIVIDUALS_LOCK_KEY})


def _write_snapshot(root: Path, payload: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    d = root / stamp
    d.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = d / "before.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    return path


def _audit(conn, run_id: str, iid: str, action: str, rule: str, crop_ids, before: dict, after: dict,
           snapshot: Optional[str]) -> None:
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual_reanchor (reanchor_id, run_id, individual_id, action, rule, crop_ids, "
        "before, after, snapshot_path) VALUES (:id, :run, :i, :a, :r, CAST(:c AS jsonb), CAST(:b AS jsonb), "
        "CAST(:af AS jsonb), :p)"),
        {"id": f"reanchor-{uuid.uuid4().hex}", "run": run_id, "i": iid, "a": action, "r": rule,
         "c": json.dumps(list(crop_ids)), "b": json.dumps(before), "af": json.dumps(after), "p": snapshot})


def apply_reanchor(engine, stream: str, *, face_threshold: float, line: float, now: datetime,
                   snapshot_root: Path = SNAPSHOT_ROOT, lock_timeout: str = "60s") -> dict:
    from sqlalchemy import text

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    rule = RULE.format(t=face_threshold)
    with engine.begin() as conn:
        if not _has_audit_table(conn):
            raise SystemExit("vision_individual_reanchor is missing: apply "
                             "services/orion-sql-db/manual_migration_vision_reanchor_v1.sql first")
        _lock(conn, lock_timeout)
        targets = load_targets(conn, stream, for_update=True)
        plans = plan_reanchor(conn, stream, targets, face_threshold=face_threshold, line=line, now=now)
        rows = report(plans, line)
        snap = _write_snapshot(snapshot_root, {
            "run_id": run_id, "stream": stream, "created_at": datetime.now(timezone.utc).isoformat(),
            "rule": rule, "rows": [{"individual_id": t.individual_id, "label": t.label, "centroid": t.centroid,
                                    "centroid_n": t.centroid_n, "updated_at": t.updated_at} for t in targets],
            "plan": rows})
        written = 0
        for p in plans:
            if not p.new_centroid:
                continue
            n = len(p.crop_ids)
            conn.execute(text(
                "UPDATE vision_individual SET centroid=CAST(:c AS real[]), centroid_n=:n, updated_at=now() "
                "WHERE individual_id=:i"), {"c": p.new_centroid, "n": n, "i": p.target.individual_id})
            _audit(conn, run_id, p.target.individual_id, "reanchor", rule, p.crop_ids,
                   {"centroid": p.target.centroid, "centroid_n": p.target.centroid_n},
                   {"centroid": p.new_centroid, "centroid_n": n}, str(snap))
            written += 1
    return {"run_id": run_id, "snapshot": str(snap), "written": written, "rows": rows}


def undo(engine, snapshot: Path, *, apply: bool, lock_timeout: str = "60s") -> dict:
    from sqlalchemy import text

    snap = json.loads(Path(snapshot).read_text())
    run_id = f"undo-{uuid.uuid4().hex[:12]}"
    conn = engine.connect()
    trans = conn.begin()
    out = {"snapshot": str(snapshot), "restored": [], "missing": []}
    try:
        if apply and not _has_audit_table(conn):
            raise SystemExit("vision_individual_reanchor is missing: apply the migration first")
        _lock(conn, lock_timeout)
        for r in snap["rows"]:
            cur = conn.execute(text(
                "SELECT centroid, centroid_n FROM vision_individual WHERE individual_id=:i FOR UPDATE"),
                {"i": r["individual_id"]}).fetchone()
            if cur is None:
                out["missing"].append(r["individual_id"])
                continue
            now_c = [float(v) for v in cur.centroid]
            if now_c == r["centroid"] and int(cur.centroid_n) == int(r["centroid_n"]):
                continue
            conn.execute(text(
                "UPDATE vision_individual SET centroid=CAST(:c AS real[]), centroid_n=:n, updated_at=now() "
                "WHERE individual_id=:i"), {"c": r["centroid"], "n": int(r["centroid_n"]), "i": r["individual_id"]})
            if apply:
                _audit(conn, run_id, r["individual_id"], "undo", "restore snapshot", [],
                       {"centroid": now_c, "centroid_n": int(cur.centroid_n)},
                       {"centroid": r["centroid"], "centroid_n": int(r["centroid_n"])}, str(snapshot))
            out["restored"].append({"individual_id": r["individual_id"], "centroid_n_now": int(cur.centroid_n),
                                    "centroid_n_restored": int(r["centroid_n"])})
        if apply:
            trans.commit()
        else:
            trans.rollback()
        return out
    except BaseException:
        trans.rollback()
        raise
    finally:
        conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.getenv("POSTGRES_URI") or os.getenv("DATABASE_URL"))
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run, nothing written)")
    ap.add_argument("--undo", metavar="BEFORE_JSON", help="restore a snapshot (dry run unless --apply)")
    ap.add_argument("--face-threshold", type=float, default=None)
    ap.add_argument("--line", type=float, default=None, help="body line for the last-24h check")
    ap.add_argument("--now", help="ISO time for the last-24h window (default: now)")
    ap.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not args.dsn:
        print("no DSN: pass --dsn or set POSTGRES_URI", file=sys.stderr)
        return 2
    from sqlalchemy import create_engine, text

    from app.vision_individuals import IndividualsConfig

    cfg = IndividualsConfig()
    face_t = cfg.face_match_threshold if args.face_threshold is None else args.face_threshold
    line = cfg.match_threshold if args.line is None else args.line
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    engine = create_engine(args.dsn)
    try:
        if args.undo:
            out = undo(engine, Path(args.undo), apply=args.apply)
            print(json.dumps(out, indent=2))
            print("APPLIED" if args.apply else "DRY RUN: rolled back. Re-run with --apply to restore.")
            return 0
        if args.apply:
            out = apply_reanchor(engine, args.stream, face_threshold=face_t, line=line, now=now,
                                 snapshot_root=Path(args.snapshot_root))
            print(json.dumps(out, indent=2) if args.json else "")
            if not args.json:
                _print_table(out["rows"], line)
            print(f"APPLIED run={out['run_id']} rows={out['written']} snapshot={out['snapshot']}")
            return 0
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            targets = load_targets(conn, args.stream)
            plans = plan_reanchor(conn, args.stream, targets, face_threshold=face_t, line=line, now=now)
            conn.rollback()
        rows = report(plans, line)
        if args.json:
            print(json.dumps({"rule": RULE.format(t=face_t), "line": line, "rows": rows}, indent=2))
        else:
            print(f"rule: {RULE.format(t=face_t)}")
            _print_table(rows, line)
            print("DRY RUN: nothing written. Re-run with --apply to write (snapshot first).")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
