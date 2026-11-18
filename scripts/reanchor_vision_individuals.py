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

``--undo <before.json>``: puts back the old centroid and centroid_n
(byte-for-byte) of every individual that run's committed audit rows say it
rewrote -- nothing else. Dry run unless ``--apply`` (audited as action
'undo'). Refused when the matcher has folded new crops into one of them
since; ``--force`` restores anyway and those crops drop out of its centroid
(they stay members; nothing re-adds them).

After a re-anchor, a split of the same individual (app/vision_split.py)
subtracts only moved crops that are part of the re-anchored centroid (the
run's ``crop_ids``, or crops that landed after it); an undo of an earlier
split does the same when it adds crops back.

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
    # Face centroids by size: a model change leaves two sizes side by side;
    # a crop is compared with the centroids of its own size only.
    fc_by_dim: Dict[int, List] = {}
    for r in fc_rows:
        fc_by_dim.setdefault(len(r.face_centroid), []).append(r)
    fc = {d: ([r.individual_id for r in rs], np.vstack([_unit(r.face_centroid) for r in rs]))
          for d, rs in fc_by_dim.items()}
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
            if face.shape[0] not in fc:
                continue
            ids, mat = fc[face.shape[0]]
            tid, how = ids[int(np.argmax(mat @ face))], "face_nearest"
        t = by_id.get(tid)
        if t is None or not t.face_centroid or len(t.face_centroid) != face.shape[0]:
            continue
        if len(c.embedding) != len(t.centroid):
            continue   # body vector from another model than the centroid's
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


def _same_real(a, b) -> bool:
    """Equal as Postgres real[] values (float32), whatever decimal form they came in."""
    return a is not None and b is not None and len(a) == len(b) and bool(
        np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def undo(engine, snapshot: Path, *, apply: bool, force: bool = False, lock_timeout: str = "60s") -> dict:
    """Put back the centroids one committed re-anchor run wrote.

    Driven by the audit rows of the snapshot's run (``action='reanchor'``),
    not by the snapshot's row list: a target the run skipped is never
    touched, and a run whose transaction never committed has nothing to
    undo. An individual whose (centroid, centroid_n) is no longer exactly what
    the run wrote -- the matcher folded new crops in since -- is refused
    (nothing is written) unless ``force``; with ``force`` those later crops
    are dropped from its centroid. Dry run: reads only, no lock, no write."""
    from sqlalchemy import text

    snap = json.loads(Path(snapshot).read_text())
    out: Dict[str, object] = {"snapshot": str(snapshot), "run_id": snap.get("run_id"), "restore": [],
                              "changed_since": [], "missing": [], "already_restored": []}
    conn = engine.connect()
    trans = conn.begin()
    try:
        if not _has_audit_table(conn):
            raise SystemExit("vision_individual_reanchor is missing: nothing was ever re-anchored")
        if apply:
            _lock(conn, lock_timeout)
        audits = conn.execute(text(
            "SELECT individual_id, before, after FROM vision_individual_reanchor "
            "WHERE run_id=:r AND action='reanchor' ORDER BY individual_id"), {"r": snap.get("run_id")}).fetchall()
        if not audits:
            out["ok"] = False
            out["reason"] = "no committed re-anchor rows for this run"
            trans.rollback()
            return out
        todo = []
        for a in audits:
            cur = conn.execute(text(
                "SELECT centroid, centroid_n FROM vision_individual WHERE individual_id=:i"
                + (" FOR UPDATE" if apply else "")), {"i": a.individual_id}).fetchone()
            if cur is None:
                out["missing"].append(a.individual_id)
                continue
            now_c, now_n = [float(v) for v in cur.centroid], int(cur.centroid_n)
            before, after = a.before, a.after
            if _same_real(now_c, before["centroid"]) and now_n == int(before["centroid_n"]):
                out["already_restored"].append(a.individual_id)
                continue
            entry = {"individual_id": a.individual_id, "centroid_n_now": now_n,
                     "centroid_n_written_by_run": int(after["centroid_n"]),
                     "centroid_n_restored": int(before["centroid_n"])}
            if not _same_real(now_c, after["centroid"]) or now_n != int(after["centroid_n"]):
                out["changed_since"].append(entry)
            todo.append((a.individual_id, now_c, now_n, before, entry))
        if out["changed_since"] and not force:
            out["ok"] = False
            out["reason"] = ("centroids changed since the re-anchor (new crops joined); --force drops "
                             "those crops from the restored centroid")
            trans.rollback()
            return out
        run_id = f"undo-{uuid.uuid4().hex[:12]}"
        for iid, now_c, now_n, before, entry in todo:
            out["restore"].append(entry)
            if apply:
                conn.execute(text(
                    "UPDATE vision_individual SET centroid=CAST(:c AS real[]), centroid_n=:n, updated_at=now() "
                    "WHERE individual_id=:i"), {"c": before["centroid"], "n": int(before["centroid_n"]), "i": iid})
                _audit(conn, run_id, iid, "undo", f"restore run {snap.get('run_id')}", [],
                       {"centroid": now_c, "centroid_n": now_n}, before, str(snapshot))
        out["ok"] = True
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
    ap.add_argument("--force", action="store_true",
                    help="--undo even when crops joined since the re-anchor (they drop out of the centroid)")
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
            out = undo(engine, Path(args.undo), apply=args.apply, force=args.force)
            print(json.dumps(out, indent=2))
            if not out.get("ok"):
                print(f"REFUSED: {out.get('reason')}. Nothing changed.", file=sys.stderr)
                return 1
            print("APPLIED" if args.apply else "DRY RUN: nothing written. Re-run with --apply to restore.")
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
