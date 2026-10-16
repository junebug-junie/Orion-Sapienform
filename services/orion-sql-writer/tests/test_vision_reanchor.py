"""scripts/reanchor_vision_individuals.py against a REAL Postgres.

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "Re-anchor".
Uses ``ORION_TEST_PG_URI`` (a throwaway database: it TRUNCATEs the walkway
tables) with the walkway migrations applied:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 vision_split_v1 \\
             vision_reanchor_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_reanchor.py

The world: "m" is named, but its body centroid points at a different person
(the chair-sitter, body C) -- the 62cb5fcc shape. Its own face crops carry
body A.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRIPT = REPO / "scripts" / "reanchor_vision_individuals.py"
MIGRATION = REPO / "services" / "orion-sql-db" / "manual_migration_vision_reanchor_v1.sql"
PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
NOW = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)


def _vec(*xs: float, dim: int = 8) -> list:
    v = list(xs) + [0.0] * (dim - len(xs))
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


BODY_A = _vec(1, 0.1)          # the named person's body
BODY_C = _vec(0.1, 1)          # the chair-sitter's body
MIXED = _vec(0.3, 1)           # m's stored centroid: mostly C
FACE_M = _vec(1, 0, 0)
FACE_O = _vec(0, 1, 0)
FACE_Z = _vec(0, 0, 1)


def _load_script():
    spec = importlib.util.spec_from_file_location("reanchor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reanchor"] = mod   # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


def test_migration_is_auto_apply_safe() -> None:
    from orion.sql_migration_drift import auto_apply_refusals

    text = MIGRATION.read_text()
    assert auto_apply_refusals(text, {"vision_individual", "vision_crop_observation"}) == []
    assert "ORION-MIGRATION-REQUIRED-BY: none" in text.splitlines()[0]


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE vision_individual_sighting, vision_individual, vision_crop_observation, "
                          "vision_individual_reanchor CASCADE"))
    yield engine
    # Other test files share this throwaway db and do not know this table:
    # leave no re-anchor in force behind.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE vision_individual_reanchor"))
    engine.dispose()


def _ind(conn, iid, *, label=None, copied=False, body=MIXED, face=FACE_M, n=50):
    from sqlalchemy import text

    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, label_copied, face_centroid, face_n) "
        "VALUES (:i, 'walkway', 'person', :c, :n, :t, :t, 1, 1, :l, :cp, :f, 5)"),
        {"i": iid, "c": body, "n": n, "t": NOW - timedelta(hours=2), "l": label, "cp": copied,
         "f": face})


def _crop(conn, cid, *, member=None, body=BODY_A, face=None, hours_ago=1.0):
    from sqlalchemy import text

    t = NOW - timedelta(hours=hours_ago)
    conn.execute(text(
        "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
        "box_xyxy, zone, embedding, face_embedding, individual_id, created_at) VALUES "
        "(:c, :o, 'walkway', :t, 'person', 0.9, ARRAY[0,0,10,10]::real[], 'walk', :e, :f, :m, :t)"),
        {"c": cid, "o": "obs-" + cid, "t": t, "e": body, "f": face, "m": member})


def _world(conn):
    _ind(conn, "m", label="Name A")
    _ind(conn, "cp", label="name a", copied=True, face=_vec(1, 0.5, 0))    # copied, shares a typed name
    _ind(conn, "lone", label="Name Z", copied=True, face=FACE_Z)             # copied, no typed twin: not a target
    _ind(conn, "o", face=FACE_O, body=BODY_C)                                 # unlabelled
    _crop(conn, "m-stored", member="m", face=FACE_M)
    for k in range(3):
        _crop(conn, f"m-face{k}", face=_vec(1, 0.02 * k, 0))                  # unknown membership, nearest m
    _crop(conn, "moved-out", member="o", face=FACE_M)                         # stored elsewhere: never used
    _crop(conn, "o-face", face=FACE_O, body=BODY_C)                           # nearest o
    _crop(conn, "m-bad-face", member="m", face=_vec(0.3, 1, 0), body=BODY_C)  # face < 0.60 to m: guard
    for k in range(10):
        _crop(conn, f"chair{k}", body=_vec(0.1, 1, 0.01 * k))                 # body only, chair-sitter
    _crop(conn, "old-chair", body=BODY_C, hours_ago=30)                       # outside the 24 h window


def _rows(conn):
    from sqlalchemy import text

    return {r.individual_id: ([float(v) for v in r.centroid], r.centroid_n, r.updated_at) for r in conn.execute(text(
        "SELECT individual_id, centroid, centroid_n, updated_at FROM vision_individual"))}


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), "--dsn", PG, "--now", NOW.isoformat(), *args],
                          capture_output=True, text=True)


@needs_pg
def test_plan_uses_only_the_targets_own_face_crops(pg) -> None:
    mod = _load_script()
    with pg.begin() as conn:
        _world(conn)
    with pg.connect() as conn:
        targets = mod.load_targets(conn, "walkway")
        assert [t.individual_id for t in targets] == ["cp", "m"]
        plans = {p.target.individual_id: p for p in mod.plan_reanchor(
            conn, "walkway", targets, face_threshold=0.60, line=0.90, now=NOW)}
    m = plans["m"]
    assert m.stored == ["m-stored"] and sorted(m.face_nearest) == ["m-face0", "m-face1", "m-face2"]
    assert m.crop_ids == sorted(["m-stored", "m-face0", "m-face1", "m-face2"])
    assert abs(sum(a * b for a, b in zip(m.new_centroid, BODY_A)) - 1.0) < 1e-5
    # The chair-sitter's 10 recent crops (plus "o-face", "m-bad-face") pass against the old
    # centroid and none of them against the new one; the 30 h old crop is outside the window.
    assert m.recent_total == 17
    assert m.recent_pass_old == 12 and m.recent_pass_new == 5
    assert plans["cp"].new_centroid is None   # nearest face for every crop is m, not cp


@needs_pg
def test_dry_run_writes_nothing(pg, tmp_path) -> None:
    with pg.begin() as conn:
        _world(conn)
        before = _rows(conn)
    out = _run("--snapshot-root", str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert "DRY RUN" in out.stdout and "skip: no face crop" in out.stdout
    assert " 12 -> 5 (of 17)" in out.stdout
    with pg.begin() as conn:
        assert _rows(conn) == before
        from sqlalchemy import text
        assert conn.execute(text("SELECT count(*) FROM vision_individual_reanchor")).scalar() == 0
    assert list(tmp_path.iterdir()) == []


@needs_pg
def test_apply_snapshots_writes_audits_and_undo_restores_byte_for_byte(pg, tmp_path) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        before = _rows(conn)
    out = _run("--apply", "--snapshot-root", str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert "APPLIED" in out.stdout
    (snap_dir,) = list(tmp_path.iterdir())
    snap = snap_dir / "before.json"
    assert stat.S_IMODE(snap.stat().st_mode) == 0o600
    data = json.loads(snap.read_text())
    assert {r["individual_id"]: (r["centroid"], r["centroid_n"]) for r in data["rows"]} == {
        k: before[k][:2] for k in ("m", "cp")}
    with pg.begin() as conn:
        after = _rows(conn)
        assert after["m"][1] == 4
        assert abs(sum(a * b for a, b in zip(after["m"][0], BODY_A)) - 1.0) < 1e-5
        for k in ("cp", "lone", "o"):
            assert after[k] == before[k]
        audit = conn.execute(text(
            "SELECT individual_id, action, crop_ids, before, after, snapshot_path FROM vision_individual_reanchor"
        )).fetchall()
        assert len(audit) == 1 and audit[0].action == "reanchor" and audit[0].individual_id == "m"
        assert audit[0].before["centroid_n"] == 50 and audit[0].after["centroid_n"] == 4
        assert sorted(audit[0].crop_ids) == ["m-face0", "m-face1", "m-face2", "m-stored"]
        assert audit[0].snapshot_path == str(snap)

    # A skipped target (cp) keeps moving with the loop; undo must not touch it.
    with pg.begin() as conn:
        conn.execute(text("UPDATE vision_individual SET centroid=:c, centroid_n=51 WHERE individual_id='cp'"),
                     {"c": _vec(0.5, 1)})
        cp_moved = _rows(conn)["cp"][:2]
    dry = _run("--undo", str(snap))
    assert dry.returncode == 0 and "DRY RUN" in dry.stdout, dry.stderr
    with pg.begin() as conn:
        assert _rows(conn)["m"][:2] == after["m"][:2]
    done = _run("--undo", str(snap), "--apply")
    assert done.returncode == 0 and "APPLIED" in done.stdout, done.stderr
    with pg.begin() as conn:
        restored = _rows(conn)
        for k in ("m", "lone", "o"):
            assert restored[k][:2] == before[k][:2], k   # exact float32 values, exact n
        assert restored["cp"][:2] == cp_moved
        assert conn.execute(text(
            "SELECT count(*) FROM vision_individual_reanchor WHERE action='undo' AND individual_id='m'")).scalar() == 1


@needs_pg
def test_apply_waits_for_the_individuals_loop_lock(pg, tmp_path) -> None:
    """A tick holds the loop's advisory lock while it reads and upserts
    centroids; the re-anchor must not interleave with it."""
    from sqlalchemy import text

    from app.vision_individuals import INDIVIDUALS_LOCK_KEY

    mod = _load_script()
    with pg.begin() as conn:
        _world(conn)
        before = _rows(conn)
    holder = pg.connect()
    try:
        holder.execute(text("SELECT pg_advisory_lock(:k)"), {"k": INDIVIDUALS_LOCK_KEY})
        holder.commit()
        with pytest.raises(Exception, match="lock timeout"):
            mod.apply_reanchor(pg, "walkway", face_threshold=0.60, line=0.90, now=NOW,
                               snapshot_root=tmp_path, lock_timeout="200ms")
        with pg.begin() as conn:
            assert _rows(conn) == before
        assert list(tmp_path.iterdir()) == []   # no snapshot for a run that never got the lock
    finally:
        holder.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": INDIVIDUALS_LOCK_KEY})
        holder.close()
    out = mod.apply_reanchor(pg, "walkway", face_threshold=0.60, line=0.90, now=NOW, snapshot_root=tmp_path,
                             lock_timeout="200ms")
    assert out["written"] == 1


@needs_pg
def test_undo_refuses_when_crops_joined_since_unless_forced(pg, tmp_path) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        before = _rows(conn)
    assert _run("--apply", "--snapshot-root", str(tmp_path)).returncode == 0
    snap = next(tmp_path.iterdir()) / "before.json"
    with pg.begin() as conn:   # the matcher folded a crop in
        conn.execute(text("UPDATE vision_individual SET centroid_n = centroid_n + 1 WHERE individual_id='m'"))
    out = _run("--undo", str(snap), "--apply")
    assert out.returncode == 1 and "REFUSED" in out.stderr
    with pg.begin() as conn:
        assert _rows(conn)["m"][1] == 5
    out = _run("--undo", str(snap), "--apply", "--force")
    assert out.returncode == 0, out.stderr
    with pg.begin() as conn:
        assert _rows(conn)["m"][:2] == before["m"][:2]


@needs_pg
def test_undo_of_a_run_that_never_committed_refuses(pg, tmp_path) -> None:
    snap = tmp_path / "before.json"
    snap.write_text(json.dumps({"run_id": "run-never", "rows": [{"individual_id": "m", "centroid": [1.0],
                                                                 "centroid_n": 1}]}))
    out = _run("--undo", str(snap), "--apply")
    assert out.returncode == 1 and "no committed re-anchor rows" in out.stdout


@needs_pg
def test_a_split_after_a_reanchor_subtracts_only_crops_in_the_reanchored_centroid(pg) -> None:
    """The split's moved crops (the other person's) were never part of the
    re-anchored body centroid: it must not subtract them again."""
    from sqlalchemy import text

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_vision_split as ts
    from app import vision_split as vs

    with pg.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_ask_picture_verdict, "
                          "vision_individual_split CASCADE"))
        ts._world(conn)
        src = conn.execute(text("SELECT centroid, centroid_n FROM vision_individual WHERE individual_id='src'")).one()
        conn.execute(text(
            "INSERT INTO vision_individual_reanchor (reanchor_id, run_id, individual_id, action, rule, crop_ids, "
            "before, after, created_at) VALUES ('r1', 'run-1', 'src', 'reanchor', 'test', '[\"k0\", \"f2\"]', "
            "'{}', '{}', :t)"), {"t": ts.NOW - timedelta(minutes=30)})
    with pg.begin() as conn:
        assert vs.apply_splits(conn, ts.NOW, ts.CFG)["splits"] == 1
        after = conn.execute(text("SELECT centroid, centroid_n FROM vision_individual WHERE individual_id='src'")).one()
        assert sorted(conn.execute(text(
            "SELECT crop_id FROM vision_crop_observation WHERE individual_id NOT IN ('src')")).scalars()) == ["f1", "m0"]
    assert list(after.centroid) == list(src.centroid) and after.centroid_n == src.centroid_n
