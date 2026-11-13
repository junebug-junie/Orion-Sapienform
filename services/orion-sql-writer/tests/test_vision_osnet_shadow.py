"""OSNet shadow body vector, writer side (2026-10-16).

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "A better body
vector". The writer stores ``osnet_embedding`` / ``osnet_model_id`` /
``embedding_model_id`` next to the SigLIP vector and nothing reads them except
``evals/run_osnet_vs_siglip_eval.py``.

Pure rules first (row shape, no-embed strip, the migration's auto-apply safety,
the matcher does not read OSNet, the eval's pair labelling). Then the write and
the eval against a REAL Postgres (``ORION_TEST_PG_URI``, a throwaway database:
it TRUNCATEs the walkway tables), with the walkway migrations applied:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 vision_split_v1 \\
             vision_osnet_shadow_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_osnet_shadow.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_crop_persist as vcp  # noqa: E402
from orion.schemas.vision import VisionCropObservationV1  # noqa: E402
from orion.vision.zones import Zone  # noqa: E402

MIGRATION = REPO / "services" / "orion-sql-db" / "manual_migration_vision_osnet_shadow_v1.sql"
WALK = Zone(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
NOEMBED = Zone(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=False)
BODY = "google/siglip-so400m-patch14-384"
OSNET = "osnet_ain_x1_0:ms_d_c:34c24e98b6b7"
NOW = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)


def _eval_module():
    spec = importlib.util.spec_from_file_location("run_osnet_vs_siglip_eval",
                                                  SERVICE / "evals" / "run_osnet_vs_siglip_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_osnet_vs_siglip_eval", mod)
    spec.loader.exec_module(mod)
    return mod


def _obs(crops, *, oid="obs-1", t=NOW):
    return VisionCropObservationV1(observation_id=oid, stream_id="walkway", observed_at=t,
                                   frame_width=1000, frame_height=1000, crops=crops)


def _crop(box, *, emb=(1.0, 0.0), osnet=(0.0, 1.0), label="person"):
    return {"label": label, "score": 0.9, "box_xyxy": list(box), "embedding": list(emb), "embedding_ref": "crop:x",
            "embedding_model_id": BODY, "osnet_embedding": list(osnet) if osnet is not None else None,
            "osnet_model_id": OSNET if osnet is not None else None}


# --- writer rows ------------------------------------------------------------------------------


def test_rows_carry_osnet_and_the_loaded_body_model() -> None:
    (row,) = vcp.build_crop_rows(_obs([_crop((400, 400, 500, 700))]), [WALK])
    assert row["osnet_embedding"] == [0.0, 1.0] and row["osnet_model_id"] == OSNET
    assert row["embedding_model_id"] == BODY


def test_older_producer_without_osnet_writes_nulls() -> None:
    (row,) = vcp.build_crop_rows(_obs([{"label": "person", "score": 0.9, "box_xyxy": [400, 400, 500, 700],
                                        "embedding": [1.0, 0.0]}]), [WALK])
    assert row["osnet_embedding"] is None and row["osnet_model_id"] is None and row["embedding_model_id"] is None


def test_no_embed_zone_strips_osnet_too() -> None:
    (row,) = vcp.build_crop_rows(_obs([_crop((100, 750, 200, 950))]), [NOEMBED, WALK])
    assert row["zone_no_embed"] is True
    assert row["embedding"] is None and row["osnet_embedding"] is None and row["osnet_model_id"] is None
    assert row["embedding_model_id"] is None


def test_zones_unavailable_strips_osnet() -> None:
    (row,) = vcp.build_crop_rows(_obs([_crop((400, 400, 500, 700))]), None)
    assert row["osnet_embedding"] is None and row["embedding"] is None


def test_insert_names_every_row_key() -> None:
    (row,) = vcp.build_crop_rows(_obs([_crop((400, 400, 500, 700))]), [WALK])
    for key in row:
        assert f":{key}" in vcp._INSERT_SQL, key


def test_matcher_never_reads_osnet() -> None:
    """Shadow only: the individuals reducer and the split do not touch the OSNet columns."""
    for name in ("vision_individuals.py", "vision_split.py"):
        assert "osnet" not in (SERVICE / "app" / name).read_text().lower(), name


# --- migration --------------------------------------------------------------------------------


def test_migration_passes_the_post_pull_auto_apply_filter() -> None:
    from orion.sql_migration_drift import auto_apply_refusals, parse_required_by, transaction_incompatible

    text = MIGRATION.read_text()
    assert auto_apply_refusals(text, {"vision_crop_observation"}) == []
    assert auto_apply_refusals(text) == []
    rb = parse_required_by(text)
    assert rb is not None and rb.services == ("orion-sql-writer",) and not rb.errors
    assert not transaction_incompatible(text)
    stmts = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("--")]
    assert len(stmts) == 3 and all(ln.startswith("ALTER TABLE vision_crop_observation ADD COLUMN IF NOT EXISTS ")
                                   for ln in stmts)
    assert not any("NOT NULL" in ln.upper() or "DEFAULT" in ln.upper() for ln in stmts)


def test_post_pull_applier_would_apply_it() -> None:
    from orion import sql_migration_drift as d

    spec = importlib.util.spec_from_file_location("apply_post_pull_migrations",
                                                  REPO / "scripts" / "apply_post_pull_migrations.py")
    ap = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("apply_post_pull_migrations", ap)
    spec.loader.exec_module(ap)
    result = d.FileResult(name=MIGRATION.name, in_window=True, changed_at=NOW, status="MISSING")
    item = ap.classify(f"services/orion-sql-db/{MIGRATION.name}", MIGRATION.read_text(), result,
                       {"vision_crop_observation"})
    assert item.action == "apply", item.reason


# --- eval: pair labelling from geometry only --------------------------------------------------


def _ec(ev, cid, oid, t, box, sig, osn, stream="walkway"):
    return ev.Crop(cid, oid, stream, t, tuple(float(v) for v in box), {"siglip": ev._unit(sig), "osnet": ev._unit(osn)})


def test_eval_pairs_come_from_geometry() -> None:
    ev = _eval_module()
    t0 = NOW
    a1 = _ec(ev, "a1", "f1", t0, (100, 100, 200, 400), (1, 0), (1, 0))
    b1 = _ec(ev, "b1", "f1", t0, (600, 100, 700, 400), (1, 0.2), (0, 1))
    dup = _ec(ev, "d1", "f1", t0, (105, 105, 200, 400), (1, 0), (1, 0))     # same person boxed twice
    a2 = _ec(ev, "a2", "f2", t0 + timedelta(seconds=2.5), (130, 110, 230, 410), (1, 0.05), (1, 0.05))
    b2 = _ec(ev, "b2", "f2", t0 + timedelta(seconds=2.5), (640, 100, 740, 400), (1, 0.25), (0, 1))
    late = _ec(ev, "l1", "f3", t0 + timedelta(seconds=30), (130, 110, 230, 410), (1, 0), (1, 0))
    fr = ev.frames([a1, b1, dup, a2, b2, late])
    diff = {tuple(sorted((x.crop_id, y.crop_id))) for x, y in ev.same_frame_pairs(fr)}
    assert ("a1", "d1") not in diff                      # overlap: left out
    assert {("a1", "b1"), ("a2", "b2")} <= diff
    same = {(x.crop_id, y.crop_id) for x, y in ev.track_pairs(fr, 3.0)}
    # a2's back-match into f1 finds a1 and the duplicate d1 within 1.5x of each other,
    # so neither a1 -> a2 nor d1 -> a2 is kept.
    assert ("b1", "b2") in same and ("a2", "l1") not in same
    assert all(x != "a1" for x, _ in same)


def test_a_box_without_vectors_still_vetoes_an_ambiguous_track_pair() -> None:
    ev = _eval_module()
    a1 = _ec(ev, "a1", "f1", NOW, (100, 100, 200, 400), (1, 0), (1, 0))
    a2 = _ec(ev, "a2", "f2", NOW + timedelta(seconds=2.5), (130, 110, 230, 410), (1, 0), (1, 0))
    assert [(x.crop_id, y.crop_id) for x, y in ev.track_pairs(ev.frames([a1, a2]), 3.0)] == [("a1", "a2")]
    # A second person (no vectors: never embedded) right next to a2 makes the match ambiguous.
    ghost = ev.Crop("g2", "f2", "walkway", NOW + timedelta(seconds=2.5), (140, 110, 240, 410), None)
    assert ev.track_pairs(ev.frames([a1, a2, ghost]), 3.0) == []
    # ...and it never becomes a scored pair itself.
    assert all(x.vecs and y.vecs for x, y in ev.same_frame_pairs(ev.frames([a1, a2, ghost])))


def test_bootstrap_calls_a_tie_when_the_models_score_the_same() -> None:
    ev = _eval_module()
    pairs_same = [(_ec(ev, f"s{i}a", f"o{i}", NOW, (0, 0, 1, 1), (1, 0), (1, 0)),
                   _ec(ev, f"s{i}b", f"p{i}", NOW, (0, 0, 1, 1), (1, 0.1 * (i % 5)), (1, 0.1 * (i % 5))))
                  for i in range(40)]
    pairs_diff = [(_ec(ev, f"d{i}a", f"q{i}", NOW, (0, 0, 1, 1), (1, 0), (1, 0)),
                   _ec(ev, f"d{i}b", f"q{i}", NOW, (0, 0, 1, 1), (0.2 * (i % 7), 1), (0.2 * (i % 7), 1)))
                  for i in range(40)]
    c = ev.compare(pairs_same, pairs_diff, rounds=50)
    assert c["winner"] == "tie" and c["osnet_minus_siglip_let_through_ci95"] == [0.0, 0.0]


def test_auc_and_line() -> None:
    ev = _eval_module()
    assert ev.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ev.auc([0.5], [0.5]) == 0.5
    # line = p10 of same = 0.86: 0.9 of the different pairs (0.95) pass, 0.1 (0.6) do not.
    sep = ev.separation([0.9] * 9 + [0.5], [0.95] * 9 + [0.6])
    assert sep["line_keeping_90pct_same"] == 0.86 and sep["different_let_through_at_line"] == 0.9
    assert sep["auc"] == 0.09                            # only the nine 0.9-vs-0.6 pairs rank right


def test_eval_says_insufficient_with_too_few_pairs() -> None:
    ev = _eval_module()
    rep = ev.evaluate([], [], lambda s: str(s or ""), max_gap_sec=3.0, min_pairs=1)
    assert rep["sufficient"] is False and "geometry" not in rep


# --- the whole path against real Postgres ------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_ask_picture_verdict, vision_crop_observation CASCADE"))
    yield engine
    engine.dispose()


def _unitv(angle: float, dim: int) -> list:
    v = [0.0] * dim
    v[0], v[1] = math.cos(angle), math.sin(angle)
    return v


def _write(engine, obs) -> int:
    from sqlalchemy import text

    rows = vcp.build_crop_rows(obs, [WALK])
    with engine.begin() as conn:
        conn.execute(text(vcp._INSERT_SQL), rows)
    return len(rows)


@needs_pg
def test_osnet_columns_round_trip(pg) -> None:
    from sqlalchemy import text

    osnet = _unitv(0.3, 512)
    _write(pg, _obs([_crop((400, 400, 500, 700), emb=_unitv(0.1, 1152), osnet=osnet),
                     _crop((600, 400, 700, 700), osnet=None, label="dog")]))
    with pg.connect() as conn:
        rows = {r.crop_id: r for r in conn.execute(text(
            "SELECT crop_id, osnet_embedding, osnet_model_id, embedding_model_id FROM vision_crop_observation"))}
    person, dog = rows["obs-1:0"], rows["obs-1:1"]
    assert len(person.osnet_embedding) == 512 and abs(person.osnet_embedding[0] - osnet[0]) < 1e-6
    assert person.osnet_model_id == OSNET and person.embedding_model_id == BODY
    assert dog.osnet_embedding is None and dog.osnet_model_id is None


def _street(engine, n_frames: int) -> None:
    """Two people walk side by side for n_frames, 2.5 s apart. SigLIP sees them as
    near-identical (same clothes colour); OSNet keeps them apart."""
    # Relative to the real clock: the eval's --live window is "last 8 days from now", so a
    # fixed date here silently aged out of it on 2026-10-24 and failed every branch's CI.
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    for k in range(n_frames):
        t = start + timedelta(seconds=2.5 * k)
        jitter = 0.02 * (k % 3)
        crops = [
            _crop((100 + 10 * k, 300, 200 + 10 * k, 600), emb=_unitv(0.10 + jitter, 1152), osnet=_unitv(0.0 + jitter, 512)),
            _crop((600 + 10 * k, 300, 700 + 10 * k, 600), emb=_unitv(0.14 + jitter, 1152), osnet=_unitv(1.2 + jitter, 512)),
        ]
        _write(engine, _obs(crops, oid=f"obs-{k}", t=t))


def _since_fixtures() -> list:
    """The eval looks back --since-hours from the wall clock (default 8 days), but the
    fixtures are pinned at NOW: without this the test broke for every branch on 10-24."""
    hours = (datetime.now(timezone.utc) - NOW).total_seconds() / 3600 + 48
    return ["--since-hours", f"{hours:.0f}"]


@needs_pg
def test_eval_insufficient_then_scored_on_real_postgres(pg, capsys) -> None:
    ev = _eval_module()
    _street(pg, 3)
    assert ev.main(["--live", PG, *_since_fixtures(), "--min-pairs", "50"]) == 2
    assert "INSUFFICIENT DATA" in capsys.readouterr().out
    from sqlalchemy import text

    with pg.begin() as conn:
        conn.execute(text("TRUNCATE vision_crop_observation"))
    _street(pg, 30)
    assert ev.main(["--live", PG, *_since_fixtures(), "--min-pairs", "20", "--json"]) == 0
    import json

    rep = json.loads(capsys.readouterr().out)
    assert rep["pairs"]["same_person_next_frame"] == 2 * 29
    assert rep["pairs"]["different_people_same_frame"] == 30
    assert rep["geometry"]["osnet"]["auc"] > rep["geometry"]["siglip"]["auc"]
    assert rep["winner"] == "osnet"
    assert rep["coverage"]["models"] == [{"body": BODY, "osnet": OSNET, "crops": 60}]


@needs_pg
def test_eval_scores_juniper_verdict_pairs(pg, capsys) -> None:
    from sqlalchemy import text

    ev = _eval_module()
    _street(pg, 2)
    with pg.begin() as conn:
        conn.execute(text(
            "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, answered_at) "
            "VALUES ('ask-1', 'q', 'vision_individual', 'ind', 'answered', 'Alder', now())"))
        for idx, (crop, verdict, name) in enumerate([("obs-0:0", "same", None), ("obs-1:0", "same", None),
                                                     ("obs-1:1", "different", "Cora")]):
            conn.execute(text(
                "INSERT INTO vision_ask_picture_verdict (ask_id, image_ref, picture_index, individual_id, verdict, "
                "name, crop_id, answered_at) VALUES ('ask-1', :r, :i, 'ind', :v, :n, :c, now())"),
                {"r": f"thumb:{idx}", "i": idx, "v": verdict, "n": name, "c": crop})
    assert ev.main(["--live", PG, *_since_fixtures(), "--min-pairs", "1000", "--json"]) == 2
    import json

    rep = json.loads(capsys.readouterr().out)
    assert rep["pairs"]["verdict_same"] == 1 and rep["pairs"]["verdict_different"] == 2
    assert rep["verdicts"]["osnet"]["same"]["n"] == 1
