"""Person learning, Phase 1 (2026-10-20): albums from Juniper's answers,
learned accept/reject bars, "probably X" guesses.

Pure rules first. Then the cycle against a REAL Postgres with the walkway
migrations plus manual_migration_vision_person_album_v1.sql applied. That part
runs only when ``ORION_TEST_PG_URI`` points at a throwaway database -- it
TRUNCATEs the walkway tables -- e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 \\
             vision_split_v1 vision_osnet_shadow_v1 vision_person_album_v1; do
      docker exec -i pg psql -U postgres < services/orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55499/postgres pytest tests/test_vision_person_model.py
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_person_model as pm  # noqa: E402

NOW = datetime(2026, 10, 20, 18, 0, tzinfo=timezone.utc)
CFG = pm.PersonModelConfig()


def _vec(*xs: float, dim: int = 8) -> np.ndarray:
    return pm.unit(list(xs) + [0.0] * (dim - len(xs)))


def _crop(cid: str, cluster: str, day: str, body, face=None) -> pm.Crop:
    vecs = {"body": pm.unit(body)}
    if face is not None:
        vecs["face"] = pm.unit(face)
    return pm.Crop(cid, cluster, day, vecs)


def _person(name: str, axis: int, clusters: int, per: int = 3, spread: float = 0.05, seed: int = 0):
    """``clusters`` clusters on different days, each ``per`` crops near one axis."""
    rng = np.random.default_rng(seed + axis)
    out = []
    for k in range(clusters):
        for j in range(per):
            v = np.zeros(8)
            v[axis] = 1.0
            v += rng.normal(0, spread, 8)
            out.append(_crop(f"{name}-{k}-{j}", f"{name}-c{k}", f"2026-10-{k + 1:02d}", v))
    return out


# --- conformal bars ----------------------------------------------------------------------------------


def test_accept_bar_needs_19_impostors() -> None:
    assert pm.conformal_accept_at([0.1] * 18, 0.05) is None
    imp = [i / 100 for i in range(19)]
    assert pm.conformal_accept_at(imp, 0.05) == max(imp)      # rank ceil(0.95 * 20) = 19
    imp = [i / 100 for i in range(39)]
    assert pm.conformal_accept_at(imp, 0.05) == sorted(imp)[37]  # rank ceil(0.95 * 40) = 38


def test_reject_bar_needs_19_genuine() -> None:
    assert pm.conformal_reject_below([0.9] * 18, 0.05) is None
    gen = [0.5 + i / 100 for i in range(19)]
    assert pm.conformal_reject_below(gen, 0.05) == min(gen)


def test_three_bands_never_two() -> None:
    bar = pm.Bar("Juniper", "body", accept_at=0.9, reject_below=0.5, n_genuine=19, n_impostor=19)
    assert pm.band(0.95, bar) == "accept"
    assert pm.band(0.9, bar) == "unsure"        # accept is strictly above the bound
    assert pm.band(0.7, bar) == "unsure"
    assert pm.band(0.4, bar) == "reject"
    open_bar = pm.Bar("Juniper", "body", None, None, 3, 3)
    assert {pm.band(s, open_bar) for s in (0.0, 0.5, 1.0)} == {"unsure"}   # no bars = ask only


# --- scores ------------------------------------------------------------------------------------------


def test_a_minus_one_exemplar_lowers_the_score_for_that_person() -> None:
    album = {"Juniper": {1: _person("j", 0, 3)}}
    stranger = [_crop(f"s{i}", "stranger", "2026-10-09", _vec(1, 0.6)) for i in range(3)]
    before = pm.score_cluster(stranger, "Juniper", "body", album, CFG)
    lookalike = [_crop(f"n{i}", "other", "2026-10-08", _vec(1, 0.6)) for i in range(3)]
    album["Juniper"][-1] = lookalike
    after = pm.score_cluster(stranger, "Juniper", "body", album, CFG)
    assert after < before


def test_a_cluster_never_scores_against_its_own_crops() -> None:
    own = _person("j", 0, 1)
    album = {"Juniper": {1: own}}
    assert pm.score_cluster(own, "Juniper", "body", album, CFG) is None


def test_genuine_scores_are_cross_day_only() -> None:
    """Two clusters on ONE day: no genuine score survives, so no accept band
    (a channel that only matches same-day clothes)."""
    same_day = [_crop(c.crop_id, c.cluster, "2026-10-01", c.vecs["body"]) for c in _person("j", 0, 2)]
    album = {"Juniper": {1: same_day}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 6)}}
    bar, genuine, impostor = pm.fit_bar("Juniper", "body", album, CFG)
    assert genuine == [] and len(impostor) >= 19
    assert bar.accept_at is None


def test_a_well_separated_person_gets_an_accept_bar_and_strangers_do_not_pass() -> None:
    album = {"Juniper": {1: _person("j", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 8)}}
    bar, genuine, impostor = pm.fit_bar("Juniper", "body", album, CFG)
    assert bar.accept_at is not None and len(impostor) >= 19
    new_visit = [_crop(f"jn{i}", "jnew", "2026-10-15", _vec(1, 0.02 * i)) for i in range(3)]
    stranger = [_crop(f"sn{i}", "snew", "2026-10-15", _vec(*([1.0] * 8))) for i in range(3)]  # near nobody
    bars = pm.fit_all(album, CFG)
    got = {g["individual_id"]: g for g in pm.attribute({"jnew": new_visit, "snew": stranger}, album, bars, {}, CFG)}
    assert got["jnew"]["person_label"] == "Juniper" and got["jnew"]["band"] == "accept"
    assert got.get("snew", {}).get("band") != "accept"


def test_accepted_for_two_people_is_downgraded_to_unsure() -> None:
    bars = {("A", "body"): pm.Bar("A", "body", 0.1, None, 19, 19), ("B", "body"): pm.Bar("B", "body", 0.1, None, 19, 19)}
    album = {"A": {1: _person("a", 0, 2)}, "B": {1: _person("b", 0, 2, seed=7)}}
    cl = [_crop("x", "x", "2026-10-09", _vec(1))]
    (g,) = pm.attribute({"x": cl}, album, bars, {}, CFG)
    assert g["band"] == "unsure"


def test_a_person_juniper_ruled_out_is_never_guessed() -> None:
    bars = {("A", "body"): pm.Bar("A", "body", 0.1, None, 19, 19)}
    album = {"A": {1: _person("a", 0, 2)}}
    cl = [_crop("x", "x", "2026-10-09", _vec(1))]
    assert pm.attribute({"x": cl}, album, bars, {"x": {"A"}}, CFG) == []


def test_face_channel_when_both_sides_have_faces_else_fallback() -> None:
    album = {"A": {1: [_crop("a", "ca", "2026-10-01", _vec(1), face=_vec(0, 1))]}}
    bars = {("A", "face"): pm.Bar("A", "face", None, None, 0, 0), ("A", "body"): pm.Bar("A", "body", None, None, 0, 0)}
    with_face = [_crop("x", "x", "2026-10-02", _vec(1), face=_vec(0, 1))]
    no_face = [_crop("y", "y", "2026-10-02", _vec(1))]
    assert pm.pick_channel(with_face, "A", bars, album, CFG) == "face"
    assert pm.pick_channel(no_face, "A", bars, album, CFG) == "body"


# --- purity, names, album -----------------------------------------------------------------------------


def test_split_cosine_sees_two_people_in_one_cluster() -> None:
    one = [_vec(1, 0.01 * i) for i in range(6)]
    two = [_vec(1, 0.01 * i) for i in range(3)] + [_vec(0, 1, 0.01 * i) for i in range(3)]
    assert pm.split_cosine(one) > 0.99
    assert pm.split_cosine(two) < 0.1
    assert pm.split_cosine(one[:3]) is None          # too few crops to judge


def test_same_sighting_floor_is_a_low_percentile_of_within_sighting_pairs() -> None:
    groups = [[_vec(1, 0.1 * j) for j in range(5)] for _ in range(4)]
    floor = pm.same_sighting_floor(groups)
    assert floor is not None and 0.9 < floor < 1.0
    assert pm.same_sighting_floor([[_vec(1)]]) is None


def test_names_differing_only_in_case_are_one_person() -> None:
    assert pm.canonical_names(["Juniper", "juniper", "Juniper", " Theo "]) == {"juniper": "Juniper", "theo": "Theo"}


def test_a_crop_marked_not_x_is_never_in_xs_positive_album() -> None:
    crops = {"c1": _crop("c1", "k", "2026-10-01", _vec(1)), "c2": _crop("c2", "k", "2026-10-01", _vec(1))}
    album = pm.build_album([("A", "c1", 1), ("A", "c2", 1), ("A", "c2", -1), ("A", "gone", 1)], crops)
    assert [c.crop_id for c in album["A"][1]] == ["c1"]
    assert [c.crop_id for c in album["A"][-1]] == ["c2"]


def test_migration_declares_its_consumers_and_is_additive() -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_vision_person_album_v1.sql").read_text()
    assert sql.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub"
    body = "\n".join(line for line in sql.splitlines() if not line.startswith("--")).upper()
    assert "DROP" not in body and "DELETE" not in body and "UPDATE" not in body


def test_settings_ship_the_model_on_and_off_means_no_config() -> None:
    from app.settings import Settings
    from app.vision_individuals_loop import person_model_config

    cfg = person_model_config(Settings())
    assert cfg is not None and cfg.alpha == 0.05 and cfg.top_k == 3 and cfg.fallback_channel == "body"
    assert person_model_config(Settings(VISION_PERSON_MODEL_ENABLED=False)) is None


def test_eval_mcnemar_is_exact_and_two_sided() -> None:
    sys.path.insert(0, str(SERVICE / "evals"))
    from run_person_recognition_eval import auc, mcnemar_p

    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(6, 0) == pytest.approx(2 / 64)      # 2 * 0.5**6
    assert mcnemar_p(5, 0) > 0.05                       # 5 wins and no losses is not enough
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auc([0.5], [0.5]) == 0.5


# --- the cycle against real Postgres --------------------------------------------------------------------

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")


@pytest.fixture()
def pg():
    from sqlalchemy import create_engine, text

    engine = create_engine(PG)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                          "vision_individual, vision_ask_picture_verdict, vision_crop_observation, "
                          "vision_person_exemplar, vision_person_bar, vision_person_attribution CASCADE"))
    yield engine
    engine.dispose()


def _add_cluster(conn, iid: str, label, vectors, day: int, *, copied: bool = False) -> None:
    from sqlalchemy import text

    t0 = NOW - timedelta(days=day)
    conn.execute(text(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, label_ask_id, label_copied) "
        "VALUES (:i, 'walkway', 'person', :c, :n, :t, :t, 1, 1, :l, :a, :cp)"),
        {"i": iid, "c": [float(x) for x in vectors[0]], "n": len(vectors), "t": t0, "l": label,
         "a": f"ask-{iid}" if label else None, "cp": copied})
    for j, v in enumerate(vectors):
        conn.execute(text(
            "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
            "box_xyxy, embedding, individual_id) VALUES (:c, 'obs', 'walkway', :t, 'person', 0.9, '{0,0,1,1}', "
            ":e, :i)"), {"c": f"{iid}-{j}", "t": t0 + timedelta(seconds=j), "e": [float(x) for x in v], "i": iid})


def _world(conn) -> None:
    """Juniper on 4 days, 7 other named people on 2 days each, one unnamed
    cluster that looks like Juniper and one that looks like nobody."""
    rng = np.random.default_rng(1)

    def near(axis, n=3):
        out = []
        for _ in range(n):
            v = np.zeros(8)
            v[axis] = 1.0
            out.append(pm.unit(v + rng.normal(0, 0.05, 8)))
        return out

    for d in range(4):
        _add_cluster(conn, f"jun{d}", "Juniper", near(0), day=d + 2)
    for p in range(1, 8):
        for d in range(2):
            _add_cluster(conn, f"p{p}d{d}", f"Person{p}", near(p), day=d + 2)
    _add_cluster(conn, "unk-j", None, near(0), day=0)
    _add_cluster(conn, "unk-x", None, [pm.unit([1, 1, 1, 1, 1, 1, 1, 1])] * 3, day=0)


@needs_pg
def test_cycle_builds_albums_fits_bars_and_guesses(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        s = pm.run_person_model(conn, NOW, CFG)
    assert s["person_people"] == 8 and s["person_bars_refit"] == 1
    with pg.begin() as conn:
        n = conn.execute(text("SELECT count(*) FROM vision_person_exemplar WHERE revoked_at IS NULL")).scalar()
        assert n == 4 * 3 + 7 * 2 * 3
        bar = conn.execute(text("SELECT accept_at, n_impostor FROM vision_person_bar "
                                "WHERE person_label='Juniper' AND channel='body'")).one()
        assert bar.accept_at is not None and bar.n_impostor >= 19
        guesses = {r[0]: (r[1], r[2]) for r in conn.execute(text(
            "SELECT individual_id, person_label, band FROM vision_person_attribution WHERE revoked_at IS NULL"))}
    assert guesses["unk-j"] == ("Juniper", "accept")
    assert guesses.get("unk-x", (None, None))[1] != "accept"
    # A guess never names the cluster.
    with pg.begin() as conn:
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id='unk-j'")).scalar() is None


@needs_pg
def test_unchanged_album_does_not_refit_and_a_rename_revokes(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        pm.run_person_model(conn, NOW, CFG)
        fitted = conn.execute(text("SELECT max(fitted_at) FROM vision_person_bar")).scalar()
    with pg.begin() as conn:
        assert pm.run_person_model(conn, NOW + timedelta(minutes=10), CFG)["person_bars_refit"] == 0
        assert conn.execute(text("SELECT max(fitted_at) FROM vision_person_bar")).scalar() == fitted
        conn.execute(text("UPDATE vision_individual SET label='Theo' WHERE individual_id='jun0'"))
        s = pm.run_person_model(conn, NOW + timedelta(minutes=20), CFG)
    assert s["person_exemplars_revoked"] == 3 and s["person_exemplars_added"] == 3 and s["person_bars_refit"] == 1
    with pg.begin() as conn:
        assert conn.execute(text("SELECT max(fitted_at) FROM vision_person_bar")).scalar() > fitted


@needs_pg
def test_a_different_picture_rules_its_cluster_out(pg) -> None:
    """Juniper said a crop is not Juniper; once that crop sits in an unnamed
    cluster (a split moved it), the cluster is never guessed as Juniper."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        conn.execute(text("UPDATE vision_crop_observation SET individual_id='jun0' WHERE crop_id='unk-j-0'"))
        conn.execute(text(
            "INSERT INTO vision_ask_picture_verdict (ask_id, image_ref, picture_index, individual_id, verdict, "
            "crop_id, answered_at) VALUES ('ask-x', 'thumb:x', 0, 'jun0', 'different', 'unk-j-0', :t)"), {"t": NOW})
        conn.execute(text("UPDATE vision_crop_observation SET individual_id='unk-j' WHERE crop_id='unk-j-0'"))
        pm.run_person_model(conn, NOW, CFG)
        got = conn.execute(text("SELECT person_label FROM vision_person_attribution WHERE individual_id='unk-j' "
                                "AND revoked_at IS NULL")).scalar()
    assert got != "Juniper"


@needs_pg
def test_a_mixed_named_cluster_adds_nothing_to_the_album(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        # Sightings make the same-person floor learnable; one named cluster holds two people.
        for iid in ("jun1", "jun2", "jun3", "p1d0", "p2d0", "p3d0", "p4d0"):
            conn.execute(text(
                "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, "
                "ended_at, dwell_sec, observation_count) SELECT 's-' || :i, :i, 'walkway', min(observed_at), "
                "max(observed_at), 0, count(*) FROM vision_crop_observation WHERE individual_id=:i"), {"i": iid})
        mixed = [_vec(1, 0.01 * i) for i in range(3)] + [_vec(0, 0, 0, 0, 0, 1, 0.01 * i) for i in range(3)]
        _add_cluster(conn, "mixed", "Juniper", mixed, day=9)
        s = pm.run_person_model(conn, NOW, CFG)
        n = conn.execute(text("SELECT count(*) FROM vision_person_exemplar e JOIN vision_crop_observation c "
                              "USING (crop_id) WHERE c.individual_id='mixed'")).scalar()
    assert s["person_impure_clusters"] == 1 and n == 0


@needs_pg
def test_flag_off_writes_nothing(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    with pg.begin() as conn:
        _world(conn)
    vi.run_one_individuals_cycle(postgres_uri=PG, cfg=vi.IndividualsConfig(), zones_by_stream={},
                                 now=NOW, person_cfg=None)
    with pg.begin() as conn:
        for t in ("vision_person_exemplar", "vision_person_bar", "vision_person_attribution"):
            assert conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() == 0
    summary, _ = vi.run_one_individuals_cycle(postgres_uri=PG, cfg=vi.IndividualsConfig(), zones_by_stream={},
                                              now=NOW, person_cfg=CFG)
    assert summary["person_people"] == 8
