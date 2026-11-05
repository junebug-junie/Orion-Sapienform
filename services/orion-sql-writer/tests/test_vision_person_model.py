"""Person learning, Phase 1 (2026-10-20): albums from Juniper's answers,
learned accept/reject bars, "probably X" guesses.

Pure rules first. Then the cycle against a REAL Postgres with the walkway
migrations plus manual_migration_vision_person_album_v1.sql applied. That part
runs only when ``ORION_TEST_PG_URI`` points at a throwaway database -- it
TRUNCATEs the walkway tables -- e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55499:5432 postgres:16-alpine
    for f in walkway_camera_v1 walkway_privacy_v2 vision_confirm_asks_v1 vision_ask_names_v1 \\
             vision_split_v1 vision_osnet_shadow_v1 vision_person_album_v1 vision_person_basis_v1 person_record_v1; do
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
ZONE = __import__("zoneinfo").ZoneInfo("America/Denver")
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


def _street(n: int = 170, seed: int = 3) -> dict:
    """``n`` unnamed one-visit clusters pointing nowhere in particular. With 8
    people, alpha / 8 needs about 160 impostor clusters before a bar can arm."""
    rng = np.random.default_rng(seed)
    return {f"st{i}": [_crop(f"st{i}-{j}", f"st{i}", "2026-10-12", rng.normal(0, 1, 8)) for j in range(3)]
            for i in range(n)}


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
    fit = pm.fit_bar("Juniper", "body", album, CFG, _street())
    assert fit.genuine_crops == [] and fit.genuine_clusters == [] and len(fit.impostor_clusters) >= 19
    assert fit.bar.accept_at is None
    # Three clusters on three days do earn one.
    three_days = {**album, "Juniper": {1: _person("j", 0, 3)}}
    assert pm.fit_bar("Juniper", "body", three_days, CFG, _street()).bar.accept_at is not None


def test_a_bar_needs_three_held_out_visits_and_half_of_them_above() -> None:
    album = {"Juniper": {1: _person("j", 0, 2)}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 6)}}
    assert pm.fit_bar("Juniper", "body", album, CFG, _street()).bar.accept_at is None   # 2 visits: not enough
    # 3 visits, but two look nothing like the third: fewer than half clear the bar.
    odd = [_crop(f"jo{k}-{j}", f"jo{k}", f"2026-10-0{k + 5}", _vec(*([0.0] * (k + 2)), 1.0)) for k in range(2)
           for j in range(3)]
    album["Juniper"][1] = _person("j", 0, 1) + odd
    assert pm.fit_bar("Juniper", "body", album, CFG, _street()).bar.accept_at is None


def test_a_well_separated_person_gets_an_accept_bar_and_strangers_do_not_pass() -> None:
    album = {"Juniper": {1: _person("j", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 8)}}
    new_visit = [_crop(f"jn{i}", "jnew", "2026-10-15", _vec(1, 0.02 * i)) for i in range(3)]
    stranger = [_crop(f"sn{i}", "snew", "2026-10-15", _vec(*([1.0] * 8))) for i in range(3)]  # near nobody
    # The two new clusters arrived after the last fit (they are not in its street).
    fit = pm.fit_bar("Juniper", "body", album, CFG, _street())
    assert fit.bar.accept_at is not None and len(fit.impostor_clusters) >= 19
    bars = pm.fit_all(album, CFG, _street())
    got = {g["individual_id"]: g for g in pm.attribute({"jnew": new_visit, "snew": stranger}, album, bars, {}, CFG)}
    assert got["jnew"]["person_label"] == "Juniper" and got["jnew"]["band"] == "accept"
    assert got.get("snew", {}).get("band") != "accept"


def test_the_street_raises_the_bar_named_people_alone_do_not() -> None:
    """Live 2026-10-20: calibrated only on other named people, ~9% of street
    clusters cleared a person's bar. Street look-alikes must raise it."""
    album = {"Juniper": {1: _person("j", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 3)} for i in range(1, 8)}}
    lookalikes = {f"la{i}": [_crop(f"la{i}-{j}", f"la{i}", "2026-10-12", _vec(1, 0.35 + 0.01 * i, 0.1 * j))
                             for j in range(3)] for i in range(25)}
    plain = pm.fit_bar("Juniper", "body", album, CFG, _street()).bar.accept_at
    crowded = pm.fit_bar("Juniper", "body", album, CFG, {**_street(), **lookalikes}).bar.accept_at
    assert plain is not None and (crowded is None or crowded > plain)


def test_street_accepts_stay_near_alpha_over_people() -> None:
    """Review finding: the street is fitted on AND guessed on, so about
    alpha / people of it clears an armed bar. Here the street is all strangers."""
    album = {"Juniper": {1: _person("j", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 3)} for i in range(1, 8)}}
    street = _street(400)
    bars = pm.fit_all(album, CFG, street)
    per_person: dict = {}
    for g in pm.attribute(street, album, bars, {}, CFG):
        if g["band"] == "accept":
            per_person[g["person_label"]] = per_person.get(g["person_label"], 0) + 1
    bound = len(street) * CFG.alpha / len(album)          # ~2.5 here
    assert all(n <= 2 * bound + 1 for n in per_person.values()), per_person
    # Below about 2 x people / alpha impostor clusters the bound is the street's
    # maximum and no street cluster can be accepted at all.
    small = _street(100)
    small_bars = pm.fit_all(album, CFG, small)
    assert not [g for g in pm.attribute(small, album, small_bars, {}, CFG) if g["band"] == "accept"]


def test_a_street_cluster_juniper_ruled_out_is_one_impostor_not_two() -> None:
    album = {"Juniper": {1: _person("j", 0, 3)}, "B": {1: _person("b", 1, 2)}}
    street = _street(20)
    album["Juniper"][-1] = street["st0"][:1]          # she said a crop of st0 is not Juniper
    fit = pm.fit_bar("Juniper", "body", album, CFG, street)
    assert len(fit.impostor_clusters) == 2 + 20         # B's 2 clusters + the street, st0 once


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
    assert pm.attribute({"x": cl}, album, bars, {"x": {"a"}}, CFG) == []   # blocks are casefolded


def test_face_channel_when_both_sides_have_faces_else_fallback() -> None:
    album = {"A": {1: [_crop("a", "ca", "2026-10-01", _vec(1), face=_vec(0, 1))]}}
    bars = {("A", "face"): pm.Bar("A", "face", 0.5, None, 19, 19), ("A", "body"): pm.Bar("A", "body", None, None, 0, 0)}
    with_face = [_crop("x", "x", "2026-10-02", _vec(1), face=_vec(0, 1))]
    no_face = [_crop("y", "y", "2026-10-02", _vec(1))]
    assert pm.pick_channel(with_face, "A", bars, album, CFG) == "face"
    assert pm.pick_channel(no_face, "A", bars, album, CFG) == "body"
    # A face bar that can never decide (too few impostors) does not shadow body.
    bars[("A", "face")] = pm.Bar("A", "face", None, None, 3, 8)
    assert pm.pick_channel(with_face, "A", bars, album, CFG) == "body"


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
    assert pm.canonical_names(["Juniper", "juniper", "Juniper", " PersonA "]) == {"juniper": "Juniper", "persona": "PersonA"}


def test_a_not_x_written_under_another_casing_still_counts() -> None:
    crops = {"c1": _crop("c1", "k", "2026-10-01", _vec(1)), "c2": _crop("c2", "k", "2026-10-01", _vec(1))}
    album = pm.build_album([("Juniper", "c1", 1), ("Juniper", "c2", 1), ("juniper", "c2", -1)], crops)
    assert list(album) == ["Juniper"]
    assert [c.crop_id for c in album["Juniper"][1]] == ["c1"] and [c.crop_id for c in album["Juniper"][-1]] == ["c2"]


def test_is_this_x_question_counts_in_plain_english() -> None:
    from app import vision_individuals as vi

    assert vi.person_confirm_question(person="Juniper", sighting_count=1, distinct_days=1).endswith(
        "I have seen them 1 time over 1 day.")
    assert vi.person_confirm_question(person="Juniper", sighting_count=4, distinct_days=2).endswith(
        "I have seen them 4 times over 2 days.")


def test_a_crop_marked_not_x_is_never_in_xs_positive_album() -> None:
    crops = {"c1": _crop("c1", "k", "2026-10-01", _vec(1)), "c2": _crop("c2", "k", "2026-10-01", _vec(1))}
    album = pm.build_album([("A", "c1", 1), ("A", "c2", 1), ("A", "c2", -1), ("A", "gone", 1)], crops)
    assert [c.crop_id for c in album["A"][1]] == ["c1"]
    assert [c.crop_id for c in album["A"][-1]] == ["c2"]


def test_migration_declares_its_consumers_and_is_additive() -> None:
    sql = (REPO / "services/orion-sql-db/manual_migration_vision_person_album_v1.sql").read_text()
    assert sql.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer"
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


# --- alpha split over real people, longer face street (2026-11-03) ---------------------------------------


def _things(names, clusters: int = 2, seed: int = 11) -> dict:
    """Album labels that are not one person (a decoration, a role): random directions."""
    rng = np.random.default_rng(seed)
    return {n: {1: [_crop(f"{n}-{k}-{j}", f"{n}-c{k}", f"2026-10-{k + 1:02d}", rng.normal(0, 1, 8))
                    for k in range(clusters) for j in range(3)]} for n in names}


def test_not_a_person_labels_do_not_widen_the_alpha_split() -> None:
    """Live 2026-11-03: 16 album labels, 4 not one real person (a decoration, two
    delivery roles, a pair). Counting them made a bar need 319 impostor clusters, not 259."""
    people = {"Juniper": {1: _person("j", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 8)}}
    album = {**people, **_things(["skeleton", "Mail carrier", "UPS delivery", "A and B"])}
    not_people = frozenset({"skeleton", "mail carrier", "ups delivery", "a and b"})
    assert pm.people_count(album, CFG, not_people) == 8 and pm.people_count(album, CFG) == 12
    # 170 street + 7 x 2 people + 4 x 2 things = 192 impostor clusters: alpha / 12
    # needs 193 (no bar), alpha / 8 needs 192.
    split_all = pm.fit_bar("Juniper", "body", album, CFG, _street())
    split_people = pm.fit_bar("Juniper", "body", album, CFG, _street(), not_people)
    assert len(split_people.impostor_clusters) == 192
    assert split_all.bar.accept_at is None and split_people.bar.accept_at is not None
    # The things keep their albums and bars, and still count as impostors.
    bars = pm.fit_all(album, CFG, _street(), not_people=not_people)
    assert ("skeleton", "body") in bars and bars[("Juniper", "body")].accept_at is not None
    assert pm.people_count(album, CFG, frozenset({"SKELETON"})) == 12      # names arrive casefolded


def test_a_not_a_person_label_never_gets_an_accept_bar() -> None:
    """Review finding: "probably A and B" names known people. A label outside the
    split must not be able to accept, or the split no longer bounds false accepts."""
    album = {"A and B": {1: _person("ab", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 8)}}
    armed = pm.fit_bar("A and B", "body", album, CFG, _street())          # counted as a person: arms
    assert armed.bar.accept_at is not None
    fit = pm.fit_bar("A and B", "body", album, CFG, _street(), frozenset({"a and b"}))
    assert fit.bar.accept_at is None and fit.genuine_clusters    # still scored, still asked about
    bars = pm.fit_all(album, CFG, _street(), not_people=frozenset({"a and b"}))
    near = [_crop(f"n{i}", "near", "2026-10-15", _vec(1, 0.02 * i)) for i in range(3)]
    (g,) = pm.attribute({"near": near}, album, bars, {}, CFG)
    assert g["person_label"] == "A and B" and g["band"] == "unsure"


def test_split_people_only_off_is_the_old_split_over_every_label() -> None:
    people = {"Juniper": {1: _person("j", 0, 4)}, **{f"p{i}": {1: _person(f"p{i}", i, 2)} for i in range(1, 8)}}
    album = {**people, **_things(["skeleton", "Mail carrier", "UPS delivery", "A and B"])}
    off = pm.PersonModelConfig(alpha_split_people_only=False)
    not_people = frozenset({"skeleton", "mail carrier", "ups delivery", "a and b"})
    assert pm.people_count(album, off, not_people) == len(album)
    assert pm.fit_bar("Juniper", "body", album, off, _street(), not_people).bar.accept_at is None
    # Off: not_people is ignored entirely, bars equal the old call's.
    assert pm.fit_all(album, off, _street(), not_people=not_people) == pm.fit_all(album, off, _street())


def test_missing_person_tables_mean_everyone_counts_as_a_person() -> None:
    class _Conn:
        def execute(self, *_a, **_k):
            class _R:
                def scalar(self):
                    return True       # to_regclass(...) IS NULL
            return _R()
    assert pm.not_a_person_names(_Conn()) == frozenset()


def _faced(cid: str, cluster: str, day: str, v) -> pm.Crop:
    return _crop(cid, cluster, day, v, face=v)


def test_face_bars_calibrate_on_the_longer_street_body_bars_do_not() -> None:
    rng = np.random.default_rng(5)
    album = {"J": {1: [_faced(f"j{k}{i}", f"jc{k}", f"2026-10-0{k + 1}", _vec(1, 0.01 * i)) for k in range(3)
                       for i in range(3)]}}
    st = lambda pre, n: {f"{pre}{i}": [_faced(f"{pre}{i}-{j}", f"{pre}{i}", "2026-10-12",  # noqa: E731
                                              rng.normal(0, 1, 8)) for j in range(3)] for i in range(n)}
    street = st("now", 10)
    face_street = {**street, **st("old", 25)}
    bars = pm.fit_all(album, CFG, street, face_street)
    assert bars[("J", "face")].n_impostor == 35 and bars[("J", "body")].n_impostor == 10
    old = pm.fit_all(album, CFG, street)          # no face street = the guess street, as before
    assert old[("J", "face")].n_impostor == 10


def test_face_window_not_longer_means_one_street(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(pm, "unnamed_clusters", lambda conn, now, cfg, tz, days=None: calls.append(days) or {"a": []})
    street, face = pm.street_pools(None, NOW, pm.PersonModelConfig(face_candidate_days=30.0), ZONE)
    assert street is face and calls == [None]


def test_settings_ship_the_face_window_and_people_split_on() -> None:
    from app.settings import Settings
    from app.vision_individuals_loop import person_model_config

    cfg = person_model_config(Settings())
    assert cfg.face_candidate_days == 90.0 and cfg.face_candidate_days > cfg.candidate_days
    assert cfg.alpha_split_people_only is True
    off = person_model_config(Settings(VISION_PERSON_FACE_CANDIDATE_DAYS=30, VISION_PERSON_ALPHA_SPLIT_PEOPLE_ONLY=False))
    assert off.face_candidate_days == 30.0 and off.alpha_split_people_only is False
    # The face window is pointless past the prune of unnamed clusters.
    assert cfg.face_candidate_days <= Settings().vision_sighting_retention_days


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
    for i in range(170):   # the street the accept bars are calibrated on
        _add_cluster(conn, f"st{i}", None, [pm.unit(rng.normal(0, 1, 8)) for _ in range(3)], day=1)


@needs_pg
def test_cycle_builds_albums_fits_bars_and_guesses(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        s = pm.run_person_model(conn, NOW, CFG)
    assert s["person_people"] == 8 and s["person_street_clusters"] == 172
    with pg.begin() as conn:
        n = conn.execute(text("SELECT count(*) FROM vision_person_exemplar WHERE revoked_at IS NULL")).scalar()
        assert n == 4 * 3 + 7 * 2 * 3
        bar = conn.execute(text("SELECT n_genuine, n_impostor FROM vision_person_bar "
                                "WHERE person_label='Juniper' AND channel='body'")).one()
        assert bar.n_impostor >= 160
        guesses = {r[0]: (r[1], r[2]) for r in conn.execute(text(
            "SELECT individual_id, person_label, band FROM vision_person_attribution WHERE revoked_at IS NULL"))}
    # The Juniper-like stranger's best guess is Juniper. It sits in the street the bar
    # is fitted on (~190 impostor clusters < 2 x people / alpha), so it is asked about, not accepted.
    assert guesses["unk-j"] == ("Juniper", "unsure")
    assert not [k for k, (_, b) in guesses.items() if b == "accept"]
    # A guess never names the cluster.
    with pg.begin() as conn:
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id='unk-j'")).scalar() is None


@needs_pg
def test_a_rename_revokes_and_the_refit_moves(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _world(conn)
        pm.run_person_model(conn, NOW, CFG)
        fitted = conn.execute(text("SELECT max(fitted_at) FROM vision_person_bar")).scalar()
    with pg.begin() as conn:
        again = pm.run_person_model(conn, NOW + timedelta(minutes=10), CFG)
        assert again["person_exemplars_added"] == 0 and again["person_exemplars_revoked"] == 0   # idempotent
        conn.execute(text("UPDATE vision_individual SET label='PersonA' WHERE individual_id='jun0'"))
        s = pm.run_person_model(conn, NOW + timedelta(minutes=20), CFG)
    assert s["person_exemplars_revoked"] == 3 and s["person_exemplars_added"] == 3
    with pg.begin() as conn:
        assert conn.execute(text("SELECT max(fitted_at) FROM vision_person_bar")).scalar() > fitted
        assert conn.execute(text("SELECT count(*) FROM vision_person_bar WHERE person_label='PersonA'")).scalar() >= 1


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


# --- "Is this <name>?" asks (2026-10-20) ----------------------------------------------------------------


def _thumb(conn, iid: str, day: int = 0, ref: str = "") -> None:
    """One non-stationary sighting with a picture, so an ask about ``iid`` can show it."""
    from sqlalchemy import text

    t = NOW - timedelta(days=day, minutes=5)
    conn.execute(text(
        "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
        "dwell_sec, observation_count, thumb_ref) VALUES (:s, :i, 'walkway', :t, :t, 0, 1, :th)"),
        {"s": f"s-{iid}", "i": iid, "t": t, "th": ref or "thumb:" + (iid.encode().hex() * 64)[:64]})


def _confirm_world(conn):
    from sqlalchemy import text

    _world(conn)
    pm.run_person_model(conn, NOW, CFG)
    _thumb(conn, "unk-j")
    conn.execute(text("UPDATE vision_individual SET sighting_count = 2 WHERE individual_id = 'unk-j'"))
    return {r[0]: r[1] for r in conn.execute(text(
        "SELECT individual_id, person_label FROM vision_person_attribution WHERE revoked_at IS NULL"))}


@needs_pg
def test_is_this_x_opens_for_the_best_unsure_guess_with_a_picture(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_daily_cap=2, ask_face_min_faces=0)
    with pg.begin() as conn:
        guesses = _confirm_world(conn)
        assert guesses["unk-j"] == "Juniper"
        opened = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
    # Only unk-j has a picture: the street clusters without one are skipped, never asked blank.
    assert [(a["source_kind"], a["source_ref"]) for a in opened] == [("vision_person_confirm", "unk-j|Juniper")]
    assert opened[0]["image_refs"] and opened[0]["question"].startswith("Is this Juniper?")
    with pg.begin() as conn:
        assert vi._open_person_confirm_asks(conn, NOW + timedelta(minutes=1), ZONE, cfg) == []   # already open
        day = vi.local_day_start(NOW, ZONE)
        slow = conn.execute(text(vi.ASKED_TODAY_SQL), {"d": day, "face": '["rule:face"]'}).scalar()
        own = conn.execute(text(vi.PERSON_CONFIRM_ASKED_TODAY_SQL), {"d": day}).scalar()
    assert (slow, own) == (0, 1)   # its own cap, not ORION_ASK_DAILY_CAP


@needs_pg
def test_is_this_x_still_asks_when_the_slow_rule_used_the_day(pg) -> None:
    """Regression 2026-10-22: the slow rule opens first and used the shared cap of 2
    every day, so "Is this <name>?" was never asked."""
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_daily_cap=1, person_confirm_ask_daily_cap=1, ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        conn.execute(text(
            "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, created_at) "
            "VALUES ('ask-slow', 'q', 'vision_individual', 'someone', :t)"), {"t": NOW})
        assert [a["source_ref"] for a in vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)] == ["unk-j|Juniper"]


@needs_pg
def test_is_this_x_stops_at_its_own_daily_cap(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(person_confirm_ask_daily_cap=1, ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        conn.execute(text(
            "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, created_at) "
            "VALUES ('ask-done', 'q', 'vision_person_confirm', 'other|Someone', 'answered', :t)"), {"t": NOW})
        assert vi._open_person_confirm_asks(conn, NOW, ZONE, cfg) == []
        assert len(vi._open_person_confirm_asks(conn, NOW + timedelta(days=1), ZONE, cfg)) == 1


@needs_pg
def test_yes_names_the_cluster_and_the_album_takes_it(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
        conn.execute(text("UPDATE orion_ask SET status='answered', answer='yes', answered_at=:t WHERE ask_id=:a"),
                     {"t": NOW, "a": ask["ask_id"]})
        vi._apply_answers(conn, NOW)
        row = conn.execute(text("SELECT label, label_ask_id, label_copied FROM vision_individual "
                                "WHERE individual_id='unk-j'")).one()
        assert (row.label, row.label_ask_id, row.label_copied) == ("Juniper", ask["ask_id"], False)
        pm.run_person_model(conn, NOW + timedelta(minutes=10), CFG)
        n = conn.execute(text("SELECT count(*) FROM vision_person_exemplar e JOIN vision_crop_observation c "
                              "USING (crop_id) WHERE c.individual_id='unk-j' AND e.polarity=1 "
                              "AND e.person_label='Juniper' AND e.revoked_at IS NULL")).scalar()
    assert n == 3


@needs_pg
def test_no_rules_the_name_out_for_that_cluster(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
        conn.execute(text("UPDATE orion_ask SET status='answered', answer='no', answered_at=:t WHERE ask_id=:a"),
                     {"t": NOW, "a": ask["ask_id"]})
        vi._apply_answers(conn, NOW)
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id='unk-j'")).scalar() is None
        negs = conn.execute(text("SELECT count(*) FROM vision_person_exemplar WHERE source='confirm_no' "
                                 "AND person_label='Juniper' AND polarity=-1")).scalar()
        assert negs == 3
        pm.run_person_model(conn, NOW + timedelta(minutes=10), CFG)
        # The answer survives reconciliation, and Orion stops guessing Juniper for it.
        assert conn.execute(text("SELECT count(*) FROM vision_person_exemplar WHERE source='confirm_no' "
                                 "AND revoked_at IS NULL")).scalar() == 3
        got = conn.execute(text("SELECT person_label FROM vision_person_attribution WHERE individual_id='unk-j' "
                                "AND revoked_at IS NULL")).scalar()
        assert got != "Juniper"
        # And never asks the same question again.
        later = vi._open_person_confirm_asks(conn, NOW + timedelta(days=1), ZONE, cfg)
        assert all(a["source_ref"] != "unk-j|Juniper" for a in later)


def _answer_confirm_by_picture(conn, verdicts):
    """Answer the open "Is this Juniper?" card about unk-j picture by picture:
    verdicts[k] for the crop unk-j-<k> (its thumbnail is the picture)."""
    import json

    from sqlalchemy import text

    from app import vision_individuals as vi

    (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, vi.IndividualsConfig(ask_face_min_faces=0))
    pictures = []
    for k, v in enumerate(verdicts):
        ref = "thumb:" + f"{k:x}" * 64
        conn.execute(text("UPDATE vision_crop_observation SET thumb_ref=:r WHERE crop_id=:c"),
                     {"r": ref, "c": f"unk-j-{k}"})
        pictures.append({"image_ref": ref, "index": k, "verdict": v})
    answer = "yes" if "same" in verdicts else "no"
    conn.execute(text("UPDATE orion_ask SET status='answered', answer=:ans, answered_at=:t, "
                      "answer_detail=CAST(:d AS jsonb) WHERE ask_id=:a"),
                 {"ans": answer, "t": NOW, "a": ask["ask_id"], "d": json.dumps({"pictures": pictures})})
    vi._apply_answers(conn, NOW)
    return ask


@needs_pg
def test_one_picture_of_three_is_the_person(pg) -> None:
    """2026-10-23, Juniper: "one of the three is Emerald but I can't specify which".
    Mixed answers never name the cluster (it holds more than one person); each
    marked picture teaches its own crop, "not sure" teaches nothing."""
    from sqlalchemy import text

    with pg.begin() as conn:
        _confirm_world(conn)
        _answer_confirm_by_picture(conn, ["same", "different", "unsure"])
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id='unk-j'")).scalar() is None
        assert conn.execute(text("SELECT count(*) FROM vision_person_attribution WHERE individual_id='unk-j' "
                                 "AND person_label='Juniper' AND revoked_at IS NULL")).scalar() == 0
        pm.run_person_model(conn, NOW + timedelta(minutes=10), CFG)
        ex = conn.execute(text("SELECT crop_id, polarity, source FROM vision_person_exemplar "
                               "WHERE source LIKE 'confirm_pick%' AND revoked_at IS NULL ORDER BY crop_id")).fetchall()
        # Both survive the album's reconciliation.
        assert [tuple(r) for r in ex] == [("unk-j-0", 1, "confirm_pick_yes"), ("unk-j-1", -1, "confirm_pick_no")]


@needs_pg
def test_every_picture_is_the_person_still_names_the_cluster(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        _confirm_world(conn)
        ask = _answer_confirm_by_picture(conn, ["same", "same", "same"])
        assert conn.execute(text("SELECT label, label_ask_id FROM vision_individual "
                                 "WHERE individual_id='unk-j'")).one() == ("Juniper", ask["ask_id"])
        assert conn.execute(text("SELECT count(*) FROM vision_person_exemplar "
                                 "WHERE source LIKE 'confirm_pick%'")).scalar() == 0


@needs_pg
def test_cant_tell_is_a_dismissal_and_that_pair_is_not_asked_again(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
        conn.execute(text("UPDATE orion_ask SET status='dismissed' WHERE ask_id=:a"), {"a": ask["ask_id"]})
        later = vi._open_person_confirm_asks(conn, NOW + timedelta(days=2), ZONE, cfg)
    assert all(a["source_ref"] != "unk-j|Juniper" for a in later)


@needs_pg
def test_someone_seen_once_is_not_asked_about(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    with pg.begin() as conn:
        _confirm_world(conn)
        conn.execute(text("UPDATE vision_individual SET sighting_count = 1 WHERE individual_id = 'unk-j'"))
        assert vi._open_person_confirm_asks(conn, NOW, ZONE, vi.IndividualsConfig(ask_face_min_faces=0)) == []


@needs_pg
def test_a_name_given_elsewhere_closes_the_open_is_this_x_card(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
        # Juniper names unk-j on another card.
        conn.execute(text("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, answer, "
                          "answered_at, created_at) VALUES ('ask-who', 'q', 'vision_individual', 'unk-j', "
                          "'answered', 'Maeve', :t, :t)"), {"t": NOW})
        vi._apply_answers(conn, NOW)
        row = conn.execute(text("SELECT status, answer, applied_at IS NOT NULL FROM orion_ask WHERE ask_id=:a"),
                           {"a": ask["ask_id"]}).one()
        assert tuple(row) == ("answered", None, True)
        assert conn.execute(text("SELECT label FROM vision_individual WHERE individual_id='unk-j'")).scalar() == "Maeve"


@needs_pg
def test_an_open_is_this_x_card_keeps_the_slow_rule_off_that_individual(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_face_min_faces=0, ask_min_sightings=2, ask_min_days=1, ask_daily_cap=3)
    with pg.begin() as conn:
        _confirm_world(conn)
        (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
        conn.execute(text("UPDATE vision_individual SET distinct_days = 2 WHERE individual_id = 'unk-j'"))
        conn.execute(text(
            "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
            "dwell_sec, observation_count, thumb_ref) VALUES ('s-unk-j-2', 'unk-j', 'walkway', :t, :t, 0, 1, "
            "'thumb:' || repeat('e', 64))"), {"t": NOW - timedelta(days=1)})
        opened = vi._open_sighting_asks(conn, NOW, ZONE, cfg)
    assert all(a["source_ref"] != "unk-j" for a in opened)


@needs_pg
def test_no_marks_only_the_crops_orion_guessed_on(pg) -> None:
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_face_min_faces=0)
    with pg.begin() as conn:
        _confirm_world(conn)
        for j in range(3, 30):   # an older history of 27 more crops
            conn.execute(text(
                "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
                "box_xyxy, embedding, individual_id) SELECT 'unk-j-old' || :j, 'obs', 'walkway', :t, 'person', 0.9, "
                "'{0,0,1,1}', embedding, 'unk-j' FROM vision_crop_observation WHERE crop_id = 'unk-j-0'"),
                {"j": j, "t": NOW - timedelta(days=20, minutes=j)})
        (ask,) = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
        conn.execute(text("UPDATE orion_ask SET status='answered', answer='no', answered_at=:t WHERE ask_id=:a"),
                     {"t": NOW, "a": ask["ask_id"]})
        vi._apply_answers(conn, NOW)
        n = conn.execute(text("SELECT count(*) FROM vision_person_exemplar WHERE source='confirm_no'")).scalar()
    assert n == pm.ATTRIBUTION_CROPS


@needs_pg
def test_face_street_reaches_back_further_than_the_guess_street(pg) -> None:
    with pg.begin() as conn:
        _add_cluster(conn, "recent", None, [_vec(1)], day=5)
        _add_cluster(conn, "old", None, [_vec(0, 1)], day=60)
        _add_cluster(conn, "ancient", None, [_vec(0, 0, 1)], day=120)
        street, face = pm.street_pools(conn, NOW, CFG, ZONE)
    assert set(street) == {"recent"} and set(face) == {"recent", "old"}


@needs_pg
def test_not_a_person_names_come_from_the_person_record(pg) -> None:
    from sqlalchemy import text

    with pg.begin() as conn:
        if conn.execute(text("SELECT to_regclass('person') IS NULL")).scalar():
            pytest.skip("manual_migration_person_record_v1.sql not applied")
        conn.execute(text("TRUNCATE person_fact, person_link, person CASCADE"))
        _add_cluster(conn, "deco", "Skeleton", [_vec(1)], day=1)
        _add_cluster(conn, "jun", "Juniper", [_vec(0, 1)], day=1)
        _add_cluster(conn, "gone", "Old name", [_vec(0, 0, 1)], day=1)
        _add_cluster(conn, "lag", "PersonT", [_vec(0, 0, 0, 1)], day=1)
        conn.execute(text(
            "INSERT INTO person (person_id, display_name, status) VALUES "
            "('p-deco', 'Halloween decor', 'not_a_person'), ('p-jun', 'Juniper', 'active'), "
            "('p-ups', 'UPS delivery', 'not_a_person'), ('p-ted', 'PersonT', 'active')"))
        conn.execute(text(
            "INSERT INTO person_link (link_id, person_id, source_kind, source_ref, decided_by, revoked_at) VALUES "
            "('l1', 'p-deco', 'vision_individual', 'deco', 'camera_name', NULL), "
            "('l2', 'p-jun', 'vision_individual', 'jun', 'camera_name', NULL), "
            "('l3', 'p-deco', 'vision_individual', 'gone', 'camera_name', now()), "
            "('l4', 'p-deco', 'vision_individual', 'lag', 'camera_name', NULL)"))
        names = pm.not_a_person_names(conn)
        conn.execute(text("TRUNCATE person_fact, person_link, person CASCADE"))
    # The linked label (not just the row's own name) counts; a revoked link and a
    # name an active person also carries do not.
    assert names == frozenset({"halloween decor", "skeleton", "ups delivery"})


# --- camera priors (2026-11-04, the office camera: "99% of the time it is me in there") ---

import dataclasses as _dc

PRIORS = {"cam0": {"Juniper": 0.97, "Sibling": 0.01}}


def _on(stream: str, crops):
    return [_dc.replace(c, stream=stream) for c in crops]


def _office_world():
    """Juniper's and Sibling's albums are walkway crops; an office crop scores
    higher against Sibling's (clothes and light), as on her first office visit."""
    album = {
        "Juniper": {1: _on("walkway", [_crop("j1", "jc", "2026-10-01", _vec(1, 0.6))])},
        "Sibling": {1: _on("walkway", [_crop("s1", "sc", "2026-10-01", _vec(1, 0.1))])},
    }
    bars = {("Juniper", "body"): pm.Bar("Juniper", "body", None, None, 29, 1500),
            ("Sibling", "body"): pm.Bar("Sibling", "body", None, None, 272, 1500)}
    office = {"o1": _on("cam0", [_crop("o1a", "o1", "2026-11-05", _vec(1, 0.05))])}
    return album, bars, office


def test_office_guess_follows_the_prior_when_no_match_is_confident() -> None:
    album, bars, office = _office_world()
    (raw,) = pm.attribute(office, album, bars, {}, CFG)
    assert raw["person_label"] == "Sibling" and raw["basis"] == "evidence"   # what happened on 11-05
    (g,) = pm.attribute(office, album, bars, {}, CFG, priors=PRIORS)
    assert g["person_label"] == "Juniper" and g["band"] == "unsure" and g["basis"] == "prior"


def test_a_confident_match_beats_the_prior() -> None:
    album, bars, office = _office_world()
    bars[("Sibling", "body")] = pm.Bar("Sibling", "body", 0.5, None, 272, 1500)   # armed, and the crop clears it
    (g,) = pm.attribute(office, album, bars, {}, CFG, priors=PRIORS)
    assert g["person_label"] == "Sibling" and g["band"] == "accept" and g["basis"] == "evidence"


def test_an_outdoor_reject_bound_cannot_rule_her_out_in_the_office() -> None:
    album, bars, office = _office_world()
    bars[("Juniper", "body")] = pm.Bar("Juniper", "body", None, 0.99, 29, 1500)   # learned outdoors
    (g,) = pm.attribute(office, album, bars, {}, CFG, priors=PRIORS)
    assert g["person_label"] == "Juniper" and g["basis"] == "prior"
    # Once her album holds an office crop, the bound is in-camera and counts: she is
    # ruled out, and the guess falls back to the scores (the 1% prior names nobody).
    album["Juniper"][1].append(_on("cam0", [_crop("j2", "jo", "2026-11-05", _vec(0, 1))])[0])
    (g2,) = pm.attribute(office, album, bars, {}, CFG, priors=PRIORS)
    assert g2["person_label"] == "Sibling" and g2["basis"] == "evidence"


def test_prior_names_match_album_names_ignoring_case_and_other_cameras_are_unchanged() -> None:
    album, bars, office = _office_world()
    (g,) = pm.attribute(office, album, bars, {}, CFG, priors={"cam0": {"juniper ": 0.97}})
    assert g["person_label"] == "Juniper" and g["basis"] == "prior"
    walk = {"w1": _on("walkway", office["o1"])}
    (w,) = pm.attribute(walk, album, bars, {}, CFG, priors=PRIORS)
    assert w["person_label"] == "Sibling" and w["basis"] == "evidence"


def test_after_her_no_a_small_prior_never_names_the_guess() -> None:
    """Review finding: a "no" must not walk down the 1% names ("Is this <sibling>?")."""
    album, bars, office = _office_world()
    (g,) = pm.attribute(office, album, bars, {"o1": {"juniper"}}, CFG, priors=PRIORS)
    assert g["person_label"] == "Sibling" and g["basis"] == "evidence"   # the scores', as on any camera


def test_a_small_prior_relaxes_no_reject_and_a_misspelled_prior_changes_nothing() -> None:
    album, bars, office = _office_world()
    bars[("Sibling", "body")] = pm.Bar("Sibling", "body", None, 1.01, 272, 1500)   # rejects every crop
    bars[("Juniper", "body")] = pm.Bar("Juniper", "body", None, 0.99, 29, 1500)
    # Juniper ruled out: the 1% sibling's reject still counts, so there is no guess at all.
    assert pm.attribute(office, album, bars, {"o1": {"juniper"}}, CFG, priors=PRIORS) == []
    # A prior naming nobody in the album: the same guesses as with no prior.
    assert pm.attribute(office, album, bars, {}, CFG, priors={"cam0": {"Junper": 0.97}}) == \
        pm.attribute(office, album, bars, {}, CFG)


def test_camera_priors_file_example_and_local_first(tmp_path) -> None:
    from orion.person_model.camera_priors import load_camera_priors

    ex = tmp_path / "camera_priors.example.yaml"
    ex.write_text("streams:\n  cam0:\n    Juniper: 0.97\n")
    assert load_camera_priors(ex) == {"cam0": {"Juniper": 0.97}}
    bad = tmp_path / "bad.yaml"
    bad.write_text("streams:\n  cam0:\n    A: 0.8\n    B: 0.5\n")
    with pytest.raises(ValueError):
        load_camera_priors(bad)
    dup = tmp_path / "dup.yaml"
    dup.write_text("streams:\n  cam0:\n    Juniper: 0.5\n    juniper: 0.4\n")
    with pytest.raises(ValueError):
        load_camera_priors(dup)
    assert load_camera_priors(tmp_path / "missing.yaml") == {}


@needs_pg
def test_write_attributions_stores_basis_and_works_without_the_column(pg) -> None:
    from datetime import datetime, timezone

    from sqlalchemy import text

    now = datetime(2026, 11, 5, 2, 0, tzinfo=timezone.utc)
    g = {"individual_id": "o1", "person_label": "Juniper", "channel": "body", "score": 0.7, "band": "unsure",
         "basis": "prior"}
    with pg.begin() as conn:
        pm.write_attributions(conn, [g], now)
        assert conn.execute(text("SELECT basis FROM vision_person_attribution WHERE individual_id='o1'")).scalar() == "prior"
    with pg.begin() as conn:
        conn.execute(text("ALTER TABLE vision_person_attribution DROP COLUMN basis"))
        try:
            pm.write_attributions(conn, [{**g, "individual_id": "o2"}], now)   # before the migration: still lands
            assert conn.execute(text(
                "SELECT count(*) FROM vision_person_attribution WHERE individual_id='o2'")).scalar() == 1
        finally:
            conn.execute(text("ALTER TABLE vision_person_attribution ADD COLUMN IF NOT EXISTS basis text"))


@needs_pg
def test_office_prior_asks_rank_last_and_one_is_open_per_name(pg) -> None:
    """Review finding: one office visit splits into several records, each guessed
    Juniper by the camera prior. They must not take the day's asks: evidence
    guesses go first, and only one prior "Is this Juniper?" is open at a time."""
    from sqlalchemy import text

    from app import vision_individuals as vi

    cfg = vi.IndividualsConfig(ask_daily_cap=6, ask_face_min_faces=0)
    with pg.begin() as conn:
        for iid in ("off-1", "off-2", "off-3", "walk-1"):
            _add_cluster(conn, iid, None, [_vec(1, 0.1)], 0)
            _thumb(conn, iid)
            conn.execute(text("UPDATE vision_individual SET sighting_count = 2 WHERE individual_id = :i"), {"i": iid})
        for iid, person, score, basis in (("off-1", "Juniper", 0.9, "prior"), ("off-2", "Juniper", 0.8, "prior"),
                                          ("off-3", "Juniper", 0.7, "prior"), ("walk-1", "Sibling", 0.5, "evidence")):
            conn.execute(text(
                "INSERT INTO vision_person_attribution (individual_id, person_label, channel, score, band, "
                "model_version, decided_at, basis) VALUES (:i, :p, 'body', :s, 'unsure', :v, :t, :b)"),
                {"i": iid, "p": person, "s": score, "v": pm.MODEL_VERSION, "t": NOW, "b": basis})
        opened = vi._open_person_confirm_asks(conn, NOW, ZONE, cfg)
    assert [a["source_ref"] for a in opened] == ["walk-1|Sibling", "off-1|Juniper"]
    with pg.begin() as conn:   # the next cycle: her office ask is still open, so no second one
        assert vi._open_person_confirm_asks(conn, NOW + timedelta(minutes=1), ZONE, cfg) == []
