"""Camera guesses Juniper answers in orion_ask (vision_person_confirm, vision_same_individual).

Row shapes copied from live orion_ask rows (2026-11-11). Neither ask records a confidence before it
opens, so the grade is the hit rate only: the faculty must stay ``unknown`` /
``ungradeable:no_stated_confidence`` however low that rate is. Orion only asks when unsure, so a
low hit rate is the asking rule working, not an unreliable faculty.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.self_monitor import adapters
from orion.self_monitor.catalog import UNGRADEABLE
from orion.self_monitor.collect import SOURCES, build_state
from orion.self_monitor.names import plain_name
from orion.self_monitor.record import reduce_bets

NOW = datetime(2026, 11, 11, 4, 0, tzinfo=timezone.utc)


def _confirm(i: int = 0, answer="yes", *, prior=False, pictures=None, status="answered", age_h=2.0, **kw):
    refs = [f"sighting:s{i}a", f"sighting:s{i}b"] + (["rule:camera_prior"] if prior else [])
    row = {"ask_id": f"ask-c{i}", "source_kind": "vision_person_confirm", "status": status, "answer": answer,
           "answer_detail": None if pictures is None else {"pictures": [
               {"index": n, "verdict": v, "image_ref": f"thumb:{i}{n}"} for n, v in enumerate(pictures)]},
           "evidence_refs": refs, "created_at": NOW - timedelta(hours=age_h + 5),
           "answered_at": NOW - timedelta(hours=age_h)}
    row.update(kw)
    return row


def _same(i: int = 0, answer="no", cosine="0.678"):
    return {"ask_id": f"ask-s{i}", "source_kind": "vision_same_individual", "status": "answered", "answer": answer,
            "answer_detail": {"names": {"a": "Addy", "b": "Juniper"}},
            "evidence_refs": [f"individual:a{i}", f"individual:b{i}", f"face_cosine:{cosine}"],
            "created_at": NOW - timedelta(hours=8), "answered_at": NOW - timedelta(hours=3)}


def test_a_yes_upholds_the_guess_and_no_confidence_is_invented() -> None:
    bet = adapters.camera_ask(_confirm(answer="yes"))
    assert bet.faculty == "vision.person_confirm" and bet.outcome == "upheld"
    assert bet.claim_kind == "event" and bet.settled_by == "juniper" and bet.stated is None
    assert bet.bet_id == bet.evidence_ref == "orion_ask:ask-c0"
    assert adapters.camera_ask(_confirm(answer="No.")).outcome == "refuted"


def test_face_cosine_is_never_read_as_a_stated_confidence() -> None:
    bet = adapters.camera_ask(_same(answer="yes", cosine="0.591"))
    assert bet.faculty == "vision.same_person" and bet.outcome == "upheld" and bet.stated is None
    assert adapters.camera_ask(_same(answer="no")).outcome == "refuted"


def test_a_camera_prior_guess_is_its_own_faculty() -> None:
    assert adapters.camera_ask(_confirm(prior=True)).faculty == "vision.person_confirm.camera_prior"


def test_per_picture_answers_follow_the_appliers_rule() -> None:
    assert adapters.camera_ask(_confirm(answer="yes", pictures=["same", "same"])).outcome == "upheld"
    assert adapters.camera_ask(_confirm(answer="no", pictures=["different"])).outcome == "refuted"
    # live: 4 cards answered "yes" with pictures {different, same}: the cluster held two people
    assert adapters.camera_ask(_confirm(answer="yes", pictures=["same", "different"])).outcome == "no_change"
    assert adapters.camera_ask(_confirm(answer="yes", pictures=["same", "unsure"])).outcome == "no_change"


def test_rows_that_are_not_a_settled_guess_are_skipped() -> None:
    assert adapters.camera_ask(_confirm(answer="")) is None          # live: one answered row, blank answer
    assert adapters.camera_ask(_confirm(answer="maybe")) is None
    assert adapters.camera_ask(_confirm(status="dismissed")) is None  # "I don't know" is not a verdict
    assert adapters.camera_ask(_confirm(answered_at=None)) is None
    assert adapters.camera_ask(_confirm(source_kind="vision_individual", answer="Juniper")) is None


def test_json_text_columns_are_read_like_jsonb() -> None:
    row = _confirm(prior=True, pictures=["different"])
    row["evidence_refs"] = json.dumps(row["evidence_refs"])
    row["answer_detail"] = json.dumps(row["answer_detail"])
    bet = adapters.camera_ask(row)
    assert bet.faculty == "vision.person_confirm.camera_prior" and bet.outcome == "refuted"


def test_a_low_hit_rate_without_a_stated_confidence_never_reads_unreliable() -> None:
    """Regression: the evidence lane is right about a third of the time (live, 14 days: 39 yes,
    76 no). With no stated confidence that is the hit rate of an unsure guess, never a status."""
    rows = [_confirm(i, "yes" if i % 3 == 0 else "no", age_h=1 + i) for i in range(30)]
    rows += [_same(i, "yes" if i % 2 else "no") for i in range(12)]
    state = build_state({"orion_ask": rows, **{n: [] for n, _, _ in SOURCES if n != "orion_ask"}},
                        now=NOW, window_days=7.0)
    assert state.source_errors == []
    by = {f.faculty: f for f in state.faculties}
    confirm = by["vision.person_confirm"]
    assert confirm.n_graded == 30 and confirm.status == "unknown" and confirm.candidate_status == "unknown"
    assert confirm.gradeability == "ungradeable:no_stated_confidence"
    assert 0.25 < confirm.reliability < 0.45 and confirm.reliability_hi < 0.6
    assert by["vision.same_person"].n_graded == 12 and by["vision.same_person"].sources == ["orion_ask"]
    assert by["vision.same_person"].gradeability == "ungradeable:no_stated_confidence"


def test_catalog_entries_show_only_when_no_answer_landed() -> None:
    names = {u.faculty for u in UNGRADEABLE}
    assert {"vision.same_person", "vision.person_confirm"} <= names
    assert all(u.gradeability != "ungradeable:source_bug" for u in UNGRADEABLE if u.faculty.startswith("vision."))
    empty = reduce_bets([], now=NOW, window_days=7.0, ungradeable=UNGRADEABLE)
    assert {f.faculty: f.gradeability for f in empty}["vision.same_person"] == "ungradeable:no_stated_confidence"
    bets = [adapters.camera_ask(_same(i)) for i in range(3)]
    got = [f for f in reduce_bets(bets, now=NOW, window_days=7.0, ungradeable=UNGRADEABLE)
           if f.faculty == "vision.same_person"]
    assert len(got) == 1 and got[0].n_bets == 3


def test_camera_faculties_have_plain_names() -> None:
    for faculty in ("vision.person_confirm", "vision.person_confirm.camera_prior", "vision.same_person"):
        assert not plain_name(faculty).endswith(" bets"), faculty
