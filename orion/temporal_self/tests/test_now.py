"""Orion's now (integration design 2026-11-15, Part A patch 1): the registry, salience, the now, cites
and the integration metric. Pure: no database, no clock."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from typing import get_args

import pytest

from orion.schemas.registry import resolve
from orion.schemas.temporal_self import NOW_LABEL_MAX, TemporalNowV1, TemporalSourceKind
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import CONSTRAINT_KINDS, CONTEXT_KINDS, PROCESS_KINDS
from orion.integration.organs import load_organs
from orion.temporal_self.now import (
    MAGNITUDE_RULES,
    build_now,
    magnitude,
    normalize_cite,
    now_label,
    stored_payload_surprise,
    parse_organs,
    record_use,
    salience_of,
    use_rate,
)

T = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
REG = load_organs()


def row(event_id, kind, *, sal=None, at=T, label="", surprise=None, payload=None, verdict=None):
    return {"event_id": event_id, "source_kind": kind, "occurred_at": at, "label": label, "verdict": verdict,
            "payload": payload or {}, "related_refs": [], "salience": sal, "surprise_nats": surprise}


# --------------------------------------------------------------------------- registry


def test_every_source_kind_belongs_to_exactly_one_organ():
    assert set(get_args(TemporalSourceKind)) == set(REG.organ_of)


def test_every_declared_magnitude_rule_exists_and_names_its_own_organs_kind():
    for kind, rule in REG.magnitude_rule.items():
        assert rule in MAGNITUDE_RULES
        assert kind in REG.kinds_of(REG.organ_of[kind])


def test_registry_refuses_a_kind_claimed_twice_an_unknown_kind_and_an_unknown_rule():
    with pytest.raises(ValueError, match="claimed by both"):
        parse_organs({"organs": {"a": {"temporal_sources": ["chat_turn"]}, "b": {"temporal_sources": ["chat_turn"]}}})
    with pytest.raises(ValueError, match="not a TemporalSourceKind"):
        parse_organs({"organs": {"a": {"temporal_sources": ["no_such_kind"]}}})
    with pytest.raises(ValueError, match="not a rule"):
        parse_organs({"organs": {"a": {"temporal_sources": ["chat_turn"], "salience_magnitude": {"chat_turn": "x"}}}})
    with pytest.raises(ValueError, match="not one of its temporal_sources"):
        parse_organs({"organs": {"a": {"temporal_sources": ["chat_turn"],
                                       "salience_magnitude": {"gpu_wait": "waited_ms"}}}})


def test_new_kinds_never_fold_into_an_arc():
    assert S.EXPERIENCE_ONLY_KINDS == {"reading_turn", "prediction_surprise", "prime_thought", "experiment_result",
                                       "dream_day", "reverie_question", "intention_close", "day_letter",
                                       "personality_choice"}
    for k in S.EXPERIENCE_ONLY_KINDS:
        assert k not in PROCESS_KINDS and k not in CONTEXT_KINDS and k not in CONSTRAINT_KINDS


def test_experience_only_tables_are_exactly_the_new_kinds_tables():
    base = {"act_id": "act_" + "c" * 24, "opened_at": T, "closed_at": T + timedelta(seconds=5),
            "object_ref": "intention:d8db9d7b1bba4cf7", "letter_date": "2026-11-24", "lane": "prime", "evidence_id": "ev1",
            "exceedance_id": 1, "scored_at": T, "hit_raw": False, "probability": 0.1, "n_paths": 40,
            "estimate_id": "e1", "created_at": T, "details": {"estimator": "family_c_pooled_run", "effect_z": 2.0},
            "day": "2026-11-17", "question_id": "q1"}
    tables = {S.ADAPTERS[k](base).source_table for k in S.EXPERIENCE_ONLY_KINDS}
    assert tables == S.EXPERIENCE_ONLY_TABLES


def test_now_models_are_registered():
    for name in ("TemporalNowV1", "TemporalNowItemV1", "ExperienceCitesV1", "ExperienceUseResultV1",
                 "ExperienceUseRateV1"):
        assert resolve(name).__name__ == name


# --------------------------------------------------------------------------- salience


def test_salience_is_none_under_twenty_past_events_never_zero():
    hist = [(T - timedelta(hours=i + 1), 1.0) for i in range(19)]
    assert salience_of(5.0, T, hist) is None
    assert salience_of(5.0, T, hist + [(T - timedelta(hours=30), 1.0)]) == 1.0


def test_all_equal_magnitudes_read_zero_the_rest_point():
    hist = [(T - timedelta(hours=i + 1), 3.0) for i in range(40)]
    assert salience_of(3.0, T, hist) == 0.0


def test_salience_counts_only_strictly_smaller_events_in_the_30_days_before():
    hist = [(T - timedelta(hours=i + 1), float(i % 4)) for i in range(40)]  # 10 each of 0,1,2,3
    assert salience_of(2.0, T, hist) == pytest.approx(0.5)
    # Events after the one scored, or older than 30 days, are not its history.
    later = [(T + timedelta(hours=1), -1.0)] * 50
    old = [(T - timedelta(days=31), -1.0)] * 50
    assert salience_of(2.0, T, hist + later + old) == pytest.approx(0.5)


def test_a_kind_with_no_rule_or_no_field_has_no_magnitude():
    assert magnitude(None, {"payload": {"x": 1}}) is None
    assert magnitude("chain_thought_count", {"payload": {"thought_ids": ["a", "b", "c"]}}) is None  # retired rule
    assert magnitude("surprise_nats", {"surprise_nats": None, "payload": {"surprise_nats": 0.7}}) == 0.7
    assert magnitude("surprise_nats", {"surprise_nats": float("nan"), "payload": {}}) is None
    assert magnitude("metacog_severity", {"verdict": "critical"}) == 2.0
    assert magnitude("visit_seconds", {"occurred_at": T, "payload": {"ended_at": (T + timedelta(minutes=5)).isoformat()}}) == 300.0


# --------------------------------------------------------------------------- the now


def test_build_now_excludes_the_readers_own_organ():
    rows = [row("prime_thought:a", "prime_thought", sal=0.99), row("reading_turn:b", "reading_turn", sal=0.1)]
    now = build_now(rows, "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG)
    assert [i.event_id for i in now.items] == ["reading_turn:b"]
    assert all(i.organ != "prime" for i in now.items)


def test_build_now_caps_two_per_organ_and_twelve_in_all_every_organ_seated_before_any_second():
    rows = []
    for organ_kind in ("reverie_chain", "curiosity_run", "reading_turn", "prediction_surprise", "dream_cycle",
                       "vision_percept", "gpu_wait", "metacog_observation", "memory_episode", "chat_turn"):
        for j in range(3):
            rows.append(row(f"{organ_kind}:{j}", organ_kind, sal=round(0.1 * j + 0.01 * len(rows), 3)))
    now = build_now(rows, "prime", since=T - timedelta(hours=1), as_of=T, shown_id="s", registry=REG)
    assert len(now.items) == 12
    organs = [i.organ for i in now.items]
    assert max(organs.count(o) for o in set(organs)) == 2
    # Round one: all 10 organs, each with its own best row; only then two second rows.
    assert len(set(organs[:10])) == 10
    first = now.items[:10]
    assert all(i.event_id.endswith(":2") for i in first)  # each kind's own highest salience
    assert [i.salience for i in first] == sorted((i.salience for i in first), reverse=True)
    assert now.dropped_by_organ_cap == 10 and now.dropped_by_total_cap == 8 and now.candidates == 30


def test_an_unscored_kind_reaches_the_now_past_many_scored_organs():
    """#3783: reading_turn has no magnitude rule. Before, any scored row (even 0.00) ranked above it, so with 12 or
    more scored rows a reading never reached a reader."""
    rows = [row(f"{k}:{j}", k, sal=s) for k, s in (("gpu_wait", 0.9), ("prediction_surprise", 0.0),
                                                   ("field_dominance_run", 0.97), ("curiosity_run", 0.0),
                                                   ("dream_cycle", 0.0), ("metacog_observation", 0.6))
            for j in range(2)]
    rows.append(row("reading_turn:memonce", "reading_turn", at=T - timedelta(hours=1)))
    now = build_now(rows, "prime", since=T - timedelta(hours=3), as_of=T, shown_id="s", registry=REG)
    ids = [i.event_id for i in now.items]
    assert "reading_turn:memonce" in ids
    reading = next(i for i in now.items if i.event_id == "reading_turn:memonce")
    assert reading.salience is None  # still shown as unknown, never 0
    # It stands at 0.5 (its expected place in its own history): after a 0.9, before a 0.00.
    assert ids.index("gpu_wait:0") < ids.index("reading_turn:memonce") < ids.index("prediction_surprise:0")


def test_unscored_rows_of_one_kind_are_newest_first_and_a_quiet_kind_goes_before_a_busy_one():
    rows = [row("chat_turn:old", "chat_turn", at=T - timedelta(hours=2)),
            row("chat_turn:new", "chat_turn", at=T - timedelta(hours=1)),
            *[dict(row(f"attention_row:{j}", "attention_row", at=T - timedelta(minutes=j)), kind_total=60)
              for j in range(2)],
            dict(row("faculty_status_changed:x", "faculty_status_changed", at=T - timedelta(hours=3)), kind_total=1)]
    now = build_now(rows, "prime", since=T - timedelta(hours=3), as_of=T, shown_id="s", registry=REG)
    ids = [i.event_id for i in now.items]
    assert ids.index("chat_turn:new") < ids.index("chat_turn:old")
    # metacognition: the kind with 1 row in the window goes before the kind with 60, though it is older.
    assert ids.index("faculty_status_changed:x") < ids.index("attention_row:0")


def test_a_row_about_a_held_concern_ranks_first_and_is_marked():
    seed = "world_pulse_read_seed:reading:50753c3f-d556-5a6f-bfa7-b02b95254905"
    rows = [row(f"gpu_wait:{j}", "gpu_wait", sal=0.99) for j in range(2)]
    rows += [row(f"field_dominance_run:{j}", "field_dominance_run", sal=1.0) for j in range(2)]
    about = dict(row("reading_turn:act_48c", "reading_turn", at=T - timedelta(hours=5)),
                 correlation_id="50753c3f-d556-5a6f-bfa7-b02b95254905",
                 related_refs=["act:act_48c", "https://github.com/Clare-Nie/MemOnce"])
    other = dict(row("reading_turn:act_hf", "reading_turn"), correlation_id="ee32b4c4-d556-5a6f-bfa7-b02b95254905")
    rows += [other, about]
    now = build_now(rows, "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG,
                    held=[seed, "prime_concern:520a5a9f8d084495"])
    assert now.items[0].event_id == "reading_turn:act_48c" and now.items[0].held_refs == [seed]
    assert all(i.held_refs == [] for i in now.items[1:])  # a different reading's seed is not a match
    body = now.model_dump(mode="json")
    assert body["items"][0]["held_refs"] == [seed] and "held_refs" not in body["items"][1]


def test_held_matching_is_exact_ids_only():
    from orion.temporal_self.now import NOW_MAX_HELD, held_lookup, held_ref_keys, held_refs_of

    # a #fragment names part of one event; a long last segment is an id; a short one (a date) and a path are not.
    assert held_ref_keys("dream_day:2026-11-23#new1") == {"dream_day:2026-11-23#new1", "dream_day:2026-11-23"}
    assert held_ref_keys("turn:fc22c1a0-4cb6-5292-bf3e-ffa202172e04") == {
        "turn:fc22c1a0-4cb6-5292-bf3e-ffa202172e04", "fc22c1a0-4cb6-5292-bf3e-ffa202172e04"}
    assert held_ref_keys("https://github.com/Clare-Nie/MemOnce") == {"https://github.com/Clare-Nie/MemOnce"}
    assert held_ref_keys("  ") == set()
    look = held_lookup(["dream_day:2026-11-23#new1", "arc:00112233aabbccdd"], {"00112233aabbccdd": "node:circe"})
    assert held_refs_of({"event_id": "dream_day:2026-11-23"}, look) == ["dream_day:2026-11-23#new1"]
    assert held_refs_of({"event_id": "x", "source_ref": "2026-11-23"}, look) == []  # a date alone is not an id
    assert held_refs_of({"event_id": "body_binding:1", "subject_ref": "node:circe"}, look) == ["arc:00112233aabbccdd"]
    assert held_refs_of({"event_id": "x", "label": "MemOnce dream_day:2026-11-23"}, look) == []  # never text
    # At most NOW_MAX_HELD rows rank first; the rest compete as ordinary rows (still marked).
    rows = [dict(row(f"reading_turn:{j}", "reading_turn", at=T - timedelta(minutes=j)), correlation_id="c" * 20)
            for j in range(NOW_MAX_HELD + 2)]
    rows += [row(f"gpu_wait:{j}", "gpu_wait", sal=0.9) for j in range(2)]
    now = build_now(rows, "prime", since=T - timedelta(hours=1), as_of=T, shown_id="s", registry=REG,
                    held=["seed:" + "c" * 20], max_per_organ=12)
    assert [i.event_id for i in now.items[:NOW_MAX_HELD]] == [f"reading_turn:{j}" for j in range(NOW_MAX_HELD)]
    assert now.items[NOW_MAX_HELD].event_id.startswith("gpu_wait")


def _pre_held_models():
    """TemporalNowV1 as an image built before #3783 has it: the same fields minus held_refs, extra="forbid"."""
    from pydantic import ConfigDict, create_model

    from orion.schemas.temporal_self import TemporalNowItemV1

    item_fields = {k: (f.annotation, f) for k, f in TemporalNowItemV1.model_fields.items() if k != "held_refs"}
    OldItem = create_model("OldItem", __config__=ConfigDict(extra="forbid"), **item_fields)
    now_fields = {k: (f.annotation, f) for k, f in TemporalNowV1.model_fields.items() if k != "items"}
    return create_model("OldNow", __config__=ConfigDict(extra="forbid"), items=(list[OldItem], []), **now_fields)


def test_no_held_is_the_same_body_shape_as_before():
    from pydantic import ValidationError

    rows = [row("reading_turn:b", "reading_turn", sal=0.1), row("chat_turn:c", "chat_turn")]
    a = build_now(rows, "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG)
    b = build_now(rows, "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG, held=[])
    assert a == b and all("held_refs" not in i for i in a.model_dump(mode="json")["items"])
    OldNow = _pre_held_models()
    # A reader that sends no held (reverie, curiosity, the thought service, dream day) on an old image still parses.
    OldNow.model_validate(a.model_dump(mode="json"))
    # A held match is only ever sent to the reader that asked (Prime, which reads the body as a dict); an old
    # forbid model would refuse it, which is why no other reader sends held.
    c = dict(rows[0], correlation_id="c" * 20)
    held_body = build_now([c], "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG,
                          held=["seed:" + "c" * 20]).model_dump(mode="json")
    with pytest.raises(ValidationError):
        OldNow.model_validate(held_body)


def test_a_direct_match_takes_a_held_seat_before_rows_that_only_share_a_held_arcs_subject():
    """An arc's subject is often broad (orion:body:energy: 40 rows in a day). Its rows must not push the concern's
    own event out of the held seats."""
    from orion.temporal_self.now import NOW_MAX_HELD

    subject_rows = [dict(row(f"body_binding:{j}", "body_binding", at=T - timedelta(minutes=j)),
                         subject_ref="orion:body:energy") for j in range(5)]
    own_event = dict(row("reading_turn:own", "reading_turn", at=T - timedelta(hours=8)), correlation_id="c" * 20)
    now = build_now([*subject_rows, own_event], "prime", since=T - timedelta(hours=9), as_of=T, shown_id="s",
                    registry=REG, held=["seed:" + "c" * 20, "arc:00112233aabbccdd"],
                    arc_subjects={"00112233aabbccdd": "orion:body:energy"}, max_per_organ=12)
    assert now.items[0].event_id == "reading_turn:own"
    assert [i.event_id for i in now.items[1:NOW_MAX_HELD]] == ["body_binding:0", "body_binding:1"]
    assert now.items[1].held_refs == ["arc:00112233aabbccdd"]
    # a stored "None" subject is not an id: an arc whose subject reads "None" matches nothing
    look_none = build_now([dict(row("x:1", "body_binding"), subject_ref="None")], "prime", since=T, as_of=T,
                          shown_id="s", registry=REG, held=["arc:00112233aabbccdd"],
                          arc_subjects={"00112233aabbccdd": "None"})
    assert look_none.items[0].held_refs == []


def test_a_held_seat_is_never_spent_on_a_row_its_organ_cap_drops():
    from orion.temporal_self.now import NOW_MAX_HELD

    key = "c" * 20
    readings = [dict(row(f"reading_turn:{j}", "reading_turn", at=T - timedelta(minutes=j)), correlation_id=key)
                for j in range(3)]
    old_chat = dict(row("chat_turn:old", "chat_turn", at=T - timedelta(hours=5)), correlation_id=key)
    fillers = [row(f"gpu_wait:{j}", "gpu_wait", sal=0.99) for j in range(2)]
    now = build_now([*readings, old_chat, *fillers], "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s",
                    registry=REG, held=["seed:" + key])
    ids = [i.event_id for i in now.items]
    assert set(ids[:NOW_MAX_HELD]) == {"reading_turn:0", "reading_turn:1", "chat_turn:old"}
    assert "reading_turn:2" not in ids  # the organ cap (2) drops it, and it took no held seat


def test_held_refs_are_named_in_the_readers_order():
    from orion.temporal_self.now import held_lookup, held_refs_of

    look = held_lookup(["z:" + "a" * 16, "a:" + "b" * 16])
    assert held_refs_of({"event_id": "e", "related_refs": ["a" * 16, "b" * 16]}, look) == ["z:" + "a" * 16,
                                                                                           "a:" + "b" * 16]


def test_build_now_refuses_an_unregistered_reader_and_never_shows_an_unregistered_kind():
    with pytest.raises(ValueError):
        build_now([], "nobody", since=T, as_of=T, shown_id="s", registry=REG)
    now = build_now([row("immune_finding:x", "immune_finding", sal=1.0)], "prime", since=T, as_of=T,
                    shown_id="s", registry=REG)
    assert now.items == [] and now.unregistered_kinds == ["immune_finding"]


def test_labels_are_code_written_and_capped():
    assert len(now_label("prime_thought", "x" * 500, None, {})) == NOW_LABEL_MAX
    assert now_label("curiosity_run", "", None, {"n_tested": 3, "n_moved": 1, "n_formed": 0, "arm": "value_order"}) == \
        "curiosity run: tested 3, moved 1, formed 0"
    assert now_label("visual_run", "", "done", {}) == "visual run [done]"
    TemporalNowV1.model_validate(build_now([row("chat_turn:1", "chat_turn", label="L" * 400)], "prime", since=T,
                                           as_of=T, shown_id="s", registry=REG).model_dump())


# --------------------------------------------------------------------------- cites


def test_record_use_drops_a_cite_that_was_not_shown_and_counts_it():
    r = record_use(shown_id="s1", reader_organ="prime", shown_event_ids=["reading_turn:a", "gpu_wait:b"],
                   cites=["experience:reading_turn:a", "dream_cycle:zzz", "[experience:gpu_wait:b]"],
                   citing_ref="prime_thought:t1", cited_at=T)
    assert [u.event_id for u in r.uses] == ["reading_turn:a", "gpu_wait:b"]
    assert r.rejected_not_shown == ["dream_cycle:zzz"]
    assert all(u.citing_organ == "prime" and u.shown_id == "s1" for u in r.uses)


def test_record_use_keeps_at_most_three_and_counts_duplicates_once():
    shown = [f"gpu_wait:{i}" for i in range(5)]
    r = record_use(shown_id="s", reader_organ="prime", shown_event_ids=shown, cites=shown + ["gpu_wait:0"],
                   citing_ref="x", cited_at=T)
    assert len(r.uses) == 3 and r.rejected_over_cap == ["gpu_wait:3", "gpu_wait:4"]
    again = record_use(shown_id="s", reader_organ="prime", shown_event_ids=shown, cites=shown[:1], citing_ref="x",
                       cited_at=T)
    assert again.uses[0].use_id == r.uses[0].use_id  # idempotent id per (shown_id, event_id)


def test_normalize_cite_keeps_colons_inside_event_ids():
    assert normalize_cite("experience:reading_turn:act_1") == "reading_turn:act_1"
    assert normalize_cite(" [experience:x:y] ") == "x:y"


# --------------------------------------------------------------------------- the metric


def test_use_rate_counts_other_organs_only_and_names_closed_organs():
    kinds = {"reading_turn": 10, "gpu_wait": 5, "prime_thought": 2, "immune_finding": 1}
    uses = [("reading_turn:a", "reading_turn", "prime"), ("reading_turn:a", "reading_turn", "reverie"),
            ("gpu_wait:b", "gpu_wait", "body")]  # body citing body: not integration
    m = use_rate(kinds, uses, window_start=T - timedelta(days=1), window_end=T, registry=REG,
                 cites_made={"prime": 1})
    assert m.experiences == 17 and m.used_by_other == 1 and m.rate == pytest.approx(1 / 17)
    assert m.closed_organs == ["reading"] and m.unregistered_kinds == {"immune_finding": 1}
    by = {o.organ: o for o in m.organs}
    assert by["reading"].rate == pytest.approx(0.1) and by["body"].used_by_other == 0
    assert by["painting"].rate is None and by["prime"].cites_made == 1


def test_use_rate_rest_point_is_zero_not_none_when_experiences_exist():
    m = use_rate({"reading_turn": 3}, [], window_start=T, window_end=T, registry=REG)
    assert m.rate == 0.0 and m.closed_organs == []
    assert use_rate({}, [], window_start=T, window_end=T, registry=REG).rate is None


# --------------------------------------------------------------------------- the three new sources


def test_reading_turn_is_one_closed_reading_act_with_its_surprisal():
    base = {"act_id": "act_" + "a" * 24, "object_ref": "https://x.example/a", "correlation_id": "c",
            "opened_at": T, "closed_at": None, "outcome_note": "stage1_completed", "surprise_nats": 0.002,
            "outcome_value": 1, "seed_title": "A title"}
    assert S.reading_turn(base) is None  # still open
    e = S.reading_turn({**base, "closed_at": T + timedelta(minutes=3)})
    assert e.event_id == f"reading_turn:act_{'a' * 24}" and e.label == "stage1_completed: A title"
    assert e.payload["surprise_nats"] == 0.002 and f"act:act_{'a' * 24}" in e.related_refs
    assert e.subject_ref is None


def test_prediction_surprise_is_minus_log_probability_of_what_happened():
    base = {"exceedance_id": 7, "series_id": "athena.cabinet_temp_c", "name": "shed", "threshold": 34,
            "window_start": T, "window_end": T + timedelta(minutes=30), "probability": 0.0, "n_paths": 40,
            "scored_at": T + timedelta(minutes=31), "max_raw": 24.5, "joint_ids": [3, 4]}
    assert S.prediction_surprise({**base, "hit_raw": None}) is None  # not scored yet
    calm = S.prediction_surprise({**base, "hit_raw": False})
    hit = S.prediction_surprise({**base, "hit_raw": True})
    p = 0.5 / 41  # Laplace over 40 paths
    assert calm.payload["surprise_nats"] == pytest.approx(-math.log(1 - p))
    assert hit.payload["surprise_nats"] == pytest.approx(-math.log(p)) and hit.verdict == "crossed"
    assert calm.related_refs == ["prediction_joint_surprise:3", "prediction_joint_surprise:4"]
    assert hit.payload["surprise_nats"] > 4.0 > calm.payload["surprise_nats"]


def test_experiment_result_is_one_run_with_its_effect_z_as_magnitude():
    row = {"estimate_id": "e-9", "created_at": T, "action": "run_cap:aitown.session", "series_id": "circe.gpu3_watts",
           "effect": 9.1, "ci_lo": 2.0, "ci_hi": 16.2, "p_value": 0.01,
           "details": {"estimator": "family_c_pooled_run", "effect_z": -2.5, "experiment_ids": ["n1", "n2"],
                       "nights": ["2026-11-17", "2026-11-18"], "summary": "I held my AI Town sessions ...",
                       "manipulation": {"verdict": "passed"},
                       "targets": {"lane.fast.tier1_hold_s": {"estimate": 6.0}}}}
    e = S.experiment_result(row)
    assert e.source_kind == "experiment_result" and e.event_id == "experiment_result:e-9"
    assert e.label.startswith("I held my AI Town sessions") and e.verdict == "passed"
    assert e.payload["effect_z"] == 2.5 and e.payload["tier1_hold_s"] == 6.0 and e.payload["nights"] == 2
    assert e.related_refs == ["prediction_experiment:n1", "prediction_experiment:n2"]
    assert MAGNITUDE_RULES["effect_z"]({"payload": e.payload}) == 2.5
    # Other effect rows (an observational estimate, a heat fit) are not experiences.
    assert S.experiment_result({**row, "details": {"estimator": "heat_per_watt"}}) is None


def test_dream_day_is_one_night_with_its_paragraph_and_memories():
    row = {"day": date(2026, 11, 17), "run_id": "dream-day-2026-11-17", "produced": True, "note": "2 memories written",
           "memories_written": 2, "memory_ids": ["m1", "m2"], "kept": 1, "dropped": 1, "new": 0, "items_dropped": 1,
           "refs_dropped": 1, "day_text": "A quiet day of reading.", "created_at": T}
    e = S.dream_day(row)
    assert e.source_kind == "dream_day" and e.event_id == "dream_day:2026-11-17" and e.source_table == "dream_day"
    assert e.label == "A quiet day of reading." and e.verdict == "produced"
    assert e.related_refs == ["durable_run:dream-day-2026-11-17", "episode_memory:m1", "episode_memory:m2"]
    assert e.payload["memories_written"] == 2 and e.payload["memory_ids"] == ["m1", "m2"]
    quiet = S.dream_day({**row, "produced": False, "memories_written": 0, "memory_ids": [], "day_text": None,
                         "note": "no lasting trace"})
    assert quiet.label == "no lasting trace" and quiet.verdict == "no_lasting_trace"
    assert S.dream_day({**row, "created_at": None}) is None


def test_prime_thought_labels_with_its_own_outcome_note():
    e = S.prime_thought({"act_id": "act_" + "b" * 24, "object_ref": "prime_thought:f6c0", "opened_at": T,
                         "closed_at": T + timedelta(seconds=14), "outcome_note": "woke: quiet | no move",
                         "surprise_nats": 0.0004})
    assert e.label == "woke: quiet | no move" and e.payload["surprise_nats"] == 0.0004
    assert e.related_refs == [f"act:act_{'b' * 24}", "prime_thought:f6c0"]


def test_action_outcome_carries_its_surprisal_only_when_read():
    base = {"id": 1, "observed_at": T, "claim_upheld": True, "dispatch_kind": "act:look", "target_id": "hub.x",
            "prediction_error": 0.2}
    assert "surprise_nats" not in S.action_outcome(base).payload
    assert S.action_outcome({**base, "surprise_nats": 0.3}).payload["surprise_nats"] == 0.3


def test_record_use_counts_cites_already_stored_for_the_shown_row_against_the_cap():
    shown = [f"gpu_wait:{i}" for i in range(5)]
    r = record_use(shown_id="s", reader_organ="prime", shown_event_ids=shown, cites=shown, citing_ref="x",
                   cited_at=T, already_used=shown[:2])
    assert [u.event_id for u in r.uses] == ["gpu_wait:2"] and r.rejected_over_cap == ["gpu_wait:3", "gpu_wait:4"]


# --------------------------------------------------------------------------- blinding (#3458)


def _now_from(events):
    from orion.temporal_self.now import stored_payload_surprise

    rows = []
    for e in events:
        payload = dict(e.payload, ended_at=e.ended_at.isoformat() if e.ended_at else None)
        rows.append({"event_id": e.event_id, "source_kind": e.source_kind, "occurred_at": e.occurred_at,
                     "label": e.label, "verdict": e.verdict, "payload": payload, "related_refs": e.related_refs,
                     "salience": 0.5, "surprise_nats": stored_payload_surprise(payload)})
    return build_now(rows, "prime", since=T - timedelta(days=1), as_of=T + timedelta(days=1), shown_id="s",
                     registry=REG, max_per_organ=12)


def test_the_now_never_shows_a_blinded_experiment_arm():
    """The three-arm curiosity test (#3183) and the dream/control hypothesis test are blinded. Their arms are
    derived from the code that assigns them, never listed by hand: a new arm is covered automatically."""
    from orion.curiosity.value import OFFER_ARMS
    from orion.schemas.dream_cycle import HypothesisArm

    assert set(OFFER_ARMS) >= {"value_order", "replica_order", "uncertainty_order"}
    curiosity = [S.curiosity_run({"run_id": f"r{i}", "turn_started_at": T, "completed_at": T + timedelta(minutes=5),
                                  "arm": arm, "n_tested": 2, "n_moved": 1, "n_formed": 0, "offered": []})
                 for i, arm in enumerate(OFFER_ARMS)]
    dream_arms = get_args(HypothesisArm)
    dreams = [S.dream_hypothesis({"hypothesis_id": f"h{i}", "cycle_id": "c", "arm": arm, "created_at": T})
              for i, arm in enumerate(dream_arms)]
    assert all(e.payload.get("arm") for e in curiosity + dreams)  # stored for the analysis scripts
    now = _now_from(curiosity + dreams)
    text = now.model_dump_json()
    assert len(now.items) == len(curiosity) + len(dreams)
    for arm in OFFER_ARMS:
        assert arm not in text
    assert '"arm"' not in text
    labels = {i.source_kind: set() for i in now.items}
    for i in now.items:
        labels[i.source_kind].add(i.label)
    # the same outcome under any arm renders the same words
    assert labels == {"curiosity_run": {"curiosity run: tested 2, moved 1, formed 0"},
                      "dream_hypothesis": {"dream hypothesis"}}
    assert "control" not in text


def test_self_inquiry_is_a_line_not_an_arm_and_keeps_its_name():
    e = S.curiosity_run({"run_id": "s", "turn_started_at": T, "completed_at": T + timedelta(minutes=5),
                         "arm": "self_inquiry", "n_tested": 0, "n_moved": 0, "n_formed": 3, "offered": []})
    assert _now_from([e]).items[0].label == "self-inquiry run: tested 0, moved 0, formed 3"


# --------------------------------------------------------------------------- pinned (Prime hunger, Part B)


def test_pinned_rows_come_first_outside_the_organ_cap_and_inside_the_total():
    """A need's content row (Prime hunger) is pinned into the now so a cite of it is valid against the shown row."""
    from orion.temporal_self.now import NOW_MAX_PINNED

    rows = []
    for organ_kind in ("reverie_chain", "curiosity_run", "reading_turn", "prediction_surprise", "dream_cycle",
                       "vision_percept", "gpu_wait", "metacog_observation", "memory_episode", "chat_turn"):
        for j in range(3):
            rows.append(row(f"{organ_kind}:{j}", organ_kind, sal=round(0.1 * j + 0.01 * len(rows), 3)))
    old_chat = row("chat_turn:old", "chat_turn", at=T - timedelta(days=2))  # not a candidate: older than since
    own = row("prime_thought:mine", "prime_thought", sal=1.0)
    pins = [old_chat, row("chat_turn:2", "chat_turn"), own, row("reading_turn:x", "reading_turn"),
            row("reading_turn:y", "reading_turn")]
    now = build_now(rows, "prime", since=T - timedelta(hours=1), as_of=T, shown_id="s", registry=REG, pinned=pins)
    ids = [i.event_id for i in now.items]
    # The reader's own organ is never pinned; at most NOW_MAX_PINNED; pins first, in order; no duplicates.
    assert ids[:NOW_MAX_PINNED] == ["chat_turn:old", "chat_turn:2", "reading_turn:x"]
    assert "prime_thought:mine" not in ids and len(ids) == 12 and len(set(ids)) == 12
    # Pins stand outside the per-organ cap: chat shows its 2 pins plus up to 2 ranked rows.
    assert sum(1 for i in now.items if i.organ == "chat") <= 2 + 2


def test_no_pins_is_exactly_the_old_now():
    rows = [row("reading_turn:b", "reading_turn", sal=0.1), row("chat_turn:c", "chat_turn")]
    a = build_now(rows, "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG)
    b = build_now(rows, "prime", since=T - timedelta(hours=6), as_of=T, shown_id="s", registry=REG, pinned=[])
    assert a == b


def test_a_sparse_kind_under_twenty_past_events_reaches_the_now():
    """#3540: a kind WITH a magnitude rule but fewer than 20 past events has salience None. It must reach the now past
    a full field of scored rows, including ones at 0.00, and still show salience None (never 0)."""
    hist = [(T - timedelta(days=i + 1), 300.0) for i in range(5)]  # 5 past office visits: too few to score
    sal = salience_of(600.0, T, hist)
    assert sal is None
    rows = [row(f"{k}:{j}", k, sal=s) for k, s in (("gpu_wait", 0.9), ("prediction_surprise", 0.0),
                                                   ("field_dominance_run", 0.97), ("curiosity_run", 0.0),
                                                   ("dream_cycle", 0.0), ("metacog_observation", 0.6),
                                                   ("retro_play", 0.8))
            for j in range(2)]
    rows.append(row("office_session:visit", "office_session", sal=sal, at=T - timedelta(hours=2)))
    now = build_now(rows, "prime", since=T - timedelta(hours=3), as_of=T, shown_id="s", registry=REG)
    ids = [i.event_id for i in now.items]
    assert "office_session:visit" in ids
    assert next(i for i in now.items if i.event_id == "office_session:visit").salience is None
    assert ids.index("office_session:visit") < ids.index("prediction_surprise:0")


# --- integration patch 7a (#3429): intention, day letter and personality emit -----------------------------------


def test_intention_close_is_one_closed_take_on_with_its_trace_and_run():
    row = {"act_id": "act_0418ced14d2fa1b989fe4086", "object_ref": "intention:d8db9d7b1bba4cf7", "trace_id": "tr1",
           "correlation_id": None, "opened_at": T, "closed_at": T + timedelta(hours=3), "outcome_value": 1.0,
           "outcome_note": "carried: the reply is in the chat thread 9c91 (608 chars; Juniper was not pinged)",
           "surprise_nats": 0.1126, "run_act_id": "act_9bd65d5db4e3bfb4612d03e1", "run_outcome_note": "completed"}
    e = S.intention_close(row)
    assert e.source_kind == "intention_close" and e.event_id == "intention_close:act_0418ced14d2fa1b989fe4086"
    assert e.source_table == "grammar_acts" and e.verdict == "produced" and e.subject_ref is None
    assert e.label.startswith("carried: the reply is in the chat thread")
    assert e.related_refs == ["act:act_0418ced14d2fa1b989fe4086", "intention:d8db9d7b1bba4cf7", "trace:tr1",
                              "durable_run:intention-d8db9d7b1bba4cf7", "act:act_9bd65d5db4e3bfb4612d03e1"]
    assert e.ended_at == T + timedelta(hours=3) and e.payload["concern_id"] == "d8db9d7b1bba4cf7"
    assert stored_payload_surprise(e.payload) == pytest.approx(0.1126)
    refused = S.intention_close({**row, "outcome_value": 0.0, "run_act_id": None,
                                 "outcome_note": "Hub refused the order: there is no thread to answer in"})
    assert refused.verdict == "not_produced" and "durable_run:intention-d8db9d7b1bba4cf7" not in refused.related_refs
    assert now_label("intention_close", refused.label, refused.verdict, refused.payload) == (
        "intention close: Hub refused the order: there is no thread to answer in [not_produced]")
    # an unscored close (expired, no outcome) is not produced; no note falls back to plain words
    bare = S.intention_close({**row, "outcome_value": None, "outcome_note": None})
    assert bare.verdict == "not_produced" and bare.label == "intention not carried"
    own_trace = S.intention_close({**row, "trace_id": "act:act_0418ced14d2fa1b989fe4086"})
    assert not any(r.startswith("trace:") for r in own_trace.related_refs)  # never the act twice
    assert S.intention_close({**row, "closed_at": None}) is None
    assert S.intention_close({**row, "object_ref": "concern:x"}) is None


def test_day_letter_is_one_letter_with_its_threads_as_concern_refs():
    md = "- The drain-blind learner [cu…\n- PR #3828: open, unmerged [juniper_chat:6eb9]\n"
    row = {"letter_date": date(2026, 11, 24), "run_id": "orion-day-2026-11-24-1", "carry_forward_md": md,
           "note_chars": 18787, "created_at": T}
    e = S.day_letter(row)
    assert e.source_kind == "day_letter" and e.event_id == "day_letter:2026-11-24" and e.source_table == "orion_day_letter"
    assert e.label == "wrote the letter for 2026-11-24, carrying 2 threads; first: The drain-blind learner [cu…"
    assert e.related_refs == ["durable_run:orion-day-2026-11-24-1", "orion_day_letter:2026-11-24#thread0",
                              "orion_day_letter:2026-11-24#thread1"]
    assert e.payload["threads"][1] == "PR #3828: open, unmerged [juniper_chat:6eb9]"
    none = S.day_letter({**row, "carry_forward_md": "No threads were named in this letter."})
    assert none.label == "wrote the letter for 2026-11-24, carrying 0 threads" and none.related_refs == [
        "durable_run:orion-day-2026-11-24-1"]
    assert S.day_letter({**row, "created_at": None}) is None


def test_day_letter_thread_refs_are_the_refs_prime_gives_the_thread_concerns():
    from orion.prime import day_letter as DL

    assert S.LETTER_THREAD_REF.format(day="2026-11-24", i=3) == f"{DL.LETTER_EVENT_PREFIX}2026-11-24#thread3"


def test_personality_choice_is_one_folded_prime_choice():
    meta = {"act_id": "act_1e2b", "choice_id": "prime:efd5", "alternatives": 2, "declined_all": False,
            "correlation_id": None}
    row = {"evidence_id": "ev-1", "lane": "prime", "source_ref": "act_1e2b", "occurred_at": T - timedelta(minutes=16),
           "excerpt": "My act horizons are barely claims over Consult with Claude promise", "disposition_id": "disp-ad56",
           "similarity": 0.99, "meta": meta, "created_at": T, "disposition_name": None, "disposition_status": "candidate"}
    e = S.personality_choice(row)
    assert e.source_kind == "personality_choice" and e.event_id == "personality_choice:ev-1"
    assert e.source_table == "personality_evidence" and e.occurred_at == T and e.verdict == "chose"
    assert e.label == "My act horizons are barely claims over Consult with Claude promise (picked from 2 options)"
    assert e.related_refs == ["act:act_1e2b", "personality_choice:prime:efd5", "personality_disposition:disp-ad56"]
    assert e.payload["chosen_at"] == (T - timedelta(minutes=16)).isoformat()
    waits = S.personality_choice({**row, "excerpt": "Prime waits instead of acting", "disposition_name": "patience",
                                  "meta": json.dumps({**meta, "alternatives": 3, "declined_all": True})})
    assert waits.verdict == "declined_all"
    assert waits.label == 'Prime waits instead of acting (declined all 3 options); folded into my leaning "patience"'
    assert S.personality_choice({**row, "meta": {**meta, "declined_all": "true"}}).verdict == "chose"  # only a real bool
    assert S.personality_choice({**row, "lane": "reading"}) is None  # only Prime's choices are this organ's
    assert S.personality_choice({**row, "created_at": None}) is None


def test_patch7a_labels_fall_back_to_plain_words_without_a_stored_label():
    assert now_label("day_letter", "", None, {"letter_date": "2026-11-24", "threads": ["a", "b"]}) == (
        "wrote the letter for 2026-11-24, carrying 2 threads")
    assert now_label("personality_choice", "", "declined_all", {"declined_all": True}) == (
        "Prime declined every option [declined_all]")
    assert now_label("intention_close", "", None, {"concern_id": "d8db"}) == "intention d8db closed"
    assert now_label("intention_close", "", None, {}) == "an intention closed"  # never "None"
    assert now_label("personality_choice", "", "chose", {}) == "Prime made a choice [chose]"
    assert now_label("personality_choice", "", "chose", {"alternatives": 4}) == "Prime chose one of 4 options"  # verdict word already in it

