"""Orion's now (integration design 2026-11-15, Part A patch 1): the registry, salience, the now, cites
and the integration metric. Pure: no database, no clock."""

from __future__ import annotations

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
                                       "dream_day", "reverie_question"}
    for k in S.EXPERIENCE_ONLY_KINDS:
        assert k not in PROCESS_KINDS and k not in CONTEXT_KINDS and k not in CONSTRAINT_KINDS


def test_experience_only_tables_are_exactly_the_new_kinds_tables():
    base = {"act_id": "act_" + "c" * 24, "opened_at": T, "closed_at": T + timedelta(seconds=5), "object_ref": "x",
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


def test_build_now_caps_two_per_organ_and_twelve_in_all_highest_salience_first():
    rows = []
    for organ_kind in ("reverie_chain", "curiosity_run", "reading_turn", "prediction_surprise", "dream_cycle",
                       "vision_percept", "gpu_wait", "metacog_observation", "memory_episode", "chat_turn"):
        for j in range(3):
            rows.append(row(f"{organ_kind}:{j}", organ_kind, sal=round(0.1 * j + 0.01 * len(rows), 3)))
    now = build_now(rows, "prime", since=T - timedelta(hours=1), as_of=T, shown_id="s", registry=REG)
    assert len(now.items) == 12
    organs = [i.organ for i in now.items]
    assert max(organs.count(o) for o in set(organs)) == 2
    sal = [i.salience for i in now.items]
    assert sal == sorted(sal, reverse=True)
    assert now.dropped_by_organ_cap == 10 and now.dropped_by_total_cap == 8 and now.candidates == 30


def test_unknown_salience_ranks_after_known_and_newest_first_on_ties():
    rows = [row("chat_turn:old", "chat_turn", at=T - timedelta(hours=2)),
            row("chat_turn:new", "chat_turn", at=T - timedelta(hours=1)),
            row("gpu_wait:z", "gpu_wait", sal=0.0)]
    now = build_now(rows, "prime", since=T - timedelta(hours=3), as_of=T, shown_id="s", registry=REG)
    assert [i.event_id for i in now.items] == ["gpu_wait:z", "chat_turn:new", "chat_turn:old"]


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
