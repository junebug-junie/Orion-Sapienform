"""Reducers over REAL stored row shapes (pulled from live Postgres 2026-10-16, unedited).

OFFER_ROW/OUTCOME_ROW: curiosity run ee31d36e8bfe. Orion tested two priors: rank 1 of the
ordered list, and one from the separate "tested repeatedly" (stale) section.
OVERRIDE_ROW: substrate_attention_broadcast_log, open_loops trimmed to the fields the reducer
reads (id, salience, source_refs) plus two for context.
"""

from __future__ import annotations

import copy
import json

import pytest

from orion.personality.choices import (
    curiosity_choice_from_rows,
    explain_curiosity_choice,
    explain_voluntary_override_choice,
    voluntary_override_choice_from_row,
)
from orion.schemas.personality import MAX_CHOICE_LIST, ChoiceOptionV1, PersonalityChoiceV1

OFFER_ROW = {'run_id': 'ee31d36e8bfe',
 'decided_at': '2026-10-09T09:11:55.975679+00:00',
 'arm': 'uncertainty_order',
 'offered': [{'rank': 1,
              'yield': 0.13326293962771127,
              'prior_id': 'self:induction_ledger_dangling_20261008',
              'confidence': 0.55,
              'entropy_nats': 0.6881388137135884,
              'times_tested': 0,
              'expected_nats': 0.09170340118739878},
             {'rank': 2,
              'yield': 0.13326293962771127,
              'prior_id': 'self:camera_layer_mediation_not_closeness_20261007',
              'confidence': 0.6,
              'entropy_nats': 0.6730116670092565,
              'times_tested': 0,
              'expected_nats': 0.08968751314939988},
             {'rank': 3,
              'yield': 0.13326293962771127,
              'prior_id': 'self:ring_is_doorway_projection_20261007',
              'confidence': 0.6,
              'entropy_nats': 0.6730116670092565,
              'times_tested': 0,
              'expected_nats': 0.08968751314939988},
             {'rank': 4,
              'yield': 0.13326293962771127,
              'prior_id': 'band17_wall_prelease_20260921',
              'confidence': 0.62,
              'entropy_nats': 0.6640641265641081,
              'times_tested': 2,
              'expected_nats': 0.08849513760724155},
             {'rank': 5,
              'yield': 0.13326293962771127,
              'prior_id': 'self:chat_seat_demand_collapse_is_upstream_20261008',
              'confidence': 0.65,
              'entropy_nats': 0.6474466390346325,
              'times_tested': 0,
              'expected_nats': 0.0862806423698368},
             {'rank': 6,
              'yield': 0.13326293962771127,
              'prior_id': 'self:presence_ring_is_three_kinds_20260930',
              'confidence': 0.68,
              'entropy_nats': 0.6268694575724263,
              'times_tested': 2,
              'expected_nats': 0.08353846667893036},
             {'rank': 7,
              'yield': 0.13326293962771127,
              'prior_id': 'self:graph_schema_incompatibility_20260927',
              'confidence': 0.7,
              'entropy_nats': 0.6108643020548935,
              'times_tested': 1,
              'expected_nats': 0.08140557260546526},
             {'rank': 8,
              'yield': 0.13326293962771127,
              'prior_id': 'repair_echo_finaltext_silent_mode_20260920',
              'confidence': 0.7,
              'entropy_nats': 0.6108643020548935,
              'times_tested': 1,
              'expected_nats': 0.08140557260546526}],
 'stale_offered': [{'rank': 1,
                    'yield': 0.06663146981385563,
                    'prior_id': 'self:outward_learning_in_record_not_in_loop_20260921',
                    'confidence': 0.99,
                    'entropy_nats': 0.056001534354847386,
                    'times_tested': 21,
                    'expected_nats': 0.003731464545894613},
                   {'rank': 2,
                    'yield': 0.13326293962771127,
                    'prior_id': 'self:four_wiring_points_named_20260922',
                    'confidence': 0.74,
                    'entropy_nats': 0.5730569171314204,
                    'times_tested': 16,
                    'expected_nats': 0.07636724935092681},
                   {'rank': 3,
                    'yield': 0.053305175851084505,
                    'prior_id': 'self:turn_start_context_is_stale_snapshot_20260923',
                    'confidence': 0.96,
                    'entropy_nats': 0.16794414773417304,
                    'times_tested': 15,
                    'expected_nats': 0.00895229232813061},
                   {'rank': 4,
                    'yield': 0.14895297973914373,
                    'prior_id': 'self:i_is_not_anchorable_to_a_model_282fbb9a08e4',
                    'confidence': 0.82,
                    'entropy_nats': 0.47139348681009424,
                    'times_tested': 14,
                    'expected_nats': 0.07021546448998829},
                   {'rank': 5,
                    'yield': 0.05352544743238994,
                    'prior_id': 'self:i_am_not_anchorable_to_a_model_282fbb9a08e4',
                    'confidence': 0.95,
                    'entropy_nats': 0.19851524334587267,
                    'times_tested': 11,
                    'expected_nats': 0.010625617222237604},
                   {'rank': 6,
                    'yield': 0.1514202777917096,
                    'prior_id': 'substrate_domain_isolation_two_paths',
                    'confidence': 0.95,
                    'entropy_nats': 0.19851524334587267,
                    'times_tested': 10,
                    'expected_nats': 0.030059233293320872},
                   {'rank': 7,
                    'yield': 0.13326293962771127,
                    'prior_id': 'self:alignment_verdict_is_self_graded_4255a432f394',
                    'confidence': 0.9,
                    'entropy_nats': 0.3250829733914482,
                    'times_tested': 9,
                    'expected_nats': 0.04332151265706143},
                   {'rank': 8,
                    'yield': 0.053368866355199884,
                    'prior_id': 'self:role_writes_race_a_mirrored_path_20260929',
                    'confidence': 0.65,
                    'entropy_nats': 0.6474466390346325,
                    'times_tested': 7,
                    'expected_nats': 0.03455349315076264}],
 'material_ids': ['crystallization:323d9b09-6315-4703-905d-6951a5a0e9a2',
                  'crystallization:7906e91b-4577-4a47-9585-bb4658d43e6c',
                  'crystallization:16aa4185-568b-4334-8e46-fda0ef582a2f',
                  'crystallization:ec68d348-9252-491b-8b50-374c20b47a1e',
                  'crystallization:27e0a72f-14fc-4f1e-a52d-1cbec1e3c5fc',
                  'crystallization:f9cf0147-a507-44cd-ba31-e755dded5f80',
                  'crystallization:9b306bdf-7790-468a-a9f9-7489ecbeba4f',
                  'crystallization:c1dd4e36-6341-48a5-a82b-a9318e674893',
                  'crystallization:7bb4c16c-6bab-4c87-91f3-6a949e10328c',
                  'crystallization:b0ef033a-8c30-4b62-921f-40120126c2e8',
                  'crystallization:df67a9e7-8ff3-4bc3-a5d9-d4b04a8f1f80',
                  'crystallization:02bbb053-612a-452a-8993-ed70f6733cf0',
                  'relation:1a6b4581-1bf3-491b-baa8-eeb74678c726',
                  'relation:abb13c94-3c80-4c73-a0e3-ddf9f2639a31',
                  'relation:e76af090-0bf1-4ebc-a9e6-d254a736b52f',
                  'relation:eb0dd028-1417-4cc9-9d28-e7f68e8b8201',
                  'relation:0355e19d-8d99-4f87-b3a6-f178e6201ccc',
                  'relation:81008581-613d-43e9-9486-97ca383731de'],
 'turn_started_at': '2026-10-09T21:05:42.692508+00:00'}

OUTCOME_ROW = {'run_id': 'ee31d36e8bfe',
 'completed_at': '2026-10-09T22:04:55.405601+00:00',
 'turn_ok': True,
 'realized_nats': 0.0012830495056959101,
 'n_tested': 2,
 'per_prior': [{'kind': 'tested',
                'nats': 0.0012830495056959101,
                'after': 0.82,
                'before': 0.8,
                'prior_id': 'self:induction_ledger_dangling_20261008',
                'tested_delta': 1},
               {'kind': 'tested',
                'nats': 0.0,
                'after': 0.96,
                'before': 0.96,
                'prior_id': 'self:turn_start_context_is_stale_snapshot_20260923',
                'tested_delta': 1}]}

OVERRIDE_ROW = {'log_id': 'broadcast-03864cdd93596bdf7d977f19',
 'generated_at': '2026-10-16T13:33:26.504636+00:00',
 'projection_json': {'correlation_id': None,
                     'frame': {'correlation_id': None,
                               'generated_at': '2026-10-16T13:33:26.504636Z',
                               'voluntary_override': {'applied_bias': 0.944175,
                                                      'beat_loop_id': 'open-loop-e91fa7e1c61e',
                                                      'effort_spent': 0.944175,
                                                      'beat_bottom_up': 0.9628696604600219,
                                                      'chosen_loop_id': 'open-loop-7376a3da4050',
                                                      'chosen_bottom_up': 0.9443178581600683,
                                                      'goal_artifact_id': 'artifact-e0d711ed-d709-443c-887e-c74acad1c1b0',
                                                      'goal_drive_origin': None},
                               'open_loops': [{'id': 'open-loop-e91fa7e1c61e',
                                               'salience': 0.9628696604600219,
                                               'combined_salience': 0.9628696604600219,
                                               'top_down_bias': 0.0,
                                               'description': 'Perception prediction error',
                                               'source_refs': ['node:substrate.perception']},
                                              {'id': 'open-loop-7376a3da4050',
                                               'salience': 0.9443178581600683,
                                               'combined_salience': 1.0,
                                               'top_down_bias': 0.944175,
                                               'description': 'Execution prediction error',
                                               'source_refs': ['node:substrate.execution']},
                                              {'id': 'open-loop-08d23ba68a84',
                                               'salience': 0.3936813747731397,
                                               'combined_salience': 0.3936813747731397,
                                               'top_down_bias': 0.0,
                                               'description': 'Route prediction error',
                                               'source_refs': ['node:substrate.route']}]}}}


# ------------------------------------------------------------- curiosity lane


def test_curiosity_real_rows():
    c = curiosity_choice_from_rows(OFFER_ROW, OUTCOME_ROW)
    assert c is not None
    assert c.choice_id == "curiosity_prior_selection:ee31d36e8bfe"
    assert c.source_ref == "curiosity_run:ee31d36e8bfe"
    assert c.chooser == "orion_llm"
    refs = [o.ref for o in c.chosen]
    assert refs == [
        "curiosity_prior:self:induction_ledger_dangling_20261008",
        "curiosity_prior:self:turn_start_context_is_stale_snapshot_20260923",
    ]
    # rank 1 from the ordered list; the stale-section pick has no rank in that order.
    assert [o.rank for o in c.chosen] == [1, None]
    assert c.chosen[1].score == pytest.approx(0.00895229232813061)
    assert c.chosen_best_rank == 1
    # 8 live + 8 stale shown, 2 picked.
    assert len(c.alternatives) == 14
    assert not set(refs) & {o.ref for o in c.alternatives}
    assert c.cost_nats == pytest.approx(0.0012830495056959101)
    assert c.cost_seconds == pytest.approx(3552.713093)
    assert c.occurred_at.isoformat() == "2026-10-09T21:05:42.692508+00:00"
    assert c.situation_refs[:2] == ["curiosity_arm:uncertainty_order", "curiosity_turn_ok:true"]
    assert len(c.situation_refs) == 20


def test_curiosity_choice_id_is_deterministic():
    a = curiosity_choice_from_rows(OFFER_ROW, OUTCOME_ROW)
    b = curiosity_choice_from_rows(copy.deepcopy(OFFER_ROW), copy.deepcopy(OUTCOME_ROW))
    assert a.model_dump() == b.model_dump()


def test_curiosity_accepts_jsonb_as_text():
    offer = dict(OFFER_ROW, offered=json.dumps(OFFER_ROW["offered"]))
    outcome = dict(OUTCOME_ROW, per_prior=json.dumps(OUTCOME_ROW["per_prior"]))
    assert curiosity_choice_from_rows(offer, outcome).chosen_best_rank == 1


def test_curiosity_deviation_rank_and_off_menu():
    outcome = copy.deepcopy(OUTCOME_ROW)
    outcome["per_prior"] = [
        {"kind": "tested", "prior_id": "self:presence_ring_is_three_kinds_20260930"},  # rank 6
        {"kind": "tested", "prior_id": "self:not_on_any_menu"},
    ]
    c = curiosity_choice_from_rows(OFFER_ROW, outcome)
    assert c.chosen_best_rank == 6
    assert c.chosen[1].rank is None and c.chosen[1].score is None
    assert len(c.alternatives) == 15


def test_curiosity_formed_and_moved_untested_are_not_chosen():
    outcome = copy.deepcopy(OUTCOME_ROW)
    outcome["per_prior"] = [
        {"kind": "formed", "prior_id": "self:new_one"},
        {"kind": "moved_untested", "prior_id": "self:induction_ledger_dangling_20261008"},
    ]
    choice, reason = explain_curiosity_choice(OFFER_ROW, outcome)
    assert choice is None and reason == "formed_only"
    outcome["per_prior"] = outcome["per_prior"][1:]
    assert explain_curiosity_choice(OFFER_ROW, outcome) == (None, "nothing_tested")


@pytest.mark.parametrize(
    "offer,outcome,reason",
    [
        (None, OUTCOME_ROW, "no_offer_row"),
        (OFFER_ROW, None, "no_outcome_row"),
        ({}, OUTCOME_ROW, "no_run_id"),
        (OFFER_ROW, dict(OUTCOME_ROW, run_id="other"), "run_id_mismatch"),
        (OFFER_ROW, dict(OUTCOME_ROW, per_prior="{not json"), "malformed_per_prior"),
        (OFFER_ROW, dict(OUTCOME_ROW, per_prior=[]), "nothing_tested"),
        (OFFER_ROW, dict(OUTCOME_ROW, per_prior=[None, 3, {"kind": "tested"}]), "nothing_tested"),
        (
            dict(OFFER_ROW, turn_started_at=None, decided_at=None),
            dict(OUTCOME_ROW, completed_at="garbage"),
            "no_timestamp",
        ),
        ("not a mapping", OUTCOME_ROW, "no_offer_row"),
    ],
)
def test_curiosity_malformed_returns_none(offer, outcome, reason):
    assert explain_curiosity_choice(offer, outcome) == (None, reason)
    assert curiosity_choice_from_rows(offer, outcome) is None


def test_curiosity_garbage_menu_entries_are_skipped():
    offer = dict(OFFER_ROW, offered=[None, "x", {"rank": "bad", "prior_id": "a"}], stale_offered=None)
    outcome = dict(OUTCOME_ROW, per_prior=[{"kind": "tested", "prior_id": "a"}], realized_nats="nan")
    c = curiosity_choice_from_rows(offer, outcome)
    assert c.chosen[0].rank is None and c.chosen_best_rank is None
    assert c.cost_nats is None and c.alternatives == []


def test_curiosity_negative_duration_is_not_a_cost():
    outcome = dict(OUTCOME_ROW, completed_at="2026-10-09T20:00:00+00:00")
    assert curiosity_choice_from_rows(OFFER_ROW, outcome).cost_seconds is None


def test_curiosity_caps_are_enforced():
    offered = [{"rank": i, "prior_id": f"p{i}", "expected_nats": 0.1} for i in range(1, 201)]
    per_prior = [{"kind": "tested", "prior_id": f"p{i}"} for i in range(1, 81)]
    offer = dict(OFFER_ROW, offered=offered, stale_offered=[],
                 material_ids=[f"crystallization:{i}" for i in range(200)])
    c = curiosity_choice_from_rows(offer, dict(OUTCOME_ROW, per_prior=per_prior))
    assert len(c.chosen) == MAX_CHOICE_LIST
    assert len(c.alternatives) == MAX_CHOICE_LIST
    assert len(c.situation_refs) == MAX_CHOICE_LIST
    assert c.chosen_best_rank == 1


def test_model_refuses_uncapped_lists_and_extra_fields():
    base = curiosity_choice_from_rows(OFFER_ROW, OUTCOME_ROW).model_dump()
    with pytest.raises(Exception):
        PersonalityChoiceV1(**dict(base, alternatives=base["chosen"] * 26))
    with pytest.raises(Exception):
        PersonalityChoiceV1(**dict(base, chosen=[]))
    with pytest.raises(Exception):
        PersonalityChoiceV1(**dict(base, surprise=1))


# ------------------------------------------------------- attention override lane


def test_override_real_row():
    c = voluntary_override_choice_from_row(OVERRIDE_ROW)
    assert c is not None
    assert c.choice_id == "voluntary_attention_override:broadcast-03864cdd93596bdf7d977f19"
    assert c.source_ref == "attention_broadcast:broadcast-03864cdd93596bdf7d977f19"
    assert c.chooser == "goal_biased_formula"
    assert [(o.ref, o.rank) for o in c.chosen] == [("open-loop-7376a3da4050", 2)]
    assert c.chosen[0].score == pytest.approx(0.9443178581600683)
    assert [(o.ref, o.rank) for o in c.alternatives] == [
        ("open-loop-e91fa7e1c61e", 1),
        ("open-loop-08d23ba68a84", 3),
    ]
    assert c.chosen_best_rank == 2
    assert c.cost_effort == pytest.approx(0.944175)
    assert c.cost_seconds is None and c.cost_nats is None
    assert c.situation_refs == [
        "artifact-e0d711ed-d709-443c-887e-c74acad1c1b0",
        "node:substrate.execution",
        "node:substrate.perception",
    ]
    assert c.correlation_id is None


def test_override_choice_id_is_deterministic():
    a = voluntary_override_choice_from_row(OVERRIDE_ROW)
    b = voluntary_override_choice_from_row(json.loads(json.dumps(OVERRIDE_ROW)))
    assert a.model_dump() == b.model_dump()


def test_override_projection_as_text():
    row = dict(OVERRIDE_ROW, projection_json=json.dumps(OVERRIDE_ROW["projection_json"]))
    assert voluntary_override_choice_from_row(row).chosen_best_rank == 2


def test_override_without_open_loops_falls_back_to_the_pair():
    row = copy.deepcopy(OVERRIDE_ROW)
    row["projection_json"]["frame"]["open_loops"] = []
    c = voluntary_override_choice_from_row(row)
    assert c.chosen_best_rank == 2
    assert [(o.ref, o.rank) for o in c.alternatives] == [("open-loop-e91fa7e1c61e", 1)]


def _with_override(vo):
    row = copy.deepcopy(OVERRIDE_ROW)
    row["projection_json"]["frame"]["voluntary_override"] = vo
    return row


@pytest.mark.parametrize(
    "row,reason",
    [
        (None, "no_row"),
        ({}, "no_log_id"),
        ({"log_id": "x", "projection_json": "{bad"}, "malformed_projection"),
        ({"log_id": "x", "projection_json": {}}, "no_frame"),
        (_with_override(None), "no_override"),
        (_with_override("oops"), "malformed_override"),
        (_with_override({"chosen_loop_id": "a"}), "malformed_override"),
        (_with_override({"chosen_loop_id": "a", "beat_loop_id": "a"}), "malformed_override"),
    ],
)
def test_override_malformed_returns_none(row, reason):
    assert explain_voluntary_override_choice(row) == (None, reason)
    assert voluntary_override_choice_from_row(row) is None


def test_override_alternatives_capped():
    row = copy.deepcopy(OVERRIDE_ROW)
    loops = row["projection_json"]["frame"]["open_loops"]
    loops += [{"id": f"open-loop-x{i}", "salience": 0.01, "source_refs": []} for i in range(80)]
    c = voluntary_override_choice_from_row(row)
    assert len(c.alternatives) == MAX_CHOICE_LIST
    assert c.alternatives[0].ref == "open-loop-e91fa7e1c61e"


def test_override_salience_tie_ranks_the_beat_first():
    row = copy.deepcopy(OVERRIDE_ROW)
    loops = row["projection_json"]["frame"]["open_loops"]
    loops[0]["salience"] = loops[1]["salience"] = 0.5
    loops.reverse()  # chosen listed before the beat
    c = voluntary_override_choice_from_row(row)
    assert c.chosen_best_rank == 2
    assert c.alternatives[0].ref == "open-loop-e91fa7e1c61e" and c.alternatives[0].rank == 1
