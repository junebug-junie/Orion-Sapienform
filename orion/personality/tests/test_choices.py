"""Curiosity rows -> PersonalityChoiceV1 (rev 4), and the PersonalityChoiceV1 contract itself.

OFFER_ROW/OUTCOME_ROW: REAL stored rows (live Postgres 2026-10-16, unedited), curiosity run
ee31d36e8bfe. Orion tested two priors: rank 1 of the ordered list, and one from the separate
"tested repeatedly" (stale) section.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from orion.personality.choices import (
    curiosity_choices_from_rows,
    explain_curiosity_choices,
    readable_prior,
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


def test_curiosity_real_rows_one_row_per_tested_prior():
    out = curiosity_choices_from_rows(OFFER_ROW, OUTCOME_ROW)
    assert [c.choice_id for c in out] == [
        "curiosity:ee31d36e8bfe:self:induction_ledger_dangling_20261008",
        "curiosity:ee31d36e8bfe:self:turn_start_context_is_stale_snapshot_20260923",
    ]
    a, b = out
    assert a.lane == "curiosity" and a.act_id == "curiosity_run:ee31d36e8bfe"
    assert a.chosen.text == "induction ledger dangling" and a.chosen.market_rank == 1 and a.chosen_market_rank == 1
    # The stale-section pick was shown unranked.
    assert b.chosen.market_rank is None and b.chosen_market_rank is None
    assert a.picks_in_decision == b.picks_in_decision == 2
    # 8 ordered + 8 stale shown, 2 tested: the other 14 are what Orion turned down.
    assert len(a.alternatives) == 14 and a.alternatives == b.alternatives
    tested = {a.chosen.key, b.chosen.key}
    assert not tested & {o.key for o in a.alternatives}
    assert [o.market_rank for o in a.alternatives[:7]] == [2, 3, 4, 5, 6, 7, 8]
    assert all(o.market_rank is None for o in a.alternatives[7:])
    assert a.occurred_at.isoformat() == "2026-10-09T21:05:42.692508+00:00"
    assert a.situation == {"curiosity_arm": "uncertainty_order", "turn_ok": True}
    assert not a.declined_all and not a.care_promoted


def test_curiosity_choices_are_deterministic():
    a = curiosity_choices_from_rows(OFFER_ROW, OUTCOME_ROW)
    b = curiosity_choices_from_rows(copy.deepcopy(OFFER_ROW), copy.deepcopy(OUTCOME_ROW))
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


def test_curiosity_accepts_jsonb_as_text():
    offer = dict(OFFER_ROW, offered=json.dumps(OFFER_ROW["offered"]))
    outcome = dict(OUTCOME_ROW, per_prior=json.dumps(OUTCOME_ROW["per_prior"]))
    assert curiosity_choices_from_rows(offer, outcome)[0].chosen_market_rank == 1


def test_curiosity_deviation_and_off_menu():
    outcome = copy.deepcopy(OUTCOME_ROW)
    outcome["per_prior"] = [
        {"kind": "tested", "prior_id": "self:presence_ring_is_three_kinds_20260930"},  # rank 6
        {"kind": "tested", "prior_id": "self:not_on_any_menu"},
    ]
    six, off = curiosity_choices_from_rows(OFFER_ROW, outcome)
    assert six.chosen_market_rank == 6 and off.chosen_market_rank is None
    assert off.chosen.text == "not on any menu"
    assert len(six.alternatives) == 15  # everything shown except the one tested from the menu


def test_curiosity_formed_and_moved_untested_are_not_chosen():
    outcome = copy.deepcopy(OUTCOME_ROW)
    outcome["per_prior"] = [
        {"kind": "formed", "prior_id": "self:new_one"},
        {"kind": "moved_untested", "prior_id": "self:induction_ledger_dangling_20261008"},
    ]
    assert explain_curiosity_choices(OFFER_ROW, outcome) == ([], "formed_only")
    outcome["per_prior"] = outcome["per_prior"][1:]
    assert explain_curiosity_choices(OFFER_ROW, outcome) == ([], "nothing_tested")


def test_curiosity_every_option_tested_is_no_choice():
    offer = dict(OFFER_ROW, offered=OFFER_ROW["offered"][:1], stale_offered=[])
    outcome = dict(OUTCOME_ROW, per_prior=[{"kind": "tested", "prior_id": "self:induction_ledger_dangling_20261008"}])
    assert explain_curiosity_choices(offer, outcome) == ([], "no_alternative")


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
        (dict(OFFER_ROW, turn_started_at=None, decided_at=None), dict(OUTCOME_ROW, completed_at="garbage"),
         "no_timestamp"),
        ("not a mapping", OUTCOME_ROW, "no_offer_row"),
        (dict(OFFER_ROW, arm="self_inquiry", offered=[], stale_offered=[]), OUTCOME_ROW, "self_inquiry_no_offer"),
    ],
)
def test_curiosity_malformed_yields_nothing(offer, outcome, reason):
    assert explain_curiosity_choices(offer, outcome) == ([], reason)
    assert curiosity_choices_from_rows(offer, outcome) == []


def test_curiosity_coverage_pick_is_unranked_and_garbage_skipped():
    offer = dict(OFFER_ROW, offered=[None, "x", {"rank": "bad", "prior_id": "a_b"},
                                     {"rank": 2, "slot": "coverage", "prior_id": "c_d"},
                                     {"rank": 1, "prior_id": "e_f"}], stale_offered=None)
    outcome = dict(OUTCOME_ROW, per_prior=[{"kind": "tested", "prior_id": "c_d"}])
    (c,) = curiosity_choices_from_rows(offer, outcome)
    assert c.chosen_market_rank is None and [(o.key, o.market_rank) for o in c.alternatives] == [("a_b", None), ("e_f", 1)]


def test_readable_prior():
    assert readable_prior("curiosity_prior:self:reading_link_can_cross_region_boundary_20261015") == \
        "reading link can cross region boundary"
    assert readable_prior("self:top8_20261007") == "top8"
    assert readable_prior("keeps_2026_date_mid") == "keeps 2026 date mid"
    assert readable_prior(None) == ""


# --- the contract -------------------------------------------------------------------------------

T = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def opt(key, rank=None, text=None):
    return {"key": key, "text": text or f"option {key}", "market_rank": rank}


def test_contract_minimal_prime_choice_round_trips():
    c = PersonalityChoiceV1(choice_id="prime:t1", lane="prime", occurred_at=T, chosen=opt("a", 2),
                            alternatives=[opt("b", 1), opt("c", 3)], chosen_market_rank=2,
                            situation={"arousal": "engaged"}, act_id="prime_thought:t1", correlation_id="corr-1")
    again = PersonalityChoiceV1.model_validate_json(c.model_dump_json())
    assert again == c and again.schema_version == "personality.choice.v1" and again.picks_in_decision == 1


def test_contract_declined_all():
    c = PersonalityChoiceV1(choice_id="outreach:o1", lane="outreach", occurred_at=T, declined_all=True,
                            alternatives=[opt("t1", 1), opt("t2", 2)])
    assert c.chosen is None and c.chosen_market_rank is None
    with pytest.raises(ValueError):
        PersonalityChoiceV1(choice_id="x", lane="outreach", occurred_at=T, declined_all=True, chosen=opt("a"),
                            alternatives=[opt("b")])
    with pytest.raises(ValueError):  # declining nothing is not a choice
        PersonalityChoiceV1(choice_id="x", lane="outreach", occurred_at=T, declined_all=True, alternatives=[])


@pytest.mark.parametrize("kw", [
    dict(alternatives=[]),  # no real alternative
    dict(chosen=None),  # a pick must name what was picked
    dict(chosen_market_rank=1),  # must equal chosen.market_rank (2)
    dict(alternatives=[opt("a", 1)]),  # duplicate key
    dict(lane="reverie"),  # prose lanes are retired
    dict(occurred_at=datetime(2026, 11, 13, 12, 0)),  # naive time
    dict(picks_in_decision=0),
    dict(situation={f"k{i}": i for i in range(17)}),
    dict(surprise=1),  # extra field
    dict(alternatives=[opt(f"x{i}") for i in range(MAX_CHOICE_LIST + 1)]),
])
def test_contract_refuses(kw):
    base = dict(choice_id="prime:t1", lane="prime", occurred_at=T, chosen=opt("a", 2),
                alternatives=[opt("b", 1)], chosen_market_rank=2)
    with pytest.raises(ValueError):
        PersonalityChoiceV1(**{**base, **kw})


def test_option_refuses_bad_rank_and_empty_text():
    with pytest.raises(ValueError):
        ChoiceOptionV1(key="a", text="x", market_rank=0)
    with pytest.raises(ValueError):
        ChoiceOptionV1(key="a", text="")


def test_contract_derives_the_top_level_rank_from_chosen():
    c = PersonalityChoiceV1(choice_id="prime:t1", lane="prime", occurred_at=T, chosen=opt("a", 2),
                            alternatives=[opt("b", 1)])
    assert c.chosen_market_rank == 2


@pytest.mark.parametrize("situation", [
    {"nested": {"a": 1}}, {"list": [1, 2]}, {"long": "x" * 201}, {"k" * 65: 1},
])
def test_contract_situation_must_stay_small_and_flat(situation):
    with pytest.raises(ValueError):
        PersonalityChoiceV1(choice_id="prime:t1", lane="prime", occurred_at=T, chosen=opt("a", 2),
                            alternatives=[opt("b", 1)], situation=situation)
    ok = PersonalityChoiceV1(choice_id="prime:t1", lane="prime", occurred_at=T, chosen=opt("a", 2),
                             alternatives=[opt("b", 1)], situation={"arousal": "engaged", "heat": 0.4, "company": True,
                                                                    "binding": None})
    assert ok.situation["heat"] == 0.4


def test_contract_choice_id_lives_in_its_lane_namespace():
    with pytest.raises(ValueError):
        PersonalityChoiceV1(choice_id="curiosity:r1:p", lane="prime", occurred_at=T, chosen=opt("a"),
                            alternatives=[opt("b")])
    with pytest.raises(ValueError):  # declining all is one decision
        PersonalityChoiceV1(choice_id="outreach:o1", lane="outreach", occurred_at=T, declined_all=True,
                            alternatives=[opt("b")], picks_in_decision=2)
