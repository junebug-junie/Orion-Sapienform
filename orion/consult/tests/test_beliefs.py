from __future__ import annotations

from orion.consult.beliefs import (
    MAX_BELIEF_OFFERS,
    consult_evidence_cypher,
    PRIOR_RUNS_CYPHER,
    consult_outcomes_section,
    finding_id_for,
    recorded_evidence,
    source_prior_ids,
)
from orion.curiosity.worldview import Prior

PRIOR = Prior(prior_id="p-lease", claim="The journal lease is never released.", confidence=0.6,
              status="open", times_tested=2)


def test_one_fixed_finding_id_per_outcome_kind():
    assert finding_id_for("c1", "answered") == "consult:c1:answer"
    assert finding_id_for("c1", "agreed_proposal") == "consult:c1:decision"
    assert finding_id_for("c1", "disagree") == "consult:c1:decision"


def test_the_read_back_lists_only_well_formed_consult_finding_ids():
    q = consult_evidence_cypher(["consult:9f138a849520:answer", "consult:abc123:decision",
                                 "consult:x' OR 1=1 //:answer", "consult:abc123:other", "prior:p1"])
    assert "IN ['consult:9f138a849520:answer', 'consult:abc123:decision']" in q
    assert "OR 1=1" not in q and "other" not in q
    assert consult_evidence_cypher(["consult:NOT-HEX:answer"]) is None
    assert "last_run_id" in PRIOR_RUNS_CYPHER


def test_source_priors_come_from_the_source():
    runs = [{"prior_id": "a", "run_id": "r1"}, {"prior_id": "b", "run_id": "r0", "last_run_id": "r1"},
            {"prior_id": "c", "run_id": "r2"}, {"prior_id": "a", "run_id": "r1"}]
    assert source_prior_ids({"source_kind": "curiosity_run", "source_ref": "curiosity_run:r1"}, runs) == ["a", "b"]
    assert source_prior_ids({"source_kind": "stuck_prior", "source_ref": "prior:p9"}, runs) == ["p9"]
    assert source_prior_ids({"source_kind": "self_monitor", "source_ref": "self_finding:x"}, runs) == []


def test_only_the_expected_finding_joined_by_an_allowed_edge_counts():
    consults = [{"consult_id": "c1", "ending": "answered"}, {"consult_id": "c2", "ending": "agreed_proposal"},
                {"consult_id": "c3", "ending": "disagree"}, {"consult_id": "c4", "ending": "agreed_proposal"}]
    rows = [
        {"finding_id": "consult:c1:answer", "prior_id": "p1", "edge": "SUPPORTS", "run_id": "r7"},
        {"finding_id": "consult:c1:answer", "prior_id": "p2", "edge": "ABOUT"},
        {"finding_id": "consult:c1:answer", "prior_id": "p1", "edge": "ABOUT"},
        # wrong kind for c2: a decision written as an answer is not the decision finding
        {"finding_id": "consult:c2:answer", "prior_id": "p3", "edge": "ABOUT"},
        {"finding_id": "consult:c3:decision", "prior_id": "", "edge": "ABOUT"},
        # Juniper's answer joined as proof: not recorded, reported
        {"finding_id": "consult:c4:decision", "prior_id": "p4", "edge": "SUPPORTS"},
        {"finding_id": "consult:other:decision", "prior_id": "p4", "edge": "ABOUT"},
    ]
    found, wrong = recorded_evidence(consults, rows)
    assert found == {"c1": ("consult:c1:answer", ["p1", "p2"], "r7")}
    assert wrong == {"c4": ["SUPPORTS"]}


def test_decision_section_says_not_proof_and_does_not_move_confidence():
    text = "\n".join(consult_outcomes_section([{
        "consult_id": "c2", "ending": "agreed_proposal", "decision": "no_because", "decision_reason": "too costly",
        "question": "fix the lease?", "belief_priors": [PRIOR],
    }]))
    assert "She said no, because: too costly" in text
    assert "NOT proof the claim is true" in text and "Do not move the prior's confidence" in text
    assert 'MERGE (f:Finding {finding_id: "consult:c2:decision"})' in text
    assert "[e:ABOUT]" in text and "SUPPORTS" not in text
    assert "prior_id: p-lease" in text, "the full prior id is printed so the MATCH can bind"
    assert f"{MAX_BELIEF_OFFERS} times at most" in text


def test_answered_section_asks_for_a_test_and_a_supporting_edge():
    text = "\n".join(consult_outcomes_section([{
        "consult_id": "c1", "ending": "answered", "answer": "It is by design: x.py:3", "question": "why?",
        "belief_priors": [],
    }]))
    assert "consult:c1:answer" in text and "times_tested + 1" in text and "[e:SUPPORTS]" in text
    assert "No belief is named by its source" in text


def test_unreadable_beliefs_are_said_and_nothing_claimed_is_empty():
    text = "\n".join(consult_outcomes_section([{"consult_id": "c1", "ending": "answered", "answer": "a",
                                               "belief_priors": None}]))
    assert "could not be read this time" in text
    assert consult_outcomes_section([]) == []


def test_a_repeat_offer_says_not_to_test_again():
    row = {"consult_id": "c1", "ending": "answered", "answer": "a", "belief_priors": [], "belief_offers": 2}
    assert "do not test the prior again" in "\n".join(consult_outcomes_section([row]))
    assert "do not test the prior again" not in "\n".join(consult_outcomes_section([{**row, "belief_offers": 1}]))
