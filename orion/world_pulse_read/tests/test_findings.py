"""What a read changed: links from stored Stage 2 receipts, beliefs from runs that cite the read."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from orion.world_pulse_read import findings as wf

T0 = datetime(2026, 10, 11, 14, 0, tzinfo=timezone.utc)

# Verbatim from production (world_pulse_read_seed, 2026-10-11): the one accepted reading claim.
LIVE_CLAIM = {
    "quote": "NASA Administrator Jared Isaacman will pilot one of three F-5 fighter jets",
    "receipt": {
        "reason": "accepted", "outcome": "accepted_provisional", "span_end": 2234, "span_start": 2160,
        "decision_id": "reading-aae77003-1145-5385-926f-cb78f3dc4b37",
        "proposal_id": "reading-86d3b2a0-3ae4-52d5-8561-1e1ddd03b0a9",
        "assertion_id": "assertion-f29721db-52ae-5017-8b26-107fceafbd0d",
        "content_sha256": "f758f22c88470dd4cf5f672a6639ff15ecc10ea5f93db990013a038eb5609961",
        "representation": "tool_digest",
    },
    "object_id": "sub-concept-wp-read-c79f0617a2ef6b3c",
    "predicate": "associated_with",
    "subject_id": "sub-concept-wp-read-0568d894ec192184",
    "statement_text": "Per advisory M26-080, NASA Administrator Jared Isaacman pilots one of the three F-5 "
                      "fighter jets of the Freedom 250 flyover team for the MetLife stop.",
}
LIVE_SEED = "digest_item:d7daaa9b-08e5-4db9-89ff-33819093430c:item:cluster:22b63427ceef0ac9:a6250952bd65a37d"


def _link_row(**over):
    row = {"seed_id": LIVE_SEED, "stage2_completed_at": T0, "claim": json.dumps(LIVE_CLAIM),
           "latest_state": "provisional", "projected_at": T0 + timedelta(seconds=1)}
    row.update(over)
    return row


def test_a_live_accepted_claim_is_a_projected_link():
    [f] = wf.link_findings([_link_row()])
    assert f.kind == "link_made" and f.seed_id == LIVE_SEED and f.projected is True
    assert f.finding_id == "link:assertion-f29721db-52ae-5017-8b26-107fceafbd0d"
    assert f.headline.startswith("Link made: Per advisory M26-080")
    assert f.evidence_text == LIVE_CLAIM["quote"]
    assert "reading_text:f758f22c88470dd4cf5f672a6639ff15ecc10ea5f93db990013a038eb5609961#bytes=2160-2234" in f.evidence_refs


def test_a_link_rejected_later_or_never_decided_is_not_reported():
    assert wf.link_findings([_link_row(latest_state="rejected")]) == []
    assert wf.link_findings([_link_row(latest_state=None)]) == []


def test_an_accepted_but_unprojected_link_says_so():
    [f] = wf.link_findings([_link_row(projected_at=None)])
    assert f.projected is False and "not yet in the graph" in f.headline


def _read(**over):
    base = dict(seed_id="reading:84912565-c6f6-586c-96d7-5fa9d83c012f", occurred_at=T0,
                url="https://arxiv.org/abs/2609.10712", request_id="84912565-c6f6-586c-96d7-5fa9d83c012f",
                title="Nemotron IMO", learned="It solves olympiad proofs. More text.")
    base.update(over)
    return wf.ReadRefV1(**base)


MOVE_ROW = {"run_id": "e5eb9cfcf7ea", "completed_at": T0 + timedelta(hours=3),
            "per_prior": json.dumps([
                {"kind": "tested", "before": 0.55, "after": 0.7, "prior_id": "self:proofs_need_search"},
                {"kind": "tested", "before": 0.6, "after": 0.6, "prior_id": "self:unchanged"},
                {"kind": "formed", "before": None, "after": 0.4, "prior_id": "world:new_one"},
            ])}


def test_prior_moves_keep_only_changes():
    moves = wf.prior_moves([MOVE_ROW])
    assert [(m.prior_id, m.before, m.after) for m in moves] == [
        ("self:proofs_need_search", 0.55, 0.7), ("world:new_one", None, 0.4)]


def _texts(why="Test 3: arxiv.org/abs/2609.10712 shows search-free proofs", last_run_id="e5eb9cfcf7ea"):
    return {"self:proofs_need_search": [{"prior_id": "self:proofs_need_search", "claim": "Proofs need search",
                                         "why": why, "last_run_id": last_run_id, "run_id": "older0000001"}],
            "world:new_one": [{"prior_id": "world:new_one", "claim": "A new one", "formed_from": "reading:other",
                               "run_id": "e5eb9cfcf7ea"}]}


def test_a_move_whose_own_text_cites_the_read_is_credited():
    [f] = wf.belief_findings([_read()], wf.prior_moves([MOVE_ROW]), _texts())
    assert f.kind == "belief_moved" and f.prior_id == "self:proofs_need_search"
    assert (f.confidence_before, f.confidence_after) == (0.55, 0.7)
    assert f.headline == "Belief moved (0.55 -> 0.70): Proofs need search"
    assert "cites:why=arxiv.org/abs/2609.10712" in f.evidence_refs


def test_a_move_is_not_credited_without_the_run_own_citation():
    moves = wf.prior_moves([MOVE_ROW])
    # The why on the node was written by a later run: not this run's words.
    assert wf.belief_findings([_read()], moves, _texts(last_run_id="later0000001")) == []
    # Cites something else.
    assert wf.belief_findings([_read()], moves, _texts(why="psql over curiosity_run_outcomes")) == []
    # The read landed after the run finished: it cannot have moved it.
    assert wf.belief_findings([_read(occurred_at=T0 + timedelta(hours=4))], moves, _texts()) == []


def test_a_bare_domain_never_names_a_read():
    read = _read(url="https://www.coreweave.com/", request_id="", seed_id="reading:x")
    assert read.tokens() == []


def test_first_sentences_is_verbatim_and_bounded():
    assert wf.first_sentences("One thing. Two things! Three.", 2) == "One thing. Two things!"
    assert wf.first_sentences("") == ""


class _Conn:
    def __init__(self, links, moves):
        self.links, self.moves = links, moves

    async def fetch(self, sql, *args):
        return self.links if sql == wf.READ_LINKS_SQL else self.moves


def test_findings_for_reads_groups_and_names_an_unreadable_graph():
    read = _read(seed_id=LIVE_SEED)
    ok = asyncio.run(wf.findings_for_reads(_Conn([_link_row()], []), [read], until=T0 + timedelta(days=1)))
    assert [f.kind for f in ok.by_seed[LIVE_SEED]] == ["link_made"] and ok.beliefs_unchecked == ""

    async def unreadable(ids):
        return None

    res = asyncio.run(wf.findings_for_reads(_Conn([], [MOVE_ROW]), [_read()], until=T0 + timedelta(days=1),
                                            prior_lookup=unreadable))
    assert res.beliefs_unchecked == "priors_unreadable"


def test_prior_texts_cypher_parameterises_ids():
    cypher = wf.prior_texts_cypher(['self:a"} DETACH DELETE p //'])
    assert cypher.startswith('CYPHER ids=["self:a\\"} DETACH DELETE p //"] MATCH (p:Prior)')


def test_a_url_names_only_its_own_page_not_a_longer_one():
    read = _read(url="https://www.nasa.gov/maxpower", request_id="", seed_id="reading:x")
    texts = lambda why: {"self:proofs_need_search": [{"prior_id": "self:proofs_need_search", "claim": "c",  # noqa: E731
                                                      "why": why, "last_run_id": "e5eb9cfcf7ea"}]}
    moves = wf.prior_moves([MOVE_ROW])
    assert wf.belief_findings([read], moves, texts("see nasa.gov/maxpower-x for more")) == []
    assert wf.belief_findings([read], moves, texts("see nasa.gov/maxpower/sub")) == []
    assert len(wf.belief_findings([read], moves, texts("per https://nasa.gov/maxpower."))) == 1


def test_two_reads_making_the_same_link_each_show_it():
    other = _link_row(seed_id="reading:other")
    assert sorted(f.seed_id for f in wf.link_findings([_link_row(), other])) == sorted([LIVE_SEED, "reading:other"])


def test_a_link_belongs_to_the_moment_it_was_accepted():
    [f] = wf.link_findings([_link_row(projected_at=T0 + timedelta(days=1))])
    assert f.observed_at == T0 and f.projected is True
