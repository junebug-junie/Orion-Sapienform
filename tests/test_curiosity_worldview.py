"""Orion's own graph, read from Hub -- and the ways that read can lie.

Everything here is about a boundary between two authors. Orion writes this
graph by hand, in Cypher, inside a turn; Hub reads it back and must never
mistake "I could not reach it" for "there is nothing there", never let a
malformed decision open a turn or interrupt Juniper, and never put an
unvalidated string into a query.
"""

from __future__ import annotations

import pytest

from orion.curiosity.worldview import (
    OPEN_PRIORS_CYPHER,
    RECENT_SETTLED_CYPHER,
    Prior,
    WorldviewReader,
    WorldviewUnavailable,
    build_prior,
    build_turn_outcome,
    hops_for_run_cypher,
    outcome_for_run_cypher,
    read_hop_notes,
    read_run_footprint,
    read_snapshot,
    read_turn_outcome,
    rows_from_reply,
    run_footprint_cypher,
    select_priors,
)


class _FakeReader(WorldviewReader):
    """Answers by matching on the query text, the way FalkorDB would by shape."""

    def __init__(self, *, answers=None, raises=False) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        for needle, rows in self.answers.items():
            if needle in cypher:
                return rows
        return []


def _prior_row(pid="p1", claim="a claim", confidence="0.55", status="open", tested=0):
    return {
        "prior_id": pid,
        "claim": claim,
        "confidence": confidence,
        "status": status,
        "times_tested": tested,
        "formed_from": "",
        "last_tested_at": "",
    }


# --- the reply shape is the contract with FalkorDB -------------------------


def test_rows_come_back_keyed_by_column_name() -> None:
    reply = [["claim", "confidence"], [["a", "0.5"], ["b", "0.9"]], ["stats"]]
    assert rows_from_reply(reply) == [
        {"claim": "a", "confidence": "0.5"},
        {"claim": "b", "confidence": "0.9"},
    ]


def test_a_header_of_type_name_pairs_still_resolves_to_the_name() -> None:
    """FalkorDB has shipped both shapes. Taking `h[-1]` handles either."""
    reply = [[[1, "claim"]], [["a"]], []]
    assert rows_from_reply(reply) == [{"claim": "a"}]


@pytest.mark.parametrize("reply", [None, [], ["only-a-header"], "not a list", 7])
def test_an_unrecognised_reply_is_empty_not_an_exception(reply) -> None:
    assert rows_from_reply(reply) == []


def test_a_float_written_as_a_string_still_reads_as_a_number() -> None:
    """`decode_responses=True` returns FalkorDB doubles as decimal STRINGS.

    Confirmed live against this deployment: `RETURN 1.5` comes back as `'1.5'`.
    A prior whose confidence read as a string would sort as maximally uncertain
    forever and never compare correctly against anything.
    """
    prior = build_prior(_prior_row(confidence="0.72"))
    assert prior is not None
    assert prior.confidence == pytest.approx(0.72)


# --- unreadable is not empty ------------------------------------------------


def test_an_unreachable_graph_is_not_reported_as_an_empty_world_view() -> None:
    view = read_snapshot(_FakeReader(raises=True), sample=8, stale_after=3)
    assert view.is_unavailable
    assert view.open_priors == []
    assert "ConnectionError" in (view.unavailable_reason or "")


def test_a_genuinely_empty_graph_is_available_and_empty() -> None:
    view = read_snapshot(_FakeReader(), sample=8, stale_after=3)
    assert not view.is_unavailable
    assert view.open_total == 0


def test_a_row_with_no_claim_is_dropped_rather_than_invented() -> None:
    """Inventing a claim would be the heuristic re-inference this whole arc
    deletes. The drop is counted so schema drift is visible, not silent."""
    assert build_prior(_prior_row(claim="")) is None
    assert build_prior(_prior_row(pid="")) is None
    offered, stale, dropped = select_priors(
        [_prior_row(), _prior_row(pid="p2", claim="")], sample=8, stale_after=3
    )
    assert dropped == 1
    assert [p.prior_id for p in offered] == ["p1"]
    assert stale == []


# --- uncertainty orders the presentation, and says so ----------------------


def test_the_most_uncertain_prior_comes_first() -> None:
    rows = [
        _prior_row("sure", confidence="0.95"),
        _prior_row("unsure", confidence="0.5"),
        _prior_row("middling", confidence="0.7"),
    ]
    offered, _, _ = select_priors(rows, sample=8, stale_after=3)
    assert [p.prior_id for p in offered] == ["unsure", "middling", "sure"]


def test_a_prior_with_no_confidence_sorts_as_maximally_uncertain() -> None:
    """The honest reading of "Orion never said how sure it was"."""
    assert Prior("p", "c", None, "open", 0).uncertainty == 0.0


def test_ties_break_on_least_tested_then_stably_on_id() -> None:
    rows = [
        _prior_row("b", confidence="0.5", tested=2),
        _prior_row("a", confidence="0.5", tested=2),
        _prior_row("c", confidence="0.5", tested=0),
    ]
    offered, _, _ = select_priors(rows, sample=8, stale_after=9)
    assert [p.prior_id for p in offered] == ["c", "a", "b"]


def test_a_re_litigated_prior_leaves_the_main_list_but_is_still_offered() -> None:
    """Testing one claim forever without resolving it is the "same shit over
    and over" failure in a new costume. It moves to its own bucket -- NOT
    hidden, because Hub never writes and only Orion can close it."""
    rows = [_prior_row("fresh", tested=0), _prior_row("stuck", tested=4)]
    offered, stale, _ = select_priors(rows, sample=8, stale_after=3)
    assert [p.prior_id for p in offered] == ["fresh"]
    assert [p.prior_id for p in stale] == ["stuck"]


def test_a_stale_prior_is_never_counted_in_both_buckets() -> None:
    rows = [_prior_row(f"p{i}", tested=5) for i in range(3)]
    offered, stale, _ = select_priors(rows, sample=8, stale_after=3)
    assert not ({p.prior_id for p in offered} & {p.prior_id for p in stale})


def test_a_zero_stale_threshold_disables_the_split() -> None:
    rows = [_prior_row("p", tested=99)]
    offered, stale, _ = select_priors(rows, sample=8, stale_after=0)
    assert [p.prior_id for p in offered] == ["p"] and stale == []


def test_the_sample_size_caps_what_is_offered() -> None:
    rows = [_prior_row(f"p{i}", confidence=str(0.5 + i / 100)) for i in range(20)]
    offered, _, _ = select_priors(rows, sample=3, stale_after=3)
    assert len(offered) == 3


# --- the decision that crosses the turn boundary ---------------------------


def test_absence_of_a_turn_outcome_means_no_continuation_and_no_outreach() -> None:
    """A turn that ran out of time or was refused leaves nothing behind, and
    that must read as silence rather than as a default guess."""
    assert read_turn_outcome(_FakeReader(), "abc123") is None


def test_a_malformed_boolean_fails_closed() -> None:
    """Orion writes this Cypher by hand. Anything unrecognised must NOT open a
    line of enquiry or interrupt Juniper."""
    outcome = build_turn_outcome(
        {"run_id": "r", "continue_line": "maybe", "reach_out": "sometimes"}
    )
    assert outcome is not None
    assert outcome.continue_line is False and outcome.reach_out is False


def test_a_quoted_boolean_is_still_honoured() -> None:
    """The single likeliest hand-written typo, and it means what it says."""
    outcome = build_turn_outcome({"run_id": "r", "continue_line": "true"})
    assert outcome is not None and outcome.continue_line is True


def test_an_outcome_with_no_run_id_is_not_an_outcome() -> None:
    assert build_turn_outcome({"continue_line": True}) is None


def test_this_runs_outcome_is_read_by_id_not_as_the_newest() -> None:
    """Reading "newest" would attribute a PREVIOUS run's decision to this one
    every time a turn died before writing its own."""
    assert "t.run_id = 'deadbeef'" in outcome_for_run_cypher("deadbeef")
    assert "ORDER BY t.written_at DESC" in outcome_for_run_cypher("deadbeef")


# --- no injection surface, rather than a sanitiser to trust ----------------


@pytest.mark.parametrize(
    "builder", [run_footprint_cypher, outcome_for_run_cypher, hops_for_run_cypher]
)
@pytest.mark.parametrize(
    "bad", ["'; MATCH (n) DETACH DELETE n //", "abc-123", "", "ABC123", "x" * 64, None]
)
def test_a_non_hex_run_id_never_reaches_a_query_string(builder, bad) -> None:
    with pytest.raises(ValueError):
        builder(bad)


def test_a_hex_run_id_is_accepted() -> None:
    assert "n.run_id = 'a1b2c3d4e5f6'" in run_footprint_cypher("a1b2c3d4e5f6")


def test_a_bad_run_id_degrades_to_a_safe_default_rather_than_raising() -> None:
    """The loop must survive a junk id in Redis, not crash the tick."""
    reader = _FakeReader()
    assert read_turn_outcome(reader, "not-hex") is None
    assert read_run_footprint(reader, "not-hex") == {}
    assert read_hop_notes(reader, "not-hex") == []
    assert reader.queries == []


# --- the run's own evidence -------------------------------------------------


def test_the_footprint_counts_what_orion_wrote_this_run() -> None:
    reader = _FakeReader(answers={"n.run_id": [
        {"label": "Prior", "n": 2}, {"label": "Hop", "n": 5},
    ]})
    assert read_run_footprint(reader, "abc123") == {"Prior": 2, "Hop": 5}


def test_a_zero_count_is_not_reported_as_a_write() -> None:
    reader = _FakeReader(answers={"n.run_id": [{"label": "Prior", "n": 0}]})
    assert read_run_footprint(reader, "abc123") == {}


def test_hop_notes_come_back_in_order_and_blank_ones_are_dropped() -> None:
    reader = _FakeReader(answers={"h:Hop": [
        {"n": 1, "note": "first"}, {"n": 2, "note": "   "}, {"n": 3, "note": "third"},
    ]})
    assert read_hop_notes(reader, "abc123") == [(1, "first"), (3, "third")]


# --- what replaced the dead journal hint ------------------------------------


def test_recently_settled_reads_closed_priors_from_orions_own_graph() -> None:
    reader = _FakeReader(answers={"p.status <> 'open'": [
        {"claim": "the vision tier is on demand", "status": "refuted"},
        {"claim": "", "status": "supported"},
    ]})
    view = read_snapshot(reader, sample=8, stale_after=3)
    assert view.recently_settled == [("the vision tier is on demand", "refuted")]


def test_open_and_settled_reads_are_actually_different_queries() -> None:
    assert "p.status = 'open'" in OPEN_PRIORS_CYPHER
    assert "p.status <> 'open'" in RECENT_SETTLED_CYPHER


# --- hub never writes -------------------------------------------------------


def test_every_hub_read_goes_out_as_a_read_only_command() -> None:
    """Hub connects as FalkorDB's unrestricted `default` user, so this is the
    only thing stopping a bug here corrupting Orion's own space."""
    sent: list[tuple] = []

    class _Client:
        def execute_command(self, *argv):
            sent.append(argv)
            return [[], [], []]

    reader = WorldviewReader(host="x", port=1, graph_name="g", client=_Client())
    read_snapshot(reader, sample=8, stale_after=3)
    read_turn_outcome(reader, "abc123")
    read_run_footprint(reader, "abc123")
    read_hop_notes(reader, "abc123")
    assert sent, "expected the reader to issue queries"
    assert all(argv[0] == "GRAPH.RO_QUERY" for argv in sent)


def test_a_redis_error_becomes_a_typed_failure_not_a_bare_exception() -> None:
    class _Client:
        def execute_command(self, *argv):
            raise OSError("connection reset")

    reader = WorldviewReader(host="x", port=1, graph_name="g", client=_Client())
    with pytest.raises(WorldviewUnavailable):
        reader.query("MATCH (n) RETURN n")
