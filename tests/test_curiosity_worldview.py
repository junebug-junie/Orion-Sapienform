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
    CLOSED_STATUSES,
    COUNTS_CYPHER,
    LIVE_PRIORS_CYPHER,
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
    run_edge_footprint_cypher,
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



def _cypher_admits_status(cypher: str, status) -> bool:
    """Apply a prior-status WHERE clause the way FalkorDB would.

    `_FakeReader` matches on query SHAPE and returns canned rows, which means a
    test using it passes whether or not the WHERE clause is right -- and the
    WHERE clause is exactly where the 2026-08-27 bug lived. Caught by mutating
    the cypher back to its pre-fix form and watching the incident replay stay
    green. This reads the predicate off the real query text instead.
    """
    import re

    where = cypher.split("WHERE", 1)[1] if "WHERE" in cypher else ""
    if not where:
        return True
    listed = re.search(r"p\.status IN \[([^\]]*)\]", where)
    if listed:
        names = {v.strip().strip("'\"") for v in listed.group(1).split(",")}
        if "NOT p.status IN" in where:
            if status in (None, "") and "p.status IS NULL" in where:
                return True
            return status not in names
        return status in names
    equals = re.search(r"p\.status = '([^']*)'", where)
    if equals:
        return status == equals.group(1)
    return True


class _StatusAwareReader(_FakeReader):
    """A reader that actually filters, so the WHERE clause is under test."""

    def __init__(self, prior_rows) -> None:
        super().__init__()
        self._prior_rows = list(prior_rows)

    def query(self, cypher: str):
        self.queries.append(cypher)
        if "RETURN p.prior_id AS prior_id" in cypher:
            return [
                r for r in self._prior_rows
                if _cypher_admits_status(cypher, r.get("status"))
            ]
        if "AS live_total" in cypher:
            live = [
                r for r in self._prior_rows
                if r.get("status") not in CLOSED_STATUSES
            ]
            return [{
                "live_total": len(live),
                "closed_total": len(self._prior_rows) - len(live),
            }]
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
    assert view.live_priors == []
    assert "ConnectionError" in (view.unavailable_reason or "")


def test_a_genuinely_empty_graph_is_available_and_empty() -> None:
    view = read_snapshot(_FakeReader(), sample=8, stale_after=3)
    assert not view.is_unavailable
    assert view.live_total == 0


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


def test_least_tested_still_wins_over_the_rotation() -> None:
    """Rotation is the LAST term, never a reordering of the real signal."""
    rows = [
        _prior_row("b", confidence="0.5", tested=2),
        _prior_row("a", confidence="0.5", tested=2),
        _prior_row("c", confidence="0.5", tested=0),
    ]
    for seed in ("", "run1", "run2", "aaaa", "zzzz"):
        offered, _, _ = select_priors(
            rows, sample=8, stale_after=9, rotate_seed=seed
        )
        assert offered[0].prior_id == "c", seed
        assert {p.prior_id for p in offered[1:]} == {"a", "b"}


def test_the_most_uncertain_prior_leads_whatever_the_seed_is() -> None:
    rows = [
        _prior_row("confident", confidence="0.95"),
        _prior_row("unsure", confidence="0.5"),
    ]
    for seed in ("", "run1", "run2", "aaaa", "zzzz"):
        offered, _, _ = select_priors(
            rows, sample=8, stale_after=9, rotate_seed=seed
        )
        assert [p.prior_id for p in offered] == ["unsure", "confident"], seed


def test_one_run_is_reproducible_but_the_window_moves_between_runs() -> None:
    """The whole point: with the pool no longer draining, a fixed tiebreak
    shows the same lexicographically-lowest `sample` priors every run and the
    rest are never presented, never tested, and so never retirable."""
    rows = [_prior_row(f"p{i:02d}", confidence="0.55") for i in range(40)]

    first = [p.prior_id for p in select_priors(
        rows, sample=8, stale_after=9, rotate_seed="run_aaa")[0]]
    again = [p.prior_id for p in select_priors(
        rows, sample=8, stale_after=9, rotate_seed="run_aaa")[0]]
    other = [p.prior_id for p in select_priors(
        rows, sample=8, stale_after=9, rotate_seed="run_bbb")[0]]

    assert first == again, "one run must build the same prompt twice"
    assert first != other, "a later run must be able to see different priors"
    assert first != sorted(rows[i]["prior_id"] for i in range(8)), (
        "a stable id tiebreak would pin exactly the 8 lowest ids forever"
    )


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
    assert read_run_footprint(reader, "not-hex") is None
    assert read_hop_notes(reader, "not-hex") == []
    assert reader.queries == []


def test_an_unreadable_footprint_is_none_and_no_writes_is_empty() -> None:
    """Collapsing these would put "wrote nothing to its own graph" in the
    journal for a run whose graph was simply unreachable."""
    assert read_run_footprint(_FakeReader(raises=True), "abc123") is None
    assert read_run_footprint(_FakeReader(), "abc123") == {}


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
    # Anchored on the settled query's ORDER BY, which no other query has:
    # "p.status IN ['refuted'" is a substring of LIVE_PRIORS_CYPHER's own
    # `NOT p.status IN [...]` and of COUNTS_CYPHER, so it answered all three.
    reader = _FakeReader(answers={"ORDER BY p.last_tested_at": [
        {"claim": "the vision tier is on demand", "status": "refuted"},
        {"claim": "", "status": "supported"},
    ]})
    view = read_snapshot(reader, sample=8, stale_after=3)
    assert view.recently_settled == [("the vision tier is on demand", "refuted")]


def test_live_and_settled_reads_are_actually_different_queries() -> None:
    assert "NOT p.status IN" in LIVE_PRIORS_CYPHER
    assert "NOT p.status IN" not in RECENT_SETTLED_CYPHER
    assert "p.status IN ['refuted', 'retired_unresolvable']" in RECENT_SETTLED_CYPHER


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


# --- the prompt must not print a header with nothing under it --------------


def test_a_view_with_only_stale_priors_does_not_print_an_empty_heading() -> None:
    from orion.curiosity.kickoff_prompt import _priors_section
    from orion.curiosity.worldview import WorldviewSnapshot

    view = WorldviewSnapshot(
        live_priors=[],
        stale_priors=[Prior("p", "a stuck claim", 0.5, "open", 5)],
        live_total=1,
    )
    text = "\n".join(_priors_section(view, stale_after=3))
    assert "WHAT YOU ARE STILL UNSURE OF" not in text
    assert "a stuck claim" in text
    assert "retired_unresolvable" in text


# --- review findings: the prompt must not name what this run cannot reach ---


def _prompt(**over):
    from orion.curiosity.kickoff_prompt import build_kickoff_prompt
    from orion.curiosity.study_material import assemble_study_material
    from datetime import datetime, timezone
    from orion.curiosity.worldview import WorldviewSnapshot

    material = assemble_study_material(
        now=datetime(2026, 8, 26, tzinfo=timezone.utc),
        approved_counts=[{"kind": "semantic", "n": 5}],
        approved_rows=[{
            "crystallization_id": "c1", "kind": "semantic", "subject": "a thought",
            "summary": "", "salience": 0.5, "created_at": None,
        }],
        relation_counts=[], relation_rows=[], relation_resolvable=0,
    )
    kwargs = dict(view=WorldviewSnapshot(), run_id="a1b2c3d4e5f6")
    kwargs.update(over)
    return build_kickoff_prompt(material, **kwargs)


def test_with_no_graph_the_prompt_offers_no_redis_cli_and_no_graph_env_vars() -> None:
    """Handing Orion commands whose env vars are unset is how a turn ends up
    reporting a tooling failure as a finding."""
    text = _prompt(graph_enabled=False)
    assert "redis-cli" not in text
    assert "ORION_CURIOSITY_GRAPH" not in text
    assert "GRAPH.QUERY" not in text
    # The credential-free HTTP door to the Atlas still works, so it stays.
    assert "/api/substrate/concepts/summary" in text


def test_an_unreadable_graph_is_never_invited_to_be_written_to() -> None:
    """The write sections are dropped in this state, so an invitation to
    GRAPH.QUERY would produce nodes with no run_id that nothing can see."""
    from orion.curiosity.worldview import WorldviewSnapshot

    text = _prompt(view=WorldviewSnapshot(unavailable_reason="ConnectionError"))
    assert "GRAPH.QUERY" not in text
    assert "COULD NOT BE READ" in text
    # Reading the Atlas is a different credential path and still works.
    assert "GRAPH.RO_QUERY" in text


def test_a_readable_graph_offers_both_halves() -> None:
    text = _prompt()
    assert "GRAPH.RO_QUERY" in text and "GRAPH.QUERY" in text
    assert "WRITING TO YOUR OWN GRAPH" in text


def test_unreadable_priors_are_never_reported_as_none_outstanding() -> None:
    """The counts query saw rows that `build_prior` could not read. Saying
    "none outstanding" would tell Orion the opposite of the truth."""
    from orion.curiosity.kickoff_prompt import _priors_section
    from orion.curiosity.worldview import WorldviewSnapshot

    text = "\n".join(
        _priors_section(
            WorldviewSnapshot(live_priors=[], stale_priors=[], live_total=4),
            stale_after=3,
        )
    )
    assert "NO PRIORS STILL IN PLAY" not in text
    assert "COULD NOT BE READ BACK" in text
    assert "4 LIVE PRIORS" in text


# ---------------------------------------------------------------- the clock


def test_the_prompt_states_the_budget_by_pointing_at_the_enforcer() -> None:
    """The real budget is HARNESS_FCC_TIMEOUT_SEC, in the harness-governor's
    env, which Hub cannot read. Any literal minute count in the prompt would be
    a second copy free to drift from the wall that actually kills the turn, so
    the prompt must name the env vars the motor stamps and nothing else."""
    text = _prompt()
    assert "ORION_TURN_BUDGET_SEC" in text
    assert "ORION_TURN_DEADLINE_EPOCH" in text
    assert "date +%s" in text


# Deliberately permissive. An earlier, narrower version of this pattern was
# mutation-tested and let through "a ~26-minute budget", "roughly 26 min",
# "900 sec" and "26m" -- and "a ~26-minute budget" is the exact phrasing used in
# this feature's own source comments, so it is the one most likely to be pasted
# into the prompt by someone being helpful.
_DURATION_RE = r"\b\d+\s*[-\s]?(seconds?|secs?|minutes?|mins?|hours?|hrs?|[msh])\b"


@pytest.mark.parametrize("graph_enabled", [True, False])
def test_the_budget_is_never_stated_as_a_hardcoded_duration(graph_enabled) -> None:
    """Regression guard for the drift this design exists to prevent: if anyone
    ever writes the number in, this fails.

    Scans the WHOLE assembled prompt, not just `_budget_section`. A duration
    written into the header or the hops section drifts exactly as badly, and
    scoping the guard to one function is how a gate ends up inert on the file
    it was written for.
    """
    import re

    text = _prompt(graph_enabled=graph_enabled)
    hit = re.search(_DURATION_RE, text)
    assert hit is None, f"hardcoded duration in prompt: {hit.group(0)!r}"


def test_the_continuation_note_is_only_named_when_it_can_be_written() -> None:
    """`continue_note` lives in a :TurnOutcome node, and `_outcome_section` is
    gated on `writable`. Naming the note in a state where no such node can be
    created promises a mechanism that is not there -- the exact failure the
    three-state split exists to prevent."""
    from orion.curiosity.worldview import WorldviewSnapshot

    readable = _prompt()
    assert "that is what the continuation note is for" in readable
    assert "continue_note" in readable

    for text in (
        _prompt(graph_enabled=False),
        _prompt(view=WorldviewSnapshot(unavailable_reason="ConnectionError")),
    ):
        assert "continuation note" not in text
        assert "continue_note" not in text
        # The thread still has somewhere to go -- Orion's own prose.
        assert "say where you would pick it up" in text


def test_a_sixth_hop_points_somewhere_real_in_every_state() -> None:
    """"Leave yourself a note (below)" pointed at a section that is not
    rendered when there is nothing to write to."""
    assert "a note (below)" in _prompt()
    assert "(below)" not in _prompt(graph_enabled=False)


def test_the_per_step_stall_wall_is_disclosed_next_to_the_turn_clock() -> None:
    """The whole-turn deadline is not the only wall. A single step that goes
    quiet past ORION_TURN_STEP_STALL_SEC kills the turn on its own while the
    outer clock still reads generous -- so showing only the outer number
    encourages the unbounded query that trips the inner one."""
    text = _prompt()
    assert "ORION_TURN_STEP_STALL_SEC" in text
    assert "per step, not per turn" in text
    assert "Keep individual queries bounded" in text


def test_the_clock_commands_survive_the_variables_being_unset() -> None:
    """Bash expands before it evaluates, so `$(( $UNSET - $(date +%s) ))`
    prints a confident negative and exits 0. The prompt must test for
    emptiness, and must say what absence looks like."""
    text = _prompt()
    assert 'test -n "$ORION_TURN_DEADLINE_EPOCH"' in text
    assert "no clock" in text
    assert "do not infer a deadline from a negative" in text


def test_the_clock_is_stated_even_with_no_graph_to_write_to() -> None:
    """A prose-only run is killed by the same wall. Gating the clock on
    writability would silence it in exactly the state where the whole result
    lives in the draft."""
    text = _prompt(graph_enabled=False)
    assert "YOUR CLOCK" in text
    assert "ORION_TURN_DEADLINE_EPOCH" in text


def test_writing_is_instructed_to_happen_at_formation_not_at_the_end() -> None:
    """Run 32b42392f495 was killed mid-writeup with one hop of five recorded.
    The write section previously read as an end-of-turn form."""
    text = _prompt()
    assert "AT THE MOMENT YOU FORM IT" in text
    assert "KEEP THE LAST QUARTER OF THE BUDGET FOR WRITING" in text


def test_the_number_checking_habit_is_stated_because_it_is_what_went_wrong() -> None:
    """The surviving finding compared rejected `stance` against active
    `semantic` and called it one population."""
    text = _prompt()
    assert "comparing like with like" in text
    assert "which population each number is over" in text


# --- a prior is live until Orion closes it ---------------------------------
#
# The accumulation loop went down on 2026-08-27 because the reader asked for
# `status = 'open'` only. These pin the status set from both directions: what
# must still come back, and what must not.


@pytest.mark.parametrize("status", ["open", "supported", "revised", "", "typo_status"])
def test_a_prior_orion_has_not_closed_is_still_offered(status: str) -> None:
    """`supported` and `revised` are test OUTCOMES, not closures.

    The unknown/empty cases are deliberate: an unrecognised status must read
    as live, because losing a belief to a spelling mistake in hand-written
    Cypher is worse than showing one claim too many. Re-litigation already has
    a bound (`stale_after`); a silently vanished prior has none.
    """
    offered, stale, dropped = select_priors(
        [_prior_row(status=status)], sample=8, stale_after=3
    )
    assert dropped == 0
    assert [p.prior_id for p in offered] == ["p1"]
    assert stale == []


@pytest.mark.parametrize("status", CLOSED_STATUSES)
def test_the_two_closing_statuses_are_the_only_ones_the_query_excludes(
    status: str,
) -> None:
    """Excluded in the CYPHER, not in Python -- so this asserts on the text
    that FalkorDB will actually run, which is where the bug lived."""
    assert f"'{status}'" in LIVE_PRIORS_CYPHER
    assert "NOT p.status IN" in LIVE_PRIORS_CYPHER
    assert "p.status IS NULL OR" in LIVE_PRIORS_CYPHER


def test_a_prior_with_no_status_survives_the_cypher_null_trap() -> None:
    """`NOT null IN [...]` is null in Cypher, and a null WHERE filters the row
    OUT. Without the explicit `IS NULL` arm a prior written with no status at
    all would vanish -- a silent loss, not an error."""
    where = LIVE_PRIORS_CYPHER.split("WHERE", 1)[1]
    assert where.strip().startswith("(p.status IS NULL OR")


def test_the_counts_split_live_from_closed_not_open_from_everything() -> None:
    assert "AS live_total" in COUNTS_CYPHER
    assert "AS closed_total" in COUNTS_CYPHER
    assert "'open'" not in COUNTS_CYPHER


def test_a_supported_prior_is_offered_and_not_also_listed_as_settled() -> None:
    """It must appear in exactly one place. Offering a claim for testing while
    also printing it under RECENTLY SETTLED shows Orion the same prior twice
    under two contradictory headings."""
    assert "supported" not in RECENT_SETTLED_CYPHER
    assert "revised" not in RECENT_SETTLED_CYPHER


def test_the_2026_08_27_graph_state_is_not_read_back_as_an_empty_worldview() -> None:
    """REPLAY OF THE REAL INCIDENT, not a mutation invented to go red.

    These are the three priors that were actually in `orion_worldview` when run
    `0a14e9531089` was handed `priors=0/0` and had nothing to continue: run 3's
    prior revised by run 5, run 5's own, and run 6's -- written `supported` on
    formation, so it was never `open` for even one run.
    """
    live = [
        _prior_row(pid="editorial_bias_concrete_over_atmospheric_32b42392f495",
                   claim="the gate prefers concrete over atmospheric",
                   confidence="0.85", status="revised", tested=1),
        _prior_row(pid="gate_bias_manual_review_7736d5271d97",
                   claim="the stance gate is manual review",
                   confidence="0.85", status="supported", tested=1),
        _prior_row(pid="automated_intake_gate",
                   claim="an automated formation policy gate runs before review",
                   confidence="0.85", status="supported", tested=1),
    ]
    reader = _StatusAwareReader(live)
    view = read_snapshot(reader, sample=8, stale_after=3)

    assert view.live_total == 3
    assert view.closed_total == 0
    assert len(view.live_priors) == 3, "the whole loop stops if this is empty"

    from orion.curiosity.kickoff_prompt import _priors_section

    text = "\n".join(_priors_section(view, stale_after=3))
    assert "NO PRIORS STILL IN PLAY" not in text
    assert "YOUR OWN GRAPH IS EMPTY" not in text
    assert "manual review" in text


def test_the_prompt_tells_orion_which_statuses_actually_close_a_prior() -> None:
    """The reader's rule is only half the fix. Run `0a14e9531089` wrote its new
    prior as `supported` in the same breath it formed it, because nothing said
    that meant anything other than 'confirmed'."""
    text = _prompt()
    assert "ONLY TWO STATUSES CLOSE A PRIOR" in text
    for status in CLOSED_STATUSES:
        assert status in text
    assert "confidence" in text.lower()


def test_a_closed_prior_does_not_come_back_through_the_live_read() -> None:
    """The other direction of the same predicate: widening liveness must not
    have widened it to everything."""
    rows = [
        _prior_row(pid="live_one", status="revised"),
        _prior_row(pid="dead_one", status="refuted"),
        _prior_row(pid="gone_one", status="retired_unresolvable"),
    ]
    view = read_snapshot(_StatusAwareReader(rows), sample=8, stale_after=3)
    assert [p.prior_id for p in view.live_priors] == ["live_one"]
    assert view.live_total == 1
    assert view.closed_total == 2


def test_a_dead_prior_pool_is_logged_not_silently_started_from(caplog) -> None:
    """Every prior closed is legal but is also what the 2026-08-27 filter bug
    looked like from outside: a run quietly beginning with nothing."""
    rows = [_prior_row(pid="d1", status="refuted")]
    with caplog.at_level("WARNING", logger="orion.curiosity.worldview"):
        view = read_snapshot(_StatusAwareReader(rows), sample=8, stale_after=3)
    assert view.live_total == 0 and view.closed_total == 1
    assert "curiosity_worldview_pool_dead" in caplog.text


def test_an_empty_graph_is_not_reported_as_a_dead_pool(caplog) -> None:
    """Never written a prior and closed every prior are different states, and
    only one of them is a fault."""
    with caplog.at_level("WARNING", logger="orion.curiosity.worldview"):
        read_snapshot(_StatusAwareReader([]), sample=8, stale_after=3)
    assert "curiosity_worldview_pool_dead" not in caplog.text


def test_the_settled_read_does_not_also_answer_the_live_read(caplog) -> None:
    """A fake-reader needle that is a substring of another query silently
    answers both. Found in review: `"p.status IN ['refuted'"` is inside
    LIVE_PRIORS_CYPHER's own `NOT p.status IN [...]`, so the settled rows came
    back as priors too and `build_prior` dropped them -- a green test that had
    stopped isolating the thing it named."""
    reader = _FakeReader(answers={"ORDER BY p.last_tested_at": [
        {"claim": "a closed claim", "status": "refuted"},
    ]})
    with caplog.at_level("WARNING", logger="orion.curiosity.worldview"):
        view = read_snapshot(reader, sample=8, stale_after=3)
    assert view.recently_settled == [("a closed claim", "refuted")]
    assert view.live_priors == []
    assert "curiosity_worldview_unreadable_priors" not in caplog.text


def test_a_pool_too_big_for_one_read_is_not_truncated_silently(caplog) -> None:
    """The live set no longer drains on first test, so rows past the limit
    would never be shown, never accumulate times_tested, and never become
    retirable -- an invisible ceiling rather than an error."""
    from orion.curiosity.worldview import LIVE_PRIORS_LIMIT

    rows = [_prior_row(pid=f"p{i:05d}") for i in range(LIVE_PRIORS_LIMIT)]
    with caplog.at_level("WARNING", logger="orion.curiosity.worldview"):
        read_snapshot(_StatusAwareReader(rows), sample=8, stale_after=3)
    assert "curiosity_worldview_priors_truncated" in caplog.text


def test_an_ordinary_pool_does_not_warn_about_truncation(caplog) -> None:
    with caplog.at_level("WARNING", logger="orion.curiosity.worldview"):
        read_snapshot(
            _StatusAwareReader([_prior_row(pid="p1")]), sample=8, stale_after=3
        )
    assert "curiosity_worldview_priors_truncated" not in caplog.text


def test_the_live_read_never_orders_on_a_value_orion_might_quote() -> None:
    """The obvious repair for the limit is a server-side
    `ORDER BY abs(p.confidence - 0.5)`. FalkorDB rejects the WHOLE query on the
    first string-typed confidence ("Type mismatch: expected ... but was
    String", reproduced live 2026-08-27), which costs Orion its entire world
    view rather than mis-ordering it. `_as_float` tolerates the same value in
    Python, which is where the sort belongs."""
    assert "ORDER BY" not in LIVE_PRIORS_CYPHER
    assert "abs(" not in LIVE_PRIORS_CYPHER


# --- continuity is a thread, not a pointer ---------------------------------


def test_the_last_few_runs_are_shown_as_subjects_not_just_the_last_note() -> None:
    """Continuity was one run deep and pointed INWARD -- the note a run leaves
    itself is always some form of "go deeper on X", so the run that follows
    cannot tell whether X is new or the fourth consecutive visit. Three runs on
    memory-crystallization gating is what that produced."""
    from orion.curiosity.kickoff_prompt import _thread_section
    from orion.curiosity.worldview import RecentRun, WorldviewSnapshot, build_recent_runs

    rows = [
        {"run_id": "r3", "written_at": 300, "continue_note": "n", "claim": "third claim"},
        {"run_id": "r1", "written_at": 100, "continue_note": "n", "claim": "first claim"},
        {"run_id": "r2", "written_at": 200, "continue_note": "n", "claim": "second claim"},
    ]
    runs = build_recent_runs(rows, limit=4)
    assert [r.run_id for r in runs] == ["r3", "r2", "r1"], "newest first"

    text = "\n".join(_thread_section(WorldviewSnapshot(recent_runs=runs)))
    assert "THE LAST 3 RUNS" in text
    assert "third claim" in text and "first claim" in text
    # It states the fact and stops. Code choosing Orion's subject for it is the
    # thing this whole arc deleted.
    assert "pick something else" not in text.lower()
    assert "you should" not in text.lower()


def test_one_run_is_not_a_thread() -> None:
    from orion.curiosity.kickoff_prompt import _thread_section
    from orion.curiosity.worldview import RecentRun, WorldviewSnapshot

    view = WorldviewSnapshot(recent_runs=[RecentRun("r1", ["only claim"], 1)])
    assert _thread_section(view) == []


def test_several_priors_from_one_run_collapse_to_that_one_run() -> None:
    """One row per (run, claim) because FalkorDB hands `collect()` back as a
    flat STRING under decode_responses -- `'[claim one, claim two]'` -- and
    claims contain commas, so there is nothing safe to split on."""
    from orion.curiosity.worldview import build_recent_runs

    rows = [
        {"run_id": "r1", "written_at": 100, "continue_note": "", "claim": "a, with comma"},
        {"run_id": "r1", "written_at": 100, "continue_note": "", "claim": "b"},
    ]
    runs = build_recent_runs(rows, limit=4)
    assert len(runs) == 1
    assert runs[0].claims == ["a, with comma", "b"]


def test_a_run_with_no_prior_falls_back_to_its_own_note() -> None:
    from orion.curiosity.worldview import build_recent_runs

    rows = [{"run_id": "r1", "written_at": 100,
             "continue_note": "trace the intake pipeline", "claim": None}]
    assert build_recent_runs(rows, limit=4)[0].claims == ["trace the intake pipeline"]


def test_an_undated_run_sorts_last_in_the_thread_too() -> None:
    from orion.curiosity.worldview import build_recent_runs

    rows = [
        {"run_id": "undated", "written_at": None, "continue_note": "", "claim": "x"},
        {"run_id": "dated", "written_at": 100, "continue_note": "", "claim": "y"},
    ]
    assert [r.run_id for r in build_recent_runs(rows, limit=4)] == ["dated", "undated"]


def test_the_thread_query_does_not_use_collect() -> None:
    from orion.curiosity.worldview import RECENT_RUNS_CYPHER

    assert "collect(" not in RECENT_RUNS_CYPHER
    assert "p.claim AS claim" in RECENT_RUNS_CYPHER


# --- edges are part of the footprint ----------------------------------------
#
# They were not, and that is why nobody noticed Orion had never drawn one.
# Measured live 2026-08-29: 8 runs, 12 Hops, 9 Findings, 5 Priors, 1 Concept,
# and `db.relationshipTypes()` empty on `orion_worldview`. The footprint --
# the journal line, the `wrote=` log, the atlas growth panel -- matched nodes
# only, so a connection would have been invisible even once written.


def test_the_footprint_reports_edges_alongside_nodes() -> None:
    # The two needles are what FalkorDB would actually discriminate on: the
    # node query binds `(n)` and the edge query binds `()-[r]->()`. A fake
    # keyed on a looser substring would pass here whether or not the edge
    # query is issued at all.
    reader = _FakeReader(answers={
        "MATCH (n) WHERE n.run_id": [{"label": "Finding", "n": 3}],
        "MATCH ()-[r]->() WHERE r.run_id": [
            {"label": "SUPPORTS", "n": 2}, {"label": "CONTRADICTS", "n": 1},
        ],
    })
    assert read_run_footprint(reader, "abc123") == {
        "Finding": 3, "SUPPORTS": 2, "CONTRADICTS": 1,
    }


def test_a_run_that_wrote_nodes_but_drew_no_edges_says_so_by_omission() -> None:
    # The live case for every run so far. Nodes present, no edge key at all --
    # NOT an edge key sitting at zero, which would read as "tried and failed".
    reader = _FakeReader(answers={
        "MATCH (n) WHERE n.run_id": [{"label": "Finding", "n": 3}],
    })
    assert read_run_footprint(reader, "abc123") == {"Finding": 3}


def test_an_unreadable_edge_query_makes_the_whole_footprint_unknown() -> None:
    """Nodes-but-no-edges and nodes-with-an-unreadable-edge-query must not
    render identically. The second is the unreadable-vs-empty conflation this
    module refuses everywhere else, one level down."""

    class _EdgesFail(_FakeReader):
        def query(self, cypher: str):
            if "-[r]->" in cypher:
                raise WorldviewUnavailable("ConnectionError: nope")
            return super().query(cypher)

    reader = _EdgesFail(answers={
        "MATCH (n) WHERE n.run_id": [{"label": "Finding", "n": 3}],
    })
    assert read_run_footprint(reader, "abc123") is None


def test_the_edge_footprint_refuses_a_non_hex_run_id() -> None:
    # Same injection guard the node query has: the run id is interpolated into
    # Cypher, so it must never come from anywhere but the loop's own hex id.
    with pytest.raises(ValueError):
        run_edge_footprint_cypher("'; MATCH (n) DETACH DELETE n //")


def test_the_edge_footprint_counts_by_type_not_in_total() -> None:
    # CONTRADICTS is the edge that costs something to write, because it cuts
    # against a claim Orion holds. Averaged into a total it is invisible.
    cypher = run_edge_footprint_cypher("a1b2c3d4e5f6")
    assert "type(r) AS label" in cypher
    assert "r.run_id = 'a1b2c3d4e5f6'" in cypher
