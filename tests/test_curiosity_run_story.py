"""The cross-store join behind the Curiosity tab's run story.

Every fixture here is a row list shaped exactly as the store hands it back
(asyncpg rows as dicts with a datetime `created_at` and a JSON-text `detail`;
FalkorDB rows with an epoch-ms `written_at`). The thing under test is the
join, and the failure that matters is a confident story assembled from the
wrong rows: a reflect run leaking into the strip, a retried run's hops
collapsed to one attempt, a reach-out reported as blocked when nothing
recorded a decision at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.curiosity.run_story import (
    OUTCOME_CANCELLED,
    DECISION_NOT_RECORDED,
    DECISION_SENT,
    LINE_LABELS,
    OUTCOME_DIED,
    OUTCOME_FINISHED,
    OUTCOME_REACH_BLOCKED,
    OUTCOME_SENT,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    RunStoryRows,
    build_stories,
    outreach_key,
    reach_out_totals,
    run_to_payload,
    story_to_payload,
    summaries,
)
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS

T0 = datetime(2026, 9, 21, 15, 48, tzinfo=timezone.utc)
MS0 = int(T0.timestamp() * 1000)


def _at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


def _ms(sec: float) -> int:
    return MS0 + int(sec * 1000)


def _life(run_id, node, status, sec, *, workflow="curiosity.investigate", detail=None, resumed=None):
    return {
        "run_id": run_id, "workflow": workflow, "node": node, "next_node": "",
        "status": status, "resumed_from_node": resumed, "correlation_id": "c",
        "created_at": _at(sec), "detail": json.dumps(detail or {}),
    }


def _completed(run_id, sec, **detail):
    base = {"line": "investigate", "attempts": 1, "reach_out": False,
            "reach_out_why": "", "journal_entry_id": f"curiosity-investigation:{run_id}",
            "finding_text": "found it"}
    base.update(detail)
    return _life(run_id, "finish", "completed", sec, detail=base)


def _happy_run(run_id="446ddd7165d5", reach_out=False, why=""):
    """Role -> 2 hops -> finding -> revision 0.60 -> 0.68 -> outcome -> journal
    -> finish, the shape of the live example run."""
    return RunStoryRows(
        lifecycle=[_life(run_id, "harness_turn", "running", 0), _completed(run_id, 850, reach_out=reach_out, reach_out_why=why)],
        roles=[{"run_id": run_id, "choice": "local_crawl", "why": "queue pressure was high", "written_at": _ms(2)}],
        hops=[
            {"run_id": run_id, "n": 2, "note": "second", "written_at": _ms(714.05)},
            {"run_id": run_id, "n": 1, "note": "first", "written_at": _ms(714.0)},
        ],
        findings=[{"run_id": run_id, "finding_id": "f1", "text": "three authors only", "evidence": "psql", "written_at": _ms(740)}],
        revisions=[{"run_id": run_id, "prior_id": "p1", "from_confidence": 0.6, "to_confidence": 0.68,
                    "from_status": "revised", "to_status": "supported", "written_at": _ms(740.02)}],
        outcomes=[{"run_id": run_id, "continue_line": True, "continue_note": "read the 7 rows",
                   "reach_out": reach_out, "reach_out_why": why, "written_at": _ms(845)}],
        priors=[{"prior_id": "p1", "claim": "who matters is a singleton", "line": ""}],
        journals=[{"entry_id": f"curiosity-investigation:{run_id}", "source_ref": f"curiosity:{run_id}",
                   "title": "Curiosity", "body": "the write-up", "created_at": _at(849)}],
    )


def test_the_happy_run_tells_its_story_in_clock_order() -> None:
    stories = build_stories(_happy_run())
    story = stories["446ddd7165d5"]
    kinds = [it.kind for it in story.timeline]
    assert kinds[0] == "starting_prior", kinds
    assert kinds.count("starting_prior") == 1
    assert kinds[-1] == "lifecycle" and story.timeline[-1].data["status"] == "completed"
    hops = [it for it in story.timeline if it.kind == "hop"]
    assert [h.data["n"] for h in hops] == [1, 2], "hops come back unordered from the graph"
    assert kinds.index("finding") < kinds.index("revision") < kinds.index("outcome") < kinds.index("journal")
    run = story.run
    assert run.line == "investigate" and run.line_known is True
    assert run.plain_line_label == "World question"
    assert run.status == STATUS_COMPLETED
    assert run.started_from == "lifecycle"
    assert story.starting_prior is not None
    assert story.starting_prior["prior_id"] == "p1"
    assert story.starting_prior["claim"] == "who matters is a singleton"
    assert story.starting_prior["source"] == "prior_revision"
    assert story.prior_outcome is not None
    assert story.prior_outcome["verdict"] == "supported"
    assert story.summary["hops"] == 2
    assert story.summary["findings"] == 1
    assert story.summary["revisions"] == 1

    assert run.duration_sec == 850.0
    assert run.prior_touched == {"prior_id": "p1", "claim": "who matters is a singleton",
                                 "from": 0.6, "to": 0.68, "from_status": "revised", "to_status": "supported"}
    assert run.hops == 2 and run.findings == 1 and run.revisions == 1
    assert run.outcome_kind == OUTCOME_FINISHED
    assert run.reach_out.wanted is False and run.reach_out.decision is None
    assert story.journal_body == "the write-up"
    assert run.harness is None, "patch C's fields are absent -> not recorded, not zero"


def test_offsets_are_relative_to_the_start_and_none_when_unknown() -> None:
    payload = story_to_payload(build_stories(_happy_run())["446ddd7165d5"])
    hop1 = next(it for it in payload["timeline"] if it["kind"] == "hop" and it["n"] == 1)
    assert hop1["offset_sec"] == 714.0
    start = next(it for it in payload["timeline"] if it["kind"] == "lifecycle")
    assert start["offset_sec"] == 0.0
    prior = next(it for it in payload["timeline"] if it["kind"] == "starting_prior")
    assert prior["offset_sec"] is None, "undated starting prior prints with no fake clock"
    assert json.dumps(payload), "json-safe"


def test_a_retried_run_shows_hops_by_clock_with_attempt_boundaries() -> None:
    """A retried turn restarts `n` at 1 under the same run_id. Sorted by `n`
    alone (what the atlas did) this reads 1,1,2,2 -- two turns' notes
    interleaved. By clock it is 1,2 | 1,2, and the boundary is marked."""
    run_id = "retried"
    rows = RunStoryRows(
        lifecycle=[
            _life(run_id, "harness_turn", "running", 0),
            _life(run_id, "harness_turn", "failed", 400, detail={"error": "HarnessTurnFailed: no_final_frame"}),
            _life(run_id, "harness_turn", "resumed", 401, resumed="harness_turn"),
            _completed(run_id, 900, attempts=2),
        ],
        hops=[
            {"run_id": run_id, "n": 2, "note": "a2 second", "written_at": _ms(800)},
            {"run_id": run_id, "n": 1, "note": "a1 first", "written_at": _ms(100)},
            {"run_id": run_id, "n": 1, "note": "a2 first", "written_at": _ms(600)},
            {"run_id": run_id, "n": 2, "note": "a1 second", "written_at": _ms(200)},
        ],
    )
    story = build_stories(rows)[run_id]
    hops = [it for it in story.timeline if it.kind == "hop"]
    assert [h.data["n"] for h in hops] == [1, 2, 1, 2]
    assert [h.attempt for h in hops] == [1, 1, 2, 2]
    boundaries = [it for it in story.timeline if it.kind == "attempt"]
    assert len(boundaries) == 1 and boundaries[0].data["attempt"] == 2
    # The death and the resume sit inline, between the two attempts' hops.
    kinds = [(it.kind, it.data.get("status")) for it in story.timeline]
    fail_i = kinds.index(("lifecycle", "failed"))
    assert kinds.index(("hop", None)) < fail_i < kinds.index(("attempt", None))
    assert story.run.attempts == 2
    assert story.run.status == STATUS_COMPLETED, "a retried run that finished is finished"
    assert story.run.error == "HarnessTurnFailed: no_final_frame"


def test_legacy_hops_without_a_clock_stay_ordered_by_n_and_get_no_attempt() -> None:
    """Before 2026-09-19 hops carried no `written_at`. `n` is all there is,
    and inventing attempt numbers from it would be a guess."""
    rows = RunStoryRows(hops=[
        {"run_id": "old", "n": 3, "note": "c"},
        {"run_id": "old", "n": 1, "note": "a"},
        {"run_id": "old", "n": 2, "note": "b"},
    ])
    story = build_stories(rows)["old"]
    hops = [it for it in story.timeline if it.kind == "hop"]
    assert [h.data["n"] for h in hops] == [1, 2, 3]
    assert all(h.attempt is None for h in hops)
    assert not [it for it in story.timeline if it.kind == "attempt"]
    assert story.run.started_at is None and story.run.started_from == "none"


def test_a_reflect_row_is_excluded_from_the_strip() -> None:
    rows = RunStoryRows(lifecycle=[
        _life("reflect1", "finish", "completed", 0, workflow="self_study.reflect", detail={"line": "reflect"}),
        _completed("real1", 10),
    ])
    assert set(build_stories(rows)) == {"real1"}


def test_start_falls_back_to_the_first_graph_node_when_lifecycle_rows_are_missing() -> None:
    """Since 2026-09-14 only `completed` rows land. The run still has a
    clock: its role choice was written at the start of the turn."""
    rows = _happy_run()
    rows.lifecycle = [r for r in rows.lifecycle if r["status"] == "completed"]
    run = build_stories(rows)["446ddd7165d5"].run
    assert run.started_at == _ms(2)
    assert run.started_from == "graph"
    assert run.finished_at == _ms(850)
    assert run.status == STATUS_COMPLETED


def test_a_graph_only_run_is_unknown_not_completed() -> None:
    rows = _happy_run()
    rows.lifecycle = []
    run = build_stories(rows)["446ddd7165d5"].run
    assert run.status == "unknown"
    assert run.finished_at == _ms(845), "the outcome node is the best end clock we have"
    assert run.line_known is False, "the line lives only on the finish row"
    assert run.plain_line_label in LINE_LABELS.values()


def test_an_in_flight_run_reads_as_running() -> None:
    rows = RunStoryRows(lifecycle=[_life("live", "harness_turn", "running", 0)])
    run = build_stories(rows)["live"].run
    assert run.status == STATUS_RUNNING
    assert run.outcome_kind == "running"
    assert run.finished_at is None and run.duration_sec is None


def test_a_wanted_reach_out_with_no_decision_row_is_not_recorded_never_blocked() -> None:
    """Live on 2026-09-22: six runs wanted to reach out, every one blocked at
    a pre-check that logs a line and writes no row. The story must not
    invent a gate for them."""
    run = build_stories(_happy_run(reach_out=True, why="the repair step recorded its own prompt"))["446ddd7165d5"].run
    assert run.reach_out.wanted is True
    assert run.reach_out.decision == DECISION_NOT_RECORDED
    assert run.reach_out.gate is None
    assert run.reach_out.why == "the repair step recorded its own prompt"
    assert run.outcome_kind == OUTCOME_REACH_BLOCKED


def test_a_blocked_reach_out_names_its_gate() -> None:
    rows = _happy_run(reach_out=True, why="w")
    key = outreach_key("446ddd7165d5")
    rows.outreach = [{"decision_id": "d1", "decided_at": _at(851), "reason": "daily_cap",
                      "correlation_id": key, "session_id": None,
                      "result_json": json.dumps({"source": "curiosity_outreach", "run_id": "446ddd7165d5"})}]
    story = build_stories(rows)["446ddd7165d5"]
    assert story.run.reach_out.decision == "blocked:daily_cap"
    assert story.run.reach_out.gate == "daily_cap"
    assert story.run.reach_out.sent_at is None
    outreach = next(it for it in story.timeline if it.kind == "outreach")
    assert outreach.at == _ms(851) and outreach.data["gate"] == "daily_cap"
    assert story.timeline[-1].kind == "outreach", "the decision is the last thing that happened"


def test_a_sent_reach_out_with_a_reply_shows_both() -> None:
    rows = _happy_run(reach_out=True, why="w")
    key = outreach_key("446ddd7165d5")
    rows.outreach = [{"decision_id": "d1", "decided_at": _at(900), "reason": "sent",
                      "correlation_id": key, "session_id": "s", "result_json": "{}"}]
    rows.chat = [
        {"correlation_id": key, "session_id": "s", "prompt": "", "response": "Juniper, the repair step...",
         "client_meta": json.dumps({"unsolicited": True}), "created_at": _at(900).replace(tzinfo=None)},
        {"correlation_id": "some-uuid4", "session_id": "s", "prompt": "oh no, which step?", "response": "...",
         "client_meta": json.dumps({"in_reply_to": key}), "created_at": _at(1800).replace(tzinfo=None)},
        {"correlation_id": "unrelated", "session_id": "s", "prompt": "hi", "response": "hello",
         "client_meta": "{}", "created_at": _at(50).replace(tzinfo=None)},
    ]
    story = build_stories(rows)["446ddd7165d5"]
    r = story.run.reach_out
    assert r.decision == DECISION_SENT and r.sent is True
    assert r.sent_at == _ms(900), "naive chat timestamps are UTC"
    assert r.composed_text == "Juniper, the repair step..."
    assert r.reply_at == _ms(1800) and r.reply_text == "oh no, which step?"
    assert story.run.outcome_kind == OUTCOME_SENT
    kinds = [it.kind for it in story.timeline]
    assert kinds[-2:] == ["outreach", "reply"]
    payload = run_to_payload(story.run)
    assert payload["reach_out"]["reply"] == {"at": _ms(1800), "text": "oh no, which step?"}


def test_a_self_sense_run_comes_from_its_score_rows_alone() -> None:
    """The self-sense line writes no graph and no journal; today it does not
    even write a lifecycle row. Its four score rows are the whole run."""
    rows = RunStoryRows(self_sense=[
        {"run_id": "20260921T174910Z-10faa7", "created_at": _at(i * 300), "question_key": k, "question": q,
         "answer_text": "...", "answer_source": "chat", "self_label_score": 3, "grounded_record_score": 2,
         "self_definition_version": 24}
        for i, (k, q) in enumerate([("what_are_you", "What are you?"), ("unasked", "..."), ("cant", "..."), ("who_matters", "...")])
    ])
    story = build_stories(rows)["20260921T174910Z-10faa7"]
    run = story.run
    assert run.line == "self_sense_eval" and run.plain_line_label == "Self-sense check"
    assert run.status == STATUS_COMPLETED
    assert run.started_from == "self_sense"
    assert run.started_at == _ms(0) and run.finished_at == _ms(900)
    assert run.self_sense["questions_answered"] == 4
    assert run.self_sense["self_definition_version"] == 24
    assert [it.kind for it in story.timeline] == ["self_sense_answer"] * 4
    assert run.outcome_kind == OUTCOME_FINISHED


def test_a_self_sense_run_with_fewer_answers_than_questions_is_not_called_complete() -> None:
    rows = RunStoryRows(self_sense=[
        {"run_id": "half", "created_at": _at(0), "question_key": "a", "question": "?", "self_label_score": 1, "grounded_record_score": 1},
        {"run_id": "half", "created_at": _at(1), "question_key": "b", "question": "?", "self_label_score": 1, "grounded_record_score": 1},
    ])
    assert build_stories(rows)["half"].run.status == "unknown"


def test_a_self_inquiry_run_is_labelled_from_its_finish_row() -> None:
    rows = _happy_run("selfrun")
    rows.lifecycle[-1] = _completed("selfrun", 850, line="self_inquiry", self_definition="I am the loop that notices.")
    rows.journals[0]["title"] = "Self-inquiry"
    run = build_stories(rows)["selfrun"].run
    assert run.line == "self_inquiry" and run.plain_line_label == "Self question"
    assert run.self_written == {"kind": "self_definition", "text": "I am the loop that notices."}


def test_a_self_inquiry_run_without_a_finish_row_is_recognised_by_its_journal_title() -> None:
    rows = _happy_run("selfrun")
    rows.lifecycle = []
    rows.journals[0]["title"] = "Self-inquiry"
    run = build_stories(rows)["selfrun"].run
    assert run.line == "self_inquiry" and run.line_known is True


def test_the_last_resort_line_fallback_reads_the_revised_priors_own_line_field() -> None:
    """No finish-row `line`, no admission brief, no self-sense marker, no
    Self-inquiry journal title -- the only thing left to ask is the prior
    the run actually revised. `line_known=False` here: this is a guess from
    a different node's field, not something the run itself recorded."""
    rows = _happy_run("guessed")
    rows.lifecycle = []
    rows.journals[0]["title"] = "Curiosity"  # not the self-inquiry title
    rows.priors[0]["line"] = "self"
    run = build_stories(rows)["guessed"].run
    assert run.line == "self_inquiry" and run.line_known is False
    assert run.plain_line_label == "Self question"


def test_the_last_resort_fallback_defaults_to_investigate_when_nothing_names_a_line() -> None:
    rows = _happy_run("unnamed")
    rows.lifecycle = []
    rows.journals[0]["title"] = "Curiosity"
    run = build_stories(rows)["unnamed"].run
    assert run.line == "investigate" and run.line_known is False


def test_a_failed_run_reads_as_died() -> None:
    rows = RunStoryRows(lifecycle=[
        _life("dead", "harness_turn", "running", 0),
        _life("dead", "harness_turn", "failed", 420, detail={"error": "rpc:TimeoutError"}),
    ])
    run = build_stories(rows)["dead"].run
    assert run.status == "failed" and run.error == "rpc:TimeoutError"
    assert run.outcome_kind == OUTCOME_DIED
    assert run.duration_sec == 420.0


def test_hop_readings_ride_beside_their_hop_and_absence_is_stated() -> None:
    rows = _happy_run()
    assert build_stories(rows)["446ddd7165d5"].readings_available is False
    rows.readings = [{"hop_run_id": "446ddd7165d5", "hop_n": 1, "hop_written_at": _ms(714.0),
                      "about_prior_id": "p1", "kind": "confirming", "moved_the_claim": True,
                      "reading_confidence": 0.7, "reasoning": "it did"}]
    story = build_stories(rows)["446ddd7165d5"]
    assert story.readings_available is True
    hop1 = next(it for it in story.timeline if it.kind == "hop" and it.data["n"] == 1)
    assert hop1.data["readings"][0]["kind"] == "confirming"
    hop2 = next(it for it in story.timeline if it.kind == "hop" and it.data["n"] == 2)
    assert hop2.data["readings"] == []


def test_patch_c_timing_fields_are_read_when_present() -> None:
    rows = _happy_run()
    rows.lifecycle[-1] = _completed("446ddd7165d5", 850, harness_elapsed_sec=712.4, turn_correlation_id="abc")
    run = build_stories(rows)["446ddd7165d5"].run
    assert run.harness == {"elapsed_sec": 712.4, "turn_correlation_id": "abc"}


def test_summaries_sort_newest_first_with_undated_last_and_filter_by_line() -> None:
    rows = RunStoryRows(
        lifecycle=[_completed("older", 0), _completed("newer", 5000),
                   _completed("selfy", 2000, line="self_inquiry")],
        hops=[{"run_id": "undated", "n": 1, "note": "x"}],
    )
    stories = build_stories(rows)
    assert [r.run_id for r in summaries(stories)] == ["newer", "selfy", "older", "undated"]
    assert [r.run_id for r in summaries(stories, line="self_inquiry")] == ["selfy"]
    assert [r.run_id for r in summaries(stories, line="self_sense_eval")] == []


def test_reach_out_totals_count_wanted_sent_and_the_top_gate() -> None:
    a = _happy_run("a", reach_out=True, why="w")
    stories = build_stories(a)
    b = _happy_run("b", reach_out=True, why="w")
    b.outreach = [{"correlation_id": outreach_key("b"), "reason": "daily_cap", "decided_at": _at(1), "result_json": "{}"}]
    stories.update(build_stories(b))
    c = _happy_run("c", reach_out=True, why="w")
    c.outreach = [{"correlation_id": outreach_key("c"), "reason": "daily_cap", "decided_at": _at(1), "result_json": "{}"}]
    stories.update(build_stories(c))
    d = _happy_run("d", reach_out=True, why="w")
    d.outreach = [{"correlation_id": outreach_key("d"), "reason": "sent", "decided_at": _at(1), "result_json": "{}"}]
    stories.update(build_stories(d))
    stories.update(build_stories(_happy_run("e")))
    totals = reach_out_totals(summaries(stories))
    assert totals == {"wanted": 4, "sent": 1, "blocked_by": {"blocked:daily_cap": 2},
                      "top_block_reason": "blocked:daily_cap", "not_recorded": 1}


def test_the_outreach_key_is_the_loops_uuid5() -> None:
    from uuid import NAMESPACE_URL, uuid5

    assert outreach_key("446ddd7165d5") == str(uuid5(NAMESPACE_URL, "curiosity_outreach:446ddd7165d5"))
    assert outreach_key("446ddd7165d5") == "48b2b73b-aea5-55ee-8d06-111b4cc3df60", "checked live 2026-09-22"


def test_orion_prose_is_bounded_not_trusted() -> None:
    rows = _happy_run()
    rows.hops[0]["note"] = "x" * 10_000
    story = build_stories(rows)["446ddd7165d5"]
    hop = next(it for it in story.timeline if it.kind == "hop" and it.data["n"] == 2)
    assert len(hop.data["note"]) == 2000


def test_an_undated_finding_sits_after_the_last_dated_hop_not_at_the_top() -> None:
    """95 of 117 live Findings carry no `written_at`. Undated-first (the
    legacy-hop rule) put the finding above the role choice and told the
    story backwards."""
    rows = _happy_run()
    del rows.findings[0]["written_at"]
    story = build_stories(rows)["446ddd7165d5"]
    kinds = [it.kind for it in story.timeline]
    assert kinds.index("finding") > kinds.index("hop")
    assert kinds.index("finding") < kinds.index("outcome")
    finding = next(it for it in story.timeline if it.kind == "finding")
    assert finding.at is None
    payload = story_to_payload(story)
    assert next(it for it in payload["timeline"] if it["kind"] == "finding")["offset_sec"] is None


def test_an_unrecorded_reach_out_decision_sits_last() -> None:
    story = build_stories(_happy_run(reach_out=True, why="w"))["446ddd7165d5"]
    assert story.timeline[-1].kind == "outreach"
    assert story.timeline[-1].at is None


def test_a_self_inquiry_lived_answer_counts_as_having_written_something() -> None:
    """Run `3dc94088912b` (live 2026-09-22): a completed finish row, a
    journal and one `:LivedAnswer` node -- and it rendered as "wrote
    nothing" with no start clock, because the reader only knew five labels."""
    rows = RunStoryRows(
        lifecycle=[_completed("3dc94088912b", 900, line="self_inquiry")],
        self_writes=[{"run_id": "3dc94088912b", "kind": "lived_answer", "question_id": "q7",
                      "family": "lived", "text": "I hold it against the records.", "evidence": "journal",
                      "written_at": _ms(30)}],
    )
    story = build_stories(rows)["3dc94088912b"]
    assert story.run.outcome_kind == OUTCOME_FINISHED
    assert story.run.started_at == _ms(30) and story.run.started_from == "graph"
    assert story.run.self_written == {"kind": "lived_answer", "text": "I hold it against the records.", "family": "lived"}
    assert [it.kind for it in story.timeline] == ["self_write", "lifecycle"]


# --- the admission path (since 2026-09-14) ------------------------------------


def _event(run_id, event, sec, detail=None, entry_id=None):
    return {"entry_id": entry_id or f"{run_id}:{event}:{sec}", "run_id": run_id, "event": event,
            "generated_at": _at(sec), "payload": json.dumps({"event": event, "detail": detail or {}, "run_id": run_id})}


def _admission(run_id, sec, *, workflow="curiosity.investigate", line="investigate", terminal="completed", updated=None):
    return {"run_id": run_id, "request": json.dumps({"workflow": workflow, "brief": {"line": line}}),
            "created_at": _at(sec), "control": None, "terminal": terminal,
            "updated_at": _at(updated if updated is not None else sec)}


def _admission_run(run_id="446ddd7165d5", *, bridge=True, line="investigate", workflow="curiosity.investigate"):
    """The live shape of 446ddd7165d5: accepted, waited 7h for the agent
    lane, admitted, ran, completed; the bridge row is a copy of the terminal
    event with the workflow mislabelled."""
    lease = {"lease": {"lane": "agent", "lease_id": "L1"}}
    rows = RunStoryRows(
        admission=[_admission(run_id, 0, workflow=workflow, line=line, updated=25600)],
        resource_events=[
            _event(run_id, "run.accepted", 0),
            _event(run_id, "run.waiting_resource", 0.02, {"requested_lane": "agent", "demand_id": "d"}),
            _event(run_id, "run.lane_swap_suppressed", 0.3, {"reason": "waiting_capacity"}),
            _event(run_id, "run.waiting_resource", 7, {"node": "resource_request"}),
            _event(run_id, "run.resource_granted", 25419, lease),
            _event(run_id, "run.lane_assigned", 25419, lease),
            _event(run_id, "run.resumed", 25419.1, {"node": "resource_wait"}),
            _event(run_id, "run.admitted", 25419.2, {"node": "resource_wait"}),
            _event(run_id, "run.running", 25419.3, {"node": "run_started"}),
            _event(run_id, "run.started", 25419.4, lease),
            _event(run_id, "run.running", 27409, {"node": "harness_turn"}),
            _event(run_id, "run.running", 27409.2, {"node": "read_turn_result"}),
            _event(run_id, "run.running", 27409.4, {"node": "journal"}),
            _event(run_id, "resource.lease_released", 27409.5, {"lane": "agent", "reason": "completed"}),
            _event(run_id, "run.completed", 27410, {"line": line, "attempts": 1, "reach_out": False,
                                                    "finding_text": "from the event", "journal_entry_id": "j"},
                   entry_id=f"{run_id}:terminal:completed"),
        ],
        event_counts=[{"run_id": run_id, "event": "run.checkpoint_resume_failed", "n": 12}],
        roles=[{"run_id": run_id, "choice": "local_crawl", "why": "w", "written_at": _ms(25420)}],
        hops=[{"run_id": run_id, "n": 1, "note": "first", "written_at": _ms(26000)}],
    )
    if bridge:
        rows.lifecycle = [_life(run_id, "finish", "completed", 27410,
                                detail={"line": line, "finding_text": "from the bridge", "journal_entry_id": "j"})]
    return rows


def test_an_admission_path_run_reads_its_lifecycle_from_the_events() -> None:
    story = build_stories(_admission_run(bridge=False))["446ddd7165d5"]
    run = story.run
    assert run.status == STATUS_COMPLETED
    assert run.started_at == _ms(0) and run.started_from == "admission"
    assert run.accepted_at == _ms(0) and run.admitted_at == _ms(25419)
    assert run.lane == "agent"
    assert run.lane_wait_sec == 25419.0
    assert run.active_sec == 27410.0 - 25419.0
    assert run.duration_sec == 27410.0
    assert run.finding_text == "from the event"
    assert run.retries == 0 and run.attempts == 1
    assert run.anomalies == {"run.checkpoint_resume_failed": 12}
    statuses = [(it.data.get("status"), it.data.get("node")) for it in story.timeline if it.kind == "lifecycle"]
    assert statuses[:3] == [("accepted", ""), ("waiting", ""), ("admitted", "")], statuses
    assert statuses.count(("waiting", "")) == 1, "consecutive waits collapse to one"
    assert ("running", "harness_turn") in statuses
    assert statuses[-2:] == [("lease_released", ""), ("completed", "finish")]
    admitted = next(it for it in story.timeline if it.data.get("status") == "admitted")
    assert admitted.data["lane"] == "agent" and admitted.data["wait_sec"] == 25419.0
    assert not any(it.data.get("status") in ("resource_granted", "lane_swap_suppressed") for it in story.timeline)
    # Graph items interleave by clock: role choice sits after admission.
    kinds = [it.kind for it in story.timeline]
    assert kinds.index("role_choice") > kinds.index("lifecycle")


def test_the_bridge_row_is_ignored_when_the_admission_path_has_the_run() -> None:
    """The bridge copies the terminal event and mislabels it. With both
    present the story must not show two completions or two starts."""
    story = build_stories(_admission_run(bridge=True))["446ddd7165d5"]
    completions = [it for it in story.timeline if it.data.get("status") == "completed"]
    assert len(completions) == 1
    assert story.run.finding_text == "from the event"
    assert story.run.started_from == "admission"


def test_the_bridge_detail_is_the_fallback_when_the_completed_event_carried_none() -> None:
    rows = _admission_run(bridge=True)
    rows.resource_events[-1] = _event("446ddd7165d5", "run.completed", 27410, {}, entry_id="446ddd7165d5:terminal:completed")
    run = build_stories(rows)["446ddd7165d5"].run
    assert run.finding_text == "from the bridge"


def test_a_legacy_only_run_is_still_read_from_the_bridge_table() -> None:
    rows = RunStoryRows(lifecycle=[
        _life("old", "harness_turn", "running", 0),
        _life("old", "harness_turn", "resumed", 400, resumed="harness_turn"),
        _completed("old", 900, attempts=2),
    ])
    run = build_stories(rows)["old"].run
    assert run.started_from == "lifecycle" and run.started_at == _ms(0)
    assert run.status == STATUS_COMPLETED and run.attempts == 2
    assert run.accepted_at is None and run.lane_wait_sec is None and run.retries == 0


def test_a_self_sense_run_is_labelled_from_the_admission_workflow_despite_the_mislabelled_bridge_row() -> None:
    rows = _admission_run("20260922T025855Z-b88ac9", workflow="self_sense_eval", line="self_sense_eval")
    # The bridge row says curiosity.investigate and carries no line at all.
    rows.lifecycle = [_life("20260922T025855Z-b88ac9", "finish", "completed", 27410, detail={"finding_text": "x"})]
    rows.resource_events[-1] = _event("20260922T025855Z-b88ac9", "run.completed", 27410, {"finding_text": "x"},
                                      entry_id="20260922T025855Z-b88ac9:terminal:completed")
    run = build_stories(rows)["20260922T025855Z-b88ac9"].run
    assert run.line == "self_sense_eval" and run.line_known is True
    assert run.plain_line_label == "Self-sense check"


def test_an_in_flight_admission_run_has_a_line_from_its_brief() -> None:
    rows = RunStoryRows(
        admission=[_admission("live", 0, line="self_inquiry", terminal=None)],
        resource_events=[_event("live", "run.accepted", 0), _event("live", "run.waiting_resource", 1, {"requested_lane": "agent"})],
    )
    run = build_stories(rows)["live"].run
    assert run.status == STATUS_RUNNING
    assert run.line == "self_inquiry" and run.line_known is True
    assert run.outcome_kind == "running"
    assert run.admitted_at is None and run.lane_wait_sec is None


def test_a_retried_turn_is_a_failure_mark_not_a_death() -> None:
    rows = _admission_run(bridge=False)
    rows.resource_events.insert(11, _event("446ddd7165d5", "run.retrying", 26500, {"node": "harness_turn", "reason": "no_final_frame"}))
    story = build_stories(rows)["446ddd7165d5"]
    assert story.run.status == STATUS_COMPLETED
    assert story.run.retries == 1
    assert story.run.outcome_kind == OUTCOME_FINISHED, "died means terminal failed"
    retry = next(it for it in story.timeline if it.data.get("status") == "retrying")
    assert retry.data["node"] == "harness_turn" and retry.data["error"] == "no_final_frame"


def test_a_failed_and_a_cancelled_admission_run() -> None:
    dead = RunStoryRows(
        admission=[_admission("dead", 0, terminal="failed", updated=500)],
        resource_events=[_event("dead", "run.accepted", 0), _event("dead", "run.failed", 500, {"error": "rpc:TimeoutError"},
                                                                    entry_id="dead:terminal:failed")],
    )
    run = build_stories(dead)["dead"].run
    assert run.status == "failed" and run.error == "rpc:TimeoutError" and run.outcome_kind == OUTCOME_DIED
    gone = RunStoryRows(admission=[_admission("gone", 0, terminal="cancelled", updated=50)])
    run = build_stories(gone)["gone"].run
    assert run.status == "cancelled" and run.outcome_kind == OUTCOME_CANCELLED
    assert run.finished_at == _ms(50), "terminal without an event falls back to the admission row's updated_at"


def test_a_reflect_admission_row_and_its_events_are_excluded() -> None:
    rows = RunStoryRows(
        admission=[_admission("refl", 0, workflow="self_study.reflect", line="reflect")],
        resource_events=[_event("refl", "run.accepted", 0), _event("refl", "run.completed", 5, {}, entry_id="refl:terminal:completed")],
        event_counts=[{"run_id": "refl", "event": "run.checkpoint_resume_failed", "n": 3}],
    )
    assert build_stories(rows) == {}


def test_the_payload_carries_the_admission_fields() -> None:
    payload = run_to_payload(build_stories(_admission_run())["446ddd7165d5"].run)
    assert payload["accepted_at"] == _ms(0) and payload["admitted_at"] == _ms(25419)
    assert payload["lane"] == "agent" and payload["lane_wait_sec"] == 25419.0
    assert payload["retries"] == 0 and payload["anomalies"] == {"run.checkpoint_resume_failed": 12}
    assert json.dumps(payload)


def test_help_request_about_is_the_starting_prior_ahead_of_revisions() -> None:
    """The subject of the sitting is the HelpRequest ABOUT prior, not the
    first revision's prior — those can diverge when a run revises something
    else along the way."""
    run_id = "c9649dc67459"
    rows = RunStoryRows(
        lifecycle=[_life(run_id, "harness_turn", "running", 0), _completed(run_id, 100)],
        help_requests=[{
            "run_id": run_id, "help_id": "h1",
            "prior_id": "self:outward_learning_in_record_not_in_loop_20260921",
            "prior_claim": "outward learning lives in the record but not the loop",
            "prior_status": "open", "prior_confidence": 0.42,
            "written_at": _ms(5),
        }],
        revisions=[{
            "run_id": run_id, "prior_id": "other:prior", "from_confidence": 0.5,
            "to_confidence": 0.55, "from_status": "open", "to_status": "revised",
            "written_at": _ms(80),
        }],
        priors=[
            {"prior_id": "self:outward_learning_in_record_not_in_loop_20260921",
             "claim": "outward learning lives in the record but not the loop",
             "status": "open", "confidence": 0.42, "line": "self_inquiry"},
            {"prior_id": "other:prior", "claim": "something else", "line": ""},
        ],
        self_writes=[{
            "run_id": run_id, "kind": "lived_answer",
            "text": "The loop still does not ingest outward learning as a first-class hop.",
            "evidence": "graph read", "written_at": _ms(90),
            "question_id": "q1",
        }],
        peer_briefs=[{
            "run_id": run_id, "brief_id": "b1", "help_id": "h1", "peer": "claude",
            "status": "ok", "summary": "peer agrees the loop gap is real",
            "written_at": _ms(95),
        }],
    )
    story = build_stories(rows)[run_id]
    assert story.starting_prior["prior_id"].startswith("self:outward_learning")
    assert story.starting_prior["source"] == "help_request_about"
    assert story.starting_prior["help_id"] == "h1"
    assert story.starting_prior["confidence"] == 0.42
    prior_items = [it for it in story.timeline if it.kind == "starting_prior"]
    assert len(prior_items) == 1
    assert "outward learning" in prior_items[0].data["claim"]

    assert story.prior_outcome["outcome_text"].startswith("The loop still does not")
    assert story.prior_outcome["outcome_kind"] == "lived_answer"
    assert story.prior_outcome["verdict"] == "answered"
    assert "lived answer" in story.prior_outcome["verdict_basis"]
    assert story.prior_outcome["peer"]["status"] == "ok"

    assert story.summary["helps"] == 1
    assert story.summary["peer_briefs"] == [{"status": "ok", "peer": "claude"}]
    assert story.summary["has_starting_prior"] is True
    assert story.summary["verdict"] == "answered"
    assert story.summary["hops"] == 0

    payload = story_to_payload(story)
    assert payload["starting_prior"]["prior_id"].startswith("self:outward")
    assert payload["prior_outcome"]["verdict"] == "answered"
    assert payload["summary"]["helps"] == 1
    writes = [it for it in payload["timeline"] if it["kind"] == "self_write"]
    assert len(writes) == 1
    assert writes[0]["write_kind"] == "lived_answer"
    assert "kind" not in writes[0] or writes[0].get("kind") == "self_write"
    assert writes[0]["text"].startswith("The loop still")


def test_lived_answer_write_kind_survives_payload_nesting() -> None:
    """Regression: nested `kind: lived_answer` used to overwrite timeline
    kind and blank the UI row."""
    run_id = "lived1"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 10)],
        self_writes=[{
            "run_id": run_id, "kind": "lived_answer",
            "text": "I noticed the queue pressure but kept going.",
            "written_at": _ms(5),
        }],
    )
    payload = story_to_payload(build_stories(rows)[run_id])
    row = next(it for it in payload["timeline"] if it["kind"] == "self_write")
    assert row["write_kind"] == "lived_answer"
    assert row["text"].startswith("I noticed")
    assert payload["prior_outcome"]["verdict"] == "answered"
    assert payload["prior_outcome"]["outcome_kind"] == "lived_answer"


def test_peer_failure_labels_the_prior_outcome_verdict() -> None:
    run_id = "peerfail"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 10)],
        help_requests=[{
            "run_id": run_id, "help_id": "h", "prior_id": "p",
            "prior_claim": "claim", "prior_status": "open", "written_at": _ms(1),
        }],
        priors=[{"prior_id": "p", "claim": "claim", "status": "open", "line": ""}],
        peer_briefs=[{
            "run_id": run_id, "brief_id": "b", "help_id": "h", "peer": "cursor",
            "status": "refused_budget", "summary": "", "refusal_reason": "budget_limited",
            "written_at": _ms(8),
        }],
    )
    po = build_stories(rows)[run_id].prior_outcome
    assert po["verdict"] == "peer_failed"
    assert "refused_budget" in po["verdict_basis"]


def test_subject_prior_revision_wins_over_a_later_side_revision() -> None:
    """Help ABOUT A was supported; a later revision of B must not erase that."""
    run_id = "side-rev"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 20)],
        help_requests=[{
            "run_id": run_id, "help_id": "h", "prior_id": "A",
            "prior_claim": "subject claim", "written_at": _ms(1),
        }],
        revisions=[
            {"run_id": run_id, "prior_id": "A", "from_confidence": 0.4, "to_confidence": 0.7,
             "from_status": "open", "to_status": "supported", "written_at": _ms(10)},
            {"run_id": run_id, "prior_id": "B", "from_confidence": 0.5, "to_confidence": 0.55,
             "from_status": "open", "to_status": "revised", "written_at": _ms(15)},
        ],
        priors=[
            {"prior_id": "A", "claim": "subject claim", "status": "supported", "confidence": 0.7, "line": ""},
            {"prior_id": "B", "claim": "side", "status": "revised", "line": ""},
        ],
    )
    story = build_stories(rows)[run_id]
    assert story.starting_prior["prior_id"] == "A"
    assert story.starting_prior["status"] == "open"
    assert story.starting_prior["confidence"] == 0.4
    assert story.prior_outcome["verdict"] == "supported"
    assert story.prior_outcome["revision"]["prior_id"] == "A"


def test_flat_revised_status_is_a_measured_verdict() -> None:
    run_id = "flat-rev"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 10)],
        revisions=[{
            "run_id": run_id, "prior_id": "p", "from_confidence": 0.5, "to_confidence": 0.5,
            "from_status": "open", "to_status": "revised", "written_at": _ms(5),
        }],
        priors=[{"prior_id": "p", "claim": "c", "status": "revised", "confidence": 0.5, "line": ""}],
    )
    po = build_stories(rows)[run_id].prior_outcome
    assert po["verdict"] == "revised"
    assert "revised" in po["verdict_basis"].lower()


def test_about_prefers_help_question_then_prior_then_brief() -> None:
    run_id = "about-help"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 10)],
        admission=[{
            "run_id": run_id,
            "request": json.dumps({
                "workflow": "curiosity.investigate",
                "brief": {"line": "investigate", "prompt": "fallback brief prompt that is long enough"},
            }),
            "created_at": _at(0), "control": None, "terminal": "completed", "updated_at": _at(10),
        }],
        help_requests=[{
            "run_id": run_id, "help_id": "h1", "prior_id": "p1",
            "question": "Does outward learning land in the working loop?",
            "prior_claim": "outward learning lives only in the journal",
            "written_at": _ms(1),
        }],
        priors=[{"prior_id": "p1", "claim": "outward learning lives only in the journal",
                 "status": "open", "line": "investigate"}],
    )
    story = build_stories(rows)[run_id]
    assert story.about["source"] == "help_request"
    assert "outward learning" in story.about["text"]
    assert story.run.about["text"] == story.about["text"]
    payload = story_to_payload(story)
    assert payload["about"]["source"] == "help_request"


def test_about_uses_prior_claim_when_no_help_question() -> None:
    run_id = "about-prior"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 10)],
        help_requests=[{
            "run_id": run_id, "help_id": "h1", "prior_id": "p1",
            "question": "",  # hire without a question string
            "prior_claim": "who matters is a singleton, not a crowd",
            "written_at": _ms(1),
        }],
        priors=[{"prior_id": "p1", "claim": "who matters is a singleton, not a crowd",
                 "status": "open", "line": "self_inquiry"}],
    )
    about = build_stories(rows)[run_id].about
    assert about["source"] == "prior"
    assert "who matters" in about["text"]


def test_about_for_self_sense_lists_the_four_fixed_questions() -> None:
    run_id = "20260922T211540Z-ff890d"
    rows = _admission_run(run_id, workflow="self_sense_eval", line="self_sense_eval", bridge=False)
    # Placeholder prompt only -- the bug that left Juniper with no subject.
    rows.admission[0] = {
        "run_id": run_id,
        "request": json.dumps({
            "workflow": "self_sense_eval",
            "brief": {
                "line": "self_sense_eval",
                "prompt": "self-sense eval: four fixed questions",
                "questions": [list(q) for q in SELF_SENSE_QUESTIONS],
            },
        }),
        "created_at": _at(0), "control": None, "terminal": "completed", "updated_at": _at(10),
    }
    about = build_stories(rows)[run_id].about
    assert about["source"] == "self_sense_questions"
    assert about["text"] == "Four fixed self-sense questions"
    assert len(about["detail"]) == 4
    assert any("what are you" in q.lower() for q in about["detail"])


def test_about_ignores_self_sense_placeholder_prompt_on_investigate_misroute() -> None:
    """If a self-sense job was driven as investigate (ff890d), the placeholder
    prompt must not masquerade as the subject -- fall through to finding."""
    run_id = "misroute"
    rows = RunStoryRows(
        lifecycle=[_completed(run_id, 10, finding_text="I looked at my own self_sense_eval_log instead.")],
        admission=[{
            "run_id": run_id,
            "request": json.dumps({
                "workflow": "curiosity.investigate",
                "brief": {"line": "investigate", "prompt": "self-sense eval: four fixed questions"},
            }),
            "created_at": _at(0), "control": None, "terminal": "completed", "updated_at": _at(10),
        }],
    )
    about = build_stories(rows)[run_id].about
    assert about["source"] == "finding"
    assert "self_sense_eval_log" in about["text"]
