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
    assert kinds[0] == "lifecycle", kinds
    assert kinds[1] == "role_choice"
    assert kinds[-1] == "lifecycle" and story.timeline[-1].data["status"] == "completed"
    hops = [it for it in story.timeline if it.kind == "hop"]
    assert [h.data["n"] for h in hops] == [1, 2], "hops come back unordered from the graph"
    assert kinds.index("finding") < kinds.index("revision") < kinds.index("outcome") < kinds.index("journal")
    run = story.run
    assert run.line == "investigate" and run.line_known is True
    assert run.plain_line_label == "World question"
    assert run.status == STATUS_COMPLETED
    assert run.started_from == "lifecycle"
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
    assert payload["timeline"][0]["offset_sec"] == 0.0
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
