from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timezone

import pytest

NOW = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
BODY = "Orion investigated the delayed journal writes. The observed records support a delivery gap, while its cause remains uncertain and needs another check."


def row(material=None):
    from orion.curiosity.compaction import batches, due_window, window_id
    start, end = due_window(NOW, "weekly", None)
    return dict(compaction_id=window_id("weekly", start), cadence="weekly", window_start=start,
                window_end=end, cycle_id="dc-test", status="pending", updated_at=NOW,
                state={"phase": "topics", "sources": [{"entry_id": "j1", "run_ref": "curiosity:r1", "created_at": NOW.isoformat()}],
                       "batches": batches(material or [{"ref": "j1", "body": BODY, "sources": ["j1"]}]), "partials": []})


@pytest.mark.parametrize("cadence,start,end", [
    ("weekly", "2026-10-05T06:00:00+00:00", "2026-10-12T06:00:00+00:00"),
    ("monthly", "2026-09-01T06:00:00+00:00", "2026-10-01T06:00:00+00:00"),
])
def test_closed_calendar_windows_and_no_repeat(cadence, start, end):
    from orion.curiosity.compaction import due_window
    window = due_window(NOW, cadence, None)
    assert tuple(d.isoformat() for d in window) == (start, end)
    latest = dict(window_start=window[0], window_end=window[1], status="completed")
    assert due_window(NOW, cadence, latest) is None
    latest["status"] = "pending"
    assert due_window(NOW, cadence, latest) == window


def test_catchup_preserves_oldest_gap_and_dst_calendar_boundaries():
    from orion.curiosity.compaction import due_window
    now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
    start, end = due_window(now, "weekly", None)
    assert (end - start).total_seconds() == 169 * 3600
    next_start, next_end = due_window(datetime(2027, 1, 1, tzinfo=timezone.utc), "weekly",
                                     dict(window_start=start, window_end=end, status="empty"))
    assert next_start == end
    assert next_end.day == 9 and next_end.month == 11
    start, end = due_window(datetime(2027, 1, 14, tzinfo=timezone.utc), "monthly", None)
    assert (start.year, start.month, end.year, end.month) == (2026, 12, 2027, 1)


def test_month_uses_weeks_plus_uncovered_and_late_journals_once():
    from orion.curiosity.compaction import source_material
    journals = [{"entry_id": f"j{i}", "source_ref": f"curiosity:r{i}", "body": f"body {i}", "created_at": NOW} for i in range(4)]
    week = {"compaction_id": "week1", "state": {"sources": [{"entry_id": "j1"}, {"entry_id": "j2"}], "body": BODY, "topics": [{"topic": "Delivery", "body": BODY, "sources": ["j1", "j2"]}]}}
    material, refs = source_material(journals, [week])
    assert [m["ref"] for m in material] == ["j0", "j3", "week1:topic:0"]
    assert len(refs) == 4
    # A mismatching or overlapping summary must not suppress original evidence.
    invalid = copy.deepcopy(week)
    invalid["state"]["sources"].append({"entry_id": "outside-month"})
    material, _ = source_material(journals, [invalid])
    assert [m["ref"] for m in material] == [f"j{i}" for i in range(4)]


def test_oversized_unicode_sources_are_lossless_and_bounded():
    from orion.curiosity.compaction import batches, INPUT_BUDGET
    from orion.cognition.compactor.chunking import json_char_len
    body = "a😀<&中文" * 10000
    result = batches([{"ref": "j1", "body": body}])
    assert "".join(item["body"] for batch in result for item in batch) == body
    assert all(sum(json_char_len(item) for item in batch) <= INPUT_BUDGET for batch in result)


def answer_for(r):
    from orion.curiosity.compaction import next_prompt
    prompt = next_prompt(r)
    if r["state"]["phase"] == "overall":
        return json.dumps({"body": BODY})
    records = json.loads(prompt.split("SOURCE RECORDS:\n")[1])
    return json.dumps({"topics": [{"topic": "Delivery reliability", "body": BODY,
                                   "refs": [item["ref"] for item in records]}]})


def test_topic_checkpoint_then_overall_keeps_provenance_and_reduces():
    from orion.curiosity.compaction import record_answer
    r = row([{"ref": str(i), "body": "evidence " * 1500, "sources": [f"j{i}"]} for i in range(7)])
    r["state"]["sources"] = [{"entry_id": f"j{i}", "run_ref": f"curiosity:r{i}", "created_at": NOW.isoformat()} for i in range(7)]
    phases = []
    for _ in range(40):
        phases.append(r["state"]["phase"])
        r = record_answer(r, answer_for(r))
        if r["status"] == "completed":
            break
    assert r["status"] == "completed" and phases[-1] == "overall"
    assert {s for t in r["state"]["topics"] for s in t["sources"]} == {f"j{i}" for i in range(7)}


@pytest.mark.parametrize("raw", ["", '{"body":""}', '{"body":"No findings."}', 'not json',
                                  '{"topics":[{"topic":"Invented", "body":"' + BODY + '","refs":["made-up"]}]}'])
def test_failed_empty_or_ungrounded_generation_does_not_change_checkpoint(raw):
    from orion.curiosity.compaction import record_answer
    r = row()
    before = copy.deepcopy(r)
    from orion.cognition.model_output import ModelOutputRejected
    with pytest.raises(ModelOutputRejected):   # the model's fault, so a durable run retries it
        record_answer(r, raw)
    assert r == before


def _topic_answer(body, refs=("s0",)):
    return json.dumps({"topics": [{"topic": "Delivery reliability", "body": body, "refs": list(refs)}]})


def test_body_over_the_cap_is_a_rejected_model_answer_not_a_code_error():
    """Live 2026-10-24..27: bodies written just past 1200 chars failed the run outright."""
    from orion.cognition.model_output import ModelOutputRejected
    from orion.curiosity.compaction import record_answer
    with pytest.raises(ModelOutputRejected, match="body"):
        record_answer(row(), _topic_answer(BODY + " more." * 200))
    assert issubclass(ModelOutputRejected, ValueError)   # existing `except ValueError` callers still catch it


def test_body_cut_off_by_the_length_cap_is_rejected_but_a_finished_one_at_the_cap_is_kept():
    from orion.cognition.model_output import ModelOutputRejected
    from orion.curiosity.compaction import TOPIC_BODY_CHARS, record_answer
    stump = (BODY + " ") * 20
    with pytest.raises(ModelOutputRejected, match="cut off"):
        record_answer(row(), _topic_answer(stump[:TOPIC_BODY_CHARS]))
    finished = stump[:TOPIC_BODY_CHARS - 1] + "."
    with pytest.raises(ModelOutputRejected, match="cut off"):   # trailing space does not hide a cut
        record_answer(row(), _topic_answer(stump[:TOPIC_BODY_CHARS - 1] + " "))
    assert record_answer(row(), _topic_answer(stump[:TOPIC_BODY_CHARS - 1] + "\u201d"))["state"]["phase"] == "overall"
    assert len(finished) == TOPIC_BODY_CHARS
    assert record_answer(row(), _topic_answer(finished))["state"]["phase"] == "overall"


def test_output_schema_encodes_the_limits_and_this_batchs_refs():
    from orion.curiosity.compaction import Topic, TopicPass, output_schema
    r = row([{"ref": str(i), "body": f"evidence {i} " * 20, "sources": ["j1"]} for i in range(3)])
    name, schema = output_schema(r)
    assert name == "CuriosityCompactionTopicPassV1"
    topics = schema["properties"]["topics"]
    item = topics["items"]["properties"]
    assert (topics["minItems"], topics["maxItems"]) == (1, TopicPass.model_fields["topics"].metadata[1].max_length) == (1, 4)
    assert item["body"]["maxLength"] == 1200 == Topic.model_fields["body"].metadata[1].max_length
    assert item["body"]["minLength"] == 80 and "pattern" not in item["body"] and "enum" not in item["body"]
    assert item["refs"]["minItems"] == 1
    assert item["refs"]["items"]["enum"] == [f"s{i}" for i in range(len(r["state"]["batches"][0]))]
    assert topics["items"]["additionalProperties"] is False
    r["state"]["phase"] = "overall"
    name, schema = output_schema(r)
    assert name == "CuriosityCompactionSummaryV1" and schema["properties"]["body"]["maxLength"] == 2400


def test_prompt_asks_for_words_not_a_character_count_at_the_cap():
    from orion.curiosity.compaction import next_prompt
    prompt = next_prompt(row())
    assert "about 120 to 150 words" in prompt and "hard limit: 1200 characters" in prompt
    assert "80 to 1200 characters" not in prompt and "1 to 4 topic areas" in prompt


def test_journal_contract_is_idempotent_indexable_and_silent():
    from orion.curiosity.compaction import journal_entry
    from orion.journaler import build_journal_entry_index_payload
    from orion.journaler.dispatch_registry import resolve_policy
    r = row()
    r["state"]["body"] = BODY
    r["state"]["topics"] = [{"topic": "Delivery", "body": BODY, "sources": ["j1"]}]
    entry = journal_entry(r)
    assert entry.entry_id == journal_entry(r).entry_id
    assert entry.source_ref.startswith("curiosity-compaction:")
    assert not entry.source_ref.startswith("curiosity:")  # cannot ingest itself
    assert "curiosity:r1 (journal j1)" in entry.body
    assert build_journal_entry_index_payload(entry)["body"] == entry.body
    assert not resolve_policy(entry.trigger_kind or "").email_enabled


def test_readout_public_path_on_full_app(monkeypatch):
    from fastapi.testclient import TestClient
    from app import curiosity_compaction_store, main

    seen = []
    def recent(limit):
        seen.append(limit)
        return [{"compaction_id": "weekly:test", "status": "pending"}]
    monkeypatch.setattr(curiosity_compaction_store, "recent", recent)
    # No lifespan: exercise the actual mounted HTTP route without starting workers.
    client = TestClient(main.app)
    response = client.get("/dreams/curiosity-compactions?limit=2")
    assert response.status_code == 200
    assert response.json() == {"items": [{"compaction_id": "weekly:test", "status": "pending"}]}
    assert seen == [2]
    assert client.get("/dreams/curiosity-compactions?limit=101").status_code == 422


def test_readout_unavailable_is_503(monkeypatch):
    from app import curiosity_compaction_store, dream_api
    from fastapi import HTTPException
    def unavailable(limit):
        raise RuntimeError("missing migration")
    monkeypatch.setattr(curiosity_compaction_store, "recent", unavailable)
    with pytest.raises(HTTPException) as exc:
        dream_api.curiosity_compactions(20)
    assert exc.value.status_code == 503


@pytest.mark.parametrize("block_at", ["acquire", "prepare"])
def test_shutdown_waits_for_db_worker_and_closes_late_session(block_at):
    import threading
    from app.curiosity_compaction_submit import submit_due
    async def run():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        events = []
        def block():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            events.append("worker-finished")
        class Session:
            def prepare(self, *args):
                if block_at == "prepare":
                    block()
            def unjournaled(self):
                return []
            def close(self):
                events.append("closed")
        class Store:
            def acquire(self, cadence):
                if block_at == "acquire":
                    block()
                return Session()
        task = asyncio.create_task(submit_due(Store(), None, None, now=NOW, cycle_id="dc-test"))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["worker-finished", "closed"]
    asyncio.run(run())


def test_dates_survive_topic_reduction_and_overall_even_if_model_omits_dates():
    from orion.curiosity.compaction import source_material, record_answer, next_prompt, journal_entry
    from datetime import timedelta
    # IDs intentionally oppose chronology.
    journals = [{"entry_id": "z-early", "source_ref": "curiosity:early", "body": BODY, "created_at": NOW},
                {"entry_id": "a-late", "source_ref": "curiosity:late", "body": BODY, "created_at": NOW + timedelta(days=1)}]
    material, sources = source_material(journals, [])
    r = row(material)
    r["state"]["sources"] = sources
    r = record_answer(r, answer_for(r))
    assert r["state"]["topics"][0]["sources"] == ["z-early", "a-late"]
    prompt = next_prompt(r)
    assert NOW.isoformat() in prompt and (NOW + timedelta(days=1)).isoformat() in prompt
    r = record_answer(r, answer_for(r))
    output = journal_entry(r).body
    assert output.index("curiosity:early") < output.index("curiosity:late")


def test_due_window_submission_uses_durable_ingress_without_narrative_prompt():
    from types import SimpleNamespace
    from app.curiosity_compaction_submit import request_for
    from app.carry_submit import submit_via_cortex
    from orion.core.bus.bus_schemas import ServiceRef
    r = row()
    r["run_id"] = r["compaction_id"] + ":r0"
    req = request_for(r)
    sent = []
    class Bus:
        async def rpc_request(self, channel, envelope, **kwargs):
            sent.append(envelope.payload)
            return {"data": "receipt"}
        codec = SimpleNamespace(decode=lambda raw: SimpleNamespace(ok=True, envelope=SimpleNamespace(payload={
            "status": "accepted", "metadata": {"durable_run": {"run_id": req.run_id,
                "workflow_kind": "curiosity.compaction", "requested_resource": "llm.route.agent"}}})))
    result = asyncio.run(submit_via_cortex(bus=Bus(), source=ServiceRef(name="orion-dream", version="test", node="test"),
                                         request=req, request_channel="orion:cortex:request",
                                         user_text="Consolidate the due Curiosity calendar window."))
    assert result is None
    assert sent[0]["context"]["metadata"]["durable_run"]["workflow"] == "curiosity.compaction"
    assert "words and pictures" not in json.dumps(sent)


def test_distinct_topic_areas_keep_their_own_sources_before_overall_synthesis():
    from orion.curiosity.compaction import source_material, record_answer, next_prompt, journal_entry
    material, sources = source_material([
        {"entry_id": "delivery", "source_ref": "curiosity:delivery", "body": BODY, "created_at": NOW},
        {"entry_id": "self-inquiry", "source_ref": "curiosity:self-inquiry", "body": BODY, "created_at": NOW},
    ], [])
    r = row(material)
    r["state"]["sources"] = sources
    r = record_answer(r, json.dumps({"topics": [
        {"topic": "Delivery reliability", "body": BODY, "refs": ["s0"]},
        {"topic": "Self-understanding", "body": BODY, "refs": ["s1"]},
    ]}))
    assert {t["topic"]: t["sources"] for t in r["state"]["topics"]} == {
        "Delivery reliability": ["delivery"], "Self-understanding": ["self-inquiry"]}
    assert "Delivery reliability" in next_prompt(r) and "Self-understanding" in next_prompt(r)
    r = record_answer(r, json.dumps({"body": BODY}))
    journal = journal_entry(r).body
    assert "## Delivery reliability" in journal and "## Self-understanding" in journal


def test_overall_body_cut_off_at_its_cap_is_rejected():
    from orion.cognition.model_output import ModelOutputRejected
    from orion.curiosity.compaction import OUTPUT_CHARS, record_answer
    r = row()
    r["state"]["phase"] = "overall"
    with pytest.raises(ModelOutputRejected, match="cut off"):
        record_answer(r, json.dumps({"body": ((BODY + " ") * 30)[:OUTPUT_CHARS]}))


def test_a_full_pass_the_schema_allows_fits_the_pass_budget():
    """Review finding: 6 x 1200-char topics could only ever be rejected as over budget."""
    from orion.curiosity.compaction import MAX_TOPICS, PASS_BUDGET, TOPIC_BODY_CHARS
    from orion.cognition.compactor.chunking import json_char_len
    full = {"topics": [{"topic": "t" * 100, "body": "b" * TOPIC_BODY_CHARS, "refs": ["s0", "s1", "s2"]}
                       for _ in range(MAX_TOPICS)]}
    assert MAX_TOPICS >= 1 and json_char_len(full) <= PASS_BUDGET


def test_reduction_that_does_not_shrink_is_not_a_retryable_model_rejection():
    from orion.cognition.model_output import ModelOutputRejected
    from orion.curiosity import compaction
    r = row()
    r["state"]["batches"] = r["state"]["batches"] * 2
    r["state"]["partials"] = [[{"topic": "T", "body": BODY, "sources": ["j1"],
                                "first_at": NOW.isoformat(), "last_at": NOW.isoformat()}]]
    orig = compaction.batches
    compaction.batches = lambda material: [[m] for m in material] * 3
    try:
        with pytest.raises(ValueError, match="did not shrink") as exc:
            compaction.record_answer(r, answer_for(r))
        assert not isinstance(exc.value, ModelOutputRejected)
    finally:
        compaction.batches = orig
