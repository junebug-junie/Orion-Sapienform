"""Curiosity finding register (event wants step 2): judge on the curiosity lane, novelty, one insert-only row."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.curiosity_finding_register as reg
import scripts.curiosity_investigation as ci
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.journal import INVESTIGATION_TAG
from orion.curiosity.register import CORPUS_VERSION, JUDGE_PROMPT_SHA256, corpus_items
from orion.curiosity.self_inquiry import SELF_INQUIRY_TAG
from test_curiosity_investigation import _FakeBus, _FakeReader, _graph_loop

MIGRATION = Path(__file__).resolve().parents[3] / "services/orion-sql-db/manual_migration_curiosity_finding_register_v1.sql"
SRC = ServiceRef(name="hub", version="t", node="athena")
T0 = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


class _Conn:
    def __init__(self, store: "_Store") -> None:
        self.s = store

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if sql == reg.CORPUS_READ_SQL:
            keys = set(zip(args[0], args[1]))
            return [{"ref": r, "text_sha256": h, "embedding": v} for (r, h), (m, v) in self.s.corpus.items()
                    if (r, h) in keys and m == args[2]]
        if sql == reg.LETTERS_SQL:
            return self.s.letters
        if sql == reg.QUESTIONS_SQL:
            return [{"question_id": "lived.alive", "text": "Am I alive?"}]
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        if sql == reg.REGISTER_EXISTS_SQL:
            return 1 if args[0] in self.s.rows else None
        if sql == reg.PER_PRIOR_SQL:
            return self.s.per_prior
        raise AssertionError(sql)

    async def executemany(self, sql, rows):
        assert sql == reg.CORPUS_INSERT_SQL
        self.s.batches.append(len(rows))
        for args in rows:
            self.s.corpus.setdefault((args[0], args[1]), (args[5], list(args[6])))
            self.s.inserted_corpus.append(args[0])

    async def execute(self, sql, *args):
        if sql == reg.INSERT_REGISTER_SQL:
            if args[0] in self.s.rows:
                return "INSERT 0 0"
            self.s.rows[args[0]] = args
            return "INSERT 0 1"
        raise AssertionError(sql)


class _Store:
    def __init__(self) -> None:
        self.corpus: dict = {}
        self.rows: dict = {}
        self.inserted_corpus: list = []
        self.batches: list = []
        self.letters: list = []
        self.per_prior = None

    def acquire(self):
        return _Conn(self)


def _vec(*xs):
    return [float(x) for x in xs]


EMB = {"I am a pattern that keeps choosing care.": _vec(1, 0, 0), "Redis was down.": _vec(0, 1, 0),
       "Care is what I keep choosing.": _vec(0.8, 0.6, 0)}


async def _embed(text):
    return EMB[text], "bge-large-en-v1.5"


def _items():
    return corpus_items({"finding": [{"run_id": "r_old", "text": "Redis was down."},
                                     {"run_id": "r_now", "text": "Care is what I keep choosing."}],
                         "prior": [{"prior_id": "p1", "run_id": "r_old", "text": "I am a pattern that keeps choosing care."}]},
                        [])


def _judge(register="philosophical", model="qwen3.8-flash-next-iq3_s", error=None):
    async def j():
        return reg.JudgeResult(register, "about care" if register else None, error, model)
    return j


def test_migration_is_additive_and_names_its_consumer():
    from orion.sql_migration_drift import auto_apply_refusals, parse_required_by

    sql = MIGRATION.read_text()
    rb = parse_required_by(sql)
    assert rb is not None and not rb.errors
    assert auto_apply_refusals(sql, existing_tables=set()) == []  # new tables: post-pull auto-apply accepts it


def test_a_finding_is_judged_scored_against_older_writing_and_written_once():
    s = _Store()
    kw = dict(pool=s, run_id="r_now", line="self_inquiry", question="Am I alive?", question_ref="self_question:lived.alive",
              answer="Care is what I keep choosing.", items=_items(), started_at=T0, judge=_judge(), embed=_embed,
              route="agent")
    row = asyncio.run(reg.score_and_record(**kw))
    assert row["register"] == "philosophical" and row["judge_error"] is None
    # nearest earlier writing is the prior (cos 0.8); this run's own Finding node is not in the corpus.
    assert math.isclose(row["novelty"], 0.2, abs_tol=1e-6) and row["nearest_ref"] == "prior:p1"
    assert row["corpus_size"] == 2
    stored = s.rows["r_now"]
    assert stored[7] == JUDGE_PROMPT_SHA256 and stored[14] == CORPUS_VERSION and stored[6] == "agent"
    assert stored[13] == "bge-large-en-v1.5" and s.batches == [2]
    assert sorted(s.inserted_corpus) == ["finding:r_old", "prior:p1"]
    # a second turn end for the same run (retry) never re-judges
    assert asyncio.run(reg.score_and_record(**kw)) is None


def test_the_corpus_embeds_only_new_text():
    s = _Store()
    items = _items()
    asyncio.run(reg.sync_corpus(s, items[:1], _embed, model="bge-large-en-v1.5"))
    s.inserted_corpus.clear()
    asyncio.run(reg.sync_corpus(s, items, _embed, model="bge-large-en-v1.5"))
    assert sorted(s.inserted_corpus) == sorted(i["ref"] for i in items[1:])


def test_an_embed_failure_leaves_novelty_unknown_not_high():
    async def broken(text):
        raise ConnectionError("vector-host down")

    s = _Store()
    row = asyncio.run(reg.score_and_record(
        pool=s, run_id="r_now", line="investigate", question="q", question_ref=None, answer="Care is what I keep choosing.",
        items=_items(), started_at=T0, judge=_judge(), embed=broken, route="agent", retry_delays=(0.0,)))
    assert row["novelty"] is None and "vector-host down" in row["novelty_error"]
    assert row["register"] == "philosophical"


def test_a_cached_vector_from_another_embedding_model_is_not_reused():
    s = _Store()
    s.corpus[("prior:p1", _items()[2]["sha"])] = ("old-model", [0.0, 0.0, 1.0])
    row = asyncio.run(reg.score_and_record(
        pool=s, run_id="r_now", line="investigate", question="q", question_ref=None, answer="Care is what I keep choosing.",
        items=_items(), started_at=T0, judge=_judge(), embed=_embed, route="agent"))
    assert math.isclose(row["novelty"], 0.2, abs_tol=1e-6)  # re-embedded under bge, not cos 0 vs the old vector


def test_a_corpus_of_another_vector_size_is_an_error_not_a_score():
    async def short(text):
        return ([1.0, 0.0] if text == "Redis was down." else EMB[text]), "bge-large-en-v1.5"

    s = _Store()
    row = asyncio.run(reg.score_and_record(
        pool=s, run_id="r_now", line="investigate", question="q", question_ref=None, answer="Care is what I keep choosing.",
        items=_items(), started_at=T0, judge=_judge(), embed=short, route="agent", retry_delays=()))
    assert row["novelty"] is None and row["novelty_error"] == "dimension_mismatch"


def test_a_transient_judge_failure_is_retried_before_the_row_is_written():
    calls = []

    async def judge():
        calls.append(1)
        if len(calls) == 1:
            return reg.JudgeResult(None, None, "call: gpu_pool_unavailable", None)
        return reg.JudgeResult("operational", "plumbing", None, "qwen3.8-flash-next-iq3_s")

    s = _Store()
    row = asyncio.run(reg.score_and_record(
        pool=s, run_id="r_now", line="investigate", question="q", question_ref=None, answer="Redis was down.",
        items=_items(), started_at=T0, judge=judge, embed=_embed, route="agent", retry_delays=(0.0, 0.0)))
    assert len(calls) == 2 and row["register"] == "operational" and row["judge_error"] is None


def test_a_model_verdict_is_not_retried():
    calls = []

    async def judge():
        calls.append(1)
        return reg.JudgeResult(None, None, "model_not_evaluated: bonsai", "bonsai")

    s = _Store()
    asyncio.run(reg.score_and_record(
        pool=s, run_id="r_now", line="investigate", question="q", question_ref=None, answer="Redis was down.",
        items=_items(), started_at=T0, judge=judge, embed=_embed, route="agent", retry_delays=(0.0, 0.0)))
    assert len(calls) == 1


def test_no_answer_is_recorded_as_absent_without_a_judge_call():
    called = []

    async def judge():
        called.append(1)
        return reg.JudgeResult("philosophical", None, None, "x")

    s = _Store()
    row = asyncio.run(reg.score_and_record(pool=s, run_id="r1", line="investigate", question="q", question_ref=None,
                                           answer="", items=[], started_at=T0, judge=judge, embed=_embed, route="agent"))
    assert called == [] and row["register"] is None and row["judge_error"] == "no_answer"


class _RpcBus:
    def __init__(self, content, model="qwen3.8-flash-next-iq3_s", raises=None, raw=None) -> None:
        from orion.core.bus.codec import OrionCodec

        self.codec = OrionCodec()
        self.content, self.model, self.raises, self.raw = content, model, raises, raw or {}
        self.sent = []

    async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
        self.sent.append((channel, env, reply_channel))
        if self.raises:
            raise self.raises
        reply = BaseEnvelope(kind="llm.chat.result", source=SRC, correlation_id=env.correlation_id,
                             payload={"content": self.content, "model_used": self.model, "raw": self.raw})
        return {"data": self.codec.encode(reply)}


def _call(bus):
    return asyncio.run(reg.judge_finding(bus, intake_channel="orion:exec:request:LLMGatewayService",
                                         gateway_service="LLMGatewayService", source_ref=SRC, route="agent",
                                         question="Am I alive?", answer="Care is what I keep choosing."))


def test_the_judge_runs_on_the_curiosity_route_with_the_pinned_prompt():
    bus = _RpcBus('{"register": "philosophical", "reason": "about care"}')
    res = _call(bus)
    assert (res.register, res.reason, res.error) == ("philosophical", "about care", None)
    channel, env, _ = bus.sent[0]
    payload = env.payload
    assert channel == "orion:exec:request:LLMGatewayService" and payload["route"] == "agent"
    assert payload["options"]["temperature"] == 0 and payload["options"]["max_tokens"] == reg.JUDGE_MAX_TOKENS
    assert payload["options"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][0]["role"] == "system" and "FINDING: Care is what I keep choosing." in payload["messages"][1]["content"]


@pytest.mark.parametrize("model", ["/models/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf", None, "qwen3-8b"])
def test_a_reply_from_a_model_that_did_not_pass_the_eval_is_not_a_label(model):
    res = _call(_RpcBus('{"register": "philosophical", "reason": "x"}', model=model))
    assert res.register is None and res.error.startswith("model_not_evaluated")


def test_garbage_and_transport_failures_are_errors_not_operational():
    assert _call(_RpcBus("I think it is philosophical")).error.startswith("not_json")
    res = _call(_RpcBus("", raises=TimeoutError("no reply")))
    assert res.register is None and res.error.startswith("call: TimeoutError") and res.transient


def test_a_gateway_refusal_is_a_call_error_not_a_wrong_model():
    res = _call(_RpcBus("", model=None, raw={"error": "gpu_pool_unavailable"}))
    assert res.error == "call: gpu_pool_unavailable" and res.transient


# --- the loop: which turns get a row ---------------------------------------------------------------


def _loop_with_register(monkeypatch, store, judged):
    reader = _FakeReader(answers={
        "MATCH (n:Finding)": [{"run_id": "r_old", "text": "Redis was down."},
                              {"run_id": "run1", "text": "Care is what I keep choosing."}],
        "MATCH (p:Prior)": [{"prior_id": "p1", "run_id": "r_old", "text": "I am a pattern that keeps choosing care."}],
    })
    loop = _graph_loop(_FakeBus(), reader=reader, finding_register_embed_url="http://embed", llm_route="agent")
    loop._pool_provider = lambda: store
    loop.finding_register_embed_url = "http://embed"

    async def fake_judge(bus, **kw):
        judged.append(kw)
        return reg.JudgeResult("philosophical", "about care", None, "qwen3.8-flash-next-iq3_s")

    monkeypatch.setattr(ci, "judge_finding", fake_judge)
    monkeypatch.setattr(ci, "http_embedder", lambda client, url: _embed)
    return loop


def test_an_investigation_turn_gets_a_row_with_its_tested_prior_as_the_question(monkeypatch):
    store, judged = _Store(), []
    store.per_prior = json.dumps([{"prior_id": "p1", "kind": "tested", "nats": 0.2}])
    loop = _loop_with_register(monkeypatch, store, judged)

    async def go():
        loop._schedule_finding_register("run1", INVESTIGATION_TAG, "turn text", T0)
        await asyncio.gather(*loop._register_tasks)

    asyncio.run(go())
    row = store.rows["run1"]
    assert row[1] == "investigate" and row[2] == "prior:p1" and row[3] == "philosophical"
    assert judged[0]["route"] == "agent" and judged[0]["question"] == "I am a pattern that keeps choosing care."
    assert judged[0]["answer"] == "Care is what I keep choosing."


@pytest.mark.parametrize("tag,text,enabled", [("self_sense_eval", "x", True), (SELF_INQUIRY_TAG, "", True),
                                              (SELF_INQUIRY_TAG, "x", False)])
def test_no_row_for_other_turns_empty_turns_or_when_off(monkeypatch, tag, text, enabled):
    store, judged = _Store(), []
    loop = _loop_with_register(monkeypatch, store, judged)
    loop.finding_register_enabled = enabled
    loop._schedule_finding_register("run1", tag, text, T0)
    assert loop._register_tasks == set()


def test_a_letter_written_after_the_run_started_is_not_earlier_writing():
    items = corpus_items({}, [{"run_id": "d1", "created_at": T0 - timedelta(hours=1), "note_md": "Redis was down."},
                              {"run_id": "d2", "created_at": T0 + timedelta(minutes=5), "note_md": "Care is what I keep choosing."}])
    s = _Store()
    row = asyncio.run(reg.score_and_record(pool=s, run_id="r", line="investigate", question="q", question_ref=None,
                                           answer="Care is what I keep choosing.", items=items, started_at=T0,
                                           judge=_judge(), embed=_embed, route="agent"))
    assert row["nearest_ref"] == "letter:d1" and row["corpus_size"] == 1


def test_a_second_attempt_of_a_run_in_flight_is_not_scheduled_again(monkeypatch):
    store, judged = _Store(), []
    loop = _loop_with_register(monkeypatch, store, judged)

    async def go():
        loop._schedule_finding_register("run1", INVESTIGATION_TAG, "turn text", T0)
        loop._schedule_finding_register("run1", INVESTIGATION_TAG, "retry text", T0)
        assert len(loop._register_tasks) == 1
        await asyncio.gather(*loop._register_tasks)

    asyncio.run(go())
    assert len(judged) == 1 and loop._register_inflight == set()
