"""Gate tests for the Graphiti trial scripts (scripts/graphiti_trial*.py): pure helpers and scoring.

No graphiti-core, FalkorDB, Postgres or network needed.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import graphiti_trial as trial  # noqa: E402
import graphiti_trial_clients as clients  # noqa: E402
import graphiti_trial_score as score  # noqa: E402


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# --- clients ----------------------------------------------------------------------------------

def test_strip_reasoning_drops_think_block_and_fence():
    assert clients.strip_reasoning('<think>hmm {"a":1}</think>\n```json\n{"b": 2}\n```') == '{"b": 2}'


def test_strip_reasoning_keeps_text_after_unclosed_think():
    assert clients.strip_reasoning('reasoning here</think>{"x": 1}') == '{"x": 1}'


def test_parse_json_content_refuses_empty_and_non_object():
    with pytest.raises(ValueError, match="empty"):
        clients.parse_json_content("<think>only thinking</think>")
    with pytest.raises(ValueError, match="not_object"):
        clients.parse_json_content("[1, 2]")
    assert clients.parse_json_content('{"ok": true}') == {"ok": True}


def test_response_format_uses_json_schema_when_model_given():
    fmt = clients.response_format_for({"type": "object"}, "ExtractedEntities")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "ExtractedEntities"
    assert fmt["json_schema"]["schema"] == {"type": "object"}
    assert clients.response_format_for(None) == {"type": "json_object"}
    assert clients.response_format_for({"type": "object"}, "bad name!")["json_schema"]["name"] == "bad_name_"


def test_call_stats_diff_is_per_episode():
    s = clients.CallStats(calls=3, llm_seconds=1.5)
    before = s.snapshot()
    s.calls += 2
    s.llm_seconds += 0.25
    d = s.diff(before)
    assert d["calls"] == 2 and d["llm_seconds"] == 0.25 and d["failures"] == 0


def test_header_box_sets_and_clears_hold_header():
    box = clients.HeaderBox()
    box.set_hold("abc")
    assert box.headers == {"X-Orion-Gpu-Lease": "abc"}
    box.set_hold(None)
    assert box.headers == {}


def test_embedding_request_and_strict_response():
    req = clients.embedding_request("hello")
    assert req["text"] == "hello" and req["embedding_profile"] == "default"
    good = {"doc_id": "x", "embedding": [0.1] * 4, "embedding_model": "m", "embedding_dim": 4}
    assert clients.embedding_from_response(good, 4) == [0.1] * 4
    with pytest.raises(ValueError, match="dim_mismatch"):
        clients.embedding_from_response(good, 1024)
    with pytest.raises(ValueError):
        clients.embedding_from_response({"doc_id": "x", "embedding": None, "embedding_model": "m",
                                         "embedding_dim": 0}, 4)


# --- episodes ---------------------------------------------------------------------------------

def _mem(i, body, at, purpose="happened"):
    return {"id": f"00000000-0000-0000-0000-00000000000{i}", "purpose": purpose, "voice": "juniper_said",
            "body": body, "at": at, "status": "active"}


def test_episodes_from_rows_merges_sources_in_time_order():
    mem = [_mem(2, "Juniper is in Chicago.", _t("2026-10-05T02:55:00")),
           _mem(1, "Juniper corrected me that she lives in Ogden, Utah, not Chicago.", _t("2026-10-06T03:50:00")),
           _mem(3, "   ", _t("2026-10-01T00:00:00"))]
    stance = [{"id": "s1", "body": "We live in Ogden, UT.", "at": "2026-08-15T02:38:40"}]
    eps = trial.episodes_from_rows(mem, stance)
    assert [e.origin for e in eps] == ["stance", "episode_memory", "episode_memory"]
    assert eps[0].body == "Juniper (to Orion): We live in Ogden, UT." and eps[0].source == "message"
    assert eps[1].name == "episode_memory:00000000-0000-0000-0000-000000000002"
    assert eps[1].source == "text" and eps[0].reference_time.tzinfo is not None


def test_select_sample_keeps_every_key_row_and_time_order():
    mem = [_mem(i, f"filler statement {i}", _t(f"2026-10-{i + 1:02d}T00:00:00")) for i in range(9)]
    mem.append({**_mem(9, "el Coco is a dog, not a cat", _t("2026-11-11T20:30:00")), "id": "coco"})
    eps = trial.episodes_from_rows(mem, [])
    picked = trial.select_sample(eps, 4)
    assert len(picked) == 4
    assert any(e.uuid == "coco" for e in picked)
    assert picked == sorted(picked, key=lambda e: e.reference_time)
    assert trial.select_sample(eps, 4) == picked  # seeded


def test_progress_line_has_rate_eta_and_errors():
    line = trial.progress_line(done=10, total=20, skipped=2, errors=1, elapsed=3600, processed=8, title="t")
    assert "50.0%" in line and "10/20 (skipped 2)" in line and "8.0 ep/h" in line
    assert "errors 1" in line and "ETA 1h15m00s" in line


def test_run_refuses_any_graph_but_trial():
    with pytest.raises(SystemExit, match="refusing graph"):
        trial.main(["run", "--graph", "graphiti_temporal"])


# --- scoring ----------------------------------------------------------------------------------

def _edge(fact, valid=None, invalid=None, expired=None, source="Juniper", target="X"):
    return {"fact": fact, "valid_at": valid, "invalid_at": invalid, "expired_at": expired,
            "source": source, "target": target, "name": "REL"}


def test_ogden_current_needs_a_live_residence_fact():
    assert score.check_ogden_current([_edge("Juniper lives in Ogden, Utah", target="Ogden")]).passed
    assert not score.check_ogden_current([_edge("Juniper lives in Ogden", invalid="2026-10-07")]).passed
    assert not score.check_ogden_current([_edge("Juniper visited Ogden once")]).passed
    # the real correction row's shape: a negation that governs Chicago, not Ogden
    assert score.check_ogden_current([_edge("Juniper lives in Ogden, Utah, not Chicago")]).passed
    assert not score.check_ogden_current([_edge("Juniper no longer lives in Ogden")]).passed


def test_chicago_home_fails_only_on_a_current_home_fact():
    stay = _edge("Juniper is staying at the Wade hotel in Chicago", target="Chicago")
    assert score.check_chicago_home([stay]).passed
    assert score.check_chicago_home([_edge("Juniper lives in Chicago", invalid="2026-10-06T03:50:00")]).passed
    assert score.check_chicago_home([_edge("Juniper lives in Ogden, not Chicago")]).passed
    assert not score.check_chicago_home([_edge("Juniper lives in Chicago", target="Chicago")]).passed
    assert not score.check_chicago_home([_edge("Juniper lives in Chicago, not Ogden")]).passed
    assert not score.check_chicago_home([_edge("Juniper lives in Chicago instead of Utah")]).passed
    assert score.check_chicago_home([_edge("Juniper flew home from Chicago to SLC")]).passed


def test_chicago_stay_needs_open_and_close_inside_trip():
    closed = _edge("Juniper is staying in Chicago for a team meeting", valid="2026-10-05T02:55:00",
                   invalid="2026-10-07T12:00:00")
    open_ = _edge("Juniper is staying in Chicago", valid="2026-10-05T02:55:00")
    late = _edge("Juniper is staying in Chicago", valid="2026-10-05T02:55:00", invalid="2026-11-01T00:00:00")
    assert score.check_chicago_stay([closed]).passed
    assert not score.check_chicago_stay([open_]).passed
    assert not score.check_chicago_stay([late]).passed


def test_coco_dog_current_and_cat_not():
    dog = _edge("el Coco is Juniper's dog", source="el Coco", target="dog")
    correction = _edge("el Coco is a dog, not a cat", source="el Coco")
    cat = _edge("el Coco is a cat", source="el Coco")
    assert score.check_coco_dog([dog, correction]).passed
    assert score.check_coco_dog([dog, {**cat, "invalid_at": "2026-11-11T20:30:00"}]).passed
    assert not score.check_coco_dog([dog, cat]).passed
    assert not score.check_coco_dog([correction | {"fact": "Juniper owns el Coco"}]).passed
    assert score.check_coco_dog([dog, _edge("Juniper has two cats and a dog named el Coco")]).passed
    assert score.check_coco_dog([dog, _edge("el Coco lives with the cats")]).passed
    assert not score.check_coco_dog([dog, _edge("el Coco is Juniper's cat")]).passed


def test_one_entity_counts_alias_nodes():
    nodes = [{"name": "Amanda", "uuid": "1"}, {"name": "Juniper's wife", "uuid": "2"}, {"name": "el Coco", "uuid": "3"}]
    assert not score.check_one_entity(nodes, score.AMANDA_ALIAS_RE, "Amanda").passed
    assert score.check_one_entity(nodes, score.COCO_RE, "Coco").passed
    assert not score.check_one_entity(nodes + [{"name": "Coco", "uuid": "4"}], score.COCO_RE, "Coco").passed


def test_duplicate_sample_and_rate():
    nodes = [{"uuid": str(i), "name": f"thing {i}", "summary": ""} for i in range(60)]
    sample = score.duplicate_sample(nodes, size=50)
    assert len(sample) == 50 and all(s["duplicate"] is None for s in sample)
    assert score.duplicate_sample(nodes, size=50) == sample
    assert score.duplicate_rate(sample) == (None, 0, 0)
    judged = [{**s, "duplicate": i < 4} for i, s in enumerate(sample)]
    assert score.duplicate_rate(judged) == (4 / 50, 4, 50)
    with pytest.raises(ValueError, match="true/false"):
        score.duplicate_rate([{"name": "x", "duplicate": "yes"}])


def test_run_checks_duplicate_gate_needs_full_judged_sample():
    nodes = [{"uuid": str(i), "name": f"n{i}"} for i in range(60)]
    half = [{"duplicate": False}] * 20
    dup_check = score.run_checks(nodes, [], half)[-1]
    assert not dup_check.passed
    full = [{"duplicate": False}] * 50
    assert score.run_checks(nodes, [], full)[-1].passed


def test_cost_summary_counts_every_attempt_but_last_outcome():
    rows = [{"uuid": "a", "ok": False, "wall_sec": 30, "lease_sec": 30, "llm": {"calls": 3, "failures": 2}},
            {"uuid": "a", "ok": True, "wall_sec": 10, "lease_sec": 10, "llm": {"calls": 5, "retries": 1}},
            {"uuid": "b", "ok": True, "wall_sec": 20, "lease_sec": 20, "llm": {"calls": 4}}]
    c = score.cost_summary(rows)
    assert c["episodes"] == 2 and c["attempts"] == 3 and c["ok"] == 2 and c["failed"] == 0
    assert c["wall_sec_total"] == 60 and c["llm_calls"] == 12 and c["llm_calls_per_episode"] == 6
    assert c["graphiti_retries"] == 1 and c["llm_failures"] == 2


# --- graphiti subclasses (run where graphiti-core is installed: CI and the adapter image) ------

class _FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        from types import SimpleNamespace as NS
        return NS(choices=[NS(finish_reason=r.get("finish", "stop"), message=NS(content=r["content"]))],
                  usage=NS(prompt_tokens=10, completion_tokens=5))


def _llm(replies):
    pytest.importorskip("graphiti_core")
    from types import SimpleNamespace as NS
    box, stats = clients.HeaderBox(), clients.CallStats()
    box.headers["X-Orion-Gpu-Lease"] = "ref"
    llm = clients.build_llm_client(base_url="http://x/v1", model="agent", header_box=box, stats=stats)
    fake = _FakeCompletions(replies)
    llm.client = NS(chat=NS(completions=fake))
    return llm, fake, stats


def _bad_request():
    import httpx
    import openai
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return openai.BadRequestError("json_schema not supported", response=httpx.Response(400, request=req), body=None)


def test_llm_client_sends_json_schema_and_hold_header():
    pytest.importorskip("graphiti_core")
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Out(BaseModel):
        name: str

    llm, fake, stats = _llm([{"content": '<think>x</think>{"name": "el Coco"}'}])
    out = asyncio.run(llm.generate_response([Message(role="system", content="s"), Message(role="user", content="u")], Out))
    assert out == {"name": "el Coco"}
    call = fake.calls[0]
    assert call["model"] == "agent" and call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"]["properties"]["name"]["type"] == "string"
    assert call["extra_headers"] == {"X-Orion-Gpu-Lease": "ref"}
    assert stats.calls == 1 and stats.prompt_tokens == 10 and stats.failures == 0


def test_llm_client_falls_back_to_json_object_when_schema_refused():
    pytest.importorskip("graphiti_core")
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Out(BaseModel):
        name: str

    llm, fake, stats = _llm([_bad_request(), {"content": '{"name": "Amanda"}'}])
    assert asyncio.run(llm.generate_response([Message(role="user", content="u")], Out)) == {"name": "Amanda"}
    assert fake.calls[1]["response_format"] == {"type": "json_object"} and stats.schema_fallbacks == 1


def test_llm_client_counts_graphiti_retries():
    pytest.importorskip("graphiti_core")
    from graphiti_core.prompts.models import Message

    llm, fake, stats = _llm([{"content": "not json"}, {"content": '{"a": 1}'}])
    assert asyncio.run(llm.generate_response([Message(role="user", content="u")])) == {"a": 1}
    assert stats.calls == 2 and stats.failures == 1 and stats.retries == 1


def test_llm_client_does_not_retry_truncation_or_a_dead_hold():
    pytest.importorskip("graphiti_core")
    from graphiti_core.llm_client.errors import RefusalError
    from graphiti_core.prompts.models import Message

    llm, fake, stats = _llm([{"content": "{}", "finish": "length"}])
    with pytest.raises(RefusalError):
        asyncio.run(llm.generate_response([Message(role="user", content="u")]))
    assert stats.calls == 1 and stats.retries == 0 and "truncated" in stats.errors[0]

    from types import SimpleNamespace as NS
    stats2 = clients.CallStats()
    dead = clients.build_llm_client(base_url="http://x/v1", model="agent", header_box=clients.HeaderBox(),
                                    stats=stats2, hold_usable=lambda: False)
    dead.client = NS(chat=NS(completions=_FakeCompletions([])))
    with pytest.raises(RefusalError):
        asyncio.run(dead.generate_response([Message(role="user", content="u")]))
    assert stats2.calls == 0


def test_llm_client_does_not_fall_back_on_a_non_schema_400():
    pytest.importorskip("graphiti_core")
    import httpx
    import openai
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Out(BaseModel):
        name: str

    req = httpx.Request("POST", "http://x/v1/chat/completions")
    ctx = openai.BadRequestError("exceeds the available context size", response=httpx.Response(400, request=req),
                                 body=None)
    llm, fake, stats = _llm([ctx, ctx, ctx])
    with pytest.raises(openai.BadRequestError):
        asyncio.run(llm.generate_response([Message(role="user", content="u")], Out))
    assert stats.schema_fallbacks == 0


# --- resume, isolation, schema refusal ----------------------------------------------------------

def test_resume_plan_redoes_episodes_without_an_ok_row(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text('{"uuid": "1", "name": "episode_memory:1", "ok": true}\n'
                 '{"uuid": "2", "name": "episode_memory:2", "ok": false}\n')
    ok = trial.ok_names_from_results(p)
    done, partial = trial.resume_plan({"episode_memory:1": "u1", "episode_memory:2": "u2",
                                       "episode_memory:3": "u3"}, ok)
    assert done == {"episode_memory:1"}
    assert sorted(partial) == ["u2", "u3"]
    assert trial.ok_names_from_results(tmp_path / "missing.jsonl") == set()


def test_isolation_verdict_flags_new_graphs_and_leaked_group_nodes():
    before = {"graphs": ["graphiti_temporal", "orion_worldview"], "trial_group_nodes_elsewhere": {}}
    clean = {"graphs": ["graphiti_temporal", "graphiti_trial", "orion_worldview"], "trial_group_nodes_elsewhere": {}}
    assert trial.isolation_verdict(before, clean)["isolated"]
    leaked = {**clean, "trial_group_nodes_elsewhere": {"graphiti_temporal": 3}}
    assert not trial.isolation_verdict(before, leaked)["isolated"]
    extra = {**clean, "graphs": clean["graphs"] + ["orion_trial"]}
    v = trial.isolation_verdict(before, extra)
    assert not v["isolated"] and v["new_graphs_other_than_trial"] == ["orion_trial"]


def test_is_schema_refusal_only_for_structured_output_errors():
    assert clients.is_schema_refusal("Error 400: failed to parse json_schema into grammar")
    assert not clients.is_schema_refusal("Error 400: the request exceeds the available context size")


# --- hold lifecycle (pool verbs monkeypatched) ---------------------------------------------------

class _Reply:
    def __init__(self, status, lease_id="L1", generation=1, role="agent"):
        from orion.schemas.gpu_pool import GpuLeaseGrantV1
        self.status, self.lease_id, self.reason, self.position, self.recall_by = status, lease_id, None, None, None
        self.grant = (GpuLeaseGrantV1(lease_id=lease_id, generation=generation, role=role, cards=["gpu1"],
                                      url="http://x", served_by="x")
                      if status in ("granted", "recall") else None)


@pytest.fixture
def pool(monkeypatch, tmp_path):
    import orion.gpu_pool.client as gpc
    log = {"acquire": [], "status": [], "cancel": [], "release": []}
    script = {"acquire": [], "status": []}

    async def acquire_hold(bus, **kw):
        log["acquire"].append(kw["request_id"])
        r = script["acquire"].pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    async def lease_status(bus, lease_id, **kw):
        log["status"].append(lease_id)
        r = script["status"].pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    async def release_lease(bus, lease_id, **kw):
        log["release"].append(lease_id)

    async def cancel(bus, lease_id, *, source):
        log["cancel"].append(lease_id)

    async def fast_sleep(_s):
        return None

    monkeypatch.setattr(gpc, "acquire_hold", acquire_hold)
    monkeypatch.setattr(gpc, "lease_status", lease_status)
    monkeypatch.setattr(gpc, "release_lease", release_lease)
    monkeypatch.setattr(trial, "cancel_lease", cancel)
    monkeypatch.setattr(trial, "OUT_DIR", tmp_path)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(trial.asyncio, "sleep", lambda s: real_sleep(0))
    hold = trial.TrialHold(object(), "r1", clients.HeaderBox(), min_ctx_tokens=0, out_dir=tmp_path)
    return hold, script, log, tmp_path


def test_hold_granted_after_queue_then_released(pool):
    hold, script, log, tmp = pool
    script["acquire"] = [_Reply("queued")]
    script["status"] = [_Reply("queued"), _Reply("granted")]

    async def go():
        await hold.acquire()
        assert hold.lease_id == "L1" and "X-Orion-Gpu-Lease" in hold.box.headers
        assert (tmp / "hold.json").exists()
        await hold.release()
    asyncio.run(go())
    assert log["release"] == ["L1"] and log["cancel"] == []
    assert not (tmp / "hold.json").exists() and hold.box.headers == {}


def test_hold_cancelled_when_wait_is_cancelled(pool):
    hold, script, log, tmp = pool
    script["acquire"] = [_Reply("queued")]
    script["status"] = [asyncio.CancelledError()]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(hold.acquire())
    assert log["cancel"] == ["L1"] and not (tmp / "hold.json").exists()


def test_hold_cancelled_when_refused_or_released_elsewhere(pool):
    hold, script, log, _ = pool
    script["acquire"] = [_Reply("queued")]
    script["status"] = [_Reply("ok")]
    with pytest.raises(RuntimeError, match="hold_not_granted:ok"):
        asyncio.run(hold.acquire())
    assert log["cancel"] == ["L1"]


def test_hold_learns_lease_id_when_first_acquire_rpc_fails(pool):
    hold, script, log, _ = pool
    script["acquire"] = [TimeoutError("rpc"), _Reply("queued", lease_id="L9")]
    with pytest.raises(TimeoutError):
        asyncio.run(hold.acquire())
    assert log["acquire"] == ["r1:1", "r1:1"]  # same request_id: idempotent re-ask
    assert log["cancel"] == ["L9"]


def test_recall_reacquires_with_a_new_request_id(pool):
    hold, script, log, _ = pool
    script["acquire"] = [_Reply("granted"), _Reply("granted", lease_id="L2")]

    async def go():
        await hold.acquire()
        hold.recalled.set()
        assert not hold.usable()
        await hold.release()
        await hold.acquire()
        assert hold.usable() and hold.lease_id == "L2"
        await hold.release()
    asyncio.run(go())
    assert log["acquire"] == ["r1:1", "r1:2"] and log["release"] == ["L1", "L2"]


def test_leftover_hold_file_is_cancelled(pool):
    hold, _, log, tmp = pool
    (tmp / "hold.json").write_text('{"lease_id": "OLD", "holder": "graphiti-trial:x"}')
    asyncio.run(hold.cancel_leftover())
    assert log["cancel"] == ["OLD"] and not (tmp / "hold.json").exists()
