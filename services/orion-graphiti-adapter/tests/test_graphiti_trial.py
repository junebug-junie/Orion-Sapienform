"""Gate tests for the Graphiti trial scripts (scripts/graphiti_trial*.py): pure helpers and scoring.

No graphiti-core, FalkorDB, Postgres or network needed.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import graphiti_trial as trial  # noqa: E402
import graphiti_trial_clients as clients  # noqa: E402
import graphiti_trial_score as score  # noqa: E402
import graphiti_trial2_inputs as t2  # noqa: E402


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
    mem = [_mem(2, "Juniper is in Shelbyville.", _t("2026-10-05T02:55:00")),
           _mem(1, "Juniper corrected me that she lives in Springfield, Illinois, not Shelbyville.", _t("2026-10-06T03:50:00")),
           _mem(3, "   ", _t("2026-10-01T00:00:00"))]
    stance = [{"id": "s1", "body": "We live in Springfield, IL.", "at": "2026-08-15T02:38:40"}]
    eps = trial.episodes_from_rows(mem, stance)
    assert [e.origin for e in eps] == ["stance", "episode_memory", "episode_memory"]
    assert eps[0].body == "Juniper (to Orion): We live in Springfield, IL." and eps[0].source == "message"
    assert eps[1].name == "episode_memory:00000000-0000-0000-0000-000000000002"
    assert eps[1].source == "text" and eps[0].reference_time.tzinfo is not None


def test_select_sample_keeps_every_key_row_and_time_order():
    mem = [_mem(i, f"filler statement {i}", _t(f"2026-10-{i + 1:02d}T00:00:00")) for i in range(9)]
    mem.append({**_mem(9, "Biscuit is a dog, not a cat", _t("2026-11-11T20:30:00")), "id": "coco"})
    eps = trial.episodes_from_rows(mem, [])
    picked = trial.select_sample(eps, 4)
    assert len(picked) == 4
    assert any(e.uuid == "coco" for e in picked)
    assert picked == sorted(picked, key=lambda e: e.reference_time)
    assert trial.select_sample(eps, 4) == picked  # seeded


def test_plan_episodes_reuses_the_exact_sample_and_refuses_missing_rows():
    mem = [_mem(i, f"statement {i}", _t(f"2026-10-{i + 1:02d}T00:00:00")) for i in range(5)]
    eps = trial.episodes_from_rows(mem, [])
    names = [eps[3].name, eps[1].name]
    assert [e.name for e in trial.plan_episodes(eps, names)] == [eps[1].name, eps[3].name]
    with pytest.raises(SystemExit, match="no longer in the source"):
        trial.plan_episodes(eps, names + ["episode_memory:gone"])


def test_progress_line_has_rate_eta_and_errors():
    line = trial.progress_line(done=10, total=20, skipped=2, errors=1, elapsed=3600, processed=8, title="t")
    assert "50.0%" in line and "10/20 (skipped 2)" in line and "8.0 ep/h" in line
    assert "errors 1" in line and "ETA 1h15m00s" in line


def test_run_refuses_any_graph_but_trial():
    with pytest.raises(SystemExit, match="refusing graph"):
        trial.main(["run", "--graph", "graphiti_temporal"])


T = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "graphiti_trial_terms.example.json").read_text())


# --- scoring ----------------------------------------------------------------------------------

def _edge(fact, valid=None, invalid=None, expired=None, source="Robin", target="X"):
    return {"fact": fact, "valid_at": valid, "invalid_at": invalid, "expired_at": expired,
            "source": source, "target": target, "name": "REL"}


def test_home_current_needs_a_live_residence_fact():
    assert score.check_home_current([_edge("Juniper lives in Springfield, Illinois", target="Springfield")], T).passed
    assert not score.check_home_current([_edge("Juniper lives in Springfield", invalid="2026-10-07")], T).passed
    assert not score.check_home_current([_edge("Juniper visited Springfield once")], T).passed
    # the real correction row's shape: a negation that governs Shelbyville, not Springfield
    assert score.check_home_current([_edge("Juniper lives in Springfield, Illinois, not Shelbyville")], T).passed
    assert not score.check_home_current([_edge("Juniper no longer lives in Springfield")], T).passed


def test_trip_city_home_fails_only_on_a_current_home_fact():
    stay = _edge("Juniper is staying at the Grandview hotel in Shelbyville", target="Shelbyville")
    assert score.check_trip_city_not_home([stay], T).passed
    assert score.check_trip_city_not_home([_edge("Juniper lives in Shelbyville", invalid="2026-10-06T03:50:00")], T).passed
    assert score.check_trip_city_not_home([_edge("Juniper lives in Springfield, not Shelbyville")], T).passed
    assert not score.check_trip_city_not_home([_edge("Juniper lives in Shelbyville", target="Shelbyville")], T).passed
    assert not score.check_trip_city_not_home([_edge("Juniper lives in Shelbyville, not Springfield")], T).passed
    assert not score.check_trip_city_not_home([_edge("Juniper lives in Shelbyville instead of Illinois")], T).passed
    assert score.check_trip_city_not_home([_edge("Juniper flew home from Shelbyville to SLC")], T).passed


def test_trip_stay_needs_open_and_close_inside_trip():
    closed = _edge("Juniper is staying in Shelbyville for a team meeting", valid="2026-10-05T02:55:00",
                   invalid="2026-10-07T12:00:00")
    open_ = _edge("Juniper is staying in Shelbyville", valid="2026-10-05T02:55:00")
    late = _edge("Juniper is staying in Shelbyville", valid="2026-10-05T02:55:00", invalid="2026-11-01T00:00:00")
    assert score.check_trip_stay([closed], T).passed
    still_there = _edge("Robin is staying at a hotel in Shelbyville", valid="2026-10-05T03:07:00", source="Robin")
    assert not score.check_trip_stay([closed, still_there], T).passed
    meeting = _edge("The team meeting is in Shelbyville", valid="2026-10-05T02:55:00", source="team meeting")
    assert score.check_trip_stay([closed, meeting], T).passed
    denial = _edge("Robin does not live in Shelbyville", valid="2026-10-06T03:50:00", source="Robin")
    correction = _edge("Robin corrected me that she lives in Springfield, not Shelbyville", valid="2026-10-06T03:50:00",
                       source="Robin")
    assert score.check_trip_stay([closed, denial, correction], T).passed
    assert not score.check_trip_stay([open_], T).passed
    assert not score.check_trip_stay([late], T).passed


def test_pet_dog_current_and_cat_not():
    dog = _edge("Biscuit is Juniper's dog", source="Biscuit", target="dog")
    correction = _edge("Biscuit is a dog, not a cat", source="Biscuit")
    cat = _edge("Biscuit is a cat", source="Biscuit")
    assert score.check_pet_dog([dog, correction], T).passed
    assert score.check_pet_dog([dog, {**cat, "invalid_at": "2026-11-11T20:30:00"}], T).passed
    assert not score.check_pet_dog([dog, cat], T).passed
    assert not score.check_pet_dog([correction | {"fact": "Juniper owns Biscuit"}], T).passed
    assert score.check_pet_dog([dog, _edge("Juniper has two cats and a dog named Biscuit")], T).passed
    assert score.check_pet_dog([dog, _edge("Biscuit lives with the cats")], T).passed
    assert not score.check_pet_dog([dog, _edge("Biscuit is Juniper's cat")], T).passed


def test_one_entity_counts_alias_nodes():
    nodes = [{"name": "Sam", "uuid": "1"}, {"name": "Juniper's wife", "uuid": "2"}, {"name": "Biscuit", "uuid": "3"}]
    spouse = re.compile(score.word(T["spouse"]) + "|" + score.word(T["spouse_alias"]), re.I)
    pet = re.compile(score.word(T["pet"]), re.I)
    assert not score.check_one_entity(nodes, spouse, "spouse").passed
    assert score.check_one_entity(nodes, pet, "pet").passed
    assert not score.check_one_entity(nodes + [{"name": "Biscuit the dog", "uuid": "4"}], pet, "pet").passed


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
    def dup(judged):
        return next(c for c in score.run_checks(nodes, [], judged, T) if c.name.startswith("duplicate"))
    assert not dup(half).passed
    assert dup([{"duplicate": False}] * 50).passed


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

    llm, fake, stats = _llm([{"content": '<think>x</think>{"name": "Biscuit"}'}])
    out = asyncio.run(llm.generate_response([Message(role="system", content="s"), Message(role="user", content="u")], Out))
    assert out == {"name": "Biscuit"}
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

    llm, fake, stats = _llm([_bad_request(), {"content": '{"name": "Sam"}'}])
    assert asyncio.run(llm.generate_response([Message(role="user", content="u")], Out)) == {"name": "Sam"}
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

    async def lease_rpc(bus, req, **kw):  # operator holds acquire through the raw verb
        assert req.verb == "acquire" and req.kind == "hold" and req.retryable is False
        log.setdefault("classes", []).append(req.work_class)
        return await acquire_hold(bus, request_id=req.request_id)

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
    monkeypatch.setattr(gpc, "lease_rpc", lease_rpc)
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


# --- privacy: terms file + redaction ----------------------------------------------------------

def test_load_terms_requires_every_key(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({**T, "pet": ""}))
    with pytest.raises(SystemExit, match="pet"):
        score.load_terms(p)
    p.write_text(json.dumps(T))
    assert score.load_terms(p)["home_city"] == "Springfield"


def test_redact_replaces_term_values_with_role_placeholders():
    out = score.redact("Robin lives in springfield with Sam, his wife; Biscuit stayed home.", T)
    assert out == "[subject] lives in [home city] with [spouse], his [spouse alias]; [pet] stayed home."


def test_redact_also_masks_denylisted_words():
    class PT:
        _TOKEN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")

        @staticmethod
        def _ascii_fold(t):
            return t

        @staticmethod
        def _h(w):
            return "H" if w == "zorblax" else "x"
    assert score.redact("met Zorblax today", T, {"H"}, PT) == "met [name] today"


def test_key_row_re_uses_terms():
    rx = trial.key_row_re(T)
    assert rx.search("Robin went to Shelbyville") and rx.search("Biscuit barked")
    assert not rx.search("nothing relevant here")
    assert trial.key_row_re(None).search("a dog")


def test_render_report_lists_checks_cost_and_invalidated_facts():
    edges = [_edge("Robin lives in Springfield", valid="2026-08-15T00:00:00"),
             _edge("Robin is in Shelbyville", valid="2026-10-05T00:00:00", invalid="2026-10-07T00:00:00")]
    dump = {"nodes": [{"uuid": "1", "name": "Robin"}], "edges": edges, "episodes": 2}
    checks = score.run_checks(dump["nodes"], edges, [], T)
    md = score.render_report(dump, checks, score.cost_summary([]), {"planned": 2}, [], [])
    assert "Verdict by the checks: **NO-GO**" in md
    assert "no longer true (1)" in md and "Robin is in Shelbyville" in md


def test_redact_handles_punctuation_accents_whitespace_and_emails():
    t = {**T, "trip_hotel": "Grand View", "home_city": "Springfield."}
    out = score.redact("the Grand\nView hotel; Springfield. and Spríngfield. mail pat@example.org", t)
    assert "[trip hotel] hotel" in out and out.count("[home city]") == 2 and "[email]" in out
    assert score.redaction_leaks(out, t, set(), None) == []
    assert score.redaction_leaks("still Grand View here", t, set(), None) == ["term:trip_hotel"]


def test_trip_stay_presence_catches_open_facts_the_window_or_wording_would_hide():
    closed = _edge("Robin is staying in Shelbyville for another night", valid="2026-10-07T00:25:00",
                   invalid="2026-10-07T07:30:00")
    assert score.check_trip_stay([closed], T).passed
    undated = _edge("Robin is in Shelbyville", source="Robin")
    as_target = _edge("Shelbyville hosts Robin for the week", valid="2026-10-05T00:00:00",
                      source="Shelbyville", target="Robin")
    other_negation = _edge("Robin is in Shelbyville and hasn't flown home yet", valid="2026-10-05T00:00:00")
    for stale in (undated, as_target, other_negation):
        assert not score.check_trip_stay([closed, stale], T).passed
    meeting_only = _edge("The team meeting is in Shelbyville", valid="2026-10-05T00:00:00",
                         invalid="2026-10-07T00:00:00", source="team meeting")
    assert not score.check_trip_stay([meeting_only], T).passed  # a closed stay must be the subject's


def _report_fixture(tmp_path, monkeypatch, hashes, pt):
    monkeypatch.setattr(score, "OUT_DIR", tmp_path)
    (tmp_path / "graph_dump.json").write_text(json.dumps({"nodes": [], "edges": [
        _edge("Robin lives in Springfield", valid="2026-08-15T00:00:00")], "episodes": 1}))
    terms = tmp_path / "terms.json"
    terms.write_text(json.dumps(T))
    monkeypatch.setattr(score, "_denylist_hashes", lambda: (hashes, pt))
    return ["report", "--terms", str(terms)]


def test_report_refuses_without_denylist_and_writes_nothing(tmp_path, monkeypatch, capsys):
    argv = _report_fixture(tmp_path, monkeypatch, set(), None)
    assert score.main(argv) == 1
    assert not (tmp_path / "report.redacted.md").exists()
    assert score.main(argv + ["--allow-no-denylist"]) == 0
    assert "Robin" not in (tmp_path / "report.redacted.md").read_text()


def test_report_refuses_and_removes_file_when_a_denylisted_word_survives(tmp_path, monkeypatch, capsys):
    class LeakyPT:  # a denylist hit redact() cannot mask (simulates a matcher/redactor mismatch)
        _TOKEN = re.compile(r"(?!x)x")

        @staticmethod
        def _h(w):
            return w

        @staticmethod
        def line_hits(line, hashes):
            return {"h"} if "NO-GO" in line else set()
    (tmp_path / "report.redacted.md").write_text("stale")
    argv = _report_fixture(tmp_path, monkeypatch, {"h"}, LeakyPT)
    assert score.main(argv) == 1
    assert not (tmp_path / "report.redacted.md").exists()


# --- trial 2: chat input, seed, ontology, sanitizer, thinking switch ---------------------------



def test_trim_keeps_head_and_tail_and_reports_the_cut():
    assert t2.trim("short", 10, 4, 2) == ("short", 0)
    out, cut = t2.trim("a" * 50 + "TAIL", 20, 10, 4)
    assert cut == 40 and out.startswith("a" * 10) and out.endswith("TAIL") and "[... 40 chars trimmed ...]" in out


def test_exchange_body_is_speaker_labelled_and_bounded():
    body, cut = t2.exchange_body("hi there", "x" * 5000)
    assert body.startswith("Juniper: hi there\nOrion: ") and cut == 5000 - t2.ORION_HEAD - t2.ORION_TAIL
    assert len(body) < t2.ORION_MAX_CHARS + 100


def test_seed_body_states_pronouns_and_relations_once():
    facts = [{"name": "Sam", "category": "pronouns", "value_text": "She / her"},
             {"name": "Sam", "category": "pronouns", "value_text": "she/her"},
             {"name": "Sam", "category": "relationship", "value_text": "Sam is my wife"},
             {"name": "Mail carrier", "category": "pronouns", "value_text": "I don't know the pronouns"},
             {"name": "Pat", "category": "pronouns", "value_text": "he / him"}]
    body = t2.seed_body(facts)
    assert "My pronouns are she/her" in body
    assert "Juniper: Sam uses she/her pronouns. Sam is my wife" in body and body.count("Sam uses") == 1
    assert "Pat uses he/him" in body and "Mail carrier" not in body
    assert all(line.startswith("Juniper: ") for line in body.splitlines())


def test_chat_episodes_put_the_seed_first_and_are_always_key_rows():
    rows = t2.chat_rows_to_dicts([
        {"id": "2", "prompt": "later", "response": "ok", "at": datetime(2026, 10, 5, 3, 0)},
        {"id": "1", "prompt": "first", "response": "ok", "at": "2026-10-05T02:55:00"}])
    eps = trial.episodes_from_chat(rows, "Juniper: seed")
    assert [e.name for e in eps] == [trial.SEED_NAME, "chat:1", "chat:2"]
    assert eps[0].reference_time < eps[1].reference_time and eps[1].reference_time.tzinfo is not None
    assert eps[0].key_row() and all(e.source == "message" for e in eps)
    assert trial.select_sample(eps, 1)[0].name == trial.SEED_NAME


def test_redisearch_safe_strips_operators_that_crashed_trial_1():
    assert t2.redisearch_safe("38% outreach figure") == "38 outreach figure"
    assert t2.redisearch_safe("a@b $x; y.z, <q>='r'#") == "a b x y z q r"


def test_key_row_re_can_drop_generic_animals_for_chat():
    rx = trial.key_row_re(T, generic_animals=False)
    assert rx.search("Biscuit barked") and not rx.search("the neighbours' dogs")
    assert not trial.key_row_re(None, generic_animals=False).search("anything")


def test_configure_graph_allows_only_trial_graphs_and_pairs_group_ids():
    try:
        trial.configure_graph("graphiti_trial2")
        assert (trial.TRIAL_GRAPH, trial.GROUP_ID) == ("graphiti_trial2", "orion_trial2")
        with pytest.raises(SystemExit, match="refusing graph"):
            trial.configure_graph("orion_worldview")
    finally:
        trial.configure_graph("graphiti_trial")


def test_isolation_verdict_fails_when_another_trial_graph_moves():
    before = {"graphs": ["graphiti_trial"], "trial_group_nodes_elsewhere": {},
              "other_trial_graph_counts": {"graphiti_trial": [100, 200]}}
    after_ok = {**before, "graphs": ["graphiti_trial", "graphiti_trial2"]}
    try:
        trial.configure_graph("graphiti_trial2")
        assert trial.isolation_verdict(before, after_ok)["isolated"]
        moved = {**after_ok, "other_trial_graph_counts": {"graphiti_trial": [101, 200]}}
        v = trial.isolation_verdict(before, moved)
        assert not v["isolated"] and v["other_trial_graphs_changed"]
    finally:
        trial.configure_graph("graphiti_trial")


def test_thinking_switch_body():
    assert clients.thinking_body(True) is None
    assert clients.thinking_body(False) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_ontology_is_small_and_excludes_generic_entity():
    pytest.importorskip("pydantic")
    types, excluded = t2.ontology()
    assert set(types) == {"Person", "Pet", "Place", "Event", "OrionPart"} and excluded == ["Entity"]
    assert "pronouns" in types["Person"].model_fields and "species" in types["Pet"].model_fields


def test_install_redisearch_sanitizer_wraps_fulltext_query(monkeypatch):
    pytest.importorskip("graphiti_core")
    from graphiti_core.search import search_utils
    monkeypatch.setattr(search_utils, "fulltext_query", search_utils.fulltext_query)  # restored after
    t2.install_redisearch_sanitizer()
    t2.install_redisearch_sanitizer()
    from types import SimpleNamespace as NS
    from graphiti_core.driver.driver import GraphProvider
    driver = NS(provider=GraphProvider.FALKORDB, fulltext_syntax="@")
    q = search_utils.fulltext_query("38% outreach figure", ["g"], driver)
    assert "%" not in q and "outreach" in q
    assert search_utils.fulltext_query("…?!", ["g"], driver) == ""
    assert search_utils.fulltext_query._orion_safe


def test_pronoun_violations_flag_he_for_she_people_and_orion_only():
    nodes = [{"name": "Robin", "summary": "Robin lives in Springfield. He likes trains. Alex said he would help."},
             {"name": "Alex", "summary": "Alex is Robin's son. He is ten."},
             {"name": "servers", "summary": "Servers are part of Orion's homelab, described as his body."}]
    edges = [_edge("Robin says he needs to rest"), _edge("Orion described his mesh"),
             _edge("Robin told me Alex broke his arm"), _edge("Robin says she needs to rest"),
             _edge("Biscuit wagged his tail at Robin")]
    v = score.pronoun_violations(nodes, edges, T)
    assert len(v) == 4
    assert any("He likes trains" in x for x in v) and any("summary of servers" in x for x in v) and any("he needs to rest" in x for x in v)
    assert any("his mesh" in x for x in v)


def test_invalidation_check_needs_every_invalidation_judged_and_under_two_wrong():
    edges = [_edge("a", invalid="2026-10-07"), _edge("b", invalid="2026-10-08"), _edge("c")]
    for i, e in enumerate(edges):
        e["uuid"] = str(i)
    sheet = score.invalidation_sheet(edges)
    assert [j["uuid"] for j in sheet] == ["0", "1"]
    assert not score.check_invalidations(sheet, edges).passed  # unjudged
    sheet[0]["wrong"], sheet[1]["wrong"] = True, False
    c = score.check_invalidations(sheet, edges)
    assert c.passed and "1 wrong of 2" in c.detail
    sheet[1]["wrong"] = True
    assert not score.check_invalidations(sheet, edges).passed
    sheet[1]["wrong"] = "yes"
    with pytest.raises(ValueError):
        score.check_invalidations(sheet, edges)


def test_trial_holds_are_operator_class_and_capped(pool):
    """Issue #3391: never Orion's own agent class; the hold is given back after its length cap."""
    hold, script, log, _ = pool
    script["acquire"] = [_Reply("granted")]

    async def go():
        await hold.acquire()
        assert not hold.over_cap()
        hold.max_hold_sec = 0
        assert hold.over_cap() and hold.usable()
        await hold.release()
        assert not hold.over_cap()
    asyncio.run(go())
    assert log["classes"] == ["operator_agent"]


def test_default_route_and_class_are_operator():
    seen = {}

    def fake_run(args):
        seen.update(vars(args))
        return 0
    orig = trial._run_with_sigterm
    try:
        async def capture(args):
            return fake_run(args)
        trial._run_with_sigterm = capture
        trial.main(["run"])
    finally:
        trial._run_with_sigterm = orig
    assert seen["work_class"] == "operator_agent" and seen["llm_route"] == "operator_agent"
    assert seen["max_hold_sec"] <= 1200


def test_seed_body_omits_pronoun_clause_when_none_stated():
    body = t2.seed_body([{"name": "Lee", "category": "relationship", "value_text": "a friend"}])
    assert "Juniper: Lee: a friend" in body and "not stated" not in body


def test_pronoun_violations_ignore_blank_he_names():
    v = score.pronoun_violations([], [_edge("Robin says he is tired")], {**T, "he_names": ["", " "]})
    assert len(v) == 1


def test_trial2_graph_takes_chat_and_resume_refuses_changed_settings(tmp_path, monkeypatch):
    (tmp_path / "plan-a.json").write_text(json.dumps({"graph": "graphiti_trial2", "source": "chat",
                                                      "thinking": "off", "ontology": "orion"}))
    same = {"source": "chat", "thinking": "off", "ontology": "orion"}
    assert trial.settings_conflict(tmp_path, "graphiti_trial2", same) is None
    assert "thinking" in trial.settings_conflict(tmp_path, "graphiti_trial2", {**same, "thinking": "on"})
    assert trial.settings_conflict(tmp_path, "graphiti_trial", {**same, "thinking": "on"}) is None
    monkeypatch.setenv("GRAPHITI_TRIAL_OUT", str(tmp_path))
    try:
        with pytest.raises(SystemExit, match="takes --source chat"):
            asyncio.run(trial.run(trial_args(graph="graphiti_trial2", source="memory")))
    finally:
        trial.configure_graph("graphiti_trial")


def trial_args(**kw):
    import argparse
    base = {"graph": "graphiti_trial", "source": "memory", "thinking": "on", "ontology": "none"}
    return argparse.Namespace(**{**base, **kw})


def test_scorer_refuses_a_dump_of_another_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHITI_TRIAL_OUT", str(tmp_path))
    monkeypatch.setattr(score, "OUT_DIR", tmp_path)
    (tmp_path / "graph_dump.json").write_text(json.dumps({"nodes": [], "edges": [], "episodes": 0,
                                                          "graph": "graphiti_trial"}))
    assert score.main(["report", "--graph", "graphiti_trial2"]) == 1


def test_pronoun_violations_skip_he_sentences_about_another_entity():
    nodes = [{"name": "Pat", "summary": "Pat fixed the sprinklers. He came out for Robin."},
             {"name": "Shelbyville", "summary": "Robin shared photos of his hotel room."}]
    v = score.pronoun_violations(nodes, [], T)
    assert len(v) == 1 and "his hotel room" in v[0]


def test_trip_city_home_needs_the_subject():
    bar = _edge("The corner bar is located in Shelbyville", source="corner bar", target="Shelbyville")
    assert score.check_trip_city_not_home([bar], T).passed


def test_one_entity_ignores_possessive_things():
    pat = re.compile(score.word(T["spouse"]), re.I)
    assert score.check_one_entity([{"name": "Sam"}, {"name": "Sam's surgery"}], pat, "spouse").passed
    assert not score.check_one_entity([{"name": "Sam"}, {"name": "Sam R."}], pat, "spouse").passed
