"""scripts/memory_supersede_pass.py: the ids-only manifest Orion reads never carries statements."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "memory_supersede_pass.py"
spec = importlib.util.spec_from_file_location("memory_supersede_pass", SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, sql, params):
        rows = [r for r in self.rows if r["memory_id"] in params[0]]

        class Cur:
            async def fetchall(self_inner):
                return rows

        return Cur()


def test_manifest_has_ids_and_statuses_only(tmp_path):
    secret = "Juniper told me a private thing about Biscuit."
    proposed = [{"old_id": "o1", "new_id": "n1", "reason": "r", "old_statement": secret}]
    conn = FakeConn([{"memory_id": "o1", "status": "active", "confirmation_state": "auto",
                      "supersedes_memory_id": None, "superseded_by": None}])
    path = tmp_path / "m.json"
    asyncio.run(P.write_manifest(conn, path, mode="dry_run", proposed=proposed, applied=None,
                                 watch=P.parse_watch(["pet_row=o1", "gone=missing-id", "bad"])))
    text = path.read_text()
    assert secret not in text and "Biscuit" not in text
    m = json.loads(text)
    assert m["proposed"] == [{"old_id": "o1", "new_id": "n1"}] and m["applied"] == []
    assert m["watched"][0] == {"label": "pet_row", "memory_id": "o1", "status": "active",
                               "confirmation_state": "auto", "supersedes_memory_id": None,
                               "superseded_by": None, "proposed_superseded_by": "n1"}
    assert m["watched"][1]["status"] == "missing"
    assert "superseded|<old_id>|<new_id>" in m["pair_handled_key"]
    # An apply keeps the proposed list and records what it applied.
    asyncio.run(P.write_manifest(conn, path, mode="apply", proposed=proposed,
                                 applied=[{"old_id": "o1", "new_id": "n1", "applied_at": "t"}], watch=[]))
    assert json.loads(path.read_text())["applied"][0]["old_id"] == "o1"


def test_apply_refuses_pairs_found_without_the_meaning_index(tmp_path):
    """Issue #3659: a pairs file with a word-only pair is refused before any database connection."""
    import argparse

    import pytest

    pairs = tmp_path / "pairs.jsonl"
    v = P.CHECK_VERSION
    pairs.write_text(json.dumps({"old_id": "o1", "new_id": "n1", "reason": "r", "dense": None, "dense_on": False,
                                 "check_version": v})
                     + "\n" + json.dumps({"old_id": "o2", "new_id": "n2", "reason": "r", "dense": 0.71,
                                           "check_version": v}) + "\n")
    args = argparse.Namespace(yes=True, pairs=str(pairs), allow_no_dense=False, allow_no_eval=True, only="", exclude="",
                              dsn="postgresql://nobody@127.0.0.1:1/none", no_manifest=True, watch=[])
    with pytest.raises(SystemExit, match="without the meaning index"):
        asyncio.run(P.cmd_apply(args))
    # --exclude drops the word-only pair: then the refusal does not fire (it reaches the DB connect).
    args.exclude = "o1:n1"
    with pytest.raises(Exception) as got:
        asyncio.run(P.cmd_apply(args))
    assert "meaning index" not in str(got.value)
    assert P.found_with_dense({"dense": 0.7}) and not P.found_with_dense({"dense": None})
    assert not P.found_with_dense({"dense": 0.7, "dense_on": False})


def test_apply_refuses_a_pairs_file_from_another_check_version(tmp_path):
    import argparse

    import pytest

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"old_id": "o", "new_id": "n", "reason": "r", "dense": 0.7,
                                 "check_version": "supersede_check.v2"}) + "\n")
    args = argparse.Namespace(yes=True, pairs=str(pairs), allow_no_dense=True, allow_no_eval=True, only="", exclude="",
                              dsn="postgresql://nobody@127.0.0.1:1/none", no_manifest=True, watch=[])
    with pytest.raises(SystemExit, match="another check version"):
        asyncio.run(P.cmd_apply(args))


def test_two_runs_leave_two_history_lines_and_watched_trace_has_legs_not_text(tmp_path):
    """Issue #3660: the manifest's run history is append-only, and a watched row's trace carries
    the legs and votes of each pair it was in, with no statement text or reasons."""
    from orion.memory.episode import supersede as s

    secret = "Juniper is in Austin this week for a work trip."
    old = s.MemoryRow("o1", secret, "about_juniper", None)
    cand = s.Candidate(old=old, dense=0.6612, shared_rare_tokens=("austin",), shared_rare_referents=(),
                       token_rows=(4,), rank=1)
    verdict = s.Classification(votes=(s.Vote("compatible", False, 0.9, reason=secret),), relation="compatible",
                               acts=False, juniper_acts=False, abstain="vote:compatible")
    rec = s.CheckRecord("o1", "n1", verdict, cand)
    conn = FakeConn([{"memory_id": "o1", "status": "active", "confirmation_state": "auto",
                      "supersedes_memory_id": None, "superseded_by": None}])
    path = tmp_path / "memory_supersede_pass_manifest.json"
    for i in range(2):
        asyncio.run(P.write_manifest(conn, path, mode="dry_run", proposed=[], applied=None,
                                     watch=P.parse_watch(["trip=o1"]), run_id=f"run{i}", dense="on",
                                     records=[rec], counts={"asked": 1}))
    hist = (tmp_path / "memory_supersede_pass_runs.jsonl").read_text().splitlines()
    assert [json.loads(h)["run_id"] for h in hist] == ["run0", "run1"]
    m = json.loads(path.read_text())
    trace = m["watched"][0]["trace"]
    assert trace[0]["legs"]["cosine"] == 0.6612 and trace[0]["legs"]["rare_token_rows"] == [4]
    assert trace[0]["votes"] == [{"relation": "compatible", "same_attribute": False, "confidence": 0.9}]
    assert "Austin" not in path.read_text() and "Austin" not in "".join(hist)


def test_a_malformed_answer_is_never_cached():
    assert not P.cacheable("garbage") and not P.cacheable('{"relation": "maybe"}')
    # a v3-format answer (no checklist) still parses as a non-acting vote: cacheable, never acting
    old_format = json.dumps({"relation": "correction", "same_attribute": "yes", "confidence": 0.9, "reason": "r"})
    assert P.cacheable(old_format)
    assert P.cacheable(json.dumps({"a_kind": "event", "b_says_a_wrong_or_changed": "no", "relation": "compatible",
                                   "same_attribute": "no", "confidence": 0.9, "reason": "r"}))


def test_gateway_ask_retries_a_transient_5xx(monkeypatch):
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(502, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": 1}"}}]})

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(P.asyncio, "sleep", no_sleep)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await P.gateway_ask(client, "http://gw", "memory_distill", "p", timeout=5)

    assert asyncio.run(run()) == "{\"ok\": 1}" and len(calls) == 3


def _ask_with(handler, monkeypatch):
    import httpx

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(P.asyncio, "sleep", no_sleep)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await P.gateway_ask(client, "http://gw", "memory_distill", "p", timeout=5)

    return asyncio.run(run())


def test_gateway_ask_caps_retries_and_never_retries_a_4xx_or_timeout(monkeypatch):
    import httpx
    import pytest

    calls = []

    def always_502(request):
        calls.append(1)
        return httpx.Response(502, json={})

    with pytest.raises(httpx.HTTPStatusError):
        _ask_with(always_502, monkeypatch)
    assert len(calls) == 4

    calls.clear()

    def bad_request(request):
        calls.append(1)
        return httpx.Response(400, json={})

    with pytest.raises(httpx.HTTPStatusError):
        _ask_with(bad_request, monkeypatch)
    assert len(calls) == 1

    calls.clear()

    def timeout(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    with pytest.raises(httpx.ReadTimeout):
        _ask_with(timeout, monkeypatch)
    assert len(calls) == 1


def test_gateway_ask_retries_a_dropped_connection(monkeypatch):
    import httpx

    calls = []

    def flaky(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadError("dropped")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    assert _ask_with(flaky, monkeypatch) == "{}" and len(calls) == 2


def test_an_undo_rewrites_the_manifest_so_it_never_says_an_undone_pair_is_in_effect(tmp_path):
    """Orion's question on PR #3734: the manifest still said 2 applied after both pairs were undone."""
    conn = FakeConn([])
    path = tmp_path / "m.json"
    pairs = [{"old_id": "o1", "new_id": "n1"}, {"old_id": "o2", "new_id": "n2"}]
    asyncio.run(P.write_manifest(conn, path, mode="apply", proposed=pairs, dense="on", run_id="apply-1",
                                 applied=[{**p, "applied_at": "t0"} for p in pairs], watch=[], undone=[]))
    before = json.loads(path.read_text())
    assert before["in_effect"] == pairs and before["undone"] == []
    # A later check version must not relabel the old proposal.
    before["check_version"] = "supersede_classify.v3.1"
    path.write_text(json.dumps(before))
    undone = [{"old_id": "o1", "new_id": "n1", "actor": "juniper", "undone_at": "t1", "note": "different slice"}]
    asyncio.run(P.write_manifest(conn, path, mode="undo", proposed=None, applied=None, watch=None,
                                 run_id="undo-1", counts={"undone": 1}, undone=undone))
    m = json.loads(path.read_text())
    assert m["last_mode"] == "undo" and m["check_version"] == "supersede_classify.v3.1" and m["dense"] == "on"
    assert m["proposed"] == pairs
    assert m["applied"][0] == {"old_id": "o1", "new_id": "n1", "applied_at": "t0", "undone_at": "t1"}
    assert "undone_at" not in m["applied"][1]
    assert m["in_effect"] == [{"old_id": "o2", "new_id": "n2"}]
    assert m["undone"] == undone and "never re-applied" in m["undone_rule"]
    # The history line records what this run did, not the previous proposal or apply.
    lines = [json.loads(x) for x in P.history_path(path).read_text().splitlines()]
    assert [x["mode"] for x in lines] == ["apply", "undo"]
    assert lines[1]["applied"] == [] and lines[1]["proposed"] == [] and lines[1]["counts"] == {"undone": 1}
    # A refresh with no new undo list keeps the old one.
    asyncio.run(P.write_manifest(conn, path, mode="refresh", proposed=None, applied=None, watch=None))
    assert json.loads(path.read_text())["undone"] == undone


def test_undone_events_reads_ids_actor_time_and_note():
    class Conn:
        async def execute(self, sql, params=None):
            assert "supersede_undone" in sql and params is None

            class Cur:
                description = None

                async def fetchall(self_inner):
                    return [{"old_id": "o1", "new_id": "n1", "actor": "operator", "created_at": "t", "note": ""}]

            return Cur()

    assert asyncio.run(P.undone_events(Conn())) == [
        {"old_id": "o1", "new_id": "n1", "actor": "operator", "undone_at": "t", "note": ""}]


def test_undo_refuses_without_a_note_before_touching_the_database():
    import argparse

    import pytest

    args = argparse.Namespace(dsn="postgresql://nowhere.invalid/x", old="o1", new="n1", actor="juniper", note="  ",
                              manifest="/nonexistent/m.json", no_manifest=True, watch=[])
    with pytest.raises(SystemExit, match="--note"):
        asyncio.run(P.cmd_undo(args))


def test_undo_cli_requires_actor_and_note(monkeypatch, capsys):
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["memory_supersede_pass.py", "--dsn", "x", "undo", "--old", "o", "--new", "n"])
    with pytest.raises(SystemExit):
        P.main()
    err = capsys.readouterr().err
    assert "--actor" in err and "--note" in err


def test_apply_decision_trace_has_legs_votes_and_outcome_but_no_statements():
    secret = "a private statement"
    pair = {"old_id": "o1", "new_id": "n1", "dense": 0.6415, "shared": ["project:circe"], "relation": "correction",
            "path": "model", "votes": ["correction"] * 3, "reason": "r", "check_version": "v",
            "old_statement": secret, "new_statement": secret}
    d = P.pair_decision(pair, "applied")
    assert secret not in json.dumps(d)
    assert d["legs"] == {"cosine": 0.6415, "shared": ["project:circe"]}
    assert d["votes"] == ["correction"] * 3 and d["outcome"] == "applied"


def test_in_effect_comes_from_the_caller_and_a_broken_previous_file_is_ignored(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("[1, 2]")  # valid JSON, not an object
    live = [{"old_id": "o9", "new_id": "n9", "actor": "memory.episode_distill", "superseded_at": "t"}]
    asyncio.run(P.write_manifest(FakeConn([]), path, mode="refresh", proposed=None, applied=None, watch=None,
                                 undone=[], in_effect=live))
    m = json.loads(path.read_text())
    assert m["in_effect"] == live and m["applied"] == [] and m["proposed"] == []


def test_the_meaning_index_failure_reason_is_logged(monkeypatch, caplog):
    import logging

    from orion.introspect import semantic_index
    from orion.memory.episode import supersede as S

    async def boom(*a, **k):
        raise semantic_index.SearchUnavailableError("embed host 503")

    monkeypatch.setattr(semantic_index, "embed", boom)
    with caplog.at_level(logging.WARNING, logger=S.__name__):
        assert asyncio.run(S.index_dense(object())("x")) is None
    assert "supersede_dense_unavailable" in caplog.text and "embed host 503" in caplog.text
