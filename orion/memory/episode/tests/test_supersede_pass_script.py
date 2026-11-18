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
