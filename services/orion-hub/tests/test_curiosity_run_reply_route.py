"""`POST /curiosity/api/run/{run_id}/reply` -- Juniper's explicit reply box
on a curiosity run's reach-out (design doc "Missing question 1, option (b)",
`docs/superpowers/pr-reports/2026-09-22-curiosity-run-story-pr.md`'s sibling
patch). Chat replies (PR #2290) link by time adjacency; this route instead
carries an EXPLICIT `client_meta.in_reply_to` because Juniper clicked reply
on this exact run -- no guessing.

Fakes stand in for the asyncpg pool (`conn.fetchrow`, distinct from the
`conn.fetch` fake `test_curiosity_routes_runs.py` uses for the read
endpoints) and for `handle_chat_request` so the thing under test is the
route's own gating, never-500 contract, and the exact client_meta it sends
-- not a real chat turn.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

import scripts.curiosity_routes as cr
import scripts.curiosity_run_store as store
from orion.curiosity.run_story import outreach_key


class _Conn:
    def __init__(self, decision=None, session_row=None, fail_on=None):
        self.decision = decision
        self.session_row = session_row
        self.fail_on = fail_on  # "decision" | "session" | None
        self.calls: list[tuple[str, tuple]] = []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        if "endogenous_outreach_decisions" in sql:
            if self.fail_on == "decision":
                raise RuntimeError("pg down")
            return self.decision
        if "chat_history_log" in sql:
            if self.fail_on == "session":
                raise RuntimeError("pg down")
            return self.session_row
        return None


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


class _Pool:
    def __init__(self, **kw):
        self.conn = _Conn(**kw)

    def acquire(self):
        return _Acquire(self.conn)


def _install_fake_main(monkeypatch, *, pool=None, cortex_client=object()):
    fake_main = types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(memory_pg_pool=pool)),
        bus=None,
        cortex_client=cortex_client,
    )
    monkeypatch.setitem(sys.modules, "scripts.main", fake_main)
    import scripts as scripts_pkg

    monkeypatch.setattr(scripts_pkg, "main", fake_main, raising=False)
    return fake_main


_SENT_DECISION = {"decision_id": "d1", "decided_at": None, "session_id": "ignored-column"}
_SESSION_ROW = {"session_id": "sess-1"}


# --- store: fetch_sent_outreach_session -------------------------------------


def test_fetch_sent_outreach_session_true_when_sent_and_session_resolved() -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=_SESSION_ROW)
    sent, session_id = asyncio.run(store.fetch_sent_outreach_session(pool, "r1"))
    assert sent is True
    assert session_id == "sess-1"
    # session comes from the chat_history_log row, not the decision row's own
    # session_id column -- the task's explicit contract.
    assert session_id != _SENT_DECISION["session_id"]


def test_fetch_sent_outreach_session_false_when_no_sent_decision() -> None:
    pool = _Pool(decision=None)
    sent, session_id = asyncio.run(store.fetch_sent_outreach_session(pool, "r1"))
    assert sent is False and session_id is None


def test_fetch_sent_outreach_session_true_but_unresolvable_when_chat_row_missing() -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=None)
    sent, session_id = asyncio.run(store.fetch_sent_outreach_session(pool, "r1"))
    assert sent is True and session_id is None


def test_fetch_sent_outreach_session_raises_on_db_failure() -> None:
    """A write path: a DB outage must not silently read as 'never sent'."""
    pool = _Pool(decision=None, fail_on="decision")
    with pytest.raises(RuntimeError):
        asyncio.run(store.fetch_sent_outreach_session(pool, "r1"))


def test_fetch_sent_outreach_session_raises_with_no_pool() -> None:
    with pytest.raises(RuntimeError):
        asyncio.run(store.fetch_sent_outreach_session(None, "r1"))


def test_fetch_sent_outreach_session_uses_the_run_derived_key() -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=_SESSION_ROW)
    asyncio.run(store.fetch_sent_outreach_session(pool, "some-run-id"))
    corr = outreach_key("some-run-id")
    assert pool.conn.calls[0][1] == (corr,)
    assert pool.conn.calls[1][1] == (corr,)


# --- route: request validation ----------------------------------------------


def test_reply_route_rejects_empty_text(monkeypatch) -> None:
    _install_fake_main(monkeypatch, pool=None)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "   "}))
    assert resp.status_code == 400
    assert json.loads(resp.body) == {"ok": False, "reason": "empty_text"}


def test_reply_route_rejects_oversized_text(monkeypatch) -> None:
    _install_fake_main(monkeypatch, pool=None)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "x" * 4001}))
    assert resp.status_code == 400
    assert json.loads(resp.body) == {"ok": False, "reason": "text_too_long"}


def test_reply_route_rejects_bad_run_id(monkeypatch) -> None:
    _install_fake_main(monkeypatch, pool=None)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1' OR 1=1", {"text": "hi"}))
    assert resp.status_code == 400
    assert json.loads(resp.body) == {"ok": False, "reason": "invalid_run_id"}


# --- route: outreach gating --------------------------------------------------


def test_reply_route_409s_when_outreach_never_sent(monkeypatch) -> None:
    """Blocked, not_recorded, or no row at all -- all the same refusal."""
    pool = _Pool(decision=None)
    _install_fake_main(monkeypatch, pool=pool)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "hi"}))
    assert resp.status_code == 409
    assert json.loads(resp.body) == {"ok": False, "reason": "no_sent_outreach_to_reply_to"}


def test_reply_route_500s_cleanly_when_session_unresolvable(monkeypatch) -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=None)
    _install_fake_main(monkeypatch, pool=pool)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "hi"}))
    assert resp.status_code == 500
    assert json.loads(resp.body) == {"ok": False, "reason": "session_unresolvable"}


def test_reply_route_surfaces_a_lookup_failure_as_ok_false_with_a_500(monkeypatch) -> None:
    """A DB failure on the outreach lookup is an infra failure, not a chat-
    turn refusal -- it still gets a 500, unlike a downstream chat failure
    (see the accepted-path tests below), but always the same `ok: false`
    shape so a caller checks one thing regardless of status code."""
    pool = _Pool(decision=None, fail_on="decision")
    _install_fake_main(monkeypatch, pool=pool)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "hi"}))
    body = json.loads(resp.body)
    assert resp.status_code == 500
    assert body["ok"] is False and "RuntimeError" in body["reason"]


def test_reply_route_500s_when_cortex_client_unavailable(monkeypatch) -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=_SESSION_ROW)
    _install_fake_main(monkeypatch, pool=pool, cortex_client=None)
    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "hi"}))
    assert resp.status_code == 500
    assert json.loads(resp.body) == {"ok": False, "reason": "cortex_client_unavailable"}


# --- route: happy path + explicit client_meta -------------------------------


def test_reply_route_accepted_and_sends_the_explicit_client_meta(monkeypatch) -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=_SESSION_ROW)
    _install_fake_main(monkeypatch, pool=pool)

    captured: dict = {}

    async def fake_handle_chat_request(cortex_client, payload, session_id, *, no_write, client_meta=None, http_request=None):
        captured["payload"] = payload
        captured["session_id"] = session_id
        captured["no_write"] = no_write
        captured["client_meta"] = client_meta
        return {"correlation_id": "new-corr-1", "llm_response": "ok"}

    monkeypatch.setattr(cr, "_handle_chat_request", fake_handle_chat_request)

    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "  Thanks Orion  "}))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body == {"ok": True, "correlation_id": "new-corr-1", "session_id": "sess-1"}

    assert captured["session_id"] == "sess-1"
    assert captured["no_write"] is False
    assert captured["payload"]["mode"] == "orion"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Thanks Orion"}]
    assert captured["client_meta"] == {
        "in_reply_to": outreach_key("r1"),
        "in_reply_to_source": "curiosity_outreach",
        "in_reply_to_explicit": True,
    }


def test_reply_route_reports_a_downstream_turn_error_as_ok_false(monkeypatch) -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=_SESSION_ROW)
    _install_fake_main(monkeypatch, pool=pool)

    async def fake_handle_chat_request(*a, **kw):
        return {"type": "turn_error", "phase": "harness", "error": "harness_rpc_timeout", "correlation_id": "c1"}

    monkeypatch.setattr(cr, "_handle_chat_request", fake_handle_chat_request)

    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "hi"}))
    assert resp.status_code == 200, "a downstream chat failure is ok:false, never a 500"
    assert json.loads(resp.body) == {"ok": False, "reason": "harness_rpc_timeout"}


def test_reply_route_reports_a_downstream_exception_as_ok_false_never_a_500(monkeypatch) -> None:
    pool = _Pool(decision=_SENT_DECISION, session_row=_SESSION_ROW)
    _install_fake_main(monkeypatch, pool=pool)

    async def boom(*a, **kw):
        raise RuntimeError("cortex unreachable")

    monkeypatch.setattr(cr, "_handle_chat_request", boom)

    resp = asyncio.run(cr.curiosity_run_reply_api("r1", {"text": "hi"}))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["ok"] is False
    assert "cortex unreachable" in body["reason"]


def test_reply_route_is_registered_as_post_only() -> None:
    paths = {r.path: sorted(r.methods) for r in cr.router.routes}
    assert paths["/curiosity/api/run/{run_id}/reply"] == ["POST"]
