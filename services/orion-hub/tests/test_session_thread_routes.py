"""Session threads API through the real routes (scripts/session_thread_routes.py): a message gets
Hub's receipt, the list and thread views, pending by session and by branch, delivered."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
for p in (str(HUB_ROOT.parents[1]), str(HUB_ROOT), str(HUB_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.session_threads import core  # noqa: E402
from scripts import session_roundup_loop as srl  # noqa: E402
from scripts import session_thread_routes as routes  # noqa: E402
from session_thread_fakes import Clock, FakeCommenter, FakeDurable, FakeStore  # noqa: E402

NOW = datetime(2026, 11, 15, 10, 20, tzinfo=timezone.utc)


class Sink:
    def __init__(self):
        self.frames = []

    def put_nowait(self, frame):
        self.frames.append(frame)


@pytest.fixture()
def world(monkeypatch):
    clock = Clock(NOW)
    store = FakeStore(clock)
    store.install(monkeypatch)
    sink = Sink()
    durable = FakeDurable()
    loop = srl.SessionRoundupLoop(enabled=True, pool_provider=lambda: object(), durable_client=durable,
                                  connections_provider=lambda: {"c": {"queue": sink}}, commenter=FakeCommenter(),
                                  clock=clock)
    monkeypatch.setattr(routes, "_loop", lambda: loop)
    app = FastAPI()
    app.state.memory_pg_pool = object()
    app.include_router(routes.router)
    return TestClient(app), loop, clock, sink, durable


def _msg(**kw):
    body = {"session_id": "sess-A", "session_name": "wt-a", "worktree": "/wt/a", "branch": "feat/a", "pr_number": 3456,
            "pr_url": "https://github.com/o/r/pull/3456", "title": "feat: session threads", "body": "Please read.",
            "spec": [{"path": "docs/superpowers/specs/x.md", "text": "# Spec\n\nwords"}]}
    body.update(kw)
    return body


def test_a_message_returns_hubs_receipt_and_orion_has_not_read_it(world):
    client, _loop, _clock, sink, durable = world
    resp = client.post("/api/session-threads/message", json=_msg())
    assert resp.status_code == 200
    data = resp.json()
    assert data["receipt"]["speaker"] == "hub" and "Orion has NOT read it yet" in data["receipt"]["text"]
    assert data["orion_has_read"] is False and data["roundup_enabled"] is True and data["answer_phase"] is False
    assert data["next_roundup_at"] == "2026-11-15T11:00:00+00:00"
    assert data["thread"]["status"] == "held" and data["thread"]["kind"] == "pr_spec"
    # The response does not echo the spec text back.
    assert data["turn"]["meta"]["spec"] == [{"path": "docs/superpowers/specs/x.md", "chars": 13, "cut": False}]
    assert [f["speaker"] for f in sink.frames] == ["session", "hub"] and durable.submits == []
    # Frames never carry the spec text either: only its path.
    assert sink.frames[0]["spec_paths"] == ["docs/superpowers/specs/x.md"] and "spec" not in sink.frames[0]["meta"]


def test_bad_messages_are_refused(world):
    client = world[0]
    assert client.post("/api/session-threads/message", json=_msg(body="", spec=[])).status_code == 422
    assert client.post("/api/session-threads/message", json=_msg(title="  ")).status_code == 422
    assert client.post("/api/session-threads/message", json={"title": "t", "body": "b"}).status_code == 422   # no session_id
    assert client.post("/api/session-threads/message", json=_msg(thread_id="nope")).status_code == 404


def test_a_pull_requests_thread_is_found_by_number_and_a_diff_keeps_its_kind(world):
    client, *_ = world
    diff = {"path": "1 file(s)", "text": "diff --git a/x.py b/x.py\n+x = 1", "kind": "diff"}
    tid = client.post("/api/session-threads/message", json=_msg(spec=[diff])).json()["thread"]["thread_id"]
    client.post("/api/session-threads/message", json=_msg(pr_number=9, pr_url="https://github.com/o/r/pull/9"))
    [found] = client.get("/api/session-threads?pr=3456").json()["threads"]
    assert found["thread_id"] == tid and found["pr_number"] == 3456 and found["turns"] == 2
    assert client.get("/api/session-threads?pr=4242").json()["threads"] == []
    full = client.get(f"/api/session-threads/{tid}?spec_text=1").json()
    assert full["turns"][0]["meta"]["spec"][0]["kind"] == "diff"
    short = client.get(f"/api/session-threads/{tid}").json()
    assert short["turns"][0]["meta"]["spec"][0]["kind"] == "diff" and "text" not in short["turns"][0]["meta"]["spec"][0]


def test_list_thread_pending_and_delivered(world, monkeypatch):
    monkeypatch.setattr(core, "MAX_QUESTION_ROUNDS_PER_THREAD", 1)   # the answer-phase mechanics, one round
    client, loop, clock, _sink, durable = world
    tid = client.post("/api/session-threads/message", json=_msg()).json()["thread"]["thread_id"]
    note = client.post("/api/session-threads/message",
                       json={"session_id": "sess-B", "title": "a note", "body": "hello"}).json()["thread"]["thread_id"]

    listed = client.get("/api/session-threads?hours=48").json()["threads"]
    assert {t["kind"] for t in listed} == {"pr_spec", "note"} and all(t["turns"] == 2 for t in listed)

    one = client.get(f"/api/session-threads/{tid}").json()
    assert [t["speaker"] for t in one["turns"]] == ["session", "hub"]
    assert one["turns"][0]["meta"]["spec"][0]["chars"] == 13 and "text" not in one["turns"][0]["meta"]["spec"][0]
    full = client.get(f"/api/session-threads/{tid}?spec_text=1").json()
    assert full["turns"][0]["meta"]["spec"][0]["text"] == "# Spec\n\nwords"
    assert client.get("/api/session-threads/nope").status_code == 404

    assert client.get("/api/session-threads/pending").status_code == 422
    assert client.get("/api/session-threads/pending?session_id=sess-A").json()["pending"] == []

    # The hour turns and Orion answers the PR thread.
    clock.to_next_hour()
    durable.replies = [f"### thread:{tid}\n1. Why hourly?\n\n### thread:{note}\nNO QUESTIONS"]
    import asyncio

    asyncio.run(loop.tick())
    by_session = client.get("/api/session-threads/pending?session_id=sess-A").json()["pending"]
    by_branch = client.get("/api/session-threads/pending?branch=feat/a").json()["pending"]
    assert [p["thread_id"] for p in by_session] == [tid] == [p["thread_id"] for p in by_branch]
    assert by_session[0]["body"] == "1. Why hourly?" and by_session[0]["pr_number"] == 3456
    for_b = client.get("/api/session-threads/pending?session_id=sess-B").json()["pending"]
    assert [p["thread_id"] for p in for_b] == [note] and for_b[0]["meta"]["no_questions"] is True
    assert client.get("/api/session-threads/pending?session_id=sess-Z").json()["pending"] == []

    done = client.post("/api/session-threads/delivered", json={"turn_ids": [by_session[0]["turn_id"]],
                                                                "session_id": "sess-A"})
    assert done.json() == {"delivered": 1}
    assert client.get("/api/session-threads/pending?session_id=sess-A&branch=feat/a").json()["pending"] == []
    assert client.post("/api/session-threads/delivered", json={"turn_ids": []}).status_code == 422

    # A reply by thread id goes on the same thread and holds it again.
    reply = client.post("/api/session-threads/message",
                        json={"session_id": "sess-A", "thread_id": tid, "title": "reply", "body": "Juniper's rule."})
    assert reply.json()["thread"]["thread_id"] == tid and reply.json()["thread"]["status"] == "held"
    # Orion asked on it already: the session is told this is read as answers, nothing comes back to answer.
    assert reply.json()["answer_phase"] is True and "all 1 rounds of questions" in reply.json()["receipt"]["text"]
    # The note got NO QUESTIONS: its one round is not used.
    again = client.post("/api/session-threads/message",
                        json={"session_id": "sess-B", "thread_id": note, "title": "reply", "body": "more"})
    assert again.json()["answer_phase"] is False


def test_routes_are_registered_on_the_hub_app_and_pending_is_not_read_as_a_thread_id():
    paths = [r.path for r in routes.router.routes]
    assert paths.index("/api/session-threads/pending") < paths.index("/api/session-threads/{thread_id}")
    main_src = (HUB_ROOT / "scripts" / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(session_thread_router)" in main_src
    assert "session_roundup_loop = SessionRoundupLoop(" in main_src and "await session_roundup_loop.start()" in main_src


def test_a_repeat_is_said_to_be_a_repeat_and_a_full_thread_answers_429(world, monkeypatch):
    client = world[0]
    first = client.post("/api/session-threads/message", json=_msg()).json()
    again = client.post("/api/session-threads/message", json=_msg())
    assert again.status_code == 200 and again.json()["duplicate"] is True and again.json()["turn"] is None
    assert first["duplicate"] is False and "Nothing new was stored" in again.json()["receipt"]["text"]
    from orion.session_threads import core

    monkeypatch.setattr(core, "MAX_UNREAD_TURNS_PER_THREAD", 1)
    resp = client.post("/api/session-threads/message", json=_msg(body="another"))
    assert resp.status_code == 429 and "too_many_unread_messages_on_thread" in resp.json()["detail"]
