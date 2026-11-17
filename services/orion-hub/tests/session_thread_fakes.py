"""In-memory stand-ins for session threads tests: the store (same signatures as the SQL functions in
scripts/session_roundup_loop.py), orion-durable-runs' HTTP API, and the GitHub commenter.

The store fake is checked against the real SQL: tests/test_session_roundup_loop.py runs every
lifecycle test twice, once on this fake and once on Postgres (when a test server is given)."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List

from orion.session_threads import core
from scripts import session_roundup_loop as srl

STORE_FUNCTIONS = ("intake_precheck", "add_message", "recent_threads", "get_thread", "thread_turns", "threads_with_status",
                   "roundup_exists", "mark_round", "release_threads", "store_answer", "fail_round",
                   "set_pr_comment_url", "pending_for", "mark_delivered")


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return self.now

    def advance(self, **kw) -> None:
        self.now = self.now + timedelta(**kw)

    def to_next_hour(self) -> None:
        self.now = core.next_roundup_at(self.now, 3600) + timedelta(seconds=5)


class FakeStore:
    def __init__(self, clock: Callable[[], datetime]) -> None:
        self.clock = clock
        self.threads: Dict[str, Dict[str, Any]] = {}
        self.turns: List[Dict[str, Any]] = []
        self.n = 0
        self.on_read: Callable[[], None] = lambda: None   # runs inside thread_turns (a slow read)

    def install(self, monkeypatch) -> None:
        for name in STORE_FUNCTIONS:
            monkeypatch.setattr(srl, name, getattr(self, name))

    def _id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def _add_turn(self, thread_id, speaker, body, *, session_id=None, roundup_id=None, meta=None):
        turn = dict(turn_id=self._id("u"), seq=len(self.turns) + 1, thread_id=thread_id, speaker=speaker, body=body,
                    session_id=session_id, roundup_id=roundup_id, delivered_at=None, pr_comment_url=None,
                    meta=copy.deepcopy(meta or {}), created_at=self.clock.now)
        self.turns.append(turn)
        return dict(turn)

    async def intake_precheck(self, pool, *, thread_id, source_ref):
        thread = self.threads.get(thread_id) if thread_id else next(
            (t for t in self.threads.values() if t["source_ref"] == source_ref), None)
        out = {"held_threads": sum(1 for t in self.threads.values() if t["status"] in ("held", "reading")),
               "thread": dict(thread) if thread else None, "unread": 0, "last_digest": None, "answer_phase": False}
        if thread:
            out["answer_phase"] = core.in_answer_phase(
                [u for u in self.turns if u["thread_id"] == thread["thread_id"] and u["speaker"] == "orion"])
            mine = [u for u in self.turns if u["thread_id"] == thread["thread_id"] and u["speaker"] == "session"]
            out.update(unread=sum(1 for u in mine if u["roundup_id"] is None),
                       last_digest=mine[-1]["meta"].get("digest") if mine else None)
        return out

    async def add_message(self, pool, *, thread_id, source_ref, kind, title, session_id, session_name, worktree, branch,
                          pr_number, pr_url, spec_paths, body, meta, receipt):
        if thread_id:
            thread = self.threads.get(thread_id)
            if thread is None:
                return None
        else:
            thread = next((t for t in self.threads.values() if t["source_ref"] == source_ref), None)
            if thread is None:
                tid = self._id("t")
                thread = self.threads[tid] = dict(
                    thread_id=tid, source_ref=source_ref, kind=kind, pr_number=pr_number, status="held",
                    last_roundup_id=None, created_at=self.clock.now)
            thread.update(title=title, session_id=session_id, session_name=session_name, worktree=worktree,
                          branch=branch, pr_url=pr_url, spec_paths=list(spec_paths))
        thread.update(status="reading" if thread["status"] == "reading" else "held", status_reason=None,
                      roundup_failures=0, updated_at=self.clock.now)
        turn = self._add_turn(thread["thread_id"], "session", body, session_id=session_id, meta=meta)
        hub = self._add_turn(thread["thread_id"], "hub", receipt(thread["thread_id"]), meta={"receipt": True})
        return {"thread": dict(thread), "turn": turn, "receipt": hub}

    async def recent_threads(self, pool, *, hours, limit=100):
        rows = sorted(self.threads.values(), key=lambda t: t["updated_at"], reverse=True)[:limit]
        return [{**t, "turns": sum(1 for u in self.turns if u["thread_id"] == t["thread_id"])} for t in rows]

    async def get_thread(self, pool, thread_id):
        thread = self.threads.get(thread_id)
        return dict(thread) if thread else None

    async def thread_turns(self, pool, thread_ids):
        out = [copy.deepcopy(u) for u in self.turns if u["thread_id"] in set(thread_ids)]
        self.on_read()
        return out

    async def threads_with_status(self, pool, status):
        rows = [dict(t) for t in self.threads.values() if t["status"] == status]
        return sorted(rows, key=lambda t: (t["updated_at"], t["thread_id"]))

    async def roundup_exists(self, pool, roundup_id):
        return any(t["last_roundup_id"] == roundup_id for t in self.threads.values())

    async def mark_round(self, pool, roundup_id, thread_ids, turn_ids):
        for tid in thread_ids:
            if self.threads[tid]["status"] == "held":
                self.threads[tid].update(status="reading", last_roundup_id=roundup_id, updated_at=self.clock.now)
        for u in self.turns:
            if u["turn_id"] in set(turn_ids) and u["roundup_id"] is None:
                u["roundup_id"] = roundup_id

    async def release_threads(self, pool, roundup_id, thread_ids):
        for u in self.turns:
            if u["roundup_id"] == roundup_id and u["speaker"] == "session" and u["thread_id"] in set(thread_ids):
                u["roundup_id"] = None
        for tid in thread_ids:
            t = self.threads[tid]
            if t["last_roundup_id"] == roundup_id and t["status"] == "reading":
                t.update(status="held", last_roundup_id=None, updated_at=self.clock.now)

    async def store_answer(self, pool, *, thread_id, roundup_id, body, meta, status):
        existing = next((u for u in self.turns if u["thread_id"] == thread_id and u["roundup_id"] == roundup_id
                         and u["speaker"] == "orion"), None)
        created = existing is None
        turn = existing or self._add_turn(thread_id, "orion", body, roundup_id=roundup_id, meta=meta)
        thread = self.threads[thread_id]
        if thread["status"] == "reading" and thread["last_roundup_id"] == roundup_id:
            unread = any(u["thread_id"] == thread_id and u["speaker"] == "session" and u["roundup_id"] is None
                         for u in self.turns)
            thread.update(status="held" if unread else status, status_reason=None, roundup_failures=0,
                          updated_at=self.clock.now)
        return dict(thread), dict(turn), created

    async def fail_round(self, pool, roundup_id, *, reason, text_for):
        out = []
        for u in self.turns:
            if u["roundup_id"] == roundup_id and u["speaker"] == "session":
                u["roundup_id"] = None
        for thread in self.threads.values():
            if thread["last_roundup_id"] == roundup_id and thread["status"] == "reading":
                n = thread["roundup_failures"] + 1
                thread.update(roundup_failures=n, status="failed" if n >= core.MAX_CONSECUTIVE_FAILURES else "held",
                              status_reason=reason[:300], updated_at=self.clock.now)
                turn = self._add_turn(thread["thread_id"], "hub", text_for(n), roundup_id=roundup_id,
                                      meta={"roundup_failed": True, "reason": reason[:300]})
                out.append((dict(thread), turn))
        return out

    async def set_pr_comment_url(self, pool, turn_id, url):
        next(u for u in self.turns if u["turn_id"] == turn_id)["pr_comment_url"] = url

    async def pending_for(self, pool, *, session_id="", branch="", limit=20):
        out = []
        for u in self.turns:
            if u["speaker"] != "orion" or u["delivered_at"] is not None:
                continue
            h = self.threads[u["thread_id"]]
            mine = bool(session_id) and (h["session_id"] == session_id or any(
                s["thread_id"] == u["thread_id"] and s["speaker"] == "session" and s["session_id"] == session_id
                for s in self.turns))
            if mine or (bool(branch) and h["branch"] == branch):
                out.append({**{k: u[k] for k in ("turn_id", "thread_id", "body", "meta", "roundup_id", "pr_comment_url",
                                                 "created_at")},
                            **{k: h[k] for k in ("title", "branch", "pr_number", "pr_url", "status")}})
        return out[:limit]

    async def mark_delivered(self, pool, turn_ids):
        n = 0
        for u in self.turns:
            if u["turn_id"] in set(turn_ids) and u["speaker"] == "orion" and u["delivered_at"] is None:
                u["delivered_at"] = self.clock.now
                n += 1
        return n


class FakeDurable:
    """orion-durable-runs' HTTP API. Each NEW run id takes the next reply: text completes the run,
    an Exception fails it, a dict is the run row as given, ``PENDING`` leaves it running. A second
    submit of a known run id is the same run (as in durable-runs), and is counted."""

    PENDING = object()

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.submits: List[Dict[str, Any]] = []
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.down = False
        self.refuse_with: int = 0

    async def submit(self, request_json):
        from orion.schemas.durable_run import DurableRunRequestV1

        if self.down:
            raise RuntimeError("durable runs unreachable")
        req = DurableRunRequestV1.model_validate(request_json)
        assert req.workflow == "session_roundup.orion_turn" and req.admission is not None
        self.submits.append(dict(run_id=req.run_id, prompt=req.brief.prompt, thread_ids=list(req.brief.thread_ids),
                                 session_id=req.brief.session_id, deadline_at=req.admission.deadline_at,
                                 resource=req.admission.resource, priority=req.admission.priority,
                                 timeout_sec=req.brief.timeout_sec, json=request_json))
        if self.refuse_with:
            return self.refuse_with, "refused"
        if req.run_id in self.runs:
            return 202, "{}"
        reply = self.replies.pop(0) if self.replies else self.PENDING
        if reply is self.PENDING:
            self.runs[req.run_id] = {"status": "running"}
        elif isinstance(reply, Exception):
            self.runs[req.run_id] = {"status": "failed", "error": str(reply)}
        elif isinstance(reply, dict):
            self.runs[req.run_id] = reply
        else:
            self.runs[req.run_id] = {"status": "completed", "harness_steps": {"turn": {"text": reply}}}
        return 202, "{}"

    def finish(self, run_id, text):
        self.runs[run_id] = {"status": "completed", "harness_steps": {"turn": {"text": text}}}

    async def get_run(self, run_id):
        if self.down:
            raise RuntimeError("durable runs unreachable")
        return self.runs.get(run_id)

    @property
    def new_runs(self) -> List[str]:
        seen: List[str] = []
        for s in self.submits:
            if s["run_id"] not in seen:
                seen.append(s["run_id"])
        return seen


class FakeCommenter:
    def __init__(self, *, fail=False):
        self.posts: List[tuple] = []
        self.branches: List[Any] = []
        self.fail = fail

    async def post(self, pr_number, body, *, branch=None):
        self.branches.append(branch)
        self.posts.append((pr_number, body))
        if self.fail:
            return None
        return f"https://github.com/o/r/pull/{pr_number}#issuecomment-{len(self.posts)}"
