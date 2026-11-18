"""Session threads API: coding sessions write to Orion, and read Orion's questions back.

``POST /api/session-threads/message``     a session's message (a PR's spec, a note, or a reply on a
                                          thread). Stores it, stores Hub's receipt, returns the receipt.
                                          No model runs: Orion reads it at the next hourly round-up.
``GET  /api/session-threads?hours=48``    threads touched in the window (Hub chat's page-load view)
``GET  /api/session-threads/pending``     ``?session_id=&branch=``: Orion's turns not yet handed back to
                                          that session OR that branch
``POST /api/session-threads/delivered``   {"turn_ids": [...], "session_id": "..."}: those turns were handed
                                          back to that session (a question turn stays pending for OTHER
                                          sessions until one replies on the thread)
``GET  /api/session-threads/{id}``        one thread with every turn
``GET  /api/debug/session-threads``       the loop's last tick

Design: docs/superpowers/specs/2026-11-15-session-threads-design.md. Loop and store:
scripts/session_roundup_loop.py.
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from scripts import session_roundup_loop as store

router = APIRouter()

# Above the stored caps on purpose: Hub cuts an over-long text and says so, instead of refusing it.
_WIRE_TEXT_MAX = 400000


class SpecFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(default="", max_length=1000)
    text: str = Field(default="", max_length=_WIRE_TEXT_MAX)


class MessageBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(min_length=1, max_length=200)
    session_name: Optional[str] = Field(default=None, max_length=400)
    worktree: Optional[str] = Field(default=None, max_length=1000)
    branch: Optional[str] = Field(default=None, max_length=600)
    pr_number: Optional[int] = Field(default=None, ge=1, le=10_000_000)
    pr_url: Optional[str] = Field(default=None, max_length=600)
    title: str = Field(default="", max_length=1000)
    body: str = Field(default="", max_length=_WIRE_TEXT_MAX)
    spec: List[SpecFile] = Field(default_factory=list, max_length=20)
    # A reply on an existing thread (scripts/orion_session_thread.py reply --thread ID).
    thread_id: Optional[str] = Field(default=None, max_length=64)


class DeliveredBody(BaseModel):
    turn_ids: List[str] = Field(min_length=1, max_length=200)
    # The claiming session. Empty (an older client): the turn is claimed once in total.
    session_id: str = Field(default="", max_length=200)


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "memory_pg_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres_pool_unavailable")
    return pool


def _loop() -> Any:
    from scripts import main as hub_main

    loop = getattr(hub_main, "session_roundup_loop", None)
    if loop is None:
        raise HTTPException(status_code=503, detail="session_threads_not_started")
    return loop


def _public(row: dict, *, with_spec_text: bool = False) -> dict:
    out = dict(row)
    meta = out.get("meta")
    if isinstance(meta, dict) and "spec" in meta and not with_spec_text:
        # The list view and the session hooks need the paths, not 40,000 characters a file.
        out["meta"] = {**meta, "spec": [{"path": s.get("path"), "chars": len(str(s.get("text") or "")),
                                         "cut": bool(s.get("cut"))} for s in meta["spec"] or []]}
    if isinstance(out.get("meta"), dict) and "delivered_to" in out["meta"]:
        # Which sessions claimed a turn is Hub's bookkeeping, not another session's business.
        out["meta"] = {k: v for k, v in out["meta"].items() if k != "delivered_to"}
    for key, value in list(out.items()):
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
    return out


@router.post("/api/session-threads/message")
async def post_message(request: Request, body: MessageBody) -> dict:
    _pool(request)
    loop = _loop()
    try:
        result = await loop.intake(
            session_id=body.session_id, session_name=body.session_name, worktree=body.worktree, branch=body.branch,
            pr_number=body.pr_number, pr_url=body.pr_url, title=body.title, body=body.body,
            spec=[s.model_dump() for s in body.spec], thread_id=(body.thread_id or "").strip() or None)
    except store.IntakeRefused as exc:
        raise HTTPException(status_code=429 if exc.too_many else 422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="thread_not_found")
    return {
        "thread": _public(result["thread"]),
        "turn": _public(result["turn"]) if result["turn"] else None,
        # True: this exact message is already the newest on the thread; nothing new was stored.
        "duplicate": bool(result.get("duplicate")),
        # Hub's receipt, not Orion's words: Orion has not read the message.
        "receipt": {"speaker": "hub", "text": result["receipt"]["body"], "turn_id": result["receipt"]["turn_id"]},
        "next_roundup_at": result["next_roundup_at"].isoformat(),
        "roundup_enabled": result["roundup_enabled"],
        # True: Orion already asked its one round of questions on this thread; this message is read as
        # answers and Orion writes only a closing note (orion/session_threads/core.py, one-round rule).
        "answer_phase": bool(result.get("answer_phase")),
        "orion_has_read": False,
    }


@router.get("/api/session-threads")
async def list_threads(request: Request, hours: float = 48.0, limit: int = 100) -> dict:
    rows = await store.recent_threads(_pool(request), hours=max(0.1, min(hours, 24 * 30)),
                                      limit=max(1, min(limit, 200)))
    return {"threads": [_public(r) for r in rows]}


@router.get("/api/session-threads/pending")
async def pending(request: Request, session_id: str = "", branch: str = "") -> dict:
    if not session_id.strip() and not branch.strip():
        raise HTTPException(status_code=422, detail="session_id_or_branch_required")
    rows = await store.pending_for(_pool(request), session_id=session_id.strip(), branch=branch.strip())
    return {"pending": [_public(r) for r in rows]}


@router.post("/api/session-threads/delivered")
async def delivered(request: Request, body: DeliveredBody) -> dict:
    return {"delivered": await store.mark_delivered(_pool(request), body.turn_ids, session_id=body.session_id.strip())}


@router.get("/api/session-threads/{thread_id}")
async def get_thread(request: Request, thread_id: str, spec_text: int = 0) -> dict:
    pool = _pool(request)
    row = await store.get_thread(pool, thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="thread_not_found")
    turns = await store.thread_turns(pool, [thread_id])
    return {"thread": _public(row), "turns": [_public(t, with_spec_text=bool(spec_text)) for t in turns]}


@router.get("/api/debug/session-threads")
async def session_threads_debug() -> dict:
    from scripts import main as hub_main

    loop = getattr(hub_main, "session_roundup_loop", None)
    return loop.status() if loop is not None else {"enabled": False, "reason": "not_started"}
