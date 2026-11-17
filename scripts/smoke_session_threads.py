#!/usr/bin/env python3
"""Smoke for session threads against a LIVE Hub (run after deploy, not before).

    python3 scripts/smoke_session_threads.py                 # http://localhost:8080
    ORION_HUB_URL=http://host:8080 python3 scripts/smoke_session_threads.py

Posts one small note thread, reads it back, and checks Hub's receipt: stored as speaker `hub`, says
Orion has NOT read it, the thread is `held`, and no Orion turn exists yet. It writes ONE real note
thread ("smoke test, no questions needed"), which Orion reads at the next hourly round-up; that
round-up is the second half of the proof (rerun with --check THREAD_ID an hour later).

The one-round rule (Orion asks questions on a thread once), live:

    python3 scripts/smoke_session_threads.py --rule-since 2026-11-18T00:00:00Z

reads every thread touched in the last 48 h and fails if any Orion turn written since that time
asked questions on a thread that already had its round (read-only).

Exit 0: every check passed. Exit 1: a check failed (printed). Exit 2: Hub not reachable.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import orion_session_thread_lib as lib  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def post_and_read() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    resp = lib.send({"session_id": "smoke-session-threads", "session_name": "smoke script",
                     "title": f"Smoke test of session threads ({stamp})",
                     "body": "This is a smoke test of the session-threads path after a deploy. "
                             "Nothing here needs an answer: NO QUESTIONS is the right reply."}, timeout=10.0)
    thread = resp.get("thread") or {}
    tid = str(thread.get("thread_id") or "")
    ok = [
        check("message accepted, thread id returned", bool(tid)),
        check("receipt is Hub's", (resp.get("receipt") or {}).get("speaker") == "hub", str(resp.get("receipt"))),
        check("receipt says Orion has NOT read it", "Orion has NOT read it" in str((resp.get("receipt") or {}).get("text"))
              or resp.get("roundup_enabled") is False, str((resp.get("receipt") or {}).get("text"))),
        check("response says Orion has not read it", resp.get("orion_has_read") is False),
        check("thread is held", thread.get("status") == "held", str(thread.get("status"))),
    ]
    if not tid:
        return 1
    back = lib.request("GET", f"/api/session-threads/{tid}", timeout=10.0)
    turns = back.get("turns") or []
    ok += [
        check("thread reads back", (back.get("thread") or {}).get("thread_id") == tid),
        check("turns are session then hub", [t.get("speaker") for t in turns] == ["session", "hub"],
              str([t.get("speaker") for t in turns])),
        check("no Orion turn yet (no GPU turn at intake)", not any(t.get("speaker") == "orion" for t in turns)),
    ]
    listed = lib.request("GET", "/api/session-threads?hours=1", timeout=10.0).get("threads") or []
    ok.append(check("thread is in the 48 h list", any(t.get("thread_id") == tid for t in listed)))
    print(f"\nthread {tid}; next round-up {resp.get('next_roundup_at')} (round-up enabled: {resp.get('roundup_enabled')})")
    print(f"After that time: python3 scripts/smoke_session_threads.py --check {tid}")
    return 0 if all(ok) else 1


def check_answer(tid: str) -> int:
    back = lib.request("GET", f"/api/session-threads/{tid}", timeout=10.0)
    thread, turns = back.get("thread") or {}, back.get("turns") or []
    orion = [t for t in turns if t.get("speaker") == "orion"]
    print(f"status: {thread.get('status')}  reason: {thread.get('status_reason')}  round: {thread.get('last_roundup_id')}")
    ok = [
        check("Orion's turn is stored", bool(orion), "no orion turn yet: still held, reading, or failed"),
        check("Orion's turn has real text", all(str(t.get("body") or "").strip() for t in orion) and bool(orion)),
        check("status is asked, no_questions or closed", thread.get("status") in ("asked", "no_questions", "closed"),
              str(thread.get("status"))),
    ]
    for t in orion:
        print(f"\nOrion ({t.get('roundup_id')}):\n{t.get('body')}")
    return 0 if all(ok) else 1


def check_round_rule(since: str) -> int:
    """No Orion turn written at or after ``since`` asked questions on a thread whose round was used."""
    cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    threads = lib.request("GET", "/api/session-threads?hours=48&limit=200", timeout=10.0).get("threads") or []
    bad, closed, read = [], 0, 0
    for t in threads:
        turns = lib.request("GET", f"/api/session-threads/{t['thread_id']}", timeout=10.0).get("turns") or []
        for i, turn in enumerate(turns):
            if turn.get("speaker") != "orion":
                continue
            try:
                when = datetime.fromisoformat(str(turn.get("created_at")).replace("Z", "+00:00"))
            except ValueError:
                continue   # a turn with no readable time cannot be judged against the cutoff
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when < cutoff:
                continue
            read += 1
            closed += bool((turn.get("meta") or {}).get("closing_note"))
            if lib.answer_phase(turns[:i]) and (lib.has_questions(turn) or turn.get("pr_comment_url")):
                bad.append(f"{t['thread_id']}:{turn.get('turn_id')}")
    print(f"{len(threads)} thread(s), {read} Orion turn(s) since {cutoff.isoformat()}, {closed} closing note(s)")
    return 0 if check("no second round of questions on any thread", not bad, ", ".join(bad)) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="THREAD_ID", help="read a thread back after its round-up")
    parser.add_argument("--rule-since", metavar="ISO_TIME", help="check the one-round rule on turns since then")
    args = parser.parse_args()
    try:
        if args.rule_since:
            return check_round_rule(args.rule_since)
        return check_answer(args.check) if args.check else post_and_read()
    except Exception as exc:  # noqa: BLE001
        print(f"Hub not reachable at {lib.hub_url()}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
