#!/usr/bin/env python3
"""Write to Orion from a coding session, and read Orion's questions back.

    python3 scripts/orion_session_thread.py send --pr 3456               # the PR's design spec(s)
    python3 scripts/orion_session_thread.py send --pr 3456 --note "..."  # with your own words
    python3 scripts/orion_session_thread.py send --note "..."            # a note, no PR
    python3 scripts/orion_session_thread.py reply --thread ID --body "..."
    python3 scripts/orion_session_thread.py reply --thread ID --pr 3456 --body-file answer.md  # + PR comment
    python3 scripts/orion_session_thread.py pending                      # Orion's questions for this session

Hub answers a send with a RECEIPT (Hub's, not Orion's). Orion reads every held message together at
the next hourly round-up and comes back with questions. Hub URL: ORION_HUB_URL (default
http://localhost:8080). Design: docs/superpowers/specs/2026-11-15-session-threads-design.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import orion_session_thread_lib as lib  # noqa: E402


def cmd_send(args: argparse.Namespace) -> int:
    session_id = lib.session_id_from()
    ident = lib.identity(session_id)
    if args.pr:
        pr = lib.pr_info(str(args.pr), cwd=ident["worktree"] or lib.REPO_ROOT)
        if pr is None:
            print(f"No pull request found for {args.pr} (gh pr view failed).", file=sys.stderr)
            return 2
        message = lib.build_pr_message(pr, session_id=session_id, worktree=ident["worktree"], note=args.note or "")
        if message is None:
            print(f"PR #{pr['number']} has no spec file under docs/superpowers/specs/. Nothing sent. "
                  "Pass --note to send your own words.", file=sys.stderr)
            return 2
    else:
        note = (args.note or "").strip()
        if not note:
            print("Nothing to send: give --pr N or --note TEXT.", file=sys.stderr)
            return 2
        message = {"session_id": session_id, "session_name": lib.session_name(ident["worktree"]),
                   "worktree": ident["worktree"], "branch": ident["branch"],
                   "title": args.title or note.splitlines()[0][:120], "body": note, "spec": []}
    resp = lib.send(message)
    print(lib.receipt_line(resp))
    return 0


def cmd_reply(args: argparse.Namespace) -> int:
    body = args.body if args.body is not None else Path(args.body_file).read_text(encoding="utf-8")
    if not body.strip():
        print("Nothing to send: the reply is empty.", file=sys.stderr)
        return 2
    resp = lib.reply(args.thread, body, session_id=lib.session_id_from())
    print(lib.receipt_line(resp))
    if args.pr:
        # The same words on the pull request, where Juniper reads them. Hub first: Orion's copy matters most.
        try:
            proc = subprocess.run(["gh", "pr", "comment", str(args.pr), "--body-file", "-"], input=body,
                                  capture_output=True, text=True, timeout=30, check=False)
        except Exception as exc:  # noqa: BLE001 -- no gh / timeout
            proc, why = None, f"{type(exc).__name__}: {exc}"
        else:
            why = (proc.stderr or proc.stdout).strip()[:300]
        if proc is None or proc.returncode != 0:
            # Orion already has it: retry only the comment, never the whole reply.
            print(f"Sent to Orion, but the PR comment failed: {why}\nRetry only the comment: "
                  f"gh pr comment {args.pr} --body-file {args.body_file or '<file with the reply>'}", file=sys.stderr)
            return 1
        print(f"Posted on PR #{args.pr}: {proc.stdout.strip()}")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    session_id = lib.session_id_from()
    branch = args.branch or lib.identity(session_id)["branch"]
    rows = lib.pending(session_id=session_id, branch=branch, timeout=6.0)
    if not rows:
        print("Nothing from Orion is waiting for this session.")
        return 0
    if not args.keep:
        rows = lib.claim(rows, session_id=session_id, timeout=6.0)   # only what no hook has shown yet
    print(lib.pending_text(rows) or "Nothing from Orion is waiting for this session.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    send = sub.add_parser("send", help="send a PR's spec, or a note, to Orion")
    send.add_argument("--pr", help="PR number or URL")
    send.add_argument("--note", help="your own words (with --pr: sent with the spec)")
    send.add_argument("--title", help="title for a note thread")
    send.set_defaults(fn=cmd_send)
    rep = sub.add_parser("reply", help="answer Orion on a thread")
    rep.add_argument("--thread", required=True)
    body = rep.add_mutually_exclusive_group(required=True)
    body.add_argument("--body")
    body.add_argument("--body-file", help="read the reply from this file")
    rep.add_argument("--pr", type=int, help="also post the reply as a comment on this pull request")
    rep.set_defaults(fn=cmd_reply)
    pend = sub.add_parser("pending", help="Orion's questions waiting for this session")
    pend.add_argument("--branch", help="branch to look up (default: this session's)")
    pend.add_argument("--keep", action="store_true", help="do not mark them as delivered")
    pend.set_defaults(fn=cmd_pending)
    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except Exception as exc:  # noqa: BLE001 -- say what failed, in one line
        print(f"Could not reach Hub at {lib.hub_url()}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
