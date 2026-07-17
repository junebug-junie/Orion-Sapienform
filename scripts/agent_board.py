#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_board_lib import (  # noqa: E402
    add_item,
    board_config_from_env,
    change_item_status,
    close_presence,
    load_state,
    upsert_presence,
)


def _print_state(state, *, worktree: str | None = None) -> None:
    for row in state.presence.values():
        if worktree and row.get("worktree_path") != worktree:
            continue
        print(
            f"presence: {row.get('status')} {row.get('worktree_path')} "
            f"{row.get('branch', '')} :: {row.get('thread_summary', '')} "
            f":: {row.get('current_task', '')}".rstrip()
        )
    for item in state.items.values():
        if worktree and item.get("worktree_path") != worktree:
            continue
        print(
            f"item: {item.get('id')} [{item.get('status')}] "
            f"{item.get('severity')}/{item.get('kind')} :: {item.get('summary')}"
        )
        files = item.get("related_files") or []
        if files:
            print("  files: " + ", ".join(str(path) for path in files))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Host-local agent workspace board")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("--kind", required=True)
    add.add_argument("--severity", required=True)
    add.add_argument("--summary", required=True)
    add.add_argument("--scope", default="this-worktree", dest="owner_scope")
    add.add_argument("--scope-note", default="")
    add.add_argument("--parent")
    add.add_argument("--files", nargs="*", default=[])

    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("--summary")
    heartbeat.add_argument("--task")
    heartbeat.add_argument("--session-id")

    resolve = sub.add_parser("resolve")
    resolve.add_argument("item_id")
    resolve.add_argument("--status", default="resolved", choices=["resolved", "parked", "handed-off"])

    listing = sub.add_parser("list")
    listing.add_argument("--worktree")
    listing.add_argument("--all", action="store_true")

    sub.add_parser("checkout")

    args = parser.parse_args(argv)
    cfg = board_config_from_env()

    try:
        if args.command == "add":
            item_id = add_item(
                cfg,
                kind=args.kind,
                severity=args.severity,
                summary=args.summary,
                owner_scope=args.owner_scope,
                scope_note=args.scope_note,
                related_files=args.files,
                parent_id=args.parent,
            )
            print(f"item: {item_id}")
            return 0
        if args.command == "heartbeat":
            row = upsert_presence(cfg, summary=args.summary, task=args.task, session_id=args.session_id)
            print(f"presence: active {row['worktree_path']}")
            return 0
        if args.command == "resolve":
            change_item_status(cfg, args.item_id, args.status)
            print(f"item: {args.item_id} {args.status}")
            return 0
        if args.command == "list":
            _print_state(load_state(cfg), worktree=args.worktree)
            return 0
        if args.command == "checkout":
            close_presence(cfg)
            state = load_state(cfg)
            open_items = [
                item for item in state.items.values()
                if item.get("status") in {"open", "parked"}
            ]
            if open_items:
                print(f"Open items remain: {len(open_items)}")
            print("presence: closed")
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"agent-board error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
