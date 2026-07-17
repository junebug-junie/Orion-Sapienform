# Agent Workspace Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vendor-neutral, host-local real-time board so concurrent agents can see open decisions/findings, per-worktree presence, and disclosure-only collision warnings.

**Architecture:** Put all durable behavior in a plain Python CLI and library under `scripts/`; vendor hooks are thin adapters that call the CLI and fail open. Store live truth in `~/.orion/agent-board.jsonl` using append-only JSONL events protected by an `fcntl` lock, then materialize current board state on read.

**Tech Stack:** Python standard library only (`argparse`, `dataclasses`, `fcntl`, `json`, `pathlib`, `subprocess`, `uuid`, `datetime`), existing `scripts/worktree_lib.py`, existing `tests/scripts/` pytest pattern, Claude `.claude/settings.json`, Cursor `.cursor/hooks.json`.

## Global Constraints

- Live board file: `~/.orion/agent-board.jsonl`.
- Stale threshold: 30 minutes.
- Collision behavior: disclosure only; no non-zero exit and no acknowledgement requirement in v1.
- Hooks must fail open and never block session start/stop.
- The host-local board is not committed; only scripts, hooks, tests, docs, and spec/plan are committed.
- No new package dependency.
- Reuse `scripts/worktree_lib.py`; do not duplicate `git worktree list` parsing.
- No bus, schema, Docker, or env key changes.
- Keep AGENTS.md and CLAUDE.md in sync. In this repo, `AGENTS.md` points at `CLAUDE.md`; modify `CLAUDE.md` and verify both paths reflect the new text.

---

## File Structure

- Create `scripts/agent_board_lib.py`: pure library for paths, locking, JSONL events, state materialization, validation, worktree resolution, stale/closed reconciliation, collision detection, and text rendering.
- Create `scripts/agent_board.py`: CLI entry point that exposes `checkin`, `heartbeat`, `add`, `resolve`, `list`, `checkout`, and `reconcile`.
- Create `scripts/hooks/session_start_agent_board.py`: Claude SessionStart adapter that emits JSON `additionalContext` from `agent_board_lib.render_checkin_context()` and fails silently outside a git repo or on board errors.
- Create `scripts/hooks/session_stop_agent_board.py`: Claude Stop adapter that emits a short checkout reminder/context and fails open.
- Create `.cursor/hooks.json`: Cursor project hook adapter invoking the same CLI scripts for `sessionStart` and `stop`.
- Modify `.claude/settings.json`: add the SessionStart board hook next to the existing worktree summary hook; add a Stop hook for the board reminder.
- Modify `scripts/prune_merged_worktrees.py`: when a worktree is successfully removed, mark its presence closed in the board.
- Modify `CLAUDE.md`: add a short Agent Workspace Board section in the worktree rules area.
- Create `tests/scripts/test_agent_board_lib.py`: unit tests for locking/materialization/validation/reconcile/collision/rendering.
- Create `tests/scripts/test_agent_board_cli.py`: subprocess tests for CLI behavior.
- Create `tests/scripts/test_session_agent_board_hooks.py`: hook JSON/fail-open tests.
- Modify `tests/scripts/test_prune_merged_worktrees.py`: assert prune closes board presence for a removed worktree.

---

### Task 1: Board Storage, Schema, Locking, and Validation

**Files:**
- Create: `scripts/agent_board_lib.py`
- Test: `tests/scripts/test_agent_board_lib.py`

**Interfaces:**
- Produces:
  - `DEFAULT_BOARD_PATH: Path`
  - `STALE_AFTER_MINUTES: int`
  - `BoardConfig(board_path: Path = DEFAULT_BOARD_PATH, stale_after_minutes: int = 30)`
  - `BoardEvent(type: str, at: str, payload: dict[str, object])`
  - `BoardState(presence: dict[str, dict[str, object]], items: dict[str, dict[str, object]])`
  - `board_config_from_env() -> BoardConfig`
  - `append_event(config: BoardConfig, event_type: str, payload: dict[str, object], *, now: datetime | None = None) -> BoardEvent`
  - `load_state(config: BoardConfig, *, now: datetime | None = None, live_worktrees: set[str] | None = None) -> BoardState`
  - `validate_item_payload(payload: dict[str, object]) -> None`
- Consumes:
  - Python standard library only.
  - Tests may set `ORION_AGENT_BOARD_PATH` to isolate state.

- [ ] **Step 1: Write failing storage and validation tests**

Create `tests/scripts/test_agent_board_lib.py` with these tests:

```python
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from agent_board_lib import (  # noqa: E402
    BoardConfig,
    append_event,
    load_state,
    validate_item_payload,
)


def _config(tmp_path: Path) -> BoardConfig:
    return BoardConfig(board_path=tmp_path / "agent-board.jsonl", stale_after_minutes=30)


def test_append_event_creates_parent_dir_and_valid_jsonl(tmp_path: Path) -> None:
    cfg = _config(tmp_path / "nested")
    event = append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
        now=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
    )

    assert cfg.board_path.exists()
    raw = cfg.board_path.read_text(encoding="utf-8").strip()
    decoded = json.loads(raw)
    assert decoded["type"] == "presence_upserted"
    assert decoded["at"] == "2026-07-16T12:00:00+00:00"
    assert decoded["payload"]["worktree_path"] == "/repo/wt-a"
    assert event.type == "presence_upserted"


def test_load_state_materializes_last_presence_event(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(cfg, "presence_upserted", {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"})
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/wt-a",
            "branch": "feat/a",
            "status": "active",
            "thread_summary": "Working on the board.",
            "current_task": "Writing tests.",
        },
    )

    state = load_state(cfg, live_worktrees={"/repo/wt-a"})

    assert state.presence["/repo/wt-a"]["thread_summary"] == "Working on the board."
    assert state.presence["/repo/wt-a"]["current_task"] == "Writing tests."


def test_owner_scope_requires_scope_note_when_not_this_worktree() -> None:
    with pytest.raises(ValueError, match="scope_note is required"):
        validate_item_payload(
            {
                "id": "item-1",
                "kind": "finding",
                "severity": "should",
                "owner_scope": "juniper",
                "worktree_path": "/repo/wt-a",
                "summary": "Needs a scope note.",
                "status": "open",
            }
        )


def test_item_validation_accepts_this_worktree_without_scope_note() -> None:
    validate_item_payload(
        {
            "id": "item-1",
            "kind": "finding",
            "severity": "should",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/wt-a",
            "summary": "Scoped to current worktree.",
            "status": "open",
        }
    )


def test_old_active_presence_becomes_stale(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
        now=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
    )

    state = load_state(
        cfg,
        now=datetime(2026, 7, 16, 12, 31, tzinfo=UTC),
        live_worktrees={"/repo/wt-a"},
    )

    assert state.presence["/repo/wt-a"]["status"] == "stale"


def test_missing_live_worktree_becomes_closed(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/wt-a", "branch": "feat/a", "status": "active"},
    )

    state = load_state(cfg, live_worktrees={"/repo/other"})

    assert state.presence["/repo/wt-a"]["status"] == "closed"


def test_concurrent_appends_do_not_corrupt_jsonl(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    def write_one(index: int) -> None:
        append_event(
            cfg,
            "item_upserted",
            {
                "id": f"item-{index}",
                "kind": "finding",
                "severity": "note",
                "owner_scope": "this-worktree",
                "worktree_path": "/repo/wt-a",
                "summary": f"Finding {index}",
                "status": "open",
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_one, range(50)))

    lines = cfg.board_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50
    decoded = [json.loads(line) for line in lines]
    assert {row["payload"]["id"] for row in decoded} == {f"item-{index}" for index in range(50)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_lib.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'agent_board_lib'`.

- [ ] **Step 3: Implement minimal library**

Create `scripts/agent_board_lib.py`:

```python
"""Host-local agent workspace board.

The board is intentionally vendor-neutral: a locked JSONL event log plus
plain functions that CLI scripts and hook adapters can call.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

DEFAULT_BOARD_PATH = Path.home() / ".orion" / "agent-board.jsonl"
STALE_AFTER_MINUTES = 30

_PRESENCE_STATUSES = {"active", "stale", "closed"}
_ITEM_KINDS = {"decision", "finding", "blocker", "followup", "theme"}
_ITEM_SEVERITIES = {"blocker", "should", "note"}
_OWNER_SCOPES = {"this-worktree", "other-worktree", "juniper", "unassigned"}
_ITEM_STATUSES = {"open", "parked", "resolved", "handed-off"}


@dataclass(frozen=True)
class BoardConfig:
    board_path: Path = DEFAULT_BOARD_PATH
    stale_after_minutes: int = STALE_AFTER_MINUTES


@dataclass(frozen=True)
class BoardEvent:
    type: str
    at: str
    payload: dict[str, object]


@dataclass(frozen=True)
class BoardState:
    presence: dict[str, dict[str, object]]
    items: dict[str, dict[str, object]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def board_config_from_env() -> BoardConfig:
    raw = os.environ.get("ORION_AGENT_BOARD_PATH")
    if raw:
        return BoardConfig(board_path=Path(raw).expanduser())
    return BoardConfig()


@contextmanager
def _locked_board(config: BoardConfig) -> Iterator[None]:
    config.board_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = config.board_path.with_suffix(config.board_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_event(
    config: BoardConfig,
    event_type: str,
    payload: dict[str, object],
    *,
    now: datetime | None = None,
) -> BoardEvent:
    event = BoardEvent(type=event_type, at=_iso(now or _utc_now()), payload=dict(payload))
    with _locked_board(config):
        with config.board_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.__dict__, sort_keys=True) + "\n")
    return event


def validate_item_payload(payload: dict[str, object]) -> None:
    kind = payload.get("kind")
    severity = payload.get("severity")
    owner_scope = payload.get("owner_scope")
    status = payload.get("status")
    if kind not in _ITEM_KINDS:
        raise ValueError(f"invalid kind: {kind!r}")
    if severity not in _ITEM_SEVERITIES:
        raise ValueError(f"invalid severity: {severity!r}")
    if owner_scope not in _OWNER_SCOPES:
        raise ValueError(f"invalid owner_scope: {owner_scope!r}")
    if status not in _ITEM_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    if owner_scope != "this-worktree" and not str(payload.get("scope_note") or "").strip():
        raise ValueError("scope_note is required when owner_scope is not this-worktree")
    for required in ("id", "worktree_path", "summary"):
        if not str(payload.get(required) or "").strip():
            raise ValueError(f"{required} is required")


def _read_events(config: BoardConfig) -> list[BoardEvent]:
    if not config.board_path.exists():
        return []
    events: list[BoardEvent] = []
    with _locked_board(config):
        for line in config.board_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            events.append(BoardEvent(type=row["type"], at=row["at"], payload=row["payload"]))
    return events


def _apply_stale_and_closed(
    presence: dict[str, dict[str, object]],
    *,
    now: datetime,
    live_worktrees: set[str] | None,
    stale_after_minutes: int,
) -> None:
    stale_cutoff = now - timedelta(minutes=stale_after_minutes)
    for worktree_path, row in presence.items():
        if row.get("status") == "closed":
            continue
        if live_worktrees is not None and worktree_path not in live_worktrees:
            row["status"] = "closed"
            continue
        heartbeat_raw = str(row.get("heartbeat_at") or row.get("updated_at") or "")
        if heartbeat_raw:
            heartbeat = datetime.fromisoformat(heartbeat_raw)
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=UTC)
            if heartbeat < stale_cutoff:
                row["status"] = "stale"


def load_state(
    config: BoardConfig,
    *,
    now: datetime | None = None,
    live_worktrees: set[str] | None = None,
) -> BoardState:
    presence: dict[str, dict[str, object]] = {}
    items: dict[str, dict[str, object]] = {}
    for event in _read_events(config):
        payload = dict(event.payload)
        payload.setdefault("updated_at", event.at)
        if event.type == "presence_upserted":
            worktree_path = str(payload["worktree_path"])
            existing = presence.get(worktree_path, {})
            merged = {**existing, **payload}
            merged.setdefault("status", "active")
            merged.setdefault("heartbeat_at", event.at)
            presence[worktree_path] = merged
        elif event.type == "presence_closed":
            worktree_path = str(payload["worktree_path"])
            existing = presence.get(worktree_path, {"worktree_path": worktree_path})
            presence[worktree_path] = {**existing, "status": "closed", "updated_at": event.at}
        elif event.type == "item_upserted":
            validate_item_payload(payload)
            items[str(payload["id"])] = payload
        elif event.type == "item_status_changed":
            item_id = str(payload["id"])
            if item_id in items:
                items[item_id] = {**items[item_id], "status": payload["status"], "updated_at": event.at}
    _apply_stale_and_closed(
        presence,
        now=now or _utc_now(),
        live_worktrees=live_worktrees,
        stale_after_minutes=config.stale_after_minutes,
    )
    return BoardState(presence=presence, items=items)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_lib.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_board_lib.py tests/scripts/test_agent_board_lib.py
git commit -m "feat: add agent board storage"
```

---

### Task 2: Worktree Identity, CLI Add/List/Resolve/Heartbeat/Checkout

**Files:**
- Modify: `scripts/agent_board_lib.py`
- Create: `scripts/agent_board.py`
- Test: `tests/scripts/test_agent_board_cli.py`

**Interfaces:**
- Consumes:
  - `BoardConfig`, `append_event`, `load_state`, `validate_item_payload` from Task 1.
  - `worktree_lib.repo_toplevel()` for cwd worktree identity.
- Produces:
  - `current_worktree_identity(cwd: Path | None = None) -> dict[str, str]`
  - `upsert_presence(config: BoardConfig, *, summary: str | None, task: str | None, session_id: str | None) -> dict[str, object]`
  - CLI command `python3 scripts/agent_board.py add --kind finding --severity note --summary "Example finding."`
  - CLI command `python3 scripts/agent_board.py list --all`
  - CLI command `python3 scripts/agent_board.py resolve ID`
  - CLI command `python3 scripts/agent_board.py heartbeat --summary "One paragraph." --task "Specific task."`
  - CLI command `python3 scripts/agent_board.py checkout`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/scripts/test_agent_board_cli.py`:

```python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "agent_board.py"


def _run(cwd: Path, board_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)}
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_add_and_list_this_worktree_item(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    add = _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "should",
        "--summary",
        "Review found an unaddressed edge case.",
        "--files",
        "scripts/agent_board.py",
    )
    assert add.returncode == 0, add.stderr
    assert "item:" in add.stdout

    listed = _run(primary_repo, board, "list", "--worktree", str(primary_repo))

    assert listed.returncode == 0
    assert "Review found an unaddressed edge case." in listed.stdout
    assert "scripts/agent_board.py" in listed.stdout


def test_add_rejects_juniper_scope_without_note(primary_repo: Path, tmp_path: Path) -> None:
    proc = _run(
        primary_repo,
        tmp_path / "agent-board.jsonl",
        "add",
        "--kind",
        "decision",
        "--severity",
        "blocker",
        "--scope",
        "juniper",
        "--summary",
        "Needs owner decision.",
    )

    assert proc.returncode == 2
    assert "scope_note is required" in proc.stderr


def test_heartbeat_updates_presence(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    heartbeat = _run(
        primary_repo,
        board,
        "heartbeat",
        "--summary",
        "Building the board CLI.",
        "--task",
        "Task 2 CLI commands.",
    )
    assert heartbeat.returncode == 0

    listed = _run(primary_repo, board, "list", "--all")

    assert "Building the board CLI." in listed.stdout
    assert "Task 2 CLI commands." in listed.stdout


def test_resolve_marks_item_resolved(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    add = _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "followup",
        "--severity",
        "note",
        "--summary",
        "Close after verification.",
    )
    item_id = add.stdout.strip().split("item:", 1)[1].strip()

    resolved = _run(primary_repo, board, "resolve", item_id)
    listed = _run(primary_repo, board, "list", "--all")

    assert resolved.returncode == 0
    assert "resolved" in listed.stdout


def test_checkout_closes_presence_but_does_not_fail_on_open_items(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    _run(primary_repo, board, "heartbeat", "--summary", "Short session.", "--task", "Parking an item.")
    _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "finding",
        "--severity",
        "note",
        "--summary",
        "Open but non-blocking.",
    )

    checkout = _run(primary_repo, board, "checkout")
    listed = _run(primary_repo, board, "list", "--all")

    assert checkout.returncode == 0
    assert "Open items remain" in checkout.stdout
    assert "closed" in listed.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_cli.py -q
```

Expected: FAIL because `scripts/agent_board.py` does not exist.

- [ ] **Step 3: Add worktree helpers to the library**

Append these functions to `scripts/agent_board_lib.py`:

```python
import subprocess
import uuid
from pathlib import Path

from worktree_lib import WorktreeLibError, repo_toplevel


def _git_branch(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def current_worktree_identity(cwd: Path | None = None) -> dict[str, str]:
    root = Path(repo_toplevel()).resolve()
    branch = _git_branch(root)
    return {"worktree_path": str(root), "branch": branch}


def upsert_presence(
    config: BoardConfig,
    *,
    summary: str | None = None,
    task: str | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    identity = current_worktree_identity()
    payload: dict[str, object] = {
        "worktree_path": identity["worktree_path"],
        "branch": identity["branch"],
        "status": "active",
        "heartbeat_at": _iso(_utc_now()),
    }
    if summary is not None:
        payload["thread_summary"] = summary
    if task is not None:
        payload["current_task"] = task
    if session_id is not None:
        payload["session_id"] = session_id
    append_event(config, "presence_upserted", payload)
    return payload


def add_item(
    config: BoardConfig,
    *,
    kind: str,
    severity: str,
    summary: str,
    owner_scope: str = "this-worktree",
    scope_note: str = "",
    related_files: list[str] | None = None,
    parent_id: str | None = None,
) -> str:
    identity = current_worktree_identity()
    item_id = str(uuid.uuid4())
    payload: dict[str, object] = {
        "id": item_id,
        "kind": kind,
        "severity": severity,
        "owner_scope": owner_scope,
        "scope_note": scope_note,
        "worktree_path": identity["worktree_path"],
        "branch": identity["branch"],
        "summary": summary,
        "status": "open",
        "related_files": related_files or [],
    }
    if parent_id:
        payload["parent_id"] = parent_id
    validate_item_payload(payload)
    append_event(config, "item_upserted", payload)
    return item_id


def change_item_status(config: BoardConfig, item_id: str, status: str = "resolved") -> None:
    if status not in _ITEM_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    append_event(config, "item_status_changed", {"id": item_id, "status": status})


def close_presence(config: BoardConfig, worktree_path: str | None = None) -> None:
    path = worktree_path
    if path is None:
        path = current_worktree_identity()["worktree_path"]
    append_event(config, "presence_closed", {"worktree_path": path, "status": "closed"})
```

- [ ] **Step 4: Create the CLI**

Create `scripts/agent_board.py`:

```python
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
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Run library tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_lib.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/agent_board.py scripts/agent_board_lib.py tests/scripts/test_agent_board_cli.py
git commit -m "feat: add agent board cli"
```

---

### Task 3: Checkin Rendering, Global Strip, Reconcile, and Collision Warnings

**Files:**
- Modify: `scripts/agent_board_lib.py`
- Modify: `scripts/agent_board.py`
- Test: `tests/scripts/test_agent_board_lib.py`
- Test: `tests/scripts/test_agent_board_cli.py`

**Interfaces:**
- Consumes:
  - Task 1/2 library and CLI.
  - `worktree_lib.list_worktrees()`.
- Produces:
  - `live_worktree_paths() -> set[str]`
  - `reconcile_closed_worktrees(config: BoardConfig, live_paths: set[str] | None = None) -> BoardState`
  - `detect_collisions(state: BoardState, current_worktree_path: str) -> list[str]`
  - `render_checkin_context(config: BoardConfig) -> str`
  - CLI command `checkin`.
  - CLI command `reconcile`.

- [ ] **Step 1: Add failing render/reconcile/collision tests**

Append these tests to `tests/scripts/test_agent_board_lib.py`:

```python
from agent_board_lib import (  # noqa: E402
    detect_collisions,
    reconcile_closed_worktrees,
    render_checkin_context,
)


def test_render_checkin_includes_this_worktree_global_strip_and_presence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/current",
            "branch": "feat/current",
            "status": "active",
            "thread_summary": "Current thread.",
            "current_task": "Implementing checkin.",
        },
    )
    append_event(
        cfg,
        "presence_upserted",
        {
            "worktree_path": "/repo/other",
            "branch": "feat/other",
            "status": "active",
            "thread_summary": "Other agent summary.",
            "current_task": "Touching scripts.",
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "current-item",
            "kind": "finding",
            "severity": "should",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/current",
            "summary": "Current worktree finding.",
            "status": "open",
            "related_files": ["scripts/agent_board.py"],
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "global-blocker",
            "kind": "blocker",
            "severity": "blocker",
            "owner_scope": "juniper",
            "scope_note": "Needs Juniper decision.",
            "worktree_path": "/repo/other",
            "summary": "Global blocker.",
            "status": "open",
            "related_files": [],
        },
    )
    monkeypatch.setattr(
        "agent_board_lib.current_worktree_identity",
        lambda: {"worktree_path": "/repo/current", "branch": "feat/current"},
    )
    monkeypatch.setattr("agent_board_lib.live_worktree_paths", lambda: {"/repo/current", "/repo/other"})

    output = render_checkin_context(cfg)

    assert "This worktree" in output
    assert "Current worktree finding." in output
    assert "Global strip" in output
    assert "Global blocker." in output
    assert "Workspace presence" in output
    assert "Other agent summary." in output


def test_detect_collisions_reports_overlapping_related_files(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/current", "branch": "feat/current", "status": "active"},
    )
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/other", "branch": "feat/other", "status": "active"},
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "current-item",
            "kind": "finding",
            "severity": "note",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/current",
            "summary": "Current file touch.",
            "status": "open",
            "related_files": ["scripts/agent_board.py"],
        },
    )
    append_event(
        cfg,
        "item_upserted",
        {
            "id": "other-item",
            "kind": "finding",
            "severity": "note",
            "owner_scope": "this-worktree",
            "worktree_path": "/repo/other",
            "summary": "Other file touch.",
            "status": "open",
            "related_files": ["scripts/agent_board.py"],
        },
    )
    state = load_state(cfg, live_worktrees={"/repo/current", "/repo/other"})

    collisions = detect_collisions(state, "/repo/current")

    assert collisions == [
        "Potential collision with /repo/other: overlapping files scripts/agent_board.py"
    ]


def test_reconcile_closed_worktrees_persists_closed_event(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    append_event(
        cfg,
        "presence_upserted",
        {"worktree_path": "/repo/old", "branch": "feat/old", "status": "active"},
    )

    state = reconcile_closed_worktrees(cfg, live_paths={"/repo/current"})
    reloaded = load_state(cfg, live_worktrees={"/repo/current"})

    assert state.presence["/repo/old"]["status"] == "closed"
    assert reloaded.presence["/repo/old"]["status"] == "closed"
```

Append this CLI test to `tests/scripts/test_agent_board_cli.py`:

```python
def test_checkin_prints_three_layer_context(primary_repo: Path, tmp_path: Path) -> None:
    board = tmp_path / "agent-board.jsonl"
    _run(
        primary_repo,
        board,
        "add",
        "--kind",
        "blocker",
        "--severity",
        "blocker",
        "--summary",
        "Global issue visible at checkin.",
        "--scope",
        "juniper",
        "--scope-note",
        "Needs Juniper decision.",
    )

    proc = _run(primary_repo, board, "checkin")

    assert proc.returncode == 0
    assert "This worktree" in proc.stdout
    assert "Global strip" in proc.stdout
    assert "Workspace presence" in proc.stdout
    assert "Global issue visible at checkin." in proc.stdout
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_lib.py tests/scripts/test_agent_board_cli.py -q
```

Expected: FAIL because render/reconcile functions and `checkin` command are not defined.

- [ ] **Step 3: Implement live worktree, reconcile, collision, and rendering functions**

Append to `scripts/agent_board_lib.py`:

```python
from worktree_lib import list_worktrees


def live_worktree_paths() -> set[str]:
    return {str(info.path.resolve()) for info in list_worktrees()}


def reconcile_closed_worktrees(
    config: BoardConfig,
    live_paths: set[str] | None = None,
) -> BoardState:
    live = live_paths if live_paths is not None else live_worktree_paths()
    state = load_state(config, live_worktrees=live)
    for worktree_path, row in state.presence.items():
        if row.get("status") == "closed":
            continue
        if worktree_path not in live:
            append_event(config, "presence_closed", {"worktree_path": worktree_path, "status": "closed"})
    return load_state(config, live_worktrees=live)


def _active_item_files(state: BoardState, worktree_path: str) -> set[str]:
    files: set[str] = set()
    for item in state.items.values():
        if item.get("worktree_path") != worktree_path:
            continue
        if item.get("status") not in {"open", "parked"}:
            continue
        for path in item.get("related_files") or []:
            files.add(str(path))
    return files


def detect_collisions(state: BoardState, current_worktree_path: str) -> list[str]:
    current_files = _active_item_files(state, current_worktree_path)
    if not current_files:
        return []
    warnings: list[str] = []
    for worktree_path, presence in sorted(state.presence.items()):
        if worktree_path == current_worktree_path:
            continue
        if presence.get("status") not in {"active", "stale"}:
            continue
        overlap = sorted(current_files & _active_item_files(state, worktree_path))
        if overlap:
            warnings.append(
                f"Potential collision with {worktree_path}: overlapping files {', '.join(overlap)}"
            )
    return warnings


def _open_items_for(state: BoardState, worktree_path: str) -> list[dict[str, object]]:
    return [
        item for item in state.items.values()
        if item.get("worktree_path") == worktree_path and item.get("status") in {"open", "parked"}
    ]


def _global_items(state: BoardState) -> list[dict[str, object]]:
    return [
        item for item in state.items.values()
        if item.get("status") in {"open", "parked"}
        and (item.get("severity") == "blocker" or item.get("owner_scope") == "juniper")
    ]


def render_checkin_context(config: BoardConfig) -> str:
    identity = current_worktree_identity()
    live = live_worktree_paths()
    state = reconcile_closed_worktrees(config, live_paths=live)
    upsert_presence(config)
    state = load_state(config, live_worktrees=live)
    current = identity["worktree_path"]
    lines: list[str] = ["Agent workspace board"]

    lines.append("This worktree:")
    current_items = _open_items_for(state, current)
    if current_items:
        for item in current_items:
            lines.append(f"- {item['severity']}/{item['kind']}: {item['summary']}")
    else:
        lines.append("- No open items for this worktree.")

    lines.append("Global strip:")
    global_items = _global_items(state)
    if global_items:
        for item in global_items:
            lines.append(f"- {item['severity']}/{item['kind']}: {item['summary']} ({item['worktree_path']})")
    else:
        lines.append("- No global blockers or Juniper escalations.")

    lines.append("Workspace presence:")
    others = [
        row for path, row in sorted(state.presence.items())
        if path != current and row.get("status") in {"active", "stale"}
    ]
    if others:
        for row in others:
            lines.append(
                f"- {row.get('status')} {row.get('worktree_path')} "
                f"{row.get('branch', '')}: {row.get('thread_summary', '')} "
                f"| task: {row.get('current_task', '')}".rstrip()
            )
    else:
        lines.append("- No other active or stale worktrees on the board.")

    collisions = detect_collisions(state, current)
    if collisions:
        lines.append("Collision warnings:")
        for warning in collisions:
            lines.append(f"- {warning}")
    return "\n".join(lines)
```

- [ ] **Step 4: Add checkin and reconcile CLI commands**

Modify `scripts/agent_board.py` imports:

```python
from agent_board_lib import (  # noqa: E402
    add_item,
    board_config_from_env,
    change_item_status,
    close_presence,
    load_state,
    reconcile_closed_worktrees,
    render_checkin_context,
    upsert_presence,
)
```

Add subcommands before parsing:

```python
sub.add_parser("checkin")
sub.add_parser("reconcile")
```

Add command handling before `heartbeat`:

```python
        if args.command == "checkin":
            print(render_checkin_context(cfg))
            return 0
        if args.command == "reconcile":
            reconcile_closed_worktrees(cfg)
            print("reconciled")
            return 0
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_agent_board_lib.py tests/scripts/test_agent_board_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/agent_board.py scripts/agent_board_lib.py tests/scripts/test_agent_board_lib.py tests/scripts/test_agent_board_cli.py
git commit -m "feat: render agent board checkin"
```

---

### Task 4: Claude and Cursor Hook Adapters

**Files:**
- Create: `scripts/hooks/session_start_agent_board.py`
- Create: `scripts/hooks/session_stop_agent_board.py`
- Modify: `.claude/settings.json`
- Create: `.cursor/hooks.json`
- Test: `tests/scripts/test_session_agent_board_hooks.py`

**Interfaces:**
- Consumes:
  - `agent_board_lib.board_config_from_env()`
  - `agent_board_lib.render_checkin_context()`
  - `agent_board_lib.current_worktree_identity()`
  - `agent_board_lib.load_state()`
- Produces:
  - Claude SessionStart JSON payload shape: `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "Agent workspace board\nThis worktree:"} }`
  - Claude Stop JSON payload shape: `{"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": "Agent board checkout reminder: no open board items are recorded for this worktree."} }`
  - Cursor project hooks that call the same scripts.

- [ ] **Step 1: Write failing hook tests**

Create `tests/scripts/test_session_agent_board_hooks.py`:

```python
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_HOOK = ROOT / "scripts" / "hooks" / "session_start_agent_board.py"
STOP_HOOK = ROOT / "scripts" / "hooks" / "session_stop_agent_board.py"


def _run_hook(hook: Path, cwd: Path, board_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)}
    return subprocess.run(
        [sys.executable, str(hook)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_session_start_hook_emits_valid_json(primary_repo: Path, tmp_path: Path) -> None:
    proc = _run_hook(START_HOOK, primary_repo, tmp_path / "agent-board.jsonl")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Agent workspace board" in payload["hookSpecificOutput"]["additionalContext"]


def test_session_stop_hook_emits_checkout_context(primary_repo: Path, tmp_path: Path) -> None:
    proc = _run_hook(STOP_HOOK, primary_repo, tmp_path / "agent-board.jsonl")

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert "agent board checkout" in payload["hookSpecificOutput"]["additionalContext"].lower()


def test_hooks_fail_open_outside_git_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()

    start = _run_hook(START_HOOK, outside, tmp_path / "agent-board.jsonl")
    stop = _run_hook(STOP_HOOK, outside, tmp_path / "agent-board.jsonl")

    assert start.returncode == 0
    assert start.stdout == ""
    assert stop.returncode == 0
    assert stop.stdout == ""


def test_cursor_hooks_file_calls_agent_board_scripts() -> None:
    hooks_path = ROOT / ".cursor" / "hooks.json"
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    rendered = json.dumps(data)
    assert "sessionStart" in rendered
    assert "stop" in rendered
    assert "session_start_agent_board.py" in rendered
    assert "session_stop_agent_board.py" in rendered
```

- [ ] **Step 2: Run hook tests to verify they fail**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_session_agent_board_hooks.py -q
```

Expected: FAIL because hook scripts and `.cursor/hooks.json` do not exist.

- [ ] **Step 3: Create Claude hook scripts**

Create `scripts/hooks/session_start_agent_board.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import board_config_from_env, render_checkin_context  # noqa: E402


def main() -> None:
    try:
        context = render_checkin_context(board_config_from_env())
    except Exception:
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

Create `scripts/hooks/session_stop_agent_board.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_board_lib import board_config_from_env, current_worktree_identity, load_state  # noqa: E402


def main() -> None:
    try:
        cfg = board_config_from_env()
        current = current_worktree_identity()["worktree_path"]
        state = load_state(cfg)
        open_items = [
            item for item in state.items.values()
            if item.get("worktree_path") == current and item.get("status") in {"open", "parked"}
        ]
    except Exception:
        return
    if open_items:
        detail = f"Agent board checkout reminder: {len(open_items)} open item(s) remain for this worktree. Run `python3 scripts/agent_board.py checkout` or resolve/park them."
    else:
        detail = "Agent board checkout reminder: no open board items are recorded for this worktree."
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": detail,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Wire Claude settings**

Modify `.claude/settings.json` so the existing `SessionStart` hook list contains both worktree summary and board checkin commands:

```json
{
  "type": "command",
  "command": "python3 \"$CLAUDE_PROJECT_DIR/scripts/hooks/session_start_agent_board.py\"",
  "timeout": 25
}
```

Add a top-level `Stop` hook entry:

```json
"Stop": [
  {
    "matcher": "",
    "hooks": [
      {
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR/scripts/hooks/session_stop_agent_board.py\"",
        "timeout": 10
      }
    ]
  }
]
```

Keep existing `permissions`, `SessionStart`, and `PreToolUse` entries intact.

- [ ] **Step 5: Create Cursor hooks**

Create `.cursor/hooks.json`:

```json
{
  "version": 1,
  "hooks": {
    "sessionStart": [
      {
        "type": "command",
        "command": "python3 scripts/hooks/session_start_agent_board.py",
        "timeout": 25
      }
    ],
    "stop": [
      {
        "type": "command",
        "command": "python3 scripts/hooks/session_stop_agent_board.py",
        "timeout": 10
      }
    ]
  }
}
```

- [ ] **Step 6: Run hook tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_session_agent_board_hooks.py -q
```

Expected: PASS.

- [ ] **Step 7: Validate JSON files**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
python3 -m json.tool .claude/settings.json >/tmp/claude-settings.json
python3 -m json.tool .cursor/hooks.json >/tmp/cursor-hooks.json
```

Expected: both commands exit 0.

- [ ] **Step 8: Commit**

```bash
git add scripts/hooks/session_start_agent_board.py scripts/hooks/session_stop_agent_board.py .claude/settings.json .cursor/hooks.json tests/scripts/test_session_agent_board_hooks.py
git commit -m "feat: wire agent board hooks"
```

---

### Task 5: Prune Close-on-Remove and Agent Contract Docs

**Files:**
- Modify: `scripts/prune_merged_worktrees.py`
- Modify: `tests/scripts/test_prune_merged_worktrees.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes:
  - `agent_board_lib.close_presence(config, worktree_path=str(w.path))`
  - `agent_board_lib.board_config_from_env()`
- Produces:
  - Successful prune records `presence_closed` for each removed worktree.
  - AGENTS/CLAUDE guidance describing `checkin`, `heartbeat`, `add`, and `checkout`.

- [ ] **Step 1: Add failing prune-close test**

Append this test to `tests/scripts/test_prune_merged_worktrees.py`:

```python
def test_yes_closes_agent_board_presence_for_removed_worktree(
    repo_with_worktrees: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary, merged_wt, _ = repo_with_worktrees
    board_path = tmp_path / "agent-board.jsonl"
    monkeypatch.setenv("ORION_AGENT_BOARD_PATH", str(board_path))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "agent_board.py"),
            "heartbeat",
            "--summary",
            "Merged worktree presence.",
            "--task",
            "Will be pruned.",
        ],
        cwd=merged_wt,
        check=True,
        capture_output=True,
        text=True,
    )

    proc = _run_prune(primary, "--yes")

    assert proc.returncode == 0
    listed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agent_board.py"), "list", "--all"],
        cwd=primary,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "ORION_AGENT_BOARD_PATH": str(board_path)},
    )
    assert str(merged_wt) in listed.stdout
    assert "closed" in listed.stdout
```

Add `import os` near the top of the file:

```python
import os
```

- [ ] **Step 2: Run prune tests to verify failure**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_prune_merged_worktrees.py::test_yes_closes_agent_board_presence_for_removed_worktree -q
```

Expected: FAIL because `prune_merged_worktrees.py` does not close board presence yet.

- [ ] **Step 3: Wire close-on-remove**

Modify `scripts/prune_merged_worktrees.py` imports after the `worktree_lib` import:

```python
from agent_board_lib import board_config_from_env, close_presence  # noqa: E402
```

Inside the `if result.returncode == 0:` block, after `removed += 1`, add:

```python
            try:
                close_presence(board_config_from_env(), worktree_path=str(w.path))
            except Exception as exc:
                print(f"[agent-board close skipped] {w.path} -- {exc}", file=sys.stderr)
```

- [ ] **Step 4: Add AGENTS/CLAUDE guidance**

In `CLAUDE.md`, in section `## 2. Clean git and worktree rules`, after the paragraph that introduces `make worktree-status`, add:

```markdown
### Agent workspace board

Concurrent agents should keep the host-local board current:

```bash
python3 scripts/agent_board.py checkin
python3 scripts/agent_board.py heartbeat --summary "Working on the agent board implementation; current risk is hook wiring." --task "Wire Claude and Cursor hooks"
python3 scripts/agent_board.py add --kind finding --severity should --summary "Review found a collision risk in hook ordering." --files scripts/hooks/session_start_agent_board.py
python3 scripts/agent_board.py checkout
```

The live board is `~/.orion/agent-board.jsonl`, protected by a lock and intentionally not committed. `checkin` shows this-worktree items, global blockers/Juniper escalations, other active/stale worktrees, and disclosure-only collision warnings. Track current-worktree items by default; if the item belongs elsewhere, set an explicit scope note.
```

Verify `AGENTS.md` reflects the same text:

```bash
grep -n "Agent workspace board" AGENTS.md CLAUDE.md
```

Expected: both paths print the heading.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest tests/scripts/test_prune_merged_worktrees.py tests/scripts/test_agent_board_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/prune_merged_worktrees.py tests/scripts/test_prune_merged_worktrees.py CLAUDE.md
git commit -m "feat: close agent board rows on prune"
```

---

### Task 6: Verification, Live Smoke, Graph Update, Review, and PR Report

**Files:**
- Modify if needed: `docs/superpowers/pr-reports/2026-07-16-agent-workspace-board-pr.md`
- Modify if needed: `graphify-out/*` from safe graphify update

**Interfaces:**
- Consumes all prior tasks.
- Produces final verification evidence, review fix log, and PR report.

- [ ] **Step 1: Run full focused test suite**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
orion_dev/bin/pytest \
  tests/scripts/test_agent_board_lib.py \
  tests/scripts/test_agent_board_cli.py \
  tests/scripts/test_session_agent_board_hooks.py \
  tests/scripts/test_prune_merged_worktrees.py \
  tests/scripts/test_session_start_worktree_summary.py \
  tests/scripts/test_worktree_lib.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run syntax/JSON checks**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
python3 -m py_compile scripts/agent_board.py scripts/agent_board_lib.py scripts/hooks/session_start_agent_board.py scripts/hooks/session_stop_agent_board.py
python3 -m json.tool .claude/settings.json >/tmp/claude-settings.json
python3 -m json.tool .cursor/hooks.json >/tmp/cursor-hooks.json
```

Expected: all commands exit 0.

- [ ] **Step 3: Run live two-worktree smoke**

Use two existing worktrees from the same repo or create a temporary one. Keep the board isolated:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
export ORION_AGENT_BOARD_PATH=/tmp/orion-agent-board-smoke.jsonl
rm -f "$ORION_AGENT_BOARD_PATH" "$ORION_AGENT_BOARD_PATH.lock"
python3 scripts/agent_board.py heartbeat --summary "Agent A smoke summary." --task "Writing board smoke."
python3 scripts/agent_board.py add --kind blocker --severity blocker --scope juniper --scope-note "Smoke test global item." --summary "Smoke blocker visible globally." --files scripts/agent_board.py
python3 scripts/agent_board.py checkin
```

Expected output contains:

```text
This worktree
Global strip
Smoke blocker visible globally.
Workspace presence
```

If a second worktree is available, run from it:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
export ORION_AGENT_BOARD_PATH=/tmp/orion-agent-board-smoke.jsonl
python3 scripts/agent_board.py checkin
```

Expected: the global blocker created by the first worktree is visible without git pull or commit.

- [ ] **Step 4: Run safe graphify update**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
scripts/safe_graphify_update.sh
```

Expected: exits 0 or refuses safely with a node-loss warning. If it refuses, do not re-run blindly; include the refusal in the PR report.

- [ ] **Step 5: Run code review subagent**

Launch an Orion repo review agent with this prompt:

```markdown
You are reviewing the `docs/agent-workspace-board` branch in `Orion-Sapienform`.

Focus on:
- JSONL locking correctness and crash safety
- whether hooks fail open
- whether presence stale/closed reconciliation can accidentally hide live rows
- whether collision warnings remain disclosure-only
- whether the CLI can run from plain terminal and from hooks
- whether tests cover the live multi-agent goal

Return findings ordered by severity, with file references and concrete fixes.
```

Expected: review returns either no material findings or fixable findings.

- [ ] **Step 6: Fix review findings**

For each material finding, write a failing test first, implement the smallest fix, rerun focused tests, and commit:

```bash
git add scripts/agent_board_lib.py tests/scripts/test_agent_board_lib.py
git commit -m "fix: address agent board review finding"
```

Expected: every material finding is either fixed or explicitly carried into PR report concerns.

- [ ] **Step 7: Write PR report**

Create `docs/superpowers/pr-reports/2026-07-16-agent-workspace-board-pr.md` with this content shape:

```markdown
# PR report: agent workspace board

## Summary

- Added host-local `~/.orion/agent-board.jsonl` workspace board.
- Added agent board CLI for checkin, heartbeat, open items, resolve, checkout, and reconcile.
- Added Claude/Cursor hook adapters that fail open.
- Added prune close-on-remove integration.
- Added focused script tests and live smoke evidence.

## Outcome moved

Concurrent agents can see this-worktree items, global blockers/Juniper escalations, other active/stale worktrees, and disclosure-only collision warnings without git round-trips.

## Current architecture

Before this patch, worktree SessionStart only surfaced counts, PR reports buried follow-ups in prose, and no host-shared real-time agent board existed.

## Architecture touched

Scripts and hooks only: `scripts/agent_board*.py`, `scripts/hooks/session_*_agent_board.py`, `.claude/settings.json`, `.cursor/hooks.json`, `scripts/prune_merged_worktrees.py`, and AGENTS/CLAUDE guidance.

## Files changed

- `scripts/agent_board_lib.py`: board storage, validation, reconciliation, collision detection, rendering
- `scripts/agent_board.py`: vendor-neutral CLI
- `scripts/hooks/session_start_agent_board.py`: SessionStart adapter
- `scripts/hooks/session_stop_agent_board.py`: Stop adapter
- `.claude/settings.json`: Claude hook wiring
- `.cursor/hooks.json`: Cursor hook wiring
- `scripts/prune_merged_worktrees.py`: close presence when removing worktrees
- `CLAUDE.md`: operator/agent guidance
- `tests/scripts/*agent_board*`: deterministic tests

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: local hook context now includes agent board state
- Compatibility notes: board file is host-local and outside git

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion_dev/bin/pytest tests/scripts/test_agent_board_lib.py tests/scripts/test_agent_board_cli.py tests/scripts/test_session_agent_board_hooks.py tests/scripts/test_prune_merged_worktrees.py tests/scripts/test_session_start_worktree_summary.py tests/scripts/test_worktree_lib.py -q
PASS
```

## Evals run

```text
No eval harness applies; this is agent-ops tooling, not Orion cognition behavior.
```

## Docker/build/smoke checks

```text
ORION_AGENT_BOARD_PATH=/tmp/orion-agent-board-smoke.jsonl python3 scripts/agent_board.py checkin
PASS: output included This worktree, Global strip, Workspace presence, and the smoke blocker.
No Docker restart required.
```

## Review findings fixed

- Finding:
 - Fix:
 - Evidence:

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: Low
- Concern: Host-local board does not sync across machines.
- Mitigation: Accepted v1 tradeoff for real-time multi-agent visibility on Athena; git export is deferred.

## PR link

PR URL from `gh pr create --fill`
```

- [ ] **Step 8: Final branch checks**

Run:

```bash
cd /mnt/scripts/Orion-Sapienform-agent-workspace-board
git diff --check
git status --short
git log --oneline -5
```

Expected: `git diff --check` exits 0; status only shows intentional uncommitted graph/report changes before the final commit.

- [ ] **Step 9: Commit PR report and final graph changes**

```bash
git add docs/superpowers/pr-reports/2026-07-16-agent-workspace-board-pr.md graphify-out
git commit -m "docs: report agent workspace board"
```

If `scripts/safe_graphify_update.sh` refused and left no graph changes, omit `graphify-out` from `git add` and state that in the PR report.

- [ ] **Step 10: Push and open PR**

Run:

```bash
git push -u origin docs/agent-workspace-board
gh pr create --fill
```

Expected: PR URL printed. Paste it into the PR report if `gh pr create --fill` did not include it automatically, amend the commit, and push the amended report.

---

## Self-Review

**Spec coverage:** Covered host-local locked JSONL storage, 30-minute stale threshold, checkin/heartbeat/add/resolve/list/checkout/reconcile CLI, this-worktree/global/presence rendering, prune close-on-remove, Claude and Cursor thin hooks, disclosure-only collision warnings, AGENTS/CLAUDE pointer, and focused tests. Deferred items from the spec remain deferred: git export, PR-report import, hard gates, UI, and graphify board export.

**Placeholder scan:** This plan contains no unfilled placeholder markers and no unspecified test commands. Deferred work is explicitly named as non-v1 scope rather than left as implementation blanks.

**Type consistency:** The plan consistently uses `BoardConfig`, `BoardEvent`, `BoardState`, `append_event`, `load_state`, `upsert_presence`, `add_item`, `change_item_status`, `close_presence`, `reconcile_closed_worktrees`, `detect_collisions`, and `render_checkin_context`.
