"""Host-local agent workspace board.

The board is intentionally vendor-neutral: a locked JSONL event log plus
plain functions that CLI scripts and hook adapters can call.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

from worktree_lib import list_worktrees, repo_toplevel

DEFAULT_BOARD_PATH = Path.home() / ".orion" / "agent-board.jsonl"
STALE_AFTER_MINUTES = 30

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


def _chmod_best_effort(path: Path, mode: int) -> None:
    """Owner-only modes when possible; fail soft for shared parents like /tmp."""
    try:
        os.chmod(path, mode)
    except OSError:
        return


def board_config_from_env() -> BoardConfig:
    raw = os.environ.get("ORION_AGENT_BOARD_PATH")
    if raw:
        return BoardConfig(board_path=Path(raw).expanduser())
    return BoardConfig()


@contextmanager
def _locked_board(config: BoardConfig) -> Iterator[None]:
    config.board_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _chmod_best_effort(config.board_path.parent, 0o700)
    lock_path = config.board_path.with_suffix(config.board_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        _chmod_best_effort(lock_path, 0o600)
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
            _chmod_best_effort(config.board_path, 0o600)
            handle.write(json.dumps(event.__dict__, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
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
            try:
                row = json.loads(line)
                events.append(BoardEvent(type=row["type"], at=row["at"], payload=row["payload"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
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


def _parse_timestamp(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def read_session_id_from_stdin_hook_payload(stream=None) -> str | None:
    """Extract `session_id` from a Claude Code hook's own stdin JSON payload.

    Shared by session_start_agent_board.py and session_stop_agent_board.py
    (both previously duplicated this logic verbatim). Never raises and
    never blocks past whatever the caller's stream naturally does -- skips
    reading entirely when stdin is a live TTY (a human running the hook
    script by hand to debug it, not the harness piping a payload in and
    closing it), since `.read()` on an open TTY blocks until EOF/interrupt.
    """
    import sys as _sys

    stream = stream if stream is not None else _sys.stdin
    try:
        if hasattr(stream, "isatty") and stream.isatty():
            return None
        raw = stream.read()
    except Exception:
        return None
    if not raw.strip():
        return None
    try:
        payload_in = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload_in, dict):
        return None
    session_id = payload_in.get("session_id")
    return str(session_id) if session_id else None


def resolve_current_identity(
    config: BoardConfig,
    *,
    session_id: str | None = None,
) -> dict[str, str]:
    """Resolve the worktree identity to use for a board call.

    Claude Code's SessionStart/Stop hooks run with a process cwd fixed to
    the session's original project directory -- it does NOT track `cd`
    calls a Bash tool makes mid-session (confirmed live: a Stop hook's own
    stdin payload reported `cwd` as the shared/primary checkout even while
    the agent's actual git work was happening in a linked worktree several
    turns earlier). `git rev-parse --show-toplevel` from that same fixed
    cwd resolves to the same wrong answer, so it can't be fixed by cwd
    detection alone.

    When `session_id` is given (the hook's own stdin payload carries one),
    look up the most recently heartbeated presence row tagged with that
    same session_id -- git-hook-driven heartbeats (`scripts/git_hooks/
    post-commit`, `scripts/safe_docker_build.sh`) pass `$CLAUDE_CODE_SESSION_ID`
    when set, and those DO run with the correct worktree cwd (they're
    invoked by `git`/`docker`, not by the Claude Code harness's own hook
    mechanism). Falls back to the normal git-rev-parse-based resolution if
    no matching session_id is found yet (e.g. the very first hook fire in a
    session, before any git-hook-driven heartbeat has landed).
    """
    if session_id:
        state = load_state(config)
        candidates = [
            row for row in state.presence.values()
            if row.get("session_id") == session_id and row.get("worktree_path")
        ]
        if candidates:
            best = max(candidates, key=lambda row: _parse_timestamp(str(row.get("heartbeat_at") or "")))
            return {
                "worktree_path": str(best["worktree_path"]),
                "branch": str(best.get("branch") or ""),
            }
    return current_worktree_identity()


def upsert_presence(
    config: BoardConfig,
    *,
    summary: str | None = None,
    task: str | None = None,
    session_id: str | None = None,
    worktree_path: str | None = None,
    branch: str | None = None,
) -> dict[str, object]:
    if worktree_path is not None:
        identity = {"worktree_path": worktree_path, "branch": branch or ""}
    else:
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


def _new_item_id() -> str:
    """Return a UUID string without importing uuid at module load time.

    ``scripts/platform/`` shadows stdlib ``platform`` when this module is loaded
    from ``scripts/`` on ``sys.path`` (including Python's automatic script-dir
    insertion for ``agent_board.py``).
    """
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    sys.modules.pop("platform", None)
    sys.modules.pop("uuid", None)
    old_path = sys.path
    try:
        sys.path = [entry for entry in old_path if entry != scripts_dir]
        import uuid

        return str(uuid.uuid4())
    finally:
        sys.path = old_path


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
    item_id = _new_item_id()
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
    if item_id not in load_state(config).items:
        raise ValueError(f"unknown item id: {item_id}")
    append_event(config, "item_status_changed", {"id": item_id, "status": status})


def close_presence(config: BoardConfig, worktree_path: str | None = None) -> None:
    path = worktree_path
    if path is None:
        path = current_worktree_identity()["worktree_path"]
    else:
        path = str(Path(path).resolve())
    append_event(config, "presence_closed", {"worktree_path": path, "status": "closed"})


def live_worktree_paths() -> set[str]:
    return {str(info.path.resolve()) for info in list_worktrees()}


def reconcile_closed_worktrees(
    config: BoardConfig,
    live_paths: set[str] | None = None,
) -> BoardState:
    live = live_paths if live_paths is not None else live_worktree_paths()
    persisted = load_state(config)
    for worktree_path, row in persisted.presence.items():
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


def _dirty_files(worktree_path: str) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "-C", worktree_path, "status", "--short", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    files: set[str] = set()
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        files.add(path)
    return files


def _service_paths(paths: set[str]) -> set[str]:
    services: set[str] = set()
    for raw in paths:
        parts = Path(raw).parts
        if len(parts) >= 2 and parts[0] == "services":
            services.add(str(Path(parts[0]) / parts[1]))
    return services


def detect_collisions(state: BoardState, current_worktree_path: str) -> list[str]:
    current_files = _active_item_files(state, current_worktree_path) | _dirty_files(current_worktree_path)
    current_services = _service_paths(current_files)
    warnings: list[str] = []
    for worktree_path, presence in sorted(state.presence.items()):
        if worktree_path == current_worktree_path:
            continue
        if presence.get("status") not in {"active", "stale"}:
            continue
        if current_files:
            other_files = _active_item_files(state, worktree_path) | _dirty_files(worktree_path)
            overlap = sorted(current_files & other_files)
            if overlap:
                warnings.append(
                    f"Potential collision with {worktree_path}: overlapping files {', '.join(overlap)}"
                )
                continue
            service_overlap = sorted(current_services & _service_paths(other_files))
            if service_overlap:
                warnings.append(
                    f"Potential collision with {worktree_path}: shared service paths {', '.join(service_overlap)}"
                )
                continue
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


def render_checkin_context(config: BoardConfig, *, session_id: str | None = None) -> str:
    identity = resolve_current_identity(config, session_id=session_id)
    live = live_worktree_paths()
    state = reconcile_closed_worktrees(config, live_paths=live)
    upsert_presence(
        config,
        session_id=session_id,
        worktree_path=identity["worktree_path"],
        branch=identity["branch"],
    )
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
