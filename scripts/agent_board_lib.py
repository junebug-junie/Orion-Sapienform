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
