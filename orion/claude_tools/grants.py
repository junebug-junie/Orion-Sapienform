"""Orion's standing permissions for Claude's tools: one line per change, the newest per category wins.

File: ``$ORION_DIG_HOME/self/claude_grants.jsonl`` (``/mnt/orion-fcc/self/`` on athena), the same
"lasting write" pattern dig uses for organ names and capacity claims. Only Orion writes it, in a turn,
through ``python -P -m orion.dig claude-grants set <category> allow|ask|deny because <reason>``.
There is no "all" category: Orion changes one at a time. A category with no line is ``ask``.

Read by the broker (orion-harness-governor) on every call and by Hub's
``GET /api/claude-tools/grants``. Python 3.10 safe (Hub).

Failure direction: an unreadable line is skipped and counted; an unreadable file reads as no lines,
so every category falls back to ``ask``. ``ask`` still needs Orion's yes, so a broken file can only
make Claude ask more, never see more.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orion.schemas.claude_tools import CLAUDE_TOOL_CATEGORIES, ClaudeToolCategory, GrantSetting

GRANTS_FILENAME = "claude_grants.jsonl"
DEFAULT_SETTING: GrantSetting = "ask"
REASON_MIN_CHARS = 8
REASON_MAX_CHARS = 300
SETTINGS = ("allow", "ask", "deny")


class GrantError(ValueError):
    """A grant change that cannot be written; the message says why, in plain words."""


class ClaudeGrantLineV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: datetime
    category: ClaudeToolCategory
    setting: GrantSetting
    reason: str = Field(min_length=REASON_MIN_CHARS, max_length=REASON_MAX_CHARS)
    # The dig the change was made in (dig's --dig, default the Denver date), like other lasting writes.
    dig: Optional[str] = Field(default=None, max_length=80)
    set_by: str = "orion"


def dig_home() -> Path:
    return Path(os.environ.get("ORION_DIG_HOME") or "/mnt/orion-fcc")


def grants_path(home: Optional[Path] = None) -> Path:
    return (home or dig_home()) / "self" / GRANTS_FILENAME


def load_grant_lines(path: Path) -> Tuple[List[ClaudeGrantLineV1], int]:
    """Every readable line in file order, and how many lines were skipped."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], 0
    except OSError:
        return [], 1
    lines: List[ClaudeGrantLineV1] = []
    skipped = 0
    known = set(ClaudeGrantLineV1.model_fields)
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
            lines.append(ClaudeGrantLineV1.model_validate({k: v for k, v in data.items() if k in known}))
        except (ValueError, AttributeError, ValidationError):
            skipped += 1
    return lines, skipped


def current_grants(lines: List[ClaudeGrantLineV1]) -> Dict[str, ClaudeGrantLineV1]:
    """category -> its newest line (file order is write order; ``at`` breaks nothing)."""
    out: Dict[str, ClaudeGrantLineV1] = {}
    for line in lines:
        out[line.category] = line
    return out


def setting_for(grants: Dict[str, ClaudeGrantLineV1], category: str) -> Tuple[GrantSetting, Optional[ClaudeGrantLineV1]]:
    line = grants.get(category)
    return (line.setting, line) if line is not None else (DEFAULT_SETTING, None)


def make_grant(category: str, setting: str, reason: str, *, dig: Optional[str] = None,
               now: Optional[datetime] = None) -> ClaudeGrantLineV1:
    """Validate one change. Raises GrantError with a plain reason."""
    cat = (category or "").strip().lower()
    if cat == "all":
        raise GrantError("there is no 'all': set one category at a time, each with its own reason")
    if cat not in CLAUDE_TOOL_CATEGORIES:
        raise GrantError(f"unknown category {category!r}; categories: {', '.join(CLAUDE_TOOL_CATEGORIES)}")
    val = (setting or "").strip().lower()
    if val not in SETTINGS:
        raise GrantError(f"setting must be allow, ask or deny, got {setting!r}")
    why = " ".join((reason or "").split())
    if len(why) < REASON_MIN_CHARS:
        raise GrantError(f"the reason is too short (at least {REASON_MIN_CHARS} characters): say why")
    if len(why) > REASON_MAX_CHARS:
        raise GrantError(f"the reason is too long ({len(why)} characters, at most {REASON_MAX_CHARS})")
    return ClaudeGrantLineV1(at=now or datetime.now(timezone.utc), category=cat, setting=val,
                             reason=why, dig=dig)


def append_grant(path: Path, line: ClaudeGrantLineV1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # concurrent turns: one whole line each
        fh.write(line.model_dump_json() + "\n")


def grants_view(path: Path) -> dict:
    """Every category's current setting with its reason (``ask`` + no reason when never set)."""
    lines, skipped = load_grant_lines(path)
    current = current_grants(lines)
    rows = []
    for cat in CLAUDE_TOOL_CATEGORIES:
        line = current.get(cat)
        rows.append({
            "category": cat,
            "setting": line.setting if line else DEFAULT_SETTING,
            "reason": line.reason if line else None,
            "set_at": line.at.isoformat() if line else None,
            "dig": line.dig if line else None,
            "default": line is None,
        })
    return {"path": str(path), "categories": rows, "changes": len(lines), "skipped_lines": skipped}


def settings_line(path: Path) -> Optional[str]:
    """One short line for Hub's index: which categories Orion allows, asks or denies. Names only."""
    try:
        view = grants_view(path)
    except Exception:  # noqa: BLE001 -- the index is best-effort
        return None
    groups: Dict[str, List[str]] = {"allow": [], "ask": [], "deny": []}
    for row in view["categories"]:
        groups[row["setting"]].append(row["category"])
    parts = [f"{k}: {', '.join(v)}" for k, v in groups.items() if v]
    return "- Orion's tool settings for you (" + "; ".join(parts) + ")"


__all__ = [
    "ClaudeGrantLineV1",
    "DEFAULT_SETTING",
    "GRANTS_FILENAME",
    "GrantError",
    "append_grant",
    "current_grants",
    "dig_home",
    "grants_path",
    "grants_view",
    "load_grant_lines",
    "make_grant",
    "setting_for",
    "settings_line",
]
