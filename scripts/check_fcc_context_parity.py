#!/usr/bin/env python3
"""Verify FCC motor context env aligns with llamacpp profile ctx_size."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILES = _REPO_ROOT / "config" / "llm_profiles.yaml"

_CTX_RE = re.compile(r"^\s*ctx_size:\s*(\d+)\s*$", re.MULTILINE)
_TOP_LEVEL_KEY_RE = re.compile(r"^  [A-Za-z0-9_.-]+:\s*$", re.MULTILINE)


def _read_env_int(key: str, default: int = 0) -> int:
    raw = str(os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _max_profile_ctx_size() -> int:
    if not _PROFILES.is_file():
        return 0
    sizes = [int(m.group(1)) for m in _CTX_RE.finditer(_PROFILES.read_text(encoding="utf-8"))]
    return max(sizes) if sizes else 0


def _active_profile_ctx_size(profile_name: str) -> int | None:
    """ctx_size of the one profile block named `profile_name`, or None if the
    profile isn't found. Regex-scoped to that block only (from its own
    top-level key line to the next top-level key line) -- this is the check
    the file-wide max in _max_profile_ctx_size() cannot do: a drift on the
    ACTIVE chat profile specifically can hide behind an unrelated profile
    elsewhere in the file that happens to have a larger ctx_size."""
    if not _PROFILES.is_file():
        return None
    text = _PROFILES.read_text(encoding="utf-8")
    key_pattern = re.compile(rf"^  {re.escape(profile_name)}:\s*$", re.MULTILINE)
    match = key_pattern.search(text)
    if not match:
        return None
    block_start = match.end()
    next_key = _TOP_LEVEL_KEY_RE.search(text, block_start)
    block_end = next_key.start() if next_key else len(text)
    block = text[block_start:block_end]
    ctx_match = _CTX_RE.search(block)
    return int(ctx_match.group(1)) if ctx_match else None


def main() -> int:
    motor_ctx = _read_env_int("HARNESS_FCC_MAX_CONTEXT_TOKENS") or _read_env_int(
        "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS", 65536
    )
    profile_max = _max_profile_ctx_size()
    errors: list[str] = []

    active_profile_name = str(os.environ.get("ATLAS_CHAT_PROFILE_NAME") or "").strip()
    active_ctx = _active_profile_ctx_size(active_profile_name) if active_profile_name else None
    if active_ctx is not None and motor_ctx != active_ctx:
        errors.append(
            f"FCC motor context ({motor_ctx}) does not match ctx_size ({active_ctx}) "
            f"of the active chat profile '{active_profile_name}' in config/llm_profiles.yaml -- "
            "these must stay in lockstep, not just motor_ctx <= some other profile's ctx_size"
        )
    elif profile_max and motor_ctx > profile_max:
        # Fallback when ATLAS_CHAT_PROFILE_NAME isn't set in this environment
        # (e.g. CI with no operator .env) -- weaker check: only catches the
        # motor exceeding EVERY profile in the file, not a mismatch with
        # whichever one is actually deployed.
        errors.append(
            f"FCC motor context ({motor_ctx}) exceeds max ctx_size in "
            f"config/llm_profiles.yaml ({profile_max})"
        )
    if motor_ctx <= 0:
        errors.append("FCC motor context tokens must be > 0")
    if errors:
        for err in errors:
            print(f"fcc-context-parity: {err}", file=sys.stderr)
        return 1
    print(
        f"fcc-context-parity: ok motor_ctx={motor_ctx} "
        f"active_profile={active_profile_name or 'n/a'} active_ctx={active_ctx if active_ctx is not None else 'n/a'} "
        f"profile_max_ctx={profile_max or 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
