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


def main() -> int:
    motor_ctx = _read_env_int("HARNESS_FCC_MAX_CONTEXT_TOKENS") or _read_env_int(
        "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS", 65536
    )
    profile_max = _max_profile_ctx_size()
    errors: list[str] = []
    if profile_max and motor_ctx > profile_max:
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
        f"fcc-context-parity: ok motor_ctx={motor_ctx} profile_max_ctx={profile_max or 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
