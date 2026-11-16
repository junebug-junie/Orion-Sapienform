"""Regression guard for the 2026-07-14 crash-loop: `docker-compose.yml`'s
`environment:` list substituted several numeric tunables with empty strings
(because they were missing from live `.env` and had no `:-default` fallback),
which crashed pydantic Settings parsing (float_parsing/int_parsing) on every
boot -- see docs/superpowers/pr-reports/2026-07-13-journal-notification-flood-fix-pr.md
and its follow-up.

This asserts each of those keys keeps a `:-default` fallback matching
`.env_example`'s real value, so a future missing-.env-key never turns into a
hard crash again.
"""
from __future__ import annotations

import re
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
COMPOSE_PATH = SERVICE_DIR / "docker-compose.yml"
ENV_EXAMPLE_PATH = SERVICE_DIR / ".env_example"

KEY_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# Keys that crashed (or silently emptied) when missing from .env, now given a
# compose-level fallback matching .env_example.
_HARDENED_KEYS = (
    "RECALL_SKIP_MAX_NOVELTY",
    "RECALL_SKIP_SHIFT_NOVELTY_FLOOR",
    "RECALL_CONTINUITY_SQL_MINUTES",
    "RECALL_CONTINUITY_RENDER_BUDGET",
    "RECALL_BELIEF_RENDER_BUDGET",
    "RECALL_MAX_QUERY_CHARS",
    "RECALL_MAX_SUB_QUERIES",
    "RECALL_DEADLINE_MS_DEFAULT",
    "RECALL_FETCH_CONCURRENCY",
)


def _env_example_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = KEY_LINE.match(stripped)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _compose_text() -> str:
    return COMPOSE_PATH.read_text(encoding="utf-8")


def test_hardened_keys_have_compose_default_matching_env_example() -> None:
    env_values = _env_example_values()
    compose_text = _compose_text()

    for key in _HARDENED_KEYS:
        assert key in env_values, f"{key} missing from .env_example -- can't verify its default"
        default_value = env_values[key]

        pattern = re.compile(rf"^\s*-\s*{re.escape(key)}=\$\{{{re.escape(key)}:-([^}}]*)\}}\s*$", re.MULTILINE)
        match = pattern.search(compose_text)
        assert match is not None, (
            f"{key} has no ${{{key}:-default}} fallback in docker-compose.yml -- "
            "a missing .env key would substitute an empty string and crash pydantic "
            "Settings parsing again (the exact 2026-07-14 incident)."
        )
        assert match.group(1) == default_value, (
            f"{key}'s compose fallback ({match.group(1)!r}) has drifted from "
            f".env_example ({default_value!r}) -- keep them in sync."
        )
