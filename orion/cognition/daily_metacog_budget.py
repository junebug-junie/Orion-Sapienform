"""Prompt budget for the nightly daily_metacog_v1 report.

cortex-exec refuses to send the daily_metacog_v1 prompt when it is longer than
CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS (``_enforce_daily_metacog_prompt_budget``).
From 2026-09-03 the prompt was always a few hundred chars over, because the
skills catalog orion-actions passed in had grown to 6,126 chars (21 skills as
JSON with 200-char descriptions). The report failed every night.

This module computes how many chars the skills catalog may use, from the
template itself and the largest memory digest recall is allowed to produce, so
the prompt fits by construction instead of by luck. orion-actions builds the
metacog catalog with :func:`build_daily_metacog_skill_catalog`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from orion.cognition.skills_manifest import SkillManifestEntry, build_bounded_skill_catalog

logger = logging.getLogger(__name__)

# Mirrors cortex-exec's CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS default (guarded by
# a test). If an operator lowers that env key, lower this too.
DAILY_METACOG_PROMPT_MAX_CHARS = 8192

# Fallback only: the real value is the recall profile's render_char_budget.
DAILY_METACOG_DIGEST_MAX_CHARS_FALLBACK = 1280
DAILY_METACOG_RECALL_PROFILE = "journal.daily.metacog.grounded.v1"

# Headroom for scalar fields longer than the placeholders below and for
# small template edits that forget to revisit this file.
DAILY_METACOG_PROMPT_SAFETY_MARGIN_CHARS = 256

_COGNITION_DIR = Path(__file__).resolve().parent
_TEMPLATE_PATH = _COGNITION_DIR / "prompts" / "daily_metacog_prompt.j2"
_PROFILE_PATH = _COGNITION_DIR.parent / "recall" / "profiles" / f"{DAILY_METACOG_RECALL_PROFILE}.yaml"
_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def daily_metacog_digest_max_chars(profile_path: Path | None = None) -> int:
    """Largest memory digest recall will render for the metacog profile."""
    path = profile_path or _PROFILE_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = int(raw.get("render_char_budget") or 0)
        if value > 0 and bool(raw.get("strict_prompt_budget")):
            return value
    except Exception as exc:  # pragma: no cover - logged, fallback below
        logger.warning("daily_metacog_digest_budget_unreadable path=%s error=%s", path, exc)
    return DAILY_METACOG_DIGEST_MAX_CHARS_FALLBACK


def _render_simple(template: str, ctx: dict[str, object]) -> str:
    # The template only uses plain ``{{ name }}`` substitutions (asserted in tests
    # against cortex-exec's real jinja render), so no jinja dependency here.
    if template.endswith("\n"):
        template = template[:-1]  # jinja's default keep_trailing_newline=False
    return _VAR_RE.sub(lambda m: str(ctx.get(m.group(1), "")), template)


def daily_metacog_prompt_overhead_chars(*, digest_max_chars: int | None = None) -> int:
    """Rendered prompt length with the largest digest and an empty catalog."""
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    digest = int(digest_max_chars if digest_max_chars is not None else daily_metacog_digest_max_chars())
    worst_case = {
        "request_date": "2026-12-31",
        "timezone": "America/Argentina/Buenos_Aires",
        "node": "n" * 32,
        "window_start_utc": "2026-12-31T00:00:00+00:00",
        "window_end_utc": "2026-12-31T00:00:00+00:00",
        "memory_digest": "x" * digest,
        "skills_catalog_count": 9999,
        "skills_catalog_compact": "",
    }
    return len(_render_simple(template, worst_case))


def daily_metacog_skill_catalog_budget(
    *,
    prompt_max_chars: int = DAILY_METACOG_PROMPT_MAX_CHARS,
    digest_max_chars: int | None = None,
    margin_chars: int = DAILY_METACOG_PROMPT_SAFETY_MARGIN_CHARS,
) -> int:
    overhead = daily_metacog_prompt_overhead_chars(digest_max_chars=digest_max_chars)
    return max(0, int(prompt_max_chars) - overhead - int(margin_chars))


def build_daily_metacog_skill_catalog(
    entries: list[SkillManifestEntry] | None = None,
    *,
    prompt_max_chars: int = DAILY_METACOG_PROMPT_MAX_CHARS,
) -> tuple[str, int]:
    """(catalog_text, listed_skill_count) sized to fit the metacog prompt."""
    budget = daily_metacog_skill_catalog_budget(prompt_max_chars=prompt_max_chars)
    return build_bounded_skill_catalog(max_chars=budget, entries=entries, read_only_only=True)
