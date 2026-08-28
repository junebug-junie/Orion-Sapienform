"""Eval: is the reverie-daydream lane actually usable against the LIVE corpus?

`services/orion-hub/` had no `evals/` directory at all before this file
(AGENTS.md §11 says to add the smallest useful one rather than silently skip).
This is that smallest useful one, and it exists because of a specific
mistake: the first version of the daydream lane shipped a de-duplication
mechanism whose calibration claim was falsified by the live table on the same
day it was committed. Unit tests could not have caught it -- they ran on
fixtures the same reasoning invented. This runs the real caption pipeline
over the real rows.

Read-only. Skips cleanly with no reachable Postgres, so it is safe in CI and
in a bare worktree; it is NOT a gate test and does not belong in `tests/`.

Run:
    pytest services/orion-hub/evals -q
    DAYDREAM_EVAL_DATABASE_URL=postgresql+psycopg2://... pytest services/orion-hub/evals -q
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest



def _lane():
    """Import the real lane. Deferred, not module-level: this directory's
    conftest fixes `sys.path` in `pytest_configure`, which runs after module
    import, and the repo-root `scripts/` package would otherwise win."""
    from scripts.endogenous_outreach import (
        _DAYDREAM_DETECTOR_OUTPUT_RE,
        _DAYDREAM_MAX_AGE_SEC,
        _clean_daydream,
    )

    return _clean_daydream, _DAYDREAM_DETECTOR_OUTPUT_RE, _DAYDREAM_MAX_AGE_SEC

_DEFAULT_DSN = "postgresql+psycopg2://postgres:postgres@localhost:55432/conjourney"

# Measured over all 328 rows on 2026-08-28: 296 non-NULL captions, 287 of
# which clean to a usable line -- 96.9%. The floor sits well below that
# because the point is to catch a COLLAPSE (a captioner regression that
# starts emitting grounding coordinates for everything), not to freeze
# today's exact number into a gate.
_MIN_USABLE_RATE = 0.85
# Below this many rows the rate is an artifact, not a measurement.
_MIN_SAMPLE = 40


def _rows():
    """Every (created_at, description) in the daydream window, or skip."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:  # pragma: no cover - environment without sqlalchemy
        pytest.skip("sqlalchemy not installed")

    dsn = os.getenv("DAYDREAM_EVAL_DATABASE_URL", _DEFAULT_DSN)
    try:
        engine = create_engine(dsn)
        with engine.connect() as conn:
            return (
                conn.execute(
                    text(
                        "SELECT created_at, chain_json->>'description' AS description "
                        "FROM reverie_visual_chain ORDER BY created_at DESC"
                    )
                )
                .mappings()
                .all()
            )
    except Exception as exc:  # noqa: BLE001 - any DB problem is a skip, not a failure
        pytest.skip(f"no reachable reverie_visual_chain ({type(exc).__name__})")


def test_caption_pipeline_yields_usable_text_on_the_live_corpus() -> None:
    """The lane's real failure mode is silence: if the captioner regresses,
    every row cleans to "" and the prompt block just stops appearing, with no
    error and no failing unit test."""
    _clean_daydream, _, _ = _lane()
    rows = _rows()
    captions = [r["description"] for r in rows if r["description"]]
    if len(captions) < _MIN_SAMPLE:
        pytest.skip(f"only {len(captions)} captioned rows; too few to measure")

    usable = [c for c in captions if _clean_daydream(c)]
    rate = len(usable) / len(captions)
    assert rate >= _MIN_USABLE_RATE, (
        f"only {len(usable)}/{len(captions)} ({rate:.1%}) live captions clean to a "
        f"usable line, under the {_MIN_USABLE_RATE:.0%} floor -- the daydream lane "
        f"is going silent, check orion-thought's captioner"
    )


def test_no_cleaned_live_caption_can_forge_a_prompt_line() -> None:
    """The whitespace collapse is an injection guard, and this is the one
    place it is checked against text nobody wrote by hand."""
    _clean_daydream, _, _ = _lane()
    rows = _rows()
    for r in rows:
        cleaned = _clean_daydream(r["description"])
        if not cleaned:
            continue
        assert "\n" not in cleaned and "\r" not in cleaned, (
            f"caption survived cleaning with a line break: {cleaned!r}"
        )


def test_no_cleaned_live_caption_is_grounding_debris() -> None:
    """The vision model intermittently answers with coordinates instead of a
    description. Since only ONE caption reaches the prompt, a single one of
    these getting through IS the whole lane for that tick."""
    _clean_daydream, _DAYDREAM_DETECTOR_OUTPUT_RE, _ = _lane()
    rows = _rows()
    leaked = [
        _clean_daydream(r["description"])
        for r in rows
        if _clean_daydream(r["description"])
        and _DAYDREAM_DETECTOR_OUTPUT_RE.search(_clean_daydream(r["description"]))
    ]
    assert not leaked, f"{len(leaked)} grounding-output captions passed the guard: {leaked[:3]}"


def test_the_window_currently_contains_a_usable_caption() -> None:
    """Liveness, not correctness: an empty 12h window means the visual chain
    worker is down and the lane is silently contributing nothing. Reported as
    a failure here rather than as an absent prompt block nobody notices --
    the same class of gap that left Orion blind for 21h on 2026-08-21."""
    _clean_daydream, _, _DAYDREAM_MAX_AGE_SEC = _lane()
    rows = _rows()
    now = datetime.now(timezone.utc)
    fresh = [
        r
        for r in rows
        if isinstance(r["created_at"], datetime)
        and (now - r["created_at"]).total_seconds() <= _DAYDREAM_MAX_AGE_SEC
        and _clean_daydream(r["description"])
    ]
    assert fresh, (
        f"no usable caption in the last {_DAYDREAM_MAX_AGE_SEC / 3600:.0f}h "
        f"({len(rows)} rows total) -- orion-thought's visual chain is not producing"
    )
