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
import re
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

# Rows are read over a WINDOW, not the lifetime of the table (review finding,
# 2026-08-28). A lifetime rate cannot detect a later failure: at ~6 rows/hour
# a 0.85 floor over 299 rows trips after 42 consecutive bad captions (~7h),
# but over 10,000 rows it needs 1,410 (~10 days). The eval would have decayed
# into a no-op by October. A windowed rate keeps a fixed detection latency
# forever. 24h rather than the lane's own 12h so a quiet night still leaves a
# measurable sample.
_WINDOW_HOURS = 24
# Measured over the 24h window on 2026-08-28: 94.7% of captioned rows clean to
# a usable line. The floor sits below that because the point is to catch a
# COLLAPSE (a captioner regression emitting grounding coordinates or bare tag
# dumps for everything), not to freeze today's number into a gate. With a
# windowed rate this is a real detection threshold: it trips after ~15% of one
# day's captions go bad.
_MIN_USABLE_RATE = 0.85
# Below this many rows the rate is an artifact, not a measurement. Kept small
# enough that a normal 24h (~144 rows) is never near it -- a large value here
# would silently convert this eval into a permanent skip.
_MIN_SAMPLE = 40


def _rows():
    """Captioned rows inside the window, or skip if there is no live DB.

    Only a CONNECTION problem is a skip. A missing or renamed table raises
    `ProgrammingError`, which must fail loudly -- swallowing it would turn
    the loudest possible failure (the lane's table is gone) into a silent
    pass, including for the liveness test below.
    """
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import OperationalError
    except ImportError:  # pragma: no cover - environment without sqlalchemy
        pytest.skip("sqlalchemy not installed")

    dsn = os.getenv("DAYDREAM_EVAL_DATABASE_URL", _DEFAULT_DSN)
    engine = create_engine(dsn)
    try:
        with engine.connect() as conn:
            return (
                conn.execute(
                    text(
                        "SELECT created_at, chain_json->>'description' AS description "
                        "FROM reverie_visual_chain "
                        "WHERE created_at > now() - make_interval(hours => :hours) "
                        "ORDER BY created_at DESC"
                    ),
                    {"hours": _WINDOW_HOURS},
                )
                .mappings()
                .all()
            )
    except OperationalError as exc:
        pytest.skip(f"no reachable database ({exc.__class__.__name__})")
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def rows():
    """One fetch shared by every test in this module -- each calling `_rows()`
    itself opened and discarded its own engine."""
    return _rows()


def test_caption_pipeline_yields_usable_text_on_the_live_corpus(rows) -> None:
    """The lane's real failure mode is silence: if the captioner regresses,
    every row cleans to "" and the prompt block just stops appearing, with no
    error and no failing unit test."""
    _clean_daydream, _, _ = _lane()
    captions = [r["description"] for r in rows if r["description"]]
    if len(captions) < _MIN_SAMPLE:
        pytest.skip(f"only {len(captions)} captioned rows in {_WINDOW_HOURS}h; too few to measure")

    usable = [c for c in captions if _clean_daydream(c)]
    rate = len(usable) / len(captions)
    assert rate >= _MIN_USABLE_RATE, (
        f"only {len(usable)}/{len(captions)} ({rate:.1%}) live captions in the last "
        f"{_WINDOW_HOURS}h clean to a usable line, under the {_MIN_USABLE_RATE:.0%} "
        f"floor -- the daydream lane is going silent, check orion-thought's captioner"
    )


def test_the_cleaner_actually_removes_debris_present_in_the_raw_captions(rows) -> None:
    """Asserts on the INPUT, not the output (review finding, 2026-08-28).

    Checking that `_clean_daydream`'s OUTPUT has no newline / no coordinate
    pair cannot fail by construction: the cleaner collapses whitespace and
    screens for coordinates on a superstring of its own return value, so a
    match in the output implies a match in the input implies it returned "".
    Two tests here previously did exactly that and were provably vacuous.

    What is worth checking is the real thing: that debris DOES occur in the
    raw corpus, and that none of it survives. If the first assertion ever
    fails, the producer got fixed and this test can go."""
    _clean_daydream, detector_re, _ = _lane()
    raw = [r["description"] for r in rows if r["description"]]
    if len(raw) < _MIN_SAMPLE:
        pytest.skip(f"only {len(raw)} captioned rows in {_WINDOW_HOURS}h; too few to measure")

    dirty = [
        c
        for c in raw
        if "\n" in c or "\r" in c or detector_re.search(c) or "**" in c
    ]
    assert dirty, (
        "no raw caption in the window contains a newline, a coordinate pair, or "
        "markdown -- either orion-thought's captioner was fixed (good; delete this "
        "test) or this eval is no longer reading the real column"
    )

    survivors = [
        cleaned
        for cleaned in (_clean_daydream(c) for c in dirty)
        if cleaned and ("\n" in cleaned or "\r" in cleaned or detector_re.search(cleaned) or "**" in cleaned)
    ]
    assert not survivors, f"{len(survivors)} debris captions survived cleaning: {survivors[:3]}"


def test_no_rendered_caption_echoes_the_captioner_instruction(rows) -> None:
    """Live 2026-08-28: 12 of 290 rendered captions carried the vision
    prompt's own instruction text back into Orion's prompt ("Directly visible
    objects and people include: 1. **Galaxy**: ..."). Pinned against the live
    corpus because the exact wording is the captioner's to change -- a unit
    fixture would only pin the phrasing I happened to see."""
    _clean_daydream, _, _ = _lane()
    raw = [r["description"] for r in rows if r["description"]]
    if len(raw) < _MIN_SAMPLE:
        pytest.skip(f"only {len(raw)} captioned rows in {_WINDOW_HOURS}h; too few to measure")

    # The echo signature is the LIST PREAMBLE ("...objects include:"), not the
    # words "directly visible" on their own -- a real caption legitimately
    # ends "The spiral structure and the central black hole are directly
    # visible." and must not trip this. Caught by this eval on its first run
    # against live data, which is the point of having it.
    echo_re = re.compile(r"visible objects?\b[^.]*\b(?:include|are)\s*:", re.IGNORECASE)
    echoes = [
        cleaned
        for cleaned in (_clean_daydream(c) for c in raw)
        if cleaned and echo_re.search(cleaned)
    ]
    assert not echoes, f"{len(echoes)} rendered captions echo the captioner's instructions: {echoes[:2]}"


def test_the_window_currently_contains_a_usable_caption(rows) -> None:
    """Liveness, not correctness: an empty window means the visual chain
    worker is down and the lane is silently contributing nothing. Reported as
    a failure here rather than as an absent prompt block nobody notices --
    the same class of gap that left Orion blind for 21h on 2026-08-21."""
    _clean_daydream, _, max_age_sec = _lane()
    now = datetime.now(timezone.utc)
    fresh = [
        r
        for r in rows
        if isinstance(r["created_at"], datetime)
        and (now - r["created_at"]).total_seconds() <= max_age_sec
        and _clean_daydream(r["description"])
    ]
    assert fresh, (
        f"no usable caption in the last {max_age_sec / 3600:.0f}h "
        f"({len(rows)} rows in the {_WINDOW_HOURS}h window) -- orion-thought's "
        f"visual chain is not producing"
    )


def test_the_eval_thresholds_cannot_be_quietly_neutered() -> None:
    """The floor and the sample minimum are the only two knobs here, and
    nothing else pins them: setting `_MIN_USABLE_RATE = 0.0` or
    `_MIN_SAMPLE = 100000` makes every test above pass (or silently skip)
    forever while claiming to measure something. Runs without a database, so
    it holds in CI too."""
    assert 0.5 < _MIN_USABLE_RATE < 1.0, (
        "a floor outside this range is not a collapse detector: at or below 0.5 it "
        "cannot fail on real data, at 1.0 it fails on a single bad caption"
    )
    # ~144 rows land in a normal 24h window at the chain's ~600s cadence, so a
    # minimum anywhere near that turns the eval into a permanent skip.
    assert _MIN_SAMPLE <= 60
    assert _WINDOW_HOURS <= 48, "a wider window re-introduces the lifetime-rate problem"
