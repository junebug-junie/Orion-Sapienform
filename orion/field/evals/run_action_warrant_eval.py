#!/usr/bin/env python3
"""Live-data eval for `orion/field/action_warrant.py`.

The unit tests prove the maths on synthetic inputs. This proves the signal is
non-degenerate against REAL history, which is the check CLAUDE.md 0A's metric
quality gate item 4 actually requires and the one that killed five earlier
candidates for this signal.

Fails on any of:

  NEVER_RESTS    the score effectively cannot read calm. This is the failure
                 that made every predecessor useless -- raw max, max|z| and
                 mean|z| each measured 0.00% of real ticks below mid-scale, and
                 a gate that can never rest looks identical to a busy system
                 from the outside while hiding a dead pipeline.
  NEVER_FIRES    the score effectively cannot flag anything. The mirror
                 failure, and the one this repo shipped as
                 metacog_pad_pulse_threshold (0.8 against a signal whose
                 observed max was 0.55).
  FLAT           real variation is absent -- a constant wearing a distribution.
  DEAD_INPUTS    too few dimensions are live for the score to mean anything.

Deliberately NOT asserted here: a target rate. The whole point of the signal is
that the rate is allowed to move with the field, so pinning it to a number would
reintroduce exactly the hand-typed constant this work exists to remove. The
bounds below are degeneracy detectors set far outside any plausible operating
point, not a calibration.

Run: python orion/field/evals/run_action_warrant_eval.py
Env: POSTGRES_URI (falls back to the standard local dev DSN)
     ACTION_WARRANT_EVAL_WINDOW_HOURS (default 6)
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

LOG = logging.getLogger("action_warrant_eval")

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:55432/conjourney"

# Below this many ticks the verdict is not meaningful; report rather than guess.
# Matches run_attention_bound_proposal_eval.py's own convention.
MIN_TICKS = 200

# Degeneracy brackets. Wide on purpose -- see the module docstring.
MIN_REST_FRACTION = 0.01   # at least 1% of ticks below mid-scale
MIN_FIRE_FRACTION = 0.001  # at least 0.1% of ticks above the high mark
REST_MARK = 0.5
FIRE_MARK = 0.95

# The score must show real spread, not a constant. Population stddev of a
# genuinely varying [0,1] score sits well above this; a frozen one sits at 0.
MIN_STDDEV = 0.01

# Fisher's null needs live inputs. Below this median, the signal is reporting
# on an almost-dead field and its own verdict is not trustworthy.
MIN_MEDIAN_LIVE_DIMENSIONS = 2


def open_readonly_connection(dsn: str):
    import psycopg2  # lazy: module imports cleanly without the driver

    conn = psycopg2.connect(dsn)
    # autocommit=False deliberately: the server-side (named) cursor below
    # streams `substrate_field_state` rather than loading the window into
    # client memory, and psycopg2 requires a transaction for that. readonly
    # keeps this a genuinely non-mutating eval.
    conn.set_session(readonly=True, autocommit=False)
    return conn


def fetch_fields(conn, window_start: datetime):
    """Stream real persisted FieldStateV1 rows through the REAL production
    function -- not a reimplementation of the formula here, so a regression in
    `action_warrant()` shows up in this eval rather than being masked by a copy.
    """
    from orion.field.action_warrant import action_warrant
    from orion.schemas.field_state import FieldStateV1

    scores: list[float] = []
    live_counts: list[int] = []
    exclusion_reasons: dict[str, int] = {}

    with conn.cursor(name="action_warrant_eval") as cur:
        cur.itersize = 500
        cur.execute(
            "SELECT field_json FROM substrate_field_state "
            "WHERE created_at >= %s ORDER BY created_at",
            (window_start,),
        )
        for (field_json,) in cur:
            try:
                field = FieldStateV1.model_validate(field_json)
            except Exception:  # noqa: BLE001 - a malformed historical row is not a verdict
                continue
            warrant = action_warrant(field)
            live_counts.append(len(warrant.contributing))
            for reason in warrant.excluded.values():
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
            if warrant.score is not None:
                scores.append(warrant.score)
    return scores, live_counts, exclusion_reasons


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_DSN)
    hours = float(os.environ.get("ACTION_WARRANT_EVAL_WINDOW_HOURS", "6"))
    window_start = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        conn = open_readonly_connection(dsn)
    except Exception as exc:  # noqa: BLE001 - eval script: report and exit
        LOG.error("could not connect to Postgres: %s", exc)
        return 2

    try:
        scores, live_counts, exclusions = fetch_fields(conn, window_start)
    finally:
        conn.close()

    if len(scores) < MIN_TICKS:
        LOG.info(json.dumps({
            "verdict": "INSUFFICIENT_DATA",
            "scored_ticks": len(scores),
            "min_ticks": MIN_TICKS,
            "window_hours": hours,
        }, indent=2))
        return 0

    rest = sum(1 for s in scores if s < REST_MARK) / len(scores)
    fire = sum(1 for s in scores if s > FIRE_MARK) / len(scores)
    stddev = statistics.pstdev(scores)
    median_live = statistics.median(live_counts) if live_counts else 0

    reasons: list[str] = []
    if rest < MIN_REST_FRACTION:
        reasons.append(
            f"NEVER_RESTS: only {rest:.4%} of ticks below {REST_MARK} "
            f"(need >= {MIN_REST_FRACTION:.2%})"
        )
    if fire < MIN_FIRE_FRACTION:
        reasons.append(
            f"NEVER_FIRES: only {fire:.4%} of ticks above {FIRE_MARK} "
            f"(need >= {MIN_FIRE_FRACTION:.3%})"
        )
    if stddev < MIN_STDDEV:
        reasons.append(f"FLAT: stddev {stddev:.6f} < {MIN_STDDEV}")
    if median_live < MIN_MEDIAN_LIVE_DIMENSIONS:
        reasons.append(
            f"DEAD_INPUTS: median live dimensions {median_live} "
            f"< {MIN_MEDIAN_LIVE_DIMENSIONS}"
        )

    ordered = sorted(scores)
    LOG.info(json.dumps({
        "verdict": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "scored_ticks": len(scores),
        "window_hours": hours,
        "rest_fraction": round(rest, 6),
        "fire_fraction": round(fire, 6),
        "stddev": round(stddev, 6),
        "min": round(ordered[0], 6),
        "p25": round(ordered[len(ordered) // 4], 6),
        "median": round(statistics.median(scores), 6),
        "p75": round(ordered[3 * len(ordered) // 4], 6),
        "max": round(ordered[-1], 6),
        "median_live_dimensions": median_live,
        "exclusions_by_reason": exclusions,
    }, indent=2))
    return 1 if reasons else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
