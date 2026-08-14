"""Derive the implicit ``decayed_unattended`` attention-loop verdict.

The producer counterpart to `verdicts.py` (the reader). `AttentionLoopOutcomeV1`
has always specified three verdicts -- `resolved`, `dismissed`, and
`decayed_unattended`, the last documented in `orion/schemas/attention_salience.py`
as "the human verdict (Resolve/Dismiss) or **implicit decay** -- the
sparse-but-clean label the refit later trains on."

**`decayed_unattended` has never been written once.** Confirmed live 2026-08-14:
all 4 rows in `attention_loop_outcome` are `resolved`/`juniper`. The label stream
is 100% hand-clicked, which cannot scale to judging a machine-generated candidate
stream -- and an outcome measure that only exists when a human clicks is the
reason the drives program never had one at all.

This module supplies the missing half: a loop that was scored, never explicitly
closed, and then stopped being re-scored has decayed unattended. That is a real
outcome ("surfaced, ignored, gone"), it is machine-derivable, and it is the
negative class any future salience refit needs.

Pure and synchronous: no DB, no clock of its own, no I/O. Callers pass observed
state and `now`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

# Verdicts that mean a human explicitly closed the loop. Mirrors
# `verdicts.py::TERMINAL_VERDICTS` -- a loop a human already judged must never be
# relabelled by this module.
TERMINAL_VERDICTS = frozenset({"resolved", "dismissed"})

IMPLICIT_VERDICT = "decayed_unattended"

# Absolute floor on silence before a loop is called decayed.
#
# Not a fresh guess: reuses the 24h cooldown `attention_loops_store.py
# ::suppress_loop(cooldown_sec=86400.0)` already applies when a *closed* loop is
# held out of the coalition. That is this repo's existing, shipped answer to "how
# long before an attention loop is considered done with", so the implicit
# labeller adopts it rather than inventing a second, divergent constant.
DEFAULT_MIN_SILENCE = timedelta(hours=24)

# The one un-derived number here, and it is named rather than buried: silence
# must also exceed this multiple of the loop's OWN median re-scoring gap, so a
# slow-cadence loop is not declared dead merely for being slow. Relative, for the
# same reason `DeviationGate.relative_sigma_floor` is relative -- an absolute-only
# rule silently mislabels anything whose natural rhythm differs from the constant.
# Report the sensitivity sweep rather than defending a single value.
#
# **Currently INERT on live data, disclosed rather than left to look validated**
# (measured 2026-08-14, `measure_attention_outcome_coverage.py --sweep`): all 7
# loops ever scored were scored in tight bursts, so their median gap rounds to
# ~0h, `cadence_multiple * 0 == 0`, and the absolute floor always binds. Label
# counts are identical at 1x / 3x / 10x across every floor tried. It is kept
# because it is a real guard with a real regression test
# (`test_slow_cadence_loop_is_not_declared_dead_for_being_slow`) and because a
# slow-cadence loop is exactly what a future interoceptive candidate stream would
# produce -- but nothing here has yet exercised it, and it must not be cited as
# validated. If it is still inert once real candidate diversity exists, remove it
# rather than keep a knob that provably does nothing (same call already made for
# `DeviationGate`'s `impulse_k`).
DEFAULT_SILENCE_CADENCE_MULTIPLE = 3.0


@dataclass(frozen=True)
class LoopObservation:
    """What is known about one loop from `attention_salience_trace` + labels."""

    loop_id: str
    theme_key: str
    trace_times: Sequence[datetime]
    """Every time this loop was scored, ascending. Its own cadence comes from here."""

    last_salience: float = 0.0
    last_features: dict[str, Any] | None = None
    existing_verdict: str | None = None


@dataclass(frozen=True)
class ImplicitVerdict:
    loop_id: str
    theme_key: str
    verdict: str
    silence: timedelta
    median_gap: timedelta | None
    salience_at_close: float
    features_at_close: dict[str, Any]
    reason: str


def median_gap(times: Sequence[datetime]) -> timedelta | None:
    """Median interval between successive scorings, or None if under 3 points.

    Under 3 points there is at most one gap, which is a sample of one and not a
    cadence -- the same short-window trap that has already produced two wrong
    claims in this arc. Callers fall back to the absolute floor instead.
    """
    if len(times) < 3:
        return None
    ordered = sorted(times)
    gaps = sorted((b - a).total_seconds() for a, b in zip(ordered, ordered[1:]))
    n = len(gaps)
    mid = gaps[n // 2] if n % 2 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2.0
    return timedelta(seconds=mid)


def derive_implicit_verdicts(
    loops: Iterable[LoopObservation],
    *,
    now: datetime,
    min_silence: timedelta = DEFAULT_MIN_SILENCE,
    cadence_multiple: float = DEFAULT_SILENCE_CADENCE_MULTIPLE,
) -> list[ImplicitVerdict]:
    """Return one `ImplicitVerdict` per loop that has decayed unattended.

    A loop qualifies when **all** hold:

    1. It has no terminal (human) verdict. A human judgement is never overwritten.
    2. It has at least one recorded scoring.
    3. Silence since its last scoring exceeds `min_silence` **and**, when the loop
       has enough history to have a cadence at all, `cadence_multiple` x its own
       median re-scoring gap.

    Loops still being actively re-scored return nothing -- they have not decayed,
    they are simply unresolved, which is a different state and must not be
    labelled as an outcome.
    """
    out: list[ImplicitVerdict] = []
    for loop in loops:
        if loop.existing_verdict in TERMINAL_VERDICTS:
            continue
        if not loop.trace_times:
            continue

        last_seen = max(loop.trace_times)
        silence = now - last_seen
        if silence < min_silence:
            continue

        gap = median_gap(loop.trace_times)
        if gap is not None and gap.total_seconds() > 0:
            required = timedelta(seconds=gap.total_seconds() * cadence_multiple)
            if silence < required:
                continue
            reason = (
                f"silent {silence.total_seconds() / 3600:.1f}h; "
                f">{cadence_multiple:g}x own median gap "
                f"{gap.total_seconds() / 3600:.2f}h and >{min_silence.total_seconds() / 3600:.0f}h floor"
            )
        else:
            reason = (
                f"silent {silence.total_seconds() / 3600:.1f}h; "
                f">{min_silence.total_seconds() / 3600:.0f}h floor "
                f"(too few traces for a cadence)"
            )

        out.append(
            ImplicitVerdict(
                loop_id=loop.loop_id,
                theme_key=loop.theme_key,
                verdict=IMPLICIT_VERDICT,
                silence=silence,
                median_gap=gap,
                salience_at_close=max(0.0, min(1.0, float(loop.last_salience))),
                features_at_close=dict(loop.last_features or {}),
                reason=reason,
            )
        )
    return out
