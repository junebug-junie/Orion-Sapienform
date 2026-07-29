"""Reversion-based per-domain prediction-error trend -- shared by the offline
AST/HOT replay (`scripts/analysis/measure_ast_hot_reducer.py`) and the live
`_attention_broadcast_tick()` self-model tick
(`services/orion-substrate-runtime/app/worker.py`).

Extracted 2026-07-29 from `measure_ast_hot_reducer.py`, where this formula was
originally developed and validated (see the function's own docstring for the
full backtest methodology and numbers). Moved here, unchanged, so the live
tick can call the exact same validated formula instead of re-deriving it --
`reduce_attention_self_model()` itself does no time-series math of its own
(see that module's docstring); this is the caller-side companion that
produces its `prediction_error_trend_by_domain` input.
"""

from __future__ import annotations


def compute_prediction_error_trend(
    window: list[dict[str, float]],
) -> dict[str, float]:
    """Reversion-based trend per domain, over an ordered (oldest-to-newest)
    window of per-tick `{domain: prediction_error}` snapshots. Positive =
    predicted to rise next; negative = predicted to fall next (the contract
    `reduce_attention_self_model()`'s `prediction_error_trend_by_domain`
    argument consumes).

    **This is deliberately mean(prior half) - mean(recent half) -- the
    OPPOSITE sign of the naive "continue the recent direction" formula this
    replaced.** That naive continuation formula (mean(recent) - mean(prior),
    predicting the recent direction keeps going) was empirically WORSE than
    a coin flip: back-tested against real `substrate_field_state` biometrics
    history (the only domain with enough real variance to test against --
    execution/chat/route are real but tiny, transport reads exactly 0.0 for
    entire multi-hour windows, per the already-documented transport
    narrow-scope finding) on two independent, non-overlapping 3-4h windows,
    checking whether the named domain's value actually moved in the
    predicted direction ~60s later: 37.7% accuracy (n=332, z=-4.50) and
    41.0% accuracy (n=454, z=-3.85), both far below chance. A
    decay-projection formula (compare the current value to what pure
    exponential continuation of the prior half's own trajectory would
    predict, before this fix) only marginally improved on that (43-45% on a
    separate earlier back-test), still well below chance -- the problem
    isn't the extrapolation method, it's the extrapolation *direction*: this
    signal is spike-and-settle (a burst of activity is more often followed
    by quiet than by more activity), not momentum-carrying. Predicting the
    OPPOSITE of the naive continuation direction scored 62.3% and 59.0% on
    the same two windows (sums to exactly 100% with continuation by
    construction -- same backtest pass, same sample set, strictly opposite
    predictions) -- real, reproducible, above-chance signal. Full validation
    methodology and numbers:
    `docs/superpowers/specs/2026-07-23-predicted-shift-reversion-finding.md`.

    **Validated on biometrics only, applied to all domains.** No
    independent data exists yet for execution/transport/chat/route --
    applying the same reversion sign to them is a reasoned extrapolation
    (those domains are computed the same way, as deltas between
    successive states from discrete events -- turns, exec steps, tool
    calls -- so the same spike-and-settle dynamic is plausible), not an
    independently confirmed one. `bus_synaptic` (added 2026-07-25) is a
    different computation shape (live EWMA/z-score edges, not a
    successive-state delta) and is even less covered by this
    extrapolation -- also unvalidated, not just unconfirmed by analogy. In
    practice this mostly matters for biometrics anyway (it wins the
    cross-domain argmax the overwhelming majority of the time in live
    replay -- see the reducer script's own report), but a future pass
    should back-test the other four domains separately once enough real
    variance accumulates, rather than assuming this generalizes.

    A domain only gets a trend value if it has at least one reading in BOTH
    halves -- comparing a half with zero real observations would fabricate
    a trend from nothing, not measure one. Fewer than 2 ticks in the window
    yields an empty dict (nothing to compare yet).

    **Window semantics are cadence-relative, not a fixed real-world time
    span.** The offline replay's own default (`PREDICTION_ERROR_TREND_
    WINDOW_TICKS = 30`) was sized against the field lane's ~2s tick cadence
    (~60s real-world span) -- a starting anchor, not independently
    calibrated (see that constant's own docstring). The live
    `_attention_broadcast_tick()` caller uses a differently-sized window
    (`SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS`, appended once per
    ~30s broadcast tick, not once per ~2s field tick) to keep a comparable
    real-world span at its own coarser cadence -- this function itself is
    agnostic to what a "tick" means to its caller, it only operates on
    whatever ordered window it's handed.
    """
    if len(window) < 2:
        return {}
    mid = len(window) // 2
    prior_half, recent_half = window[:mid], window[mid:]

    def _domain_means(half: list[dict[str, float]]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for snapshot in half:
            for domain, value in snapshot.items():
                sums[domain] = sums.get(domain, 0.0) + value
                counts[domain] = counts.get(domain, 0) + 1
        return {d: sums[d] / counts[d] for d in sums}

    prior_means = _domain_means(prior_half)
    recent_means = _domain_means(recent_half)
    return {
        domain: prior_means[domain] - recent_means[domain]
        for domain in recent_means
        if domain in prior_means
    }
