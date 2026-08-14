"""Turn a `substrate_field_state.field_json` payload into per-(node, channel)
observations for the deviation gate.

Pure and synchronous: takes an already-loaded dict, does no I/O. The caller
decides where `field_json` came from (Postgres replay, a live tick, a fixture).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

# Field channels decay geometrically toward zero, and a channel nothing refreshes
# eventually lands in subnormal-float territory (live 2026-08-14: every
# `node:circe` channel reads ~3e-321 while its own `staleness` still reports 0.0).
# Subnormals are not meaningfully different from rest, but they *are* different
# enough to pump artificial variance into a baseline. Collapse them to a clean
# 0.0 and count how often that happened, rather than either trusting them or
# silently dropping the observation.
SUBNORMAL_EPSILON = 1e-12


@dataclass(frozen=True)
class Observation:
    node_id: str
    channel: str
    value: float
    """The value the gate should train on -- subnormals collapsed to a clean 0.0."""

    raw_value: float
    """The value exactly as stored, before subnormal coercion.

    Carried separately because `geometric_decay_ratio()` needs the *uncoerced*
    series to work at all: it rejects any series containing a non-positive
    value, so feeding it coerced values would make it structurally incapable of
    detecting the very decay artifact it exists to detect. Found in self-review
    2026-08-14, after a first run reported 'decay-suspect series: none' for
    exactly that reason rather than because the data was clean.
    """

    coerced_subnormal: bool


def iter_observations(field_json: Mapping[str, Any]) -> Iterator[Observation]:
    """Yield one `Observation` per (node, channel) present in `node_vectors`.

    Non-numeric and non-finite values are skipped entirely -- they are not real
    readings and must not train a baseline.
    """
    node_vectors = field_json.get("node_vectors")
    if not isinstance(node_vectors, Mapping):
        return
    for node_id, channels in node_vectors.items():
        if not isinstance(channels, Mapping):
            continue
        for channel, raw in channels.items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            coerced = abs(value) < SUBNORMAL_EPSILON and value != 0.0
            yield Observation(
                node_id=str(node_id),
                channel=str(channel),
                value=0.0 if coerced else value,
                raw_value=value,
                coerced_subnormal=coerced,
            )


# Ratios of a real decay loop agree to within this *relative* tolerance.
#
# Absolute tolerance was wrong and was caught in review (2026-08-14): subnormal
# floats carry only ~10 significant bits, so a perfect 0.92-per-tick decay in the
# 3e-321 range has a successive-ratio spread of ~1.7e-3 -- far outside the 1e-6
# absolute tolerance originally used. The probe was therefore blind in exactly
# the magnitude range it exists to police, while reporting "none". 2% is
# comfortably above subnormal quantisation noise and far below the spread of any
# genuinely varying channel (whose successive ratios swing by tens of percent).
_DECAY_RATIO_REL_TOL = 2e-2

# Values at or below this are the *end state* of the decay artifact rather than
# any real reading -- ~1e-308 is the smallest normal double.
SUBNORMAL_PINNED_CEILING = 1e-300


def geometric_decay_ratio(series: list[float]) -> float | None:
    """Return the constant successive ratio of `series` if it is a clean
    geometric decay *in progress*, else None.

    Detects the decayed-to-zero artifact CLAUDE.md documents for
    `node:substrate.route` (a generic staleness-decay loop multiplying an
    unrefreshed channel by a fixed factor every tick, producing a value that
    looks *exactly* like genuinely-calm-at-zero). A channel matching this is not
    calm -- it is unmaintained -- and the measurement names it rather than
    folding it into an admission-rate denominator as if it were healthy signal.

    Catches only decay *in progress*. A series that has already bottomed out --
    pinned at a constant subnormal, or at exactly 0.0 -- has no informative
    ratio and is reported by `subnormal_pinned()` instead. Both are needed: the
    end state is what most dead channels actually look like at any given moment.

    Returns None for series that are too short, contain a non-positive value, or
    whose successive ratios are not near-constant in relative terms.
    """
    if len(series) < 4:
        return None
    ratios: list[float] = []
    for prev, cur in zip(series, series[1:]):
        if prev <= 0.0 or cur <= 0.0:
            return None
        ratios.append(cur / prev)
    mean = sum(ratios) / len(ratios)
    # A real decay loop holds its ratio; a live channel does not.
    if mean <= 0.0 or mean >= 1.0:
        return None
    if any(abs(r - mean) > _DECAY_RATIO_REL_TOL * mean for r in ratios):
        return None
    return mean


def subnormal_pinned(series: list[float]) -> bool:
    """True when `series` has bottomed out at the end state of the decay artifact.

    The shape caught: every value a non-zero magnitude at or below
    `SUBNORMAL_PINNED_CEILING`. Successive ratios are then exactly 1.0, so
    `geometric_decay_ratio()` rejects it (`mean >= 1.0`) -- yet no real channel
    naturally rests at 3e-323. Only a decay loop produces that, so it is positive
    evidence of an unmaintained channel rather than a calm one.

    **Exactly-0.0 series are deliberately NOT reported here**, even though they
    are the other end state of the same artifact. A channel genuinely at rest
    reads 0.0 too (`repair_pressure` with nothing to repair, say), and the two
    are indistinguishable without history predating the decay. Flagging them
    would trade a false-clean for a false-alarm on most of the channel set; the
    honest position is that this detector cannot separate them and does not try.

    Added in review 2026-08-14 after a live check found 15 series sitting in the
    `0 < v < 1e-300` band in a 30-minute window with zero of them flagged.
    """
    if len(series) < 4:
        return False
    return all(0.0 < abs(v) <= SUBNORMAL_PINNED_CEILING for v in series)
