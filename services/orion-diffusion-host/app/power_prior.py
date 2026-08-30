"""What this workload has actually cost, learned from its own settlements.

`PowerIntentV1.expected_watts` was None on all 26 settlements that existed
when this was written, so `residual_watts` -- the field that says "you
predicted X, reality was Y" -- was structurally incapable of being populated.
The measurement half of the loop worked; there was simply nothing to be wrong
about. See docs/superpowers/specs/2026-08-30-power-intent-prior-design.md.

Estimator: median of a bounded recent window, per
(workload_kind, node, gpu_index). Both properties are load-bearing, and both
come from the live data rather than from taste:

  full 60s window       n=24  mean 252.7  median 254.3  sd  8.0
  truncated 20s window  n= 2  mean  96.5  median  96.5  sd 67.9

MEDIAN, not mean: the two truncated rows are not outliers of the same process,
they are a superseded instrument (a sampling window that closed before the
workload finished). Pooling all 26 gives mean 240.7 / sd 45.2 -- two bad points
in 26 inflate the spread 5.6x and drag the estimate ~14W below the true centre.
A mean would carry that contamination into the first thing Orion ever predicts
about itself.

BOUNDED, not all-history: a median over everything is robust to those two
points today but has no way to forget a genuine regime change -- a model swap,
a different card, a resolution change. The window is what makes this
self-correcting rather than permanently anchored to whatever ran first.

This module is pure: no bus, no clock, no I/O. It is a dict of deques and a
median, so it can be tested without a container.
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Deque, Dict, Optional, Tuple

# (workload_kind, node, gpu_index). gpu_index is part of the key because an
# index is only meaningful per host and different cards draw differently --
# the same reasoning PowerIntentV1 uses for requiring `node`.
PriorKey = Tuple[str, str, Optional[int]]


class PowerPrior:
    """Bounded, per-workload memory of settled peak draw."""

    def __init__(self, *, window: int = 20, min_samples: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if min_samples > window:
            raise ValueError("min_samples cannot exceed window")
        self._window = window
        self._min_samples = min_samples
        self._seen: Dict[PriorKey, Deque[float]] = {}

    def observe(
        self,
        *,
        workload_kind: str,
        node: str,
        gpu_index: Optional[int],
        outcome: str,
        actual_peak_watts: Optional[float],
    ) -> bool:
        """Record one settlement. Returns whether it was counted.

        Only `settled` contributes. `no_samples` and `deadline_expired` carry
        no peak, and the settlement schema went out of its way to keep "we did
        not see" distinct from "we saw nothing drawn" -- coercing either to a
        number here would destroy exactly that distinction one layer down.
        """
        if outcome != "settled":
            return False
        if actual_peak_watts is None:
            return False
        value = float(actual_peak_watts)
        # A negative or zero peak is not a quiet card, it is a broken reading.
        # nvidia-smi returning nothing already surfaces as outcome != settled.
        if value <= 0.0:
            return False

        key: PriorKey = (workload_kind, node, gpu_index)
        bucket = self._seen.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._window)
            self._seen[key] = bucket
        bucket.append(value)
        return True

    def expected_watts(
        self, *, workload_kind: str, node: str, gpu_index: Optional[int]
    ) -> Optional[float]:
        """The prior, or None if this workload has not been measured enough.

        None here is the honest answer and propagates as None into the intent.
        PowerIntentV1's own docstring is explicit that None means UNKNOWN and
        must never be coerced to zero -- and that inventing a plausible
        constant would bake a fabricated number into a dataset that is later
        fitted against.
        """
        bucket = self._seen.get((workload_kind, node, gpu_index))
        if bucket is None or len(bucket) < self._min_samples:
            return None
        return float(statistics.median(bucket))

    def sample_count(
        self, *, workload_kind: str, node: str, gpu_index: Optional[int]
    ) -> int:
        bucket = self._seen.get((workload_kind, node, gpu_index))
        return len(bucket) if bucket else 0
