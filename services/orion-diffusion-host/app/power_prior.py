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

PERSISTED, since 2026-11-03: the GPU pool relaunches this container for
almost every generation (gpu2 goes back to the agent model after ~15 idle
minutes, then diffusion reclaims it). An in-memory-only prior restarted empty
each time, so `expected_watts` was None on 0 of 9 intents 11-02 and 1 of 17
on 10-31 although ~1,000 settlements existed. The window now round-trips
through a small JSON file (`to_state` / `from_state`), and every sample carries
the time it was observed so a sample older than `max_age_sec` is dropped
instead of anchoring the estimate after a long gap.

The class is pure: no bus, no clock (callers pass `now`), no I/O. The two file
helpers at the bottom are the only I/O, and both are best-effort.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

# (workload_kind, node, gpu_index). gpu_index is part of the key because an
# index is only meaningful per host and different cards draw differently --
# the same reasoning PowerIntentV1 uses for requiring `node`.
PriorKey = Tuple[str, str, Optional[int]]
# (peak watts, unix time the settlement was observed).
Sample = Tuple[float, float]

STATE_VERSION = 1


class PowerPrior:
    """Bounded, per-workload memory of settled peak draw."""

    def __init__(self, *, window: int = 20, min_samples: int = 3,
                 max_age_sec: Optional[float] = None) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if min_samples > window:
            raise ValueError("min_samples cannot exceed window")
        if max_age_sec is not None and max_age_sec <= 0:
            raise ValueError("max_age_sec must be > 0 (or None for no age limit)")
        self._window = window
        self._min_samples = min_samples
        self._max_age_sec = max_age_sec
        self._seen: Dict[PriorKey, Deque[Sample]] = {}

    def observe(
        self,
        *,
        workload_kind: str,
        node: str,
        gpu_index: Optional[int],
        outcome: str,
        actual_peak_watts: Optional[float],
        at: float = 0.0,
    ) -> bool:
        """Record one settlement observed at unix time `at`. Returns whether it was counted.

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
        if not math.isfinite(value) or value <= 0.0:
            return False

        key: PriorKey = (workload_kind, node, gpu_index)
        bucket = self._seen.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._window)
            self._seen[key] = bucket
        bucket.append((value, float(at)))
        return True

    def _fresh(self, key: PriorKey, now: Optional[float]) -> list:
        bucket = self._seen.get(key)
        if not bucket:
            return []
        if self._max_age_sec is None or now is None:
            return [v for v, _ in bucket]
        floor = float(now) - self._max_age_sec
        return [v for v, t in bucket if t >= floor]

    def expected_watts(
        self, *, workload_kind: str, node: str, gpu_index: Optional[int],
        now: Optional[float] = None,
    ) -> Optional[float]:
        """The prior, or None if this workload has not been measured enough.

        With `max_age_sec` set and `now` given, only samples observed within
        the last `max_age_sec` count.

        None here is the honest answer and propagates as None into the intent.
        PowerIntentV1's own docstring is explicit that None means UNKNOWN and
        must never be coerced to zero -- and that inventing a plausible
        constant would bake a fabricated number into a dataset that is later
        fitted against.
        """
        values = self._fresh((workload_kind, node, gpu_index), now)
        if len(values) < self._min_samples:
            return None
        return float(statistics.median(values))

    def sample_count(
        self, *, workload_kind: str, node: str, gpu_index: Optional[int],
        now: Optional[float] = None,
    ) -> int:
        return len(self._fresh((workload_kind, node, gpu_index), now))

    def to_state(self) -> Dict[str, Any]:
        """The whole window as plain JSON-able data."""
        return {
            "version": STATE_VERSION,
            "buckets": [
                {"workload_kind": k[0], "node": k[1], "gpu_index": k[2],
                 "samples": [[v, t] for v, t in bucket]}
                for k, bucket in self._seen.items()
            ],
        }

    def load_state(self, state: Any, *, now: Optional[float] = None) -> int:
        """Merge a saved window back in (oldest first, still bounded by `window`).

        Skips anything malformed, any sample older than `max_age_sec` when `now`
        is given, and any non-positive peak. Returns how many samples were kept.
        """
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
            return 0
        floor = None
        if self._max_age_sec is not None and now is not None:
            floor = float(now) - self._max_age_sec
        kept = 0
        for b in state.get("buckets") or []:
            if not isinstance(b, dict):
                continue
            gpu = b.get("gpu_index")
            if gpu is not None and not isinstance(gpu, int):
                continue
            samples = []
            for item in b.get("samples") or []:
                try:
                    v, t = float(item[0]), float(item[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (math.isfinite(v) and math.isfinite(t)) or v <= 0.0:
                    continue
                if floor is not None and t < floor:
                    continue
                samples.append((v, t))
            samples.sort(key=lambda s: s[1])
            for v, t in samples:
                if self.observe(workload_kind=str(b.get("workload_kind") or ""), node=str(b.get("node") or ""),
                                gpu_index=gpu, outcome="settled", actual_peak_watts=v, at=t):
                    kept += 1
        return kept


def load_prior_file(prior: PowerPrior, path: str, *, now: float) -> int:
    """Warm `prior` from `path`. A missing or unreadable file is an empty prior, never an error."""
    if not path:
        return 0
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return prior.load_state(state, now=now)


def save_prior_file(prior: PowerPrior, path: str) -> bool:
    """Write the window atomically (temp file + rename). Returns False on any OS error."""
    if not path:
        return False
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(prior.to_state()), encoding="utf-8")
        os.replace(tmp, target)
        return True
    except OSError:
        return False
