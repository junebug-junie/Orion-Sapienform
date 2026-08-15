"""Producer liveness: tell "genuinely calm" apart from "nobody is writing this".

This module exists because of a defect the rest of this package **cannot**
detect, documented in `orion/field/pressure.py` and confirmed live 2026-08-14:

    When `services/orion-biometrics` goes quiet, every input to
    `resource_pressure` decays toward 0, the dimension reads *calm*, and because
    `config/feedback/feedback_policy.v1.yaml` lists `resource_pressure:
    decrease` under `positive_delta_channels`, the in-flight action is credited
    with a POSITIVE OUTCOME for a producer outage.

A per-channel ratio check cannot see this: a channel in the field digester's
`NODE_DECAY_CHANNELS` decaying toward rest is *correct* behaviour, so "is it
decaying" is not the question. The question is **"is anything still writing
it?"** -- and the answer is visible in the series shape without needing
per-channel perturbation attribution (which `substrate_field_applied_deltas`
does not carry -- it is `delta_id -> receipt_id -> applied_at` only).

The discriminator:

- The digester's decay loop can only ever make a channel **smaller**
  (`value * 0.92` per tick).
- Any real write -- a perturbation from a live producer -- breaks monotonicity,
  whether it raises the value or merely holds it flat against the decay.
- So a series that is monotonically non-increasing across a long window, and has
  actually shrunk over it, has received **no producer writes at all** in that
  window. That is a silent producer, not a calm one.

This is the exact mirror of `orion.field.channel_glossary.classify_channel_series`'s
`ratchet_suspect` verdict (monotonically non-*decreasing* => a mode=add channel
with no decay). Deliberately shaped the same way rather than invented fresh, and
it composes with that classifier rather than replacing it: `classify_channel_series`
answers "is this channel telling me anything", this answers "is anyone still
writing it".

**Window length is load-bearing.** Judged over a short tail, an ordinary quiet
stretch looks identical to an outage. Callers must pass samples spanning the
whole window of interest, not a local tail -- see `MIN_WINDOW_SAMPLES`.
"""
from __future__ import annotations

from typing import Literal

Verdict = Literal["refreshed", "silent_producer", "at_floor", "insufficient_data"]

# Below this many samples the monotonicity signal is noise: with 4 points a
# healthy-but-quiet channel is non-increasing by chance a meaningful fraction of
# the time. Matches the spirit of `channel_glossary.RATCHET_MIN_SAMPLES`.
MIN_WINDOW_SAMPLES = 20

# A series must have shrunk by at least this factor across the window for
# "monotonically non-increasing" to mean decay rather than a flat rest state.
# 0.92/tick reaches this in ~28 ticks, so any genuinely decaying channel clears
# it easily across a real window.
MIN_DECAY_FACTOR = 0.5

# At or below this, a channel has already bottomed out: it cannot demonstrate
# further decay, so monotonicity carries no information and the honest verdict
# is "at_floor" rather than a confident silent/refreshed call.
FLOOR_MAGNITUDE = 1e-300


def classify_producer_liveness(series: list[float]) -> Verdict:
    """Classify whether anything is still writing this channel.

    Verdicts:

    - ``insufficient_data``: fewer than `MIN_WINDOW_SAMPLES` points, or the
      window is entirely non-positive. No claim either way.
    - ``at_floor``: every value is already at/below `FLOOR_MAGNITUDE`. The
      channel has finished decaying, so its shape can no longer distinguish a
      silent producer from a calm one. **Not** a clean bill of health -- it means
      the question is unanswerable from this series alone, which is exactly the
      state that makes the `resource_pressure` defect invisible.
    - ``silent_producer``: monotonically non-increasing across the window AND
      shrunk past `MIN_DECAY_FACTOR`. Only the decay loop touched it; no producer
      wrote it. **This is the finding.**
    - ``refreshed``: something broke the monotonic decay, i.e. a real write
      landed. The channel's low value (if any) is genuine rest.
    """
    if len(series) < MIN_WINDOW_SAMPLES:
        return "insufficient_data"

    if all(abs(v) <= FLOOR_MAGNITUDE for v in series):
        return "at_floor"

    positives = [v for v in series if v > 0.0]
    if not positives:
        return "insufficient_data"

    # A single upward step anywhere is proof of a write. Tolerance is relative so
    # it does not become an absolute threshold on small-scale channels -- the same
    # mistake the absolute `sigma_floor` made in `deviation_gate.py`.
    for prev, cur in zip(series, series[1:]):
        if cur > prev * (1.0 + 1e-9) + 1e-15:
            return "refreshed"

    first, last = series[0], series[-1]
    if first <= 0.0:
        return "insufficient_data"
    if last / first > MIN_DECAY_FACTOR:
        # Non-increasing but essentially flat: resting, not decaying. A live
        # producer writing the same rest value repeatedly looks like this.
        return "refreshed"

    return "silent_producer"
