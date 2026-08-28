"""Power intent and its settlement.

A workload declares what it is about to draw BEFORE it draws it, and an independent
meter says what actually happened. The declaration is a falsifiable claim about the
physical world, settled by an instrument the declaring service does not own.

WHY THIS SHAPE. Design doc
``docs/superpowers/specs/2026-08-28-consequential-action-space-and-power-budget-design.md``.
Power is the first budget in this system that is physically capped (the UPS, not a
config value), externally metered, genuinely contested between every consumer on the
circuit, and shared in failure. Overriding a workload requires knowing it is coming --
which is what makes the declaration necessary, and what incidentally makes it
falsifiable.

WHY 1 Hz SAMPLING IS PART OF THE CONTRACT AND NOT AN IMPLEMENTATION DETAIL. Measured
live 2026-08-28 on circe: the standing GPU telemetry samples every ~31s, and reverie
diffusion on GPU 2 produced 332 images in three days while registering **4 busy samples
out of 8,754** (3 above 60W). The workload is real and the watts are real; the
instrument simply steps over them. Settling 332 intents against 4 observations yields a
degenerate residual -- a confident number that measures the sampler, not the workload.
So the intent itself triggers a fast-sample window, and the achieved rate is reported
back in the settlement rather than assumed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SettlementOutcome = Literal["settled", "no_samples", "deadline_expired"]


class PowerIntentV1(BaseModel):
    """A workload announcing an imminent draw, before it starts.

    ``expected_watts`` IS OPTIONAL AND None MEANS UNKNOWN. It does not mean zero, and a
    consumer must never coerce it to zero. The first intents a new workload declares are
    expected to carry None deliberately: nobody has measured this workload yet, and
    inventing a plausible constant to fill the field would bake a fabricated number into
    the first day of the dataset that is later fitted against. Declare unknown, measure,
    then start declaring a value derived from real settlements.
    """

    model_config = ConfigDict(extra="ignore")

    intent_id: str
    workload_kind: str

    # Which meter settles this. `gpu_index` scopes it to one card; None means the whole
    # node's wall draw. Node is required because GPU indices are only meaningful
    # per-host -- and because an index is not a stable identity across a driver reload.
    node: str
    gpu_index: Optional[int] = None

    declared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expected_duration_sec: float

    # None = not yet known. See the class docstring.
    expected_watts: Optional[float] = None

    # Hard stop for the fast-sample window. REQUIRED, not a convenience: a crashed or
    # forgotten workload must not leave a sampler pinned at 1 Hz indefinitely, and an
    # unbounded window would also make "how long did this actually take" unanswerable.
    deadline: datetime

    correlation_id: Optional[str] = None


class PowerIntentSettledV1(BaseModel):
    """What the meter saw over the intent's window.

    THE OUTCOME FIELD EXISTS SO AN UNMEASURED INTENT CANNOT READ AS A CHEAP ONE. If the
    sampler produced nothing, this settles as ``no_samples`` with null watts -- never as
    0.0 W. Those are opposite claims: one says "we did not see", the other says "we saw
    nothing drawn", and collapsing them is the failure mode that has cost this repo real
    time (a producer reporting a full tank while blind, a metric decayed to zero reading
    as calm).

    ``sample_count`` and ``achieved_sample_hz`` are reported so a consumer can judge
    whether the window was resolved well enough to believe. A settlement built from two
    samples of a four-second burst is arithmetic, not measurement, and the numbers here
    are what let a reader tell the difference.
    """

    model_config = ConfigDict(extra="ignore")

    intent_id: str
    workload_kind: str
    node: str
    gpu_index: Optional[int] = None

    settled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: SettlementOutcome

    # The window actually observed, which is not necessarily the window declared.
    window_start: datetime
    window_end: datetime
    sample_count: int = 0
    achieved_sample_hz: Optional[float] = None

    # All null unless outcome == "settled".
    actual_peak_watts: Optional[float] = None
    actual_mean_watts: Optional[float] = None
    energy_joules: Optional[float] = None

    # Draw on this meter immediately BEFORE the window opened. Without it, a 220 W peak
    # on a card that idles at 42 W is indistinguishable from one that idles at 200 W --
    # and it is the delta, not the absolute, that this workload actually caused.
    baseline_watts: Optional[float] = None

    # Echoed from the intent so a settlement is self-contained.
    expected_watts: Optional[float] = None

    # actual_peak_watts - expected_watts. STAYS None WHEN EITHER SIDE IS UNKNOWN.
    # A residual computed against an unknown expectation is not a small error, it is a
    # meaningless one, and filling it with 0.0 would make an unmeasured workload look
    # perfectly predicted.
    residual_watts: Optional[float] = None
