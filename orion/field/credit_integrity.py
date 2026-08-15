"""Can a feedback-credited channel currently be fooled by a producer outage?

THE TRAP THIS WATCHES FOR
-------------------------
`config/feedback/feedback_policy.v1.yaml` lists `resource_pressure: decrease`
under `positive_delta_channels`, so a DROP is credited to the in-flight action.
If the drop happens because a producer went quiet rather than because the action
helped, the loop learns from an outage. `orion/attention/tension/liveness.py`
documents this defect; this module is the scheduled watch for its precondition.

TWO SIGNALS, AT DIFFERENT RESOLUTIONS
-------------------------------------
`classify_producer_liveness()` is reused verbatim rather than reimplemented. It
already answers "did anything write this series", and over the whole history it
is not enough: one real write anywhere makes the entire series `refreshed`.
Measured 2026-08-14, 10,497 live ticks over 6 hours -- every credited dimension
classifies `refreshed` end to end. True about six hours, silent about any 30
seconds inside it, and 30 seconds (`windows.field_after_window_sec`) is what
the feedback loop actually samples.

**But the shape signal cannot be run at 30 seconds, and this was measured
rather than assumed.** `MIN_WINDOW_SAMPLES = 20` and `MIN_DECAY_FACTOR = 0.5`
("0.92/tick reaches this in ~28 ticks"), while 30s at the live 2.04s cadence is
~15 ticks. So a 30s window fails the classifier twice over: too few samples,
and not enough elapsed decay to shrink by half. Those constants are principled
-- a short window makes an ordinary quiet stretch look like an outage, which is
the classifier's own documented reason for them -- so lowering them to fit
would manufacture false positives, not resolution.

The honest consequence: **shape has a floor on how short an outage it can see,
and that floor is longer than the window being protected.** It is run at the
span it can actually support (derived from the data's own cadence, reported as
`shape_span_seconds`) and its limit is stated rather than hidden.

Provenance carries the short end, because it needs ONE tick, not twenty.
`apply_diffusion()` recomputes `capability_provenance` every tick and CLEARS it
when nobody contributes (services/orion-field-digester/app/digestion/
diffusion.py), so `collect_field_channel_pressures()` falls back to naming the
capability itself instead of a source node. A merged value whose winner is not
a node id is a decayed remnant -- true at any window length, and true at the
floor where shape has nothing to read (the classifier's declared `at_floor`
blind spot, which it calls "exactly the state that makes the
`resource_pressure` defect invisible").

So: provenance is the detector at feedback-window resolution; shape is the
longer-horizon cross-check. Reported separately, never blended, so a reader can
see which one fired and at what resolution.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No guard, no gating of the feedback loop itself. Changing a learning loop is
proposal-mode work (CLAUDE.md 0A) and the measured case for it does not exist
yet: over those same 10,497 ticks the provenance-unbacked events were 55, all
ISOLATED SINGLE TICKS (longest run 1 tick = ~2s) against a 30s window. This
watches for the precondition so the guard can be designed against a real
instance instead of a hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from orion.attention.tension.liveness import classify_producer_liveness
from orion.field.pressure import (
    CHANNEL_DIMENSION_MAP,
    collect_field_channel_pressures,
    field_pressures,
)
from orion.schemas.field_state import FieldStateV1

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "config" / "feedback" / "feedback_policy.v1.yaml"

# The verdict that means "only the decay loop touched this".
SILENT = "silent_producer"
AT_FLOOR = "at_floor"

_EPOCH = datetime(1970, 1, 1)

# Ticks the shape classifier needs before its verdict means anything:
# MIN_WINDOW_SAMPLES samples AND enough elapsed decay to clear
# MIN_DECAY_FACTOR (~28 ticks at 0.92). Taking the larger of the two, since
# either one short makes the verdict `insufficient_data` or `refreshed` by
# default.
SHAPE_MIN_TICKS = 28


@dataclass(frozen=True)
class ChannelSample:
    at: datetime
    value: float
    provenance_backed: bool


@dataclass(frozen=True)
class WindowFinding:
    """A span in which a credited channel could have misled the loop."""

    channel: str
    kind: str  # SILENT | "unbacked_run"
    started_at: datetime
    span_seconds: float
    tick_count: int

    def describe(self) -> str:
        return (
            f"{self.channel}: {self.kind} for {self.span_seconds:.0f}s "
            f"({self.tick_count} ticks) from {self.started_at.isoformat()}"
        )


@dataclass
class CreditIntegrityReport:
    window_seconds: float
    # The span the SHAPE signal was actually run at. Larger than
    # window_seconds whenever the classifier cannot support the policy window
    # -- which is the normal case, and is reported so nobody reads a clean
    # shape result as coverage it does not have.
    shape_span_seconds: float = 0.0
    channels: dict[str, int] = field(default_factory=dict)  # channel -> samples
    findings: list[WindowFinding] = field(default_factory=list)
    # Longest observed run per channel, reported even when below threshold, so
    # a trend toward the window is visible before it crosses.
    longest_unbacked_run: dict[str, int] = field(default_factory=dict)
    whole_series_verdict: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.findings)


def load_credited_dimensions(
    policy_path: Path | None = None,
) -> tuple[dict[str, list[str]], float]:
    """DIMENSION -> contributing channels, and the window the loop samples over.

    The policy names DIMENSIONS (`resource_pressure`), and
    `collect_field_channel_pressures()` returns CHANNELS. They are not the same
    namespace: `resource_pressure` is fed by the channel `pressure`.
    `execution_pressure` and `reliability_pressure` happen to spell the same in
    both, which is a coincidence and was enough to hide the bug -- the first
    version of this module looked up policy names directly in the channel dict
    and silently reported **0 samples** for `resource_pressure`, the one
    channel this whole module exists for, while printing a healthy report.

    A dimension that maps to no channel is returned with an empty list rather
    than dropped, so `analyse_credit_integrity` can raise it as a finding
    instead of reporting zero samples as zero problems.

    Read from the policy, never hardcoded: `positive_delta_channels` is the set
    whose DELTA becomes evidence, narrower than `pressure_channels` (sampled
    but not credited).
    """
    from orion.feedback.policy import load_feedback_policy

    policy = load_feedback_policy(policy_path or POLICY_PATH)
    mapping = {
        dim: sorted(ch for ch, d in CHANNEL_DIMENSION_MAP.items() if d == dim)
        for dim in sorted(policy.positive_delta_channels)
    }
    return mapping, float(policy.windows.field_after_window_sec)


def _samples_for(
    states: list[FieldStateV1], dimensions: dict[str, list[str]]
) -> dict[str, list[ChannelSample]]:
    """Value = the DIMENSION the loop actually credits; backing = whether every
    contributing channel had a real source this tick.

    Sampling the dimension rather than a channel matters: `extractors.py`
    calls `field_pressures()` and diffs dimensions, so a per-channel series
    would be measuring something the loop never sees.
    """
    out: dict[str, list[ChannelSample]] = {dim: [] for dim in dimensions}
    for state in states:
        merged, provenance = collect_field_channel_pressures(state)
        dims = field_pressures(state)
        for dim, channels in dimensions.items():
            if dim not in dims:
                continue
            # A node id means a real source produced this tick's value. The
            # capability id means diffusion cleared provenance -- see module
            # docstring. Membership, not a string prefix, for the same reason
            # field_channel_glossary_routes._winning_write_time uses it.
            # ALL contributing channels must be backed: one unbacked
            # contributor is enough to make the dimension a partial remnant.
            backed = bool(channels) and all(
                (provenance.get(ch) or "") in state.node_vectors
                for ch in channels
                if ch in merged
            )
            out[dim].append(
                ChannelSample(at=state.generated_at, value=dims[dim], provenance_backed=backed)
            )
    return out


def _median_tick_seconds(samples: list[ChannelSample]) -> float:
    """Measured from the data, never assumed. The live median is 2.04s, but a
    detector that hardcodes it silently changes meaning if the digester's
    cadence ever moves."""
    if len(samples) < 2:
        return 0.0
    gaps = sorted(
        (samples[i + 1].at - samples[i].at).total_seconds() for i in range(len(samples) - 1)
    )
    return gaps[len(gaps) // 2]


def _shape_span(samples: list[ChannelSample], window_seconds: float) -> float:
    """The shortest span at which the shape classifier can return a real
    verdict, or the policy window if that is already long enough."""
    tick = _median_tick_seconds(samples)
    if tick <= 0:
        return window_seconds
    return max(window_seconds, SHAPE_MIN_TICKS * tick)


def _rolling_windows(samples: list[ChannelSample], window_seconds: float):
    """Yield (start_index, end_index_exclusive) spans of >= window_seconds.

    Advances the left edge one sample at a time; overlapping windows are
    intentional, since an outage that straddles a fixed boundary would be
    split in half by non-overlapping tiling and could fall under the
    classifier's minimum sample count.
    """
    n = len(samples)
    right = 0
    for left in range(n):
        if right < left:
            right = left
        while right < n and (samples[right].at - samples[left].at).total_seconds() < window_seconds:
            right += 1
        if right >= n:
            break
        yield left, right + 1


def analyse_credit_integrity(
    states: list[FieldStateV1],
    dimensions: dict[str, list[str]],
    window_seconds: float,
) -> CreditIntegrityReport:
    """Findings are spans, not tick counts. A tick rate is an implementation
    detail of the digester; the feedback window is in seconds and so is this."""
    report = CreditIntegrityReport(window_seconds=window_seconds)
    by_channel = _samples_for(states, dimensions)

    for channel, samples in by_channel.items():
        report.channels[channel] = len(samples)
        if not dimensions.get(channel):
            # A credited dimension fed by no channel cannot be watched at all,
            # and reporting it as "0 samples, 0 findings" is how the first
            # version of this module hid its own bug.
            report.findings.append(
                WindowFinding(
                    channel=channel,
                    kind="unmappable_dimension",
                    started_at=states[0].generated_at if states else _EPOCH,
                    span_seconds=0.0,
                    tick_count=0,
                )
            )
            continue
        if not samples:
            continue

        values = [s.value for s in samples]
        report.whole_series_verdict[channel] = classify_producer_liveness(values)

        # 1. Shape, run at the span the classifier can actually support --
        # NOT at window_seconds, which it cannot (see module docstring).
        shape_span = _shape_span(samples, window_seconds)
        report.shape_span_seconds = max(report.shape_span_seconds, shape_span)
        worst: tuple[int, int] | None = None
        for left, right in _rolling_windows(samples, shape_span):
            if classify_producer_liveness([s.value for s in samples[left:right]]) == SILENT:
                # Report the FIRST such window rather than every overlapping
                # one -- a 10-minute outage would otherwise emit hundreds of
                # near-identical rows and bury everything else.
                worst = (left, right)
                break
        if worst is not None:
            left, right = worst
            report.findings.append(
                WindowFinding(
                    channel=channel,
                    kind=SILENT,
                    started_at=samples[left].at,
                    span_seconds=(samples[right - 1].at - samples[left].at).total_seconds(),
                    tick_count=right - left,
                )
            )

        # 2. Provenance: the longest run with nothing contributing.
        run = 0
        run_start = 0
        longest = 0
        longest_start = 0
        for i, sample in enumerate(samples):
            if sample.provenance_backed:
                run = 0
                continue
            if run == 0:
                run_start = i
            run += 1
            if run > longest:
                longest, longest_start = run, run_start
        report.longest_unbacked_run[channel] = longest

        if longest:
            end = longest_start + longest - 1
            span = (samples[end].at - samples[longest_start].at).total_seconds()
            if span >= window_seconds:
                report.findings.append(
                    WindowFinding(
                        channel=channel,
                        kind="unbacked_run",
                        started_at=samples[longest_start].at,
                        span_seconds=span,
                        tick_count=longest,
                    )
                )
    return report
