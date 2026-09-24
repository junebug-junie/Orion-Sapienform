"""Transport baseline reducer -- per-hop log-latency EWMA that cannot learn "busy"
as normal.

Spec: ``docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-
baseline-design.md`` sections A1-A4. Pure and deterministic: no I/O, no clock
reads (every timestamp comes from the snapshot's own ``window_end``), fully
JSON-serializable state. The equilibrium service owns Redis persistence and
trigger publishing; this module only folds snapshots and names conditions.

## What it reads

One ``RpcHealthSnapshotV1`` payload (plain dict, so this module does not depend
on the schema model) carrying ``channel_latency: {hop: {success_count,
timeout_count, log_ms_sum, log_ms_sumsq, max_ms}}``. State is keyed
``(service, instance, hop)``. A snapshot without ``channel_latency`` (an old
producer) is skipped entirely -- nothing is guessed from the pooled p95.

## Per-key state (A1)

- ``fast``: EWMA mean/variance of the window log-mean, via
  ``orion/bus/ewma.py::compute_ewma_update`` with ``min_variance`` passed
  explicitly (log-ms scale; see ``TransportBaselineConfig.min_variance``).
  **Folding choice, disclosed:** each evaluated window folds its *mean*
  (``log_ms_sum / success_count``) once with a fixed alpha. It is not folded
  per-sample, so a busy window does not teach the baseline faster than a quiet
  one -- per-sample folding would make "busy" normalize itself exactly when it
  is most suspicious. ``z`` is therefore a window-level chart ("this window's
  mean vs typical window means"); ``min_calls`` keeps a window mean from being
  a single sample.
- ``level``: the same window log-mean folded *without* guard 1. See
  "Deviation from the spec" below.
- ``floor``: asymmetric EWMA of ``level`` (see "Deviation"). Time-based half-lives,
  fast down / very slow up, and frozen upward while a saturation episode is
  open.
- ``calls``: EWMA of calls-per-minute on this key, evidence only (guard 4).

Warm-up: ``fast`` and ``level`` fold with ``alpha = max(alpha, 1/n)`` (a running
mean for the first ~1/alpha windows), and ``floor`` simply equals ``level``
until ``n_warm`` evaluations -- otherwise one unlucky first window would seed
a floor that its slow-up half-life keeps for days (measured in the mesh eval:
a calm hop stuck at ratio 1.31 from a single low first draw).

Sparse keys: a key that has fewer than ``min_calls`` successes in one window
accumulates sufficient statistics across consecutive windows (up to
``max_aggregate_windows``) and is evaluated once the pool reaches
``min_calls``. This is the spec's "or across aggregated consecutive windows"
and is what lets the ~1 call / 30 s background lane be measured at all.

## Anti-normalization guards (A2)

1. A window with ``z >= spike_z``, or one whose pooled windows saw a timeout,
   does not update ``fast``. ``borderline_z <= z < spike_z`` folds the value
   clipped at ``mean + borderline_z * sigma``.
2. ``saturation_ratio = exp(level - floor)``, opens at ``>= saturation_ratio``.
3. Saturation *or spike* observed hot for ``>= regime_after_s`` (hot time,
   ``Episode.hot_s`` -- silence never counts) emits exactly one
   ``regime_shift`` and only then re-seeds ``floor`` and ``fast``'s mean to
   the new level. The spike path matters: a step too small for
   ``saturation_ratio`` (e.g. 1.5x) but at z >= 3 freezes ``fast`` under
   guard 1 for good, so without it the spike would stay open forever while
   the floor crept up unannounced. The regime_shift *ends* the latency
   episodes it explains: no separate close is emitted, so a plateau is never
   reported as cleared before its new normal has been stated.
4. Calls-per-minute is carried on every event and never gates anything.
5. Latency conditions need ``fast.count >= n_warm`` qualifying evaluations.
   Spike needs ``z >= spike_z`` on ``spike_sustain`` consecutive evaluations.
   Timeouts need no minimum. ``zero_success`` = the key has succeeded before,
   has usual traffic, and this window has 0 successes and > 0 timeouts. It
   subsumes ``timeout`` for the same failure (one outage = one row).
6. Identity is ``(service, instance or node, hop)``; a window whose
   ``window_end`` is not newer than the key's last folded window (a
   redelivery or replay) is ignored.

## Deviation from the spec, and why

The spec defines ``saturation_ratio = exp(fast.mean - floor)``. That only
works for slow creep. For a *step* change (every window at z >= 3), guard 1
freezes ``fast`` entirely, the ratio stays ~1 forever, and the step is never
reported as a regime shift -- the spike episode would simply stay open
indefinitely. ``level`` (the unguarded window-mean EWMA) tracks both creep and
steps, so the ratio uses it instead. ``fast`` is still the only thing ``z`` is
measured against, so guard 1 still protects the spike detector.

The spec defines the floor as an asymmetric EWMA of the *window* log-mean.
Measured in the mesh eval, that cannot rest at 1: with a fast-down / slow-up
filter over raw window means, the floor tracks the lower envelope of window
noise, so a perfectly calm hop read ``saturation_ratio`` 1.22-1.41 (sparse
hops worst) -- a structural floor under "calm", the same failure class as the
``sqrt(2/pi)`` floor in CLAUDE.md's metric gate. **Floor follows level:** it is
an asymmetric EWMA of ``level`` (already smoothed), so calm reads ~1.0.

The spec also quotes ``alpha_up ~= 0.002`` per 30 s window as "a half-life of
about 3 days". Those disagree: 0.002 per window is a ~2.9 h half-life, which
would absorb a 2.5x plateau before the 6 h regime check. The floor here is
time-based and uses the stated intent (3 day up / ~90 s down half-lives).

## Episodes (A4)

Each (key, condition) is an open -> escalate* -> close state machine.
- open: condition first observed (spike: after ``spike_sustain``).
- escalate: magnitude reached ``escalate_factor`` x the last announced
  magnitude (spike: z; saturation: ratio; timeout: timeouts per window).
- close: condition not observed for ``close_quiet_s`` (15 min; a key
  faulting every few minutes is one episode, not a flapping pair per
  window). Carries ``duration_s`` and ``peak_ms``. Latency episodes on a
  hop with no latency evidence arriving at all also age out this way. A hop
  evicted from state (idle > ``max_idle_s`` or over ``max_keys``) emits a
  close for every open episode rather than dropping it silently.

## Config fingerprint

The config is hashed into the state. Resuming a state produced under a
different config cold-starts (``load_state`` returns ``cold_start_reason``)
rather than silently mixing baselines calibrated with different constants --
the caveat ``trend_reducer`` disclosed and left open.

## Import cost

``import orion.metacog.transport_baseline`` executes ``orion/metacog/
__init__.py`` which pulls pydantic (same disclosed caveat as
``trend_reducer``). The consumer (orion-equilibrium-service) already depends
on pydantic, so this is accepted rather than restructured.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from orion.bus.ewma import compute_ewma_update

STATE_SCHEMA_VERSION = 2

CONDITIONS = ("timeout", "zero_success", "spike", "saturation", "regime_shift")


@dataclass(frozen=True)
class TransportBaselineConfig:
    """All reducer constants. Every field is part of the fingerprint.

    Defaults are the spec's proposals; the log-only week (acceptance check 1)
    is what sets them for real.
    """

    # fast / level EWMA, per evaluated window.
    fast_alpha: float = 0.05
    level_alpha: float = 0.05
    # Log-ms variance floor. Real per-window log-mean variance for LLM hops is
    # ~0.05-0.5; 0.01 (sigma 0.1, ~10% relative) only binds for very tight
    # hops, where it stops a near-constant key from z-spiking on jitter.
    min_variance: float = 0.01
    # calls-per-minute EWMA, per raw window (including windows with no traffic).
    calls_alpha: float = 0.05
    # floor half-lives (seconds of window time).
    floor_half_life_down_s: float = 90.0
    floor_half_life_up_s: float = 3 * 24 * 3600.0
    # Warm-up / sample minimums.
    min_calls: int = 5
    n_warm: int = 10
    max_aggregate_windows: int = 20
    # Firing.
    spike_z: float = 3.0
    # Materiality: a spike/saturation is only hot when the window is also at
    # least this many ms above its reference, in absolute terms. z and ratio
    # are scale-free, so without it a 10 ms status poll reading 14 ms opened a
    # "spike" live (2026-09-24). Provisional; the log-only week tunes it.
    min_excess_ms: float = 250.0
    borderline_z: float = 2.0
    spike_sustain: int = 2
    saturation_ratio: float = 2.0
    saturation_close_ratio: float = 1.5
    regime_after_s: float = 6 * 3600.0
    escalate_factor: float = 2.0
    # Quiet time before an episode closes. 15 min: a hop faulting every few
    # minutes is one episode, not an open/close pair per fault.
    close_quiet_s: float = 900.0
    # State bounds.
    max_idle_s: float = 7 * 24 * 3600.0
    max_keys: int = 512

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Episode:
    opened_ts: float
    last_hot_ts: float
    announced_magnitude: float
    peak_ms: float | None = None
    peak_magnitude: float = 0.0
    # Time the condition was actually observed hot, gaps longer than
    # close_quiet_s excluded -- regime_shift is judged on this, never on
    # wall-clock since open (silence is not evidence of a sustained regime).
    hot_s: float = 0.0


@dataclass
class KeyState:
    service: str
    instance: str | None
    hop: str
    last_seen_ts: float = 0.0
    # window_end of the last window folded for this key (duplicate/replay guard)
    last_window_end: float = 0.0
    last_eval_ts: float = 0.0
    # fast (guarded)
    fast_mean: float = 0.0
    fast_var: float = 0.0
    fast_count: int = 0
    # level (unguarded)
    level_mean: float = 0.0
    level_count: int = 0
    # floor
    floor: float | None = None
    floor_ts: float | None = None
    # calls per minute
    calls_ewma: float = 0.0
    calls_count: int = 0
    # sparse-key pooling
    pend_n: int = 0
    pend_sum: float = 0.0
    pend_sumsq: float = 0.0
    pend_max_ms: float | None = None
    pend_windows: int = 0
    pend_tainted: bool = False
    # spike sustain counter
    hot_streak: int = 0
    episodes: dict[str, Episode] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportConditionEvent:
    condition: str
    phase: str
    service: str
    instance: str | None
    key: str
    excluded: bool
    z: float | None
    saturation_ratio: float | None
    baseline_ms: float | None
    floor_ms: float | None
    window_mean_ms: float | None
    calls_per_min: float
    calls_per_min_usual: float | None
    duration_s: float | None
    peak_ms: float | None
    timeout_count: int


@dataclass(frozen=True)
class KeyObservation:
    """One per (key, snapshot) -- the log-only phase's per-key line."""

    service: str
    instance: str | None
    key: str
    excluded: bool
    evaluated: bool
    warm: bool
    z: float | None
    saturation_ratio: float | None
    window_mean_ms: float | None
    baseline_ms: float | None
    floor_ms: float | None
    calls_per_min: float
    calls_per_min_usual: float | None
    success_count: int
    timeout_count: int
    open_conditions: tuple[str, ...]


@dataclass
class TransportBaselineState:
    fingerprint: str
    keys: dict[str, KeyState] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldResult:
    events: list[TransportConditionEvent]
    observations: list[KeyObservation]
    skipped_reason: str | None = None


# --------------------------------------------------------------------------
# helpers


def state_key(service: str, instance: str | None, hop: str) -> str:
    return f"{service}|{instance or ''}|{hop}"


def is_excluded(hop: str, exclude_labels: Iterable[str]) -> bool:
    """A hop is excluded if its health label (after ``#``), its ``verb:`` name,
    or the whole key equals one of ``exclude_labels``."""
    labels = {str(x).strip() for x in exclude_labels if str(x).strip()}
    if not labels:
        return False
    if hop in labels:
        return True
    if "#" in hop and hop.rsplit("#", 1)[1] in labels:
        return True
    if hop.startswith("verb:") and hop[len("verb:"):] in labels:
        return True
    return False


def _parse_ts(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _as_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _half_life_alpha(dt: float, half_life: float) -> float:
    if dt <= 0 or half_life <= 0:
        return 0.0
    return 1.0 - 0.5 ** (dt / half_life)


def _exp(v: float | None) -> float | None:
    return math.exp(v) if v is not None else None


# --------------------------------------------------------------------------
# persistence


def new_state(config: TransportBaselineConfig) -> TransportBaselineState:
    return TransportBaselineState(fingerprint=config.fingerprint())


def state_to_dict(state: TransportBaselineState) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "fingerprint": state.fingerprint,
        "keys": {k: asdict(v) for k, v in state.keys.items()},
    }


_EPISODE_FLOATS = ("opened_ts", "last_hot_ts", "announced_magnitude", "peak_magnitude", "hot_s")
_KEY_FLOATS = (
    "last_seen_ts", "last_window_end", "last_eval_ts", "fast_mean", "fast_var",
    "level_mean", "calls_ewma", "pend_sum", "pend_sumsq",
)
_KEY_OPT_FLOATS = ("floor", "floor_ts", "pend_max_ms")
_KEY_INTS = ("fast_count", "level_count", "calls_count", "pend_n", "pend_windows", "hot_streak")


def _finite(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"not a finite number: {v!r}")
    return float(v)


def _strict_int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"not a non-negative int: {v!r}")
    return v


def _key_state_from_raw(raw: Any) -> KeyState:
    """Type-checked rebuild; raises on any wrong-typed field so a corrupt
    checkpoint is refused at load, not discovered by a crash on every fold."""
    if not isinstance(raw, dict):
        raise ValueError("key state is not a dict")
    service, hop = raw.get("service"), raw.get("hop")
    instance = raw.get("instance")
    if not isinstance(service, str) or not isinstance(hop, str):
        raise ValueError("service/hop must be strings")
    if instance is not None and not isinstance(instance, str):
        raise ValueError("instance must be a string or null")
    ks = KeyState(service=service, instance=instance, hop=hop)
    for name in _KEY_FLOATS:
        setattr(ks, name, _finite(raw.get(name, 0.0)))
    for name in _KEY_OPT_FLOATS:
        v = raw.get(name)
        setattr(ks, name, None if v is None else _finite(v))
    for name in _KEY_INTS:
        setattr(ks, name, _strict_int(raw.get(name, 0)))
    if not isinstance(raw.get("pend_tainted", False), bool):
        raise ValueError("pend_tainted must be bool")
    ks.pend_tainted = raw.get("pend_tainted", False)
    eps = raw.get("episodes") or {}
    if not isinstance(eps, dict):
        raise ValueError("episodes must be a dict")
    for cond, ep in eps.items():
        if cond not in CONDITIONS or not isinstance(ep, dict):
            raise ValueError(f"bad episode {cond!r}")
        pm = ep.get("peak_ms")
        ks.episodes[cond] = Episode(
            **{n: _finite(ep.get(n, 0.0)) for n in _EPISODE_FLOATS},
            peak_ms=None if pm is None else _finite(pm),
        )
    return ks


def load_state(
    data: Any, config: TransportBaselineConfig
) -> tuple[TransportBaselineState, str | None]:
    """Rebuild state from its JSON form. Never raises.

    Returns ``(state, cold_start_reason)``; the reason is ``None`` on a clean
    resume. A fingerprint mismatch is refused (cold start), not merged; so is
    any wrong-typed field.
    """
    fp = config.fingerprint()
    try:
        if data is None:
            return new_state(config), "no_saved_state"
        if not isinstance(data, dict):
            return new_state(config), "malformed_state"
        if data.get("schema_version") != STATE_SCHEMA_VERSION:
            return new_state(config), f"schema_version_mismatch:{data.get('schema_version')}"
        if data.get("fingerprint") != fp:
            return new_state(config), f"config_fingerprint_mismatch:{data.get('fingerprint')}!={fp}"
        keys_raw = data.get("keys")
        if not isinstance(keys_raw, dict):
            return new_state(config), "malformed_state"
        state = TransportBaselineState(fingerprint=fp)
        for k, raw in keys_raw.items():
            state.keys[str(k)] = _key_state_from_raw(raw)
        return state, None
    except Exception as exc:  # noqa: BLE001 -- documented never-raises contract
        return new_state(config), f"malformed_state:{type(exc).__name__}"


# --------------------------------------------------------------------------
# episode machinery


def _episode_step(
    ks: KeyState,
    cond: str,
    *,
    hot: bool,
    magnitude: float,
    now: float,
    peak_ms: float | None,
    config: TransportBaselineConfig,
) -> tuple[str, Episode] | None:
    """Advance one condition's state machine. Returns (phase, episode) when a
    phase transition must be announced, else None."""
    ep = ks.episodes.get(cond)
    if hot:
        if ep is None:
            ep = Episode(
                opened_ts=now,
                last_hot_ts=now,
                announced_magnitude=magnitude,
                peak_ms=peak_ms,
                peak_magnitude=magnitude,
            )
            ks.episodes[cond] = ep
            return "open", ep
        gap = now - ep.last_hot_ts
        if 0 < gap <= config.close_quiet_s:
            ep.hot_s += gap
        ep.last_hot_ts = now
        ep.peak_magnitude = max(ep.peak_magnitude, magnitude)
        if peak_ms is not None:
            ep.peak_ms = peak_ms if ep.peak_ms is None else max(ep.peak_ms, peak_ms)
        if ep.announced_magnitude > 0 and magnitude >= config.escalate_factor * ep.announced_magnitude:
            ep.announced_magnitude = magnitude
            return "escalate", ep
        return None
    if ep is not None and (now - ep.last_hot_ts) >= config.close_quiet_s:
        del ks.episodes[cond]
        return "close", ep
    return None


def _keep_alive(ks: KeyState, cond: str, now: float, config: TransportBaselineConfig) -> None:
    """Refresh an already-open episode without announcing anything (used when
    a stronger condition subsumes it this window)."""
    ep = ks.episodes.get(cond)
    if ep is None:
        return
    gap = now - ep.last_hot_ts
    if 0 < gap <= config.close_quiet_s:
        ep.hot_s += gap
    ep.last_hot_ts = now


# --------------------------------------------------------------------------
# fold


def _event(
    ks: KeyState,
    cond: str,
    phase: str,
    ep: Episode,
    *,
    now: float,
    excluded: bool,
    calls_per_min: float,
    usual: float | None,
    timeouts: int,
    z: float | None = None,
    ratio: float | None = None,
    wmean: float | None = None,
) -> TransportConditionEvent:
    return TransportConditionEvent(
        condition=cond,
        phase=phase,
        service=ks.service,
        instance=ks.instance,
        key=ks.hop,
        excluded=excluded,
        z=z,
        saturation_ratio=ratio,
        baseline_ms=_exp(ks.fast_mean) if ks.fast_count else None,
        floor_ms=_exp(ks.floor),
        window_mean_ms=wmean,
        calls_per_min=calls_per_min,
        calls_per_min_usual=usual,
        duration_s=(now - ep.opened_ts) if (phase != "open" or cond == "regime_shift") else None,
        peak_ms=ep.peak_ms,
        timeout_count=timeouts,
    )


def fold_snapshot(
    state: TransportBaselineState,
    payload: dict[str, Any],
    *,
    config: TransportBaselineConfig,
    exclude_labels: Iterable[str] = (),
) -> FoldResult:
    """Fold one rpc_health snapshot into ``state`` (mutated in place)."""
    if not isinstance(payload, dict):
        return FoldResult([], [], "not_a_dict")
    channel_latency = payload.get("channel_latency")
    if not isinstance(channel_latency, dict):
        return FoldResult([], [], "no_channel_latency")

    service = str(payload.get("service") or "unknown")
    # Producer identity: instance when set, else node. Two processes of one
    # service must never share a key (their windows interleave and would read
    # as duplicates / quiet windows for each other).
    ident_raw = payload.get("instance") or payload.get("node")
    instance = str(ident_raw) if ident_raw not in (None, "") else None
    now = _parse_ts(payload.get("window_end"))
    start = _parse_ts(payload.get("window_start"))
    if now is None:
        return FoldResult([], [], "no_window_end")
    window_s = (now - start) if (start is not None and now > start) else 30.0
    window_min = window_s / 60.0
    exclude = tuple(exclude_labels)

    events: list[TransportConditionEvent] = []
    observations: list[KeyObservation] = []

    # Every known key of this producer sees this window, even with no traffic,
    # so quiet windows age episodes and count as 0 calls in the usual-load EWMA.
    prefix = state_key(service, instance, "")
    hops: dict[str, dict[str, Any]] = {}
    for sk, ks in state.keys.items():
        if sk.startswith(prefix) and ks.service == service and ks.instance == instance:
            hops[ks.hop] = {}
    for hop, stats in channel_latency.items():
        hops[str(hop)] = stats if isinstance(stats, dict) else {}

    for hop, stats in hops.items():
        sk = state_key(service, instance, hop)
        ks = state.keys.get(sk)
        success = _as_int(stats.get("success_count"))
        timeouts = _as_int(stats.get("timeout_count"))
        if ks is None:
            if success == 0 and timeouts == 0:
                continue
            ks = KeyState(service=service, instance=instance, hop=hop)
            state.keys[sk] = ks
        if now <= ks.last_window_end:
            # Duplicate (redelivered) or out-of-order window: never fold twice,
            # never fold time backwards.
            continue
        ks.last_window_end = now
        excluded = is_excluded(hop, exclude)
        if success > 0 or timeouts > 0:
            ks.last_seen_ts = now

        calls = success + timeouts
        calls_per_min = calls / window_min if window_min > 0 else float(calls)
        usual = ks.calls_ewma if ks.calls_count > 0 else None
        max_ms = _as_float(stats.get("max_ms"))

        def emit(cond: str, phase: str, ep: Episode, **kw: Any) -> None:
            events.append(
                _event(
                    ks, cond, phase, ep, now=now, excluded=excluded,
                    calls_per_min=calls_per_min, usual=usual, timeouts=timeouts, **kw,
                )
            )

        # --- zero_success, then timeout: no minimum sample count ----------
        # zero_success = total loss on a hop that really succeeded before. It
        # subsumes `timeout` for the same failure: while it is open, a timeout
        # episode is not opened (one outage = one row), only kept alive if it
        # was already open.
        had_successes = ks.level_count > 0 or ks.pend_n > 0
        zero_hot = (
            timeouts > 0 and success == 0 and had_successes and usual is not None and usual > 0
        )
        step = _episode_step(
            ks, "zero_success", hot=zero_hot, magnitude=float(timeouts), now=now,
            peak_ms=None, config=config,
        )
        if step:
            emit("zero_success", step[0], step[1])
        if "zero_success" in ks.episodes and timeouts > 0:
            _keep_alive(ks, "timeout", now, config)
        else:
            step = _episode_step(
                ks, "timeout", hot=timeouts > 0, magnitude=float(timeouts), now=now,
                peak_ms=max_ms, config=config,
            )
            if step:
                emit("timeout", step[0], step[1])

        # --- calls EWMA (evidence only) ----------------------------------
        if ks.calls_count == 0:
            ks.calls_ewma = calls_per_min
        else:
            ks.calls_ewma = config.calls_alpha * calls_per_min + (1 - config.calls_alpha) * ks.calls_ewma
        ks.calls_count += 1

        # --- latency pooling ---------------------------------------------
        log_sum = _as_float(stats.get("log_ms_sum"))
        log_sumsq = _as_float(stats.get("log_ms_sumsq"))
        if success > 0 and log_sum is not None:
            ks.pend_n += success
            ks.pend_sum += log_sum
            ks.pend_sumsq += log_sumsq or 0.0
            if max_ms is not None:
                ks.pend_max_ms = max_ms if ks.pend_max_ms is None else max(ks.pend_max_ms, max_ms)
            ks.pend_windows += 1
            if timeouts > 0:
                ks.pend_tainted = True
        elif ks.pend_windows:
            ks.pend_windows += 1
            if timeouts > 0:
                # a timeout-only window inside a pool still taints it (guard 1)
                ks.pend_tainted = True

        evaluated = False
        z: float | None = None
        ratio: float | None = None
        wmean_ms: float | None = None
        warm = ks.fast_count >= config.n_warm

        if ks.pend_n >= config.min_calls:
            evaluated = True
            m = ks.pend_sum / ks.pend_n
            wmean_ms = math.exp(m)
            peak = ks.pend_max_ms if ks.pend_max_ms is not None else wmean_ms
            tainted = ks.pend_tainted
            _reset_pending(ks)
            # Two hot evaluations hours apart are not "sustained".
            if ks.last_eval_ts and (now - ks.last_eval_ts) > config.close_quiet_s:
                ks.hot_streak = 0
            ks.last_eval_ts = now

            sigma = math.sqrt(max(ks.fast_var, config.min_variance))
            if ks.fast_count > 0:
                z = (m - ks.fast_mean) / sigma
            # Absolute excess over the pre-fold baseline (materiality gate).
            spike_material = (
                ks.fast_count > 0 and wmean_ms - math.exp(ks.fast_mean) >= config.min_excess_ms
            )

            # guard 1: Phase I/II separation.
            if ks.fast_count == 0:
                fold_value: float | None = m
            elif tainted or (z is not None and z >= config.spike_z and spike_material):
                fold_value = None
            elif z is not None and z >= config.borderline_z and spike_material:
                fold_value = ks.fast_mean + config.borderline_z * sigma
            else:
                fold_value = m
            if fold_value is not None:
                upd = compute_ewma_update(
                    prev_ewma=ks.fast_mean,
                    prev_variance=ks.fast_var,
                    prev_count=ks.fast_count,
                    value=fold_value,
                    # running-mean warm-up: no single first window dominates
                    alpha=max(config.fast_alpha, 1.0 / (ks.fast_count + 1)),
                    min_variance=config.min_variance,
                )
                ks.fast_mean, ks.fast_var = upd.ewma, upd.variance
                ks.fast_count += 1

            # level: unguarded.
            if ks.level_count == 0:
                ks.level_mean = m
            else:
                la = max(config.level_alpha, 1.0 / (ks.level_count + 1))
                ks.level_mean = la * m + (1 - la) * ks.level_mean
            ks.level_count += 1

            # floor: asymmetric, time-based, frozen upward while saturated.
            # It follows the smoothed *level*, not the raw window mean -- see
            # "floor follows level" in the module docstring.
            lv = ks.level_mean
            if ks.floor is None or ks.level_count <= config.n_warm:
                # Until warm, the floor is the level: an asymmetric filter
                # seeded from one early draw would keep that draw for days.
                ks.floor, ks.floor_ts = lv, now
            else:
                dt = now - (ks.floor_ts or now)
                if lv < ks.floor:
                    a = _half_life_alpha(dt, config.floor_half_life_down_s)
                elif "saturation" in ks.episodes:
                    a = 0.0
                else:
                    a = _half_life_alpha(dt, config.floor_half_life_up_s)
                ks.floor = a * lv + (1 - a) * ks.floor
                ks.floor_ts = now

            ratio = math.exp(ks.level_mean - ks.floor)

            if warm:
                # spike: z >= spike_z sustained spike_sustain evaluations.
                spike_now = z is not None and z >= config.spike_z and spike_material
                ks.hot_streak = ks.hot_streak + 1 if spike_now else 0
                spike_hot = ks.hot_streak >= config.spike_sustain or (
                    spike_now and "spike" in ks.episodes
                )
                step = _episode_step(
                    ks, "spike", hot=spike_hot, magnitude=max(z or 0.0, 0.0), now=now,
                    peak_ms=peak, config=config,
                )
                if step:
                    emit("spike", step[0], step[1], z=z, ratio=ratio, wmean=wmean_ms)

                # saturation, with open/close hysteresis.
                sat_open = "saturation" in ks.episodes
                sat_hot = ratio >= (
                    config.saturation_close_ratio if sat_open else config.saturation_ratio
                ) and math.exp(ks.level_mean) - math.exp(ks.floor) >= config.min_excess_ms
                step = _episode_step(
                    ks, "saturation", hot=sat_hot, magnitude=ratio, now=now,
                    peak_ms=peak, config=config,
                )
                if step:
                    emit("saturation", step[0], step[1], z=z, ratio=ratio, wmean=wmean_ms)

                # regime_shift: a saturation OR a spike observed hot for
                # regime_after_s (hot time, not wall time) is a new normal.
                # The spike path covers steps too small for saturation_ratio
                # but large enough that guard 1 freezes `fast` for good.
                # Stated once; only then are floor and fast re-seeded, and the
                # latency episodes it explains end with it (no separate close:
                # the plateau is never reported as cleared).
                long_eps = [
                    ks.episodes[c] for c in ("saturation", "spike")
                    if c in ks.episodes and ks.episodes[c].hot_s >= config.regime_after_s
                ]
                if long_eps:
                    basis = min(long_eps, key=lambda e: e.opened_ts)
                    emit("regime_shift", "open", basis, z=z, ratio=ratio, wmean=wmean_ms)
                    ks.floor = ks.level_mean
                    ks.floor_ts = now
                    ks.fast_mean = ks.level_mean
                    ks.hot_streak = 0
                    ks.episodes.pop("saturation", None)
                    ks.episodes.pop("spike", None)
        elif ks.pend_windows >= config.max_aggregate_windows:
            # Too sparse to ever reach min_calls in a reasonable span: drop the
            # pool rather than evaluate an under-sampled mean.
            _reset_pending(ks)

        if not evaluated and ks.pend_windows == 0:
            # No latency evidence arriving at all (not merely pooling): open
            # latency episodes age out after close_quiet_s rather than stay
            # open forever on a hop that went quiet -- and silence never adds
            # hot time toward a regime_shift.
            for cond in ("spike", "saturation"):
                if cond in ks.episodes:
                    step = _episode_step(
                        ks, cond, hot=False, magnitude=0.0, now=now, peak_ms=None, config=config
                    )
                    if step:
                        emit(cond, step[0], step[1])

        observations.append(
            KeyObservation(
                service=service,
                instance=instance,
                key=hop,
                excluded=excluded,
                evaluated=evaluated,
                warm=ks.fast_count >= config.n_warm,
                z=z,
                saturation_ratio=ratio,
                window_mean_ms=wmean_ms,
                baseline_ms=_exp(ks.fast_mean) if ks.fast_count else None,
                floor_ms=_exp(ks.floor),
                calls_per_min=calls_per_min,
                calls_per_min_usual=usual,
                success_count=success,
                timeout_count=timeouts,
                open_conditions=tuple(sorted(ks.episodes)),
            )
        )

    events.extend(_evict(state, now, config, exclude))
    return FoldResult(events, observations)


def _reset_pending(ks: KeyState) -> None:
    ks.pend_n = 0
    ks.pend_sum = 0.0
    ks.pend_sumsq = 0.0
    ks.pend_max_ms = None
    ks.pend_windows = 0
    ks.pend_tainted = False


def _evict(
    state: TransportBaselineState,
    now: float,
    config: TransportBaselineConfig,
    exclude: tuple[str, ...],
) -> list[TransportConditionEvent]:
    """Bound state. A hop evicted with open episodes gets a close row for each
    (duration measured to eviction), so no episode silently vanishes."""
    victims = [k for k, ks in state.keys.items() if (now - ks.last_seen_ts) > config.max_idle_s]
    overflow = len(state.keys) - len(victims) - config.max_keys
    if overflow > 0:
        rest = sorted(
            ((k, ks) for k, ks in state.keys.items() if k not in victims),
            key=lambda kv: kv[1].last_seen_ts,
        )
        victims.extend(k for k, _ in rest[:overflow])
    closes: list[TransportConditionEvent] = []
    for k in victims:
        ks = state.keys.pop(k)
        for cond, ep in sorted(ks.episodes.items()):
            closes.append(
                _event(
                    ks, cond, "close", ep, now=now, excluded=is_excluded(ks.hop, exclude),
                    calls_per_min=0.0,
                    usual=ks.calls_ewma if ks.calls_count else None,
                    timeouts=0,
                )
            )
    return closes
