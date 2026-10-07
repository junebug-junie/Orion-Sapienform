"""EWMA-derived, floor-bounded thresholds for the transport lattice lane.

Plain version: the static numbers in
``config/substrate-lattice/transport_lattice_policy.v1.yaml`` say "0.25 is
worth watching". This module also learns what is NORMAL for a channel and lets
it trigger earlier than the static number when the normal is low. It can never
trigger later: the static value is a hard ceiling on every effective
threshold, so a channel that has been hot for days (whose learned baseline has
drifted up to meet it) still trips at the static number.

Two clocks per channel, both via ``orion.bus.ewma.compute_ewma_update``
(half-life alpha, ``queue_contention`` style):

* fast (default 30 min half-life): ``z_fast`` = "just changed". Reported only.
* slow (default 2 day half-life): the chronic level and its long-run spread.
  ``derived = slow_mean + k * slow_sd``. A baseline that adapts absorbs a
  chronically hot channel (z -> 0), so z alone is never used to quiet anything.

Cold start: fewer than ``min_samples`` slow samples => no derived threshold,
``z_fast`` is ``None`` (unknown, not 0.0) until ``MIN_FAST_SAMPLES``.

Staleness: state whose ``last_ts`` is older than ``max_state_age_sec`` is
reported ``stale`` and the static thresholds apply. A dead producer must not
look like a calm channel or freeze a learned threshold in place.

Only channels in ``DERIVED_CHANNELS`` are wired. ``contract_pressure`` (5
distinct values in 3.4 days, flat; its lattice row was deleted 2026-10-07)
and ``observer_failure_pressure`` (96% exact zeros, event-only; retired end to
end 2026-10-07, its lattice row replaced by the static
``transport_reliability_pressure`` row) failed the metric gate; see
``docs/superpowers/specs/2026-09-30-ewma-transport-thresholds-gate.md``.

One producer (``orion-substrate-runtime``'s bus-synaptic tick) writes the
state to Redis; the hub lattice routes and the mind recall resolver both call
``fetch_effective_thresholds`` so they see the same numbers.
"""
from __future__ import annotations

import json
import logging
import math
import os
import functools
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from orion.bus.ewma import compute_ewma_update

logger = logging.getLogger("orion.field.transport_thresholds")

RUNGS = ("watch_at", "summarize_at", "propose_at")

# Channel -> how the producer's raw value maps onto the channel's own scale.
# bus_synaptic_pressure is the transport lattice policy's row id (and this
# module's Redis state key); its policy `source:` is capability:transport.pressure,
# which is exactly
# 0.85 x node:substrate.bus_synaptic prediction_error (topology edge weight,
# verified live: ratio 0.85 over 124k rows). Pinned to the topology yaml by
# test_transport_thresholds.py.
DERIVED_CHANNELS: dict[str, float] = {"bus_synaptic_pressure": 0.85}

MIN_FAST_SAMPLES = 300  # ~3 fast half-lives at the 30s tick; below this the fast variance is under-warmed
# Cold start also needs real elapsed time, not just a sample count (a fast or
# bursty producer must not warm the baseline early). 10 s/sample is a third of
# the 30 s tick.
MIN_SECONDS_PER_SAMPLE = 10.0
_STATE_CACHE_TTL_SEC = 15.0  # producer refreshes every 30 s; readers may be hit per request
# Derived thresholds may tighten a static rung by at most 2x. Keeps a
# near-zero-variance stretch from producing a hair-trigger.
MAX_TIGHTEN_RATIO = 0.5
_MIN_VARIANCE = 25e-6  # sd floor 0.005 on a 0-1 pressure scale
_MAX_DT_SEC = 300.0  # a producer outage must not be absorbed as one big step

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ThresholdConfig:
    enabled: bool = True
    fast_half_life_sec: float = 1800.0
    slow_half_life_sec: float = 172800.0
    min_samples: int = 2880  # 24h at the 30s producer tick
    k_watch: float = 5.0
    k_step: float = 2.0  # summarize = k_watch + step, propose = k_watch + 2*step
    max_state_age_sec: float = 600.0
    state_key: str = "orion:lattice:transport_thresholds:v1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ThresholdConfig":
        e = os.environ if env is None else env
        d = cls()

        def _f(key: str, default: float) -> float:
            raw = str(e.get(key, "")).strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        enabled_raw = str(e.get("TRANSPORT_THRESHOLDS_DERIVED_ENABLED", "")).strip().lower()
        # Sane-range clamps: a garbage value must not disable cold start or
        # produce a hair-trigger (the static floor bounds the rest).
        return cls(
            enabled=(enabled_raw in _TRUE) if enabled_raw else d.enabled,
            fast_half_life_sec=max(_f("TRANSPORT_THRESHOLDS_FAST_HALF_LIFE_SEC", d.fast_half_life_sec), 60.0),
            slow_half_life_sec=max(_f("TRANSPORT_THRESHOLDS_SLOW_HALF_LIFE_SEC", d.slow_half_life_sec), 3600.0),
            min_samples=max(int(_f("TRANSPORT_THRESHOLDS_MIN_SAMPLES", d.min_samples)), 100),
            k_watch=max(_f("TRANSPORT_THRESHOLDS_K_WATCH", d.k_watch), 2.0),
            k_step=max(_f("TRANSPORT_THRESHOLDS_K_STEP", d.k_step), 0.0),
            max_state_age_sec=max(_f("TRANSPORT_THRESHOLDS_MAX_STATE_AGE_SEC", d.max_state_age_sec), 60.0),
            state_key=str(e.get("TRANSPORT_THRESHOLDS_STATE_KEY", "")).strip() or d.state_key,
        )


@dataclass(frozen=True)
class ChannelState:
    n: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    fast_ewma: float = 0.0
    fast_var: float = 0.0
    slow_ewma: float = 0.0
    slow_var: float = 0.0
    last_z_fast: float | None = None  # z of the last sample vs the prior fast baseline

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes | None) -> "ChannelState | None":
        if not raw:
            return None
        try:
            d = json.loads(raw)
            z = d.get("last_z_fast")
            return cls(
                n=int(d["n"]), first_ts=float(d["first_ts"]), last_ts=float(d["last_ts"]),
                fast_ewma=float(d["fast_ewma"]), fast_var=float(d["fast_var"]),
                slow_ewma=float(d["slow_ewma"]), slow_var=float(d["slow_var"]),
                last_z_fast=None if z is None else float(z),
            )
        except (ValueError, KeyError, TypeError):
            return None


def _alpha(dt: float, half_life: float) -> float:
    return 1.0 - 0.5 ** (max(dt, 0.0) / max(half_life, 1e-9))


def update_state(
    prev: ChannelState | None, value: float, now_ts: float, cfg: ThresholdConfig
) -> ChannelState:
    """Absorb one sample. Non-finite values are skipped (state unchanged)."""
    if not math.isfinite(value):
        return prev or ChannelState()
    if prev is None or prev.n == 0:
        return ChannelState(
            n=1, first_ts=now_ts, last_ts=now_ts, fast_ewma=value, slow_ewma=value,
            last_z_fast=None,
        )
    dt = min(max(now_ts - prev.last_ts, 0.0), _MAX_DT_SEC)
    # Warm-up: alpha is floored at 1/n so the first samples form a plain running
    # mean/variance. Without this the seed value keeps ~71% of the weight after
    # 24 h at a 2-day half-life and the "derived" threshold would depend on
    # whichever reading happened to arrive first (review finding).
    warm = 1.0 / (prev.n + 1)
    fast = compute_ewma_update(
        prev_ewma=prev.fast_ewma, prev_variance=prev.fast_var, prev_count=prev.n,
        value=value, alpha=max(_alpha(dt, cfg.fast_half_life_sec), warm), min_variance=_MIN_VARIANCE,
    )
    slow = compute_ewma_update(
        prev_ewma=prev.slow_ewma, prev_variance=prev.slow_var, prev_count=prev.n,
        value=value, alpha=max(_alpha(dt, cfg.slow_half_life_sec), warm), min_variance=_MIN_VARIANCE,
    )
    return ChannelState(
        n=prev.n + 1, first_ts=prev.first_ts, last_ts=max(now_ts, prev.last_ts),
        fast_ewma=fast.ewma, fast_var=fast.variance,
        slow_ewma=slow.ewma, slow_var=slow.variance,
        # z is only meaningful once the fast baseline has enough history.
        last_z_fast=fast.zscore if prev.n >= MIN_FAST_SAMPLES else None,
    )


def effective_thresholds(
    static_def: Mapping[str, Any],
    state: ChannelState | None,
    cfg: ThresholdConfig,
    *,
    now_ts: float | None = None,
    channel_id: str | None = None,
) -> dict[str, Any]:
    """Effective ``watch_at``/``summarize_at``/``propose_at`` plus provenance.

    Invariant (the hard floor): for every rung, ``effective <= static`` when the
    static rung is set. A ``null`` static rung stays null (derived values never
    invent a rung the policy did not define).
    """
    now_ts = time.time() if now_ts is None else now_ts
    reason = "derived"
    if not cfg.enabled:
        reason = "disabled"
    elif channel_id is not None and channel_id not in DERIVED_CHANNELS:
        reason = "channel_not_wired"
    elif state is None or state.n == 0:
        reason = "no_state"
    elif abs(now_ts - state.last_ts) > cfg.max_state_age_sec:  # abs: host clock skew reads stale too
        reason = "stale_state"
    elif (
        state.n < cfg.min_samples
        or (state.last_ts - state.first_ts) < cfg.min_samples * MIN_SECONDS_PER_SAMPLE
    ):
        reason = "cold_start"

    usable = reason == "derived"
    sd = math.sqrt(max(state.slow_var, _MIN_VARIANCE)) if state else 0.0
    window = cfg.slow_half_life_sec
    out: dict[str, Any] = {}
    prev_eff: float | None = None
    for i, rung in enumerate(RUNGS):
        static_v = static_def.get(rung)
        static_f = None if static_v is None else float(static_v)
        derived: float | None = None
        eff = static_f
        source = "static"
        if usable and static_f is not None:
            derived = state.slow_ewma + (cfg.k_watch + i * cfg.k_step) * sd
            candidate = min(static_f, max(MAX_TIGHTEN_RATIO * static_f, derived))
            if prev_eff is not None:
                candidate = max(candidate, prev_eff)  # keep the ladder monotone
                candidate = min(candidate, static_f)
            if candidate < static_f:
                eff, source = candidate, "derived"
        if eff is not None:
            prev_eff = eff
        out[rung] = {
            "value": eff,
            "source": source,
            "static": static_f,
            "derived": derived,
            "window_sec": window if usable else None,
            "n_samples": state.n if state else 0,
        }
    z = state.last_z_fast if (state and usable) else None
    static_watch = static_def.get("watch_at")
    chronic_hot = bool(
        usable and static_watch is not None and state.slow_ewma >= float(static_watch)
    )
    out["_meta"] = {
        "reason": reason,
        "z_fast": z,  # None = unknown, never 0.0 for "no data"
        "chronic_level": state.slow_ewma if state else None,
        "chronic_hot": chronic_hot,
        "min_samples": cfg.min_samples,
        "n_samples": state.n if state else 0,
        "state_age_sec": (now_ts - state.last_ts) if state and state.n else None,
    }
    return out


def flat_rungs(eff: Mapping[str, Any]) -> dict[str, float | None]:
    return {r: eff[r]["value"] for r in RUNGS}


@functools.lru_cache(maxsize=4)
def _client(redis_url: str):
    """One pooled client per URL (readers are hit per request)."""
    import redis  # lazy: pure functions above stay importable without it

    return redis.Redis.from_url(
        redis_url, socket_connect_timeout=0.5, socket_timeout=0.5, decode_responses=True
    )


_STATE_CACHE: dict[tuple[str, str, str], tuple[float, "ChannelState | None"]] = {}
_STATE_CACHE_LOCK = threading.Lock()


def clear_state_cache() -> None:
    with _STATE_CACHE_LOCK:
        _STATE_CACHE.clear()


def load_state(channel_id: str, redis_url: str, cfg: ThresholdConfig) -> ChannelState | None:
    """Reader-side load with a short TTL cache; failures are cached too so a
    down Redis costs one timeout per TTL, not one per request."""
    if not redis_url:
        return None
    key = (redis_url, cfg.state_key, channel_id)
    mono = time.monotonic()
    with _STATE_CACHE_LOCK:
        hit = _STATE_CACHE.get(key)
        if hit and mono - hit[0] < _STATE_CACHE_TTL_SEC:
            return hit[1]
    try:
        state = ChannelState.from_json(_client(redis_url).hget(cfg.state_key, channel_id))
    except Exception:
        logger.debug("transport_thresholds_state_load_failed channel=%s", channel_id, exc_info=True)
        state = None
    with _STATE_CACHE_LOCK:
        _STATE_CACHE[key] = (mono, state)
    return state


def record_sample(
    channel_id: str, value: float, redis_url: str, cfg: ThresholdConfig | None = None,
    *, now_ts: float | None = None,
) -> ChannelState | None:
    """Producer side: fold one reading into the stored state. Fail-open."""
    cfg = cfg or ThresholdConfig.from_env()
    if not cfg.enabled or not redis_url or channel_id not in DERIVED_CHANNELS:
        return None
    try:
        client = _client(redis_url)
        prev = ChannelState.from_json(client.hget(cfg.state_key, channel_id))
        new = update_state(prev, value, time.time() if now_ts is None else now_ts, cfg)
        client.hset(cfg.state_key, channel_id, new.to_json())
        return new
    except Exception:
        logger.warning("transport_thresholds_record_failed channel=%s", channel_id, exc_info=True)
        return None


def fetch_effective_thresholds(
    channel_id: str, static_def: Mapping[str, Any], redis_url: str,
    cfg: ThresholdConfig | None = None, *, now_ts: float | None = None,
) -> dict[str, Any]:
    """Reader side (hub + mind). Any failure degrades to the static thresholds."""
    cfg = cfg or ThresholdConfig.from_env()
    state = None
    if cfg.enabled and channel_id in DERIVED_CHANNELS:
        state = load_state(channel_id, redis_url, cfg)
    return effective_thresholds(static_def, state, cfg, now_ts=now_ts, channel_id=channel_id)
