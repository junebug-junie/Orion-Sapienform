"""Episode semantics for the telemetry_anomaly metacog trigger. Pure, no I/O.

Before this module, orion-equilibrium-service fired one ``telemetry_anomaly``
metacog trigger per flagged 60s scoring window: the shared 30s cooldown is
shorter than orion-field-digester's 60s scoring interval, so it never held a
sustained anomaly to one fire (10-08: 396 fires in 35 episodes). Each fire
costs one LLM metacog entry.

This module turns the per-window stream into episodes:

* An **episode** is a run of flagged windows. It ends once no flagged window
  has arrived for ``calm_gap_sec`` (default 150s, i.e. at least two calm 60s
  windows -- a single calm window between two flagged ones is flicker around
  the threshold, not a new event; see the PR report for the live gap
  histogram this was read from).
* An episode **fires once**, on its first window that is not explained by a
  chat turn, and may **re-fire** only if its severity climbs into a higher
  ``orion.metacog.evidence_map`` band (nominal -> degraded -> critical) than
  the band it already fired at.
* A window is **turn-explained** when its lead attributed channel belongs to a
  glossary category produced by turn execution (``task_execution`` in
  config/field/field_channel_glossary.v1.yaml) AND a harness turn was active
  between ``window_start - turn_carry_sec`` and ``window_end``. Those windows
  are already covered by the chat_turn trigger and execution prediction
  error; firing on them double-counts the turn. A turn-explained window does
  not consume the episode's single fire, so an anomaly that outlasts the turn
  still fires on its first window no turn explains.
* **Nominal open waits one window** (2026-10-20). An episode whose first
  flagged window is only in the ``nominal`` band does not fire on it; it fires
  on its second flagged window (if any) with that window's band, which is the
  max band of the first two windows (nominal is the lowest flagged band). So
  the opening fire carries the band the onset reached, and no
  ``severity_escalation`` can follow one minute later. Live 10-20: 4 of 29
  fires in 6h were nominal ``episode_open`` + ``severity_escalation`` on the
  next window -- one onset, two LLM metacog entries. Chosen over "suppress
  escalation for 2 windows": suppressing/absorbing the escalation leaves the
  band the onset really reached unreported, and orion-dream's novelty weights
  only degraded/critical keys (15-day replay: novelty 10.0 -> 7.0 when
  absorbing, 10.0 -> 10.0 when deferring). Degraded/critical opens still fire
  at once (no added latency on real-band onsets). Cost: a lone single-window
  nominal blip (threshold flicker, zero dream weight) no longer fires.
  15-day replay: 140 -> 112 fires.
* **Sustained override.** Once ``sustained_windows`` (default 15) flagged
  windows have landed within the trailing ``sustained_horizon_sec`` (default
  1800s) -- counted across episode boundaries, so an outage that flickers
  below threshold every few minutes still accumulates -- turn coupling no
  longer suppresses it. Fifteen minutes of flagged scoring in half an hour
  is well past the live p75 turn length (~570s), and a hung turn is itself
  the symptom of a real outage: on 10-08 the GPU outage made turns run 20-70
  minutes, so turn brackets covered every flagged window and pure
  suppression never fired it at all. With the override it fires at 14:54Z
  (onset ~14:40Z). Replay cost over 15 days: 121 -> 140 fires.

``turn_carry_sec`` (default 150s) is how long a turn-derived channel can keep
a window elevated after the turn's last delta: orion-field-digester holds a
node channel flat for FIELD_DECAY_STALENESS_THRESHOLD_SEC (90s) and then
decays it by 0.92 per 2s tick (~55s to fall to 10%).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from orion.metacog.evidence_map import (
    TELEMETRY_RATIO_CRITICAL,
    TELEMETRY_RATIO_DEGRADED,
    TELEMETRY_RATIO_SATURATE,
    banded,
    severity_rank,
)

DEFAULT_CALM_GAP_SEC: float = 150.0
DEFAULT_TURN_CARRY_SEC: float = 150.0
DEFAULT_SUSTAINED_WINDOWS: int = 15
DEFAULT_SUSTAINED_HORIZON_SEC: float = 1800.0
# A turn whose bracket never closes (lost HarnessRunV1 and lost timeout) stops
# counting as active after this long with NO harness grammar event. Measured
# live (3 days, 182 turns): the longest silence between two harness-governor
# grammar events inside one turn is p50 59s, p99 840s, max 855s.
DEFAULT_TURN_OPEN_MAX_SEC: float = 900.0
# Glossary categories whose channels are produced by turn execution
# (services/orion-field-digester/app/ingest/state_deltas.py execution_run
# deltas). Read from the glossary's own `category`, not a channel-name list.
TURN_COUPLED_CATEGORIES: frozenset[str] = frozenset({"task_execution"})


def ratio_severity(ratio: float) -> str:
    """Same banding evidence_map.map_telemetry_anomaly applies to recon_loss/threshold."""
    severity, _ = banded(
        ratio, TELEMETRY_RATIO_DEGRADED, TELEMETRY_RATIO_CRITICAL, TELEMETRY_RATIO_SATURATE
    )
    return severity


@dataclass(frozen=True)
class AnomalyEpisodeStateV1:
    """Entire carried-forward state; small enough to persist every window."""

    episode_id: str | None = None  # window_end (ISO) of the episode's first flagged window
    started_at: float | None = None  # epoch seconds of that window_end
    last_flagged_at: float | None = None
    flagged_windows: int = 0
    turn_explained_windows: int = 0
    fired_rank: int = -1  # -1: nothing published for this episode yet
    fires: int = 0
    peak_ratio: float = 0.0
    # window_end of every flagged window in the trailing sustained horizon.
    # Survives episode closes (the sustained override must see flicker).
    recent_flagged: tuple[float, ...] = ()

    @property
    def active(self) -> bool:
        return self.episode_id is not None


@dataclass(frozen=True)
class AnomalyEpisodeDecision:
    state: AnomalyEpisodeStateV1
    fire: bool
    # episode_open | severity_escalation | sustained_past_turn when fire; otherwise one of
    # calm | turn_explained | within_episode | open_deferred | stale_window
    reason: str
    severity: str
    turn_explained: bool
    new_episode: bool


def _closed_if_calm(state: AnomalyEpisodeStateV1, at: float, calm_gap_sec: float) -> AnomalyEpisodeStateV1:
    """Close on a calm gap, or on a clock step backwards larger than the calm
    gap (a producer clock correction must not mute anomalies until the clock
    catches up with a stored last_flagged_at)."""
    if not state.active or state.last_flagged_at is None:
        return state
    gap = at - state.last_flagged_at
    if gap > calm_gap_sec:
        return AnomalyEpisodeStateV1(recent_flagged=state.recent_flagged)
    if gap < -calm_gap_sec:
        return AnomalyEpisodeStateV1()  # history is on the old clock; drop it too
    return state


def step(
    state: AnomalyEpisodeStateV1,
    *,
    window_end: float,
    window_end_iso: str,
    flagged: bool,
    ratio: float,
    lead_category: str | None,
    turn_active: bool,
    calm_gap_sec: float = DEFAULT_CALM_GAP_SEC,
    turn_suppress: bool = True,
    sustained_windows: int = DEFAULT_SUSTAINED_WINDOWS,
    sustained_horizon_sec: float = DEFAULT_SUSTAINED_HORIZON_SEC,
    defer_nominal_open: bool = True,
) -> AnomalyEpisodeDecision:
    """Fold one scored window into the episode state.

    ``ratio`` is recon_loss / trigger threshold (>1 when flagged).
    ``turn_active`` is the caller's answer to "was a harness turn active in
    [window_start - turn_carry_sec, window_end]" (TurnActivityTracker).
    """
    severity = ratio_severity(ratio) if flagged else "nominal"
    state = _closed_if_calm(state, window_end, calm_gap_sec)

    if not flagged:
        return AnomalyEpisodeDecision(state, False, "calm", severity, False, False)

    # A window at or slightly before the last flagged one is a redelivery or
    # jitter: never let it open or fire anything. (A big step back already
    # closed the episode above.)
    if state.last_flagged_at is not None and window_end <= state.last_flagged_at:
        return AnomalyEpisodeDecision(state, False, "stale_window", severity, False, False)

    new_episode = not state.active
    if new_episode:
        state = AnomalyEpisodeStateV1(
            episode_id=window_end_iso, started_at=window_end, recent_flagged=state.recent_flagged
        )
    turn_explained = bool(
        turn_suppress and turn_active and lead_category in TURN_COUPLED_CATEGORIES
    )
    state = replace(
        state,
        last_flagged_at=window_end,
        flagged_windows=state.flagged_windows + 1,
        turn_explained_windows=state.turn_explained_windows + (1 if turn_explained else 0),
        peak_ratio=max(state.peak_ratio, float(ratio)),
        recent_flagged=tuple(
            t for t in state.recent_flagged if t > window_end - sustained_horizon_sec
        ) + (window_end,),
    )
    sustained = sustained_windows > 0 and len(state.recent_flagged) >= sustained_windows
    if turn_explained and not sustained:
        return AnomalyEpisodeDecision(state, False, "turn_explained", severity, True, new_episode)

    rank = severity_rank(severity)
    if state.fired_rank < 0:
        reason = "sustained_past_turn" if turn_explained else "episode_open"
        if (
            defer_nominal_open
            and reason == "episode_open"
            and severity == "nominal"
            and state.flagged_windows == 1
        ):
            # Wait one window so the open carries the band the onset reaches (module doc).
            return AnomalyEpisodeDecision(state, False, "open_deferred", severity, turn_explained, new_episode)
        return AnomalyEpisodeDecision(state, True, reason, severity, turn_explained, new_episode)
    if rank > state.fired_rank:
        return AnomalyEpisodeDecision(state, True, "severity_escalation", severity, turn_explained, new_episode)
    return AnomalyEpisodeDecision(state, False, "within_episode", severity, turn_explained, new_episode)


def mark_published(state: AnomalyEpisodeStateV1, severity: str) -> AnomalyEpisodeStateV1:
    """Record a fire only after it really reached the bus, so a failed publish
    is retried on the next flagged window instead of silently consuming the
    episode's one fire."""
    return replace(
        state,
        fired_rank=max(state.fired_rank, severity_rank(severity)),
        fires=state.fires + 1,
    )


def state_to_dict(state: AnomalyEpisodeStateV1) -> dict[str, Any]:
    return {
        "episode_id": state.episode_id,
        "started_at": state.started_at,
        "last_flagged_at": state.last_flagged_at,
        "flagged_windows": state.flagged_windows,
        "turn_explained_windows": state.turn_explained_windows,
        "fired_rank": state.fired_rank,
        "fires": state.fires,
        "peak_ratio": state.peak_ratio,
        "recent_flagged": list(state.recent_flagged),
    }


def state_from_dict(data: Any) -> AnomalyEpisodeStateV1:
    """Never raises; anything malformed restores as no open episode.

    Restoring a real open episode is what keeps a restart from re-firing an
    anomaly that is still running (the episode closes on its own once
    calm_gap_sec passes without a flagged window)."""
    if not isinstance(data, dict):
        return AnomalyEpisodeStateV1()
    try:
        recent = tuple(float(t) for t in data.get("recent_flagged") or ())
    except (TypeError, ValueError):
        recent = ()
    if not data.get("episode_id"):
        return AnomalyEpisodeStateV1(recent_flagged=recent)
    try:
        return AnomalyEpisodeStateV1(
            recent_flagged=recent,
            episode_id=str(data["episode_id"]),
            started_at=float(data["started_at"]),
            last_flagged_at=float(data["last_flagged_at"]),
            flagged_windows=int(data.get("flagged_windows", 0)),
            turn_explained_windows=int(data.get("turn_explained_windows", 0)),
            fired_rank=int(data.get("fired_rank", -1)),
            fires=int(data.get("fires", 0)),
            peak_ratio=float(data.get("peak_ratio", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return AnomalyEpisodeStateV1()


def episode_upstream(decision: AnomalyEpisodeDecision) -> dict[str, Any]:
    """Additive keys for MetacogTriggerV1.upstream (a free-form dict)."""
    s = decision.state
    return {
        "episode_id": s.episode_id,
        "episode_fire_reason": decision.reason,
        "episode_severity": decision.severity,
        "episode_windows": s.flagged_windows,
        "episode_turn_explained_windows": s.turn_explained_windows,
        "episode_duration_sec": (
            round(s.last_flagged_at - s.started_at, 3)
            if s.last_flagged_at is not None and s.started_at is not None
            else 0.0
        ),
        "episode_peak_ratio": round(s.peak_ratio, 4),
    }


@dataclass
class _TurnBracket:
    first: float
    last: float
    closed_at: float | None = None


class TurnActivityTracker:
    """Which harness turns were active when, keyed by correlation_id.

    Fed with activity timestamps (harness-governor grammar events stream
    throughout a turn; HarnessRunV1 closes it and carries fcc_elapsed_sec, so a
    turn already in flight at restart still gets its real start). Bounded:
    pruned on every observe.
    """

    def __init__(
        self,
        *,
        turn_carry_sec: float = DEFAULT_TURN_CARRY_SEC,
        open_max_sec: float = DEFAULT_TURN_OPEN_MAX_SEC,
    ) -> None:
        self.turn_carry_sec = float(turn_carry_sec)
        self.open_max_sec = float(open_max_sec)
        self._turns: dict[str, _TurnBracket] = {}

    def __len__(self) -> int:
        return len(self._turns)

    def observe(
        self,
        correlation_id: str,
        at: float,
        *,
        closed: bool = False,
        elapsed_sec: float | None = None,
    ) -> None:
        if not correlation_id:
            return
        b = self._turns.get(correlation_id)
        if b is None:
            b = self._turns[correlation_id] = _TurnBracket(first=at, last=at)
        b.first = min(b.first, at)
        b.last = max(b.last, at)
        if elapsed_sec is not None and elapsed_sec > 0:
            b.first = min(b.first, at - float(elapsed_sec))
        if closed:
            b.closed_at = max(b.closed_at or at, at)
        self._prune(at)

    def _end(self, b: _TurnBracket, now: float) -> float:
        if b.closed_at is not None:
            return b.closed_at
        # Open: active up to now unless it went silent past open_max_sec.
        return now if now - b.last <= self.open_max_sec else b.last

    def _prune(self, now: float) -> None:
        horizon = now - (self.turn_carry_sec + self.open_max_sec + 600.0)
        stale = [cid for cid, b in self._turns.items() if self._end(b, now) < horizon]
        for cid in stale:
            del self._turns[cid]

    def active_between(self, lo: float, hi: float) -> bool:
        """Any turn overlapping [lo - turn_carry_sec, hi]."""
        lo = lo - self.turn_carry_sec
        return any(b.first <= hi and self._end(b, hi) >= lo for b in self._turns.values())

    def turns(self) -> Iterable[tuple[str, float, float | None]]:
        return ((cid, b.first, b.closed_at) for cid, b in self._turns.items())
