"""self_signal_episode gate: the body's calibrated signals -> metacog episode rows.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md, section D.
Pure lifecycle: ``orion/metacog/self_signal_episode.py``. This module is the
adapter, and imports only ``orion.*`` (the evidence-map producer contract test
loads it by path):

- ``observe_signals``: every internal ``node:substrate.*`` prediction-error node
  -> ``world_first.node_candidate`` -> ``world_first.judge_candidate`` -- the
  attention seam's own eligibility rule, the same calls attention-runtime makes
  (services/orion-attention-runtime/app/worker.py ``_world_first_candidates``).
  Magnitudes come from ``orion.attention.pe_history_cache.NodePeHistoryCache``
  over ``substrate_node_prediction_error_history``, the table both contests
  already read. A read failure makes every signal ABSENT (unknown): it neither
  opens nor closes anything.
- ``SelfSignalEpisodeGate``: folds observations, owns the open budget, the
  close-row retry outbox and (de)serialization for Redis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, Mapping

from orion.attention.world_first import (
    SUBSTRATE_NODE_PREFIX,
    judge_candidate,
    node_candidate,
    source_kind_for_node,
)
from orion.metacog.self_signal_episode import (
    TRIGGER_KIND,
    EpisodeEvent,
    SelfSignalEpisodesStateV1,
    SignalObservation,
    episode_reason,
    episode_upstream,
    mark_open_published,
    mark_silent,
    needs_open_publish,
    observation_from_verdict,
    open_budget_ok,
    record_open,
    state_from_dict,
    state_to_dict,
    step_signal,
)
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.self_signal_episode_gate")

# Close rows whose publish failed, retried on later polls. Bounded: past this,
# the oldest is dropped (logged) rather than growing without limit while the
# bus is down.
MAX_PENDING_CLOSES: int = 50


def observe_signals(
    magnitudes: dict[str, tuple[Any, datetime, float | None]] | None,
    *,
    now: datetime,
    history_error: str | None = None,
    known_signal_ids: Iterable[str] = (),
) -> list[SignalObservation]:
    """One observation per internal node, judged by the attention seam.

    ``known_signal_ids`` (signals with a tracked episode) are always observed,
    so an episode whose node vanished from the history reads as absent and can
    eventually close ``went_silent`` instead of hanging open forever."""
    mags = magnitudes or {}
    ids = {
        nid
        for nid in mags
        if nid.startswith(SUBSTRATE_NODE_PREFIX) and source_kind_for_node(nid) == "internal"
    } | {nid for nid in known_signal_ids if source_kind_for_node(nid) == "internal"}
    out: list[SignalObservation] = []
    for node_id in sorted(ids):
        mag, observed_at, mid = mags.get(node_id, (None, None, None))
        cand = node_candidate(
            node_id=node_id,
            label=f"{node_id.removeprefix(SUBSTRATE_NODE_PREFIX)} prediction error",
            magnitude=mag,
            observed_at=observed_at,
            now=now,
            absent_reason=history_error,
            rank_percentile=mid,
            # The attention contests' default (ATTENTION_EVENT_DECAY_ENABLED
            # ships on): event-written nodes are news for 300 s.
            event_decay=True,
        )
        out.append(observation_from_verdict(judge_candidate(cand), now=now.timestamp()))
    return out


def build_self_signal_episode_trigger(
    event: EpisodeEvent,
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    tends_to: dict[str, Any] | None = None,
) -> MetacogTriggerV1:
    """``tends_to`` (open rows only): the signal's renderable outcome association
    (``self_signal_outcome_rollup.association_payload``), cited by its persisted
    association_id. Absent -> the row is exactly as before."""
    up = episode_upstream(event)
    if tends_to and event.phase == "open":
        up["tends_to"] = dict(tends_to)
    refs = up.get("evidence_refs") or []
    return MetacogTriggerV1(
        trigger_kind=TRIGGER_KIND,
        reason=episode_reason(event),
        zen_state=zen_state,
        pressure=max(0.0, min(1.0, float(pressure))),
        recall_enabled=recall_enabled,
        signal_refs=[event.episode.signal_id, *refs[:1]],
        upstream=up,
    )


@dataclass(frozen=True)
class Outgoing:
    phase: Literal["open", "close"]
    signal_id: str
    episode_id: str
    trigger: MetacogTriggerV1


@dataclass(frozen=True)
class GateConfig:
    poll_sec: float = 30.0
    open_dwell_sec: float = 300.0
    open_window_sec: float = 600.0
    calm_gap_sec: float = 600.0
    silent_close_sec: float = 3600.0
    max_opens_per_hour: int = 6


def config_from_settings(settings: Any) -> GateConfig:
    return GateConfig(
        poll_sec=float(settings.metacog_self_signal_episodes_poll_interval_sec),
        open_dwell_sec=float(settings.metacog_self_signal_episodes_open_dwell_sec),
        open_window_sec=float(settings.metacog_self_signal_episodes_open_window_sec),
        calm_gap_sec=float(settings.metacog_self_signal_episodes_calm_gap_sec),
        silent_close_sec=float(settings.metacog_self_signal_episodes_silent_close_sec),
        max_opens_per_hour=int(settings.metacog_self_signal_episodes_max_opens_per_hour),
    )


class SelfSignalEpisodeGate:
    def __init__(self, config: GateConfig) -> None:
        self.config = config
        self.state = SelfSignalEpisodesStateV1()
        # True until load() restores a persisted state; the first poll after a
        # cold start adopts already-hot signals silently (no boot burst).
        self.cold_start = True
        self._pending_closes: list[Outgoing] = []

    # -- persistence -------------------------------------------------------

    def load(self, raw: str | bytes | None) -> None:
        if raw is None:
            logger.info("self_signal_episode cold_start reason=no_persisted_state")
            return
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            logger.warning("self_signal_episode cold_start reason=malformed_state")
            return
        if not isinstance(data, dict):
            logger.warning("self_signal_episode cold_start reason=malformed_state")
            return
        self.state = state_from_dict(data)
        self.cold_start = False
        logger.info(
            "self_signal_episode resumed episodes=%d open=%s",
            len(self.state.episodes),
            sorted(sid for sid, ep in self.state.episodes.items() if ep.confirmed),
        )

    def dump(self) -> str:
        return json.dumps(state_to_dict(self.state))

    # -- one poll ------------------------------------------------------------

    def step(
        self,
        observations: Iterable[SignalObservation],
        *,
        zen_state: str,
        pressure: float,
        recall_enabled: bool,
        history_ok: bool = True,
        tends_to: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[Outgoing]:
        """Fold one poll. Returns the rows to publish now (retried closes
        first); call ``mark_published`` for each one that really went out.

        ``history_ok`` False (the history read failed: every observation is
        unknown) keeps a cold start cold, so signals already hot when the
        history first becomes readable are still adopted silently."""
        cfg = self.config
        cold = self.cold_start
        episodes = dict(self.state.episodes)
        outgoing: list[Outgoing] = list(self._pending_closes)
        # Opens queued THIS poll count against the budget too (published ones
        # are recorded for real in mark_published).
        budget = self.state
        for obs in observations:
            before = episodes.get(obs.signal_id)
            ep, event = step_signal(
                before,
                obs,
                poll_sec=cfg.poll_sec,
                open_dwell_sec=cfg.open_dwell_sec,
                open_window_sec=cfg.open_window_sec,
                calm_gap_sec=cfg.calm_gap_sec,
                silent_close_sec=cfg.silent_close_sec,
                cold_start=cold,
            )
            if event is not None and event.phase == "close":
                if event.episode.open_published:
                    out = Outgoing(
                        "close",
                        obs.signal_id,
                        event.episode.episode_id,
                        build_self_signal_episode_trigger(
                            event, zen_state=zen_state, pressure=pressure, recall_enabled=recall_enabled
                        ),
                    )
                    outgoing.append(out)
                    self._pending_closes.append(out)
                else:
                    logger.info(
                        "self_signal_episode_close_unpublished episode_id=%s reason=%s "
                        "(open row never reached the bus; no close row)",
                        event.episode.episode_id,
                        event.close_reason,
                    )
                self._log_event(event)
            elif ep is not None and needs_open_publish(ep):
                if not open_budget_ok(budget, obs.at, max_opens_per_hour=cfg.max_opens_per_hour):
                    ep = mark_silent(ep, "over_open_budget")
                    logger.warning(
                        "self_signal_episode_suppressed budget=%d/h episode_id=%s",
                        cfg.max_opens_per_hour,
                        ep.episode_id,
                    )
                else:
                    budget = record_open(budget, obs.at)
                    open_event = event if event is not None else EpisodeEvent("open", ep, obs)
                    outgoing.append(
                        Outgoing(
                            "open",
                            obs.signal_id,
                            ep.episode_id,
                            build_self_signal_episode_trigger(
                                open_event,
                                zen_state=zen_state,
                                pressure=pressure,
                                recall_enabled=recall_enabled,
                                tends_to=(tends_to or {}).get(obs.signal_id),
                            ),
                        )
                    )
                    if event is not None:
                        self._log_event(event)
            if ep is None:
                episodes.pop(obs.signal_id, None)
            else:
                episodes[obs.signal_id] = ep
        self.state = SelfSignalEpisodesStateV1(episodes=episodes, recent_opens=self.state.recent_opens)
        if history_ok:
            self.cold_start = False
        if len(self._pending_closes) > MAX_PENDING_CLOSES:
            dropped = self._pending_closes[: len(self._pending_closes) - MAX_PENDING_CLOSES]
            self._pending_closes = self._pending_closes[-MAX_PENDING_CLOSES:]
            logger.warning("self_signal_episode_close_outbox_full dropped=%d", len(dropped))
        return outgoing

    def mark_published(self, out: Outgoing, *, now: float) -> None:
        if out.phase == "close":
            self._pending_closes = [p for p in self._pending_closes if p is not out]
            return
        ep = self.state.episodes.get(out.signal_id)
        if ep is None or ep.episode_id != out.episode_id:
            return
        episodes = dict(self.state.episodes)
        episodes[out.signal_id] = mark_open_published(ep)
        self.state = record_open(
            SelfSignalEpisodesStateV1(episodes=episodes, recent_opens=self.state.recent_opens), now
        )

    @staticmethod
    def _log_event(event: EpisodeEvent) -> None:
        ep = event.episode
        logger.info(
            "self_signal_episode_event phase=%s signal=%s episode_id=%s open_band=%s "
            "peak_band=%s peak_raw=%s hot_polls=%d close_reason=%s duration_sec=%s",
            event.phase,
            ep.signal_id,
            ep.episode_id,
            ep.open_band,
            ep.peak_band,
            ep.peak_value,
            ep.hot_polls,
            event.close_reason,
            event.duration_sec,
        )
