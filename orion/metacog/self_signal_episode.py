"""Self-signal episodes: metacog as a reader of the body's calibrated signals. Pure, no I/O.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md, section
"D. Metacog becomes a reader" (PR #2528, approved 2026-10-10, "hit it" 2026-10-14):
``self_signal_episode`` rows are written when a calibrated INTERNAL signal enters
high/unusual in its bad direction, and when it returns. Prose once per episode.

There is ONE definition of "high/unusual in its bad direction": the attention
seam's ``orion.attention.world_first.judge_candidate`` over a node candidate
built by ``node_candidate`` (PR #2583). This module never re-derives a band,
percentile, polarity, staleness or event window -- it only folds that verdict
into an episode over time:

- **hot**     the verdict is eligible (internal, band high/unusual, bad direction),
              or high for itself but not yet sustained (``verdict.sustained`` False:
              a continuous body source, orion/attention/sustained.py).
- **calm**    a real current reading that is not eligible (band quiet/usual).
- **faded**   an event-written signal whose last event is past its orienting
              window (the value is carried forward, but it is no longer news).
              Treated as calm for closing; the close row says ``faded``.
- **unknown** absent, stale, insufficient history, a placeholder, or no
              declared polarity. Silence is not calm: unknown never opens an
              episode, never advances the calm clock, and resets it. An episode
              that stays unknown for ``silent_close_sec`` closes as
              ``went_silent`` -- explicitly NOT a recovery.

Hysteresis (both directions, read from 7 days of live history -- PR report):

- **Enter (dwell).** A hot poll starts a *pending* episode. It is confirmed --
  and the open row written -- once the signal has been hot for MORE than
  ``open_dwell_sec`` within the trailing ``open_window_sec``. The anchor is the
  seam's own ``EVENT_ORIENTING_WINDOW_SEC`` (300 s, "one event's consequence is
  gone by 300 s", world_first.py): an episode is the body staying unusual for
  longer than one event can explain. Without it, live replay opened ~110
  episodes/day, because "high" is the top decile of each signal's OWN week and
  so is hot ~10% of the time by construction; with it, ~9.5/day.
  A continuous body source (biometrics, bus_synaptic) carries its dwell over
  its own readings (``SignalObservation.hot_dwell_sec``), the same number the
  attention contest judges, so its episode confirms on the poll it first
  competes for attention (spec 2026-10-16-attention-continuous-signal-quota).
- **Exit (calm gap).** Close once observations have been calm/faded
  continuously for ``calm_gap_sec`` (600 s = twice the orienting window; on
  live data 47-64% of the calm runs between two hot polls of the per-tick and
  execution signals are shorter than this, i.e. flicker inside one episode).
  A pending episode that calms down the same way is dropped silently: it
  never wrote a row.

Restarts: the caller persists the state. A restart WITH persisted state resumes
open episodes, so it never re-opens one (no burst). A cold start with NO
persisted state marks any signal already hot on the first poll as ``silent``
(``adopted_at_cold_start``): tracked so it cannot re-open, but no open and no
close row -- we did not see it start.

The caller publishes; ``mark_open_published`` is recorded only after the open
trigger really reached the bus, so a failed publish is retried on the next hot
poll instead of silently consuming the episode's one prose. A close row is
published only for an episode whose open row was published.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from orion.attention.sustained import SUSTAIN_DWELL_SEC, SUSTAIN_WINDOW_SEC, is_sustained

ObservationKind = Literal["hot", "calm", "faded", "unknown"]

TRIGGER_KIND = "self_signal_episode"
EVIDENCE_SOURCE = "self_signal_episode"
# Rows the evidence refs resolve to: (node_id, observed_at) is that table's
# primary key (manual_migration_node_prediction_error_history_v1.sql).
SOURCE_TABLE = "substrate_node_prediction_error_history"

DEFAULT_POLL_SEC: float = 30.0
# The attention seam's sustained rule (orion/attention/sustained.py, stdlib
# only): open dwell = world_first.EVENT_ORIENTING_WINDOW_SEC, open window =
# twice it. Imported, so the episode gate and the attention contest cannot
# drift apart.
DEFAULT_OPEN_DWELL_SEC: float = SUSTAIN_DWELL_SEC
DEFAULT_OPEN_WINDOW_SEC: float = SUSTAIN_WINDOW_SEC
DEFAULT_CALM_GAP_SEC: float = 600.0
DEFAULT_SILENT_CLOSE_SEC: float = 3600.0
# Hard cap on evidence refs carried by one episode (Juniper's standing rule).
MAX_EVIDENCE_REFS: int = 200
# Hard cap on remembered hot-poll times per signal (trailing window only).
_MAX_RECENT_HOT: int = 400


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


def evidence_ref(signal_id: str, observed_at: float | None) -> str | None:
    """``<table>:<node_id>@<observed_at ISO>``: resolves to exactly one
    persisted history row via its primary key."""
    if observed_at is None:
        return None
    return f"{SOURCE_TABLE}:{signal_id}@{_iso(observed_at)}"


@dataclass(frozen=True)
class SignalObservation:
    signal_id: str
    kind: ObservationKind
    at: float  # poll time (epoch seconds)
    reason: str
    label: str = ""
    value: float | None = None
    # Bad-direction percentile (strict-below) from the verdict: the band's key.
    percentile: float | None = None
    band: str = "unknown"
    observed_at: float | None = None  # the reading's own observed_at
    n_readings_7d: int = 0
    p50_7d: float | None = None
    p90_7d: float | None = None
    trend: str | None = None
    value_kind: str | None = None
    polarity: str | None = None
    # A continuous body source's dwell over its own readings, from the verdict
    # (orion.attention.sustained.hot_dwell_sec). When set, the open dwell is
    # this number, so the episode confirms on exactly the poll the attention
    # contest first lets the signal compete. None = count hot polls.
    hot_dwell_sec: float | None = None


def observation_from_verdict(verdict: Any, *, now: float) -> SignalObservation:
    """Classify one ``world_first.CandidateVerdict`` (duck-typed so this module
    stays import-light). Only the verdict's own fields are read -- never its
    reason string -- so the eligibility rule has exactly one home."""
    cand = verdict.candidate
    mag = cand.unusualness
    observed = cand.observed_at
    observed_epoch = observed.timestamp() if observed is not None else None
    sustained = getattr(verdict, "sustained", None)
    if verdict.eligible or sustained is False:
        # ``sustained`` False: high for itself now, not yet for long enough to
        # compete. Still a hot poll here: the pending episode starts at the
        # onset and a flicker keeps resetting the calm clock, as before.
        kind: ObservationKind = "hot"
    elif verdict.score is None:
        # absent / insufficient_history / stale / placeholder / no polarity
        kind = "unknown"
    elif getattr(verdict, "event_age_sec", None) is not None and verdict.event_decay == 0.0:
        # The judge's own past-the-window verdict (decay 0 for an event-written
        # source), read rather than re-derived from age and window.
        kind = "faded"
    else:
        kind = "calm"
    return SignalObservation(
        signal_id=str(cand.source_id),
        kind=kind,
        at=float(now),
        reason=str(verdict.reason),
        label=str(cand.label or cand.source_id),
        value=None if cand.absent else float(mag.value),
        percentile=None if verdict.score is None else float(verdict.score),
        band=str(verdict.band),
        observed_at=None if cand.absent else observed_epoch,
        n_readings_7d=int(mag.n_readings_7d or 0),
        p50_7d=mag.p50_7d,
        p90_7d=mag.p90_7d,
        trend=mag.trend,
        value_kind=cand.value_kind,
        polarity=cand.polarity,
        hot_dwell_sec=(
            None if sustained is None or getattr(verdict, "hot_dwell_sec", None) is None
            else float(verdict.hot_dwell_sec)
        ),
    )


@dataclass(frozen=True)
class SignalEpisodeV1:
    """One signal's run of hot polls: pending until confirmed, then open."""

    episode_id: str
    signal_id: str
    opened_at: float  # onset: the run's first hot poll
    open_band: str
    open_percentile: float | None
    open_value: float | None
    open_observed_at: float | None
    peak_band: str
    peak_percentile: float | None
    peak_value: float | None
    peak_observed_at: float | None
    last_hot_at: float
    hot_polls: int = 1
    # Hot poll times inside the trailing open window (the dwell's evidence).
    recent_hot: tuple[float, ...] = ()
    confirmed_at: float | None = None  # None = still pending (no row yet)
    calm_since: float | None = None
    unknown_since: float | None = None
    open_published: bool = False
    # Confirmed at a cold start, or over the open budget: tracked so it
    # cannot re-open, but it writes no rows.
    silent: bool = False
    silent_reason: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @property
    def confirmed(self) -> bool:
        return self.confirmed_at is not None


@dataclass(frozen=True)
class EpisodeEvent:
    phase: Literal["open", "close"]
    episode: SignalEpisodeV1
    observation: SignalObservation
    close_reason: str | None = None  # calm | faded | went_silent
    closed_at: float | None = None

    @property
    def duration_sec(self) -> float | None:
        if self.closed_at is None:
            return None
        return round(max(0.0, self.closed_at - self.episode.opened_at), 3)


def _add_ref(refs: tuple[str, ...], ref: str | None, cap: int) -> tuple[str, ...]:
    """Distinct refs, oldest dropped past ``cap`` -- except the first (the
    onset reading), which is always kept."""
    if not ref or ref in refs:
        return refs
    out = refs + (ref,)
    if len(out) > cap:
        out = out[:1] + out[len(out) - cap + 1 :]
    return out


def _new_episode(obs: SignalObservation) -> SignalEpisodeV1:
    ref = evidence_ref(obs.signal_id, obs.observed_at)
    return SignalEpisodeV1(
        episode_id=f"{obs.signal_id}@{_iso(obs.at)}",
        signal_id=obs.signal_id,
        opened_at=obs.at,
        open_band=obs.band,
        open_percentile=obs.percentile,
        open_value=obs.value,
        open_observed_at=obs.observed_at,
        peak_band=obs.band,
        peak_percentile=obs.percentile,
        peak_value=obs.value,
        peak_observed_at=obs.observed_at,
        last_hot_at=obs.at,
        recent_hot=(obs.at,),
        evidence_refs=(ref,) if ref else (),
    )


def _is_higher_peak(ep: SignalEpisodeV1, obs: SignalObservation) -> bool:
    if obs.percentile is None:
        return False
    if ep.peak_percentile is None:
        return True
    if obs.percentile != ep.peak_percentile:
        return obs.percentile > ep.peak_percentile
    # Same percentile (pinned at the top of its week): the larger raw value.
    return (obs.value or 0.0) > (ep.peak_value or 0.0)


def dwell_met(
    recent_hot: tuple[float, ...], now: float, *, poll_sec: float, open_dwell_sec: float, open_window_sec: float
) -> bool:
    """Hot for MORE than ``open_dwell_sec`` (counted as hot polls x poll_sec)
    within the trailing ``open_window_sec``. The comparison is the attention
    seam's own ``is_sustained``; only the dwell's measurement (polls) is local."""
    n = sum(1 for t in recent_hot if t > now - open_window_sec)
    return is_sustained(n * poll_sec, hot_now=True, dwell_threshold_sec=open_dwell_sec)


def _open_dwell_met(
    episode: "SignalEpisodeV1", obs: SignalObservation, *, poll_sec: float, open_dwell_sec: float,
    open_window_sec: float,
) -> bool:
    """A continuous body source carries its dwell over its own readings (the
    number the attention contest judged); everything else counts hot polls."""
    if obs.hot_dwell_sec is not None:
        return is_sustained(obs.hot_dwell_sec, hot_now=True, dwell_threshold_sec=open_dwell_sec)
    return dwell_met(
        episode.recent_hot, obs.at, poll_sec=poll_sec, open_dwell_sec=open_dwell_sec,
        open_window_sec=open_window_sec,
    )


def step_signal(
    episode: SignalEpisodeV1 | None,
    obs: SignalObservation,
    *,
    poll_sec: float = DEFAULT_POLL_SEC,
    open_dwell_sec: float = DEFAULT_OPEN_DWELL_SEC,
    open_window_sec: float = DEFAULT_OPEN_WINDOW_SEC,
    calm_gap_sec: float = DEFAULT_CALM_GAP_SEC,
    silent_close_sec: float = DEFAULT_SILENT_CLOSE_SEC,
    cold_start: bool = False,
    max_evidence_refs: int = MAX_EVIDENCE_REFS,
) -> tuple[SignalEpisodeV1 | None, EpisodeEvent | None]:
    """Fold one observation of one signal. Returns (episode after, event).

    ``event`` is an ``open`` exactly once per episode (at confirmation, never
    for a pending run that calms down) and a ``close`` only for a confirmed,
    non-silent episode. ``cold_start`` (the first poll after a start with NO
    persisted state) makes a run that is already hot silent: no rows."""
    if episode is None:
        if obs.kind != "hot":
            return None, None
        episode = _new_episode(obs)
        if cold_start:
            # Already hot when we first looked, with no persisted state: we
            # did not see it start, so it writes no rows (no boot burst).
            episode = replace(episode, silent=True, silent_reason="adopted_at_cold_start")
        hot_now = True
    elif obs.kind == "hot":
        hot_now = True
        recent = tuple(t for t in episode.recent_hot if t > obs.at - open_window_sec)
        recent = (recent + (obs.at,))[-_MAX_RECENT_HOT:]
        ep = replace(
            episode,
            last_hot_at=max(episode.last_hot_at, obs.at),
            hot_polls=episode.hot_polls + 1,
            recent_hot=recent,
            calm_since=None,
            unknown_since=None,
            evidence_refs=_add_ref(
                episode.evidence_refs, evidence_ref(obs.signal_id, obs.observed_at), max_evidence_refs
            ),
        )
        if _is_higher_peak(episode, obs):
            ep = replace(
                ep,
                peak_band=obs.band,
                peak_percentile=obs.percentile,
                peak_value=obs.value,
                peak_observed_at=obs.observed_at,
            )
        episode = ep
    else:
        hot_now = False

    if hot_now:
        if not episode.confirmed and _open_dwell_met(
            episode,
            obs,
            poll_sec=poll_sec,
            open_dwell_sec=open_dwell_sec,
            open_window_sec=open_window_sec,
        ):
            episode = replace(episode, confirmed_at=obs.at)
            if episode.silent:
                return episode, None
            return episode, EpisodeEvent("open", episode, obs)
        return episode, None

    if obs.kind == "unknown":
        # Silence is not calm: reset the calm clock, start (or keep) the
        # silence clock, and only close when silence itself has lasted.
        since = episode.unknown_since if episode.unknown_since is not None else obs.at
        ep = replace(episode, calm_since=None, unknown_since=since)
        if obs.at - since >= silent_close_sec:
            if not ep.confirmed or ep.silent:
                return None, None
            return None, EpisodeEvent("close", ep, obs, "went_silent", obs.at)
        return ep, None

    # calm / faded
    since = episode.calm_since if episode.calm_since is not None else obs.at
    ep = replace(episode, calm_since=since, unknown_since=None)
    if obs.at - since >= calm_gap_sec:
        if not ep.confirmed or ep.silent:
            return None, None
        return None, EpisodeEvent("close", ep, obs, obs.kind, obs.at)
    return ep, None


def mark_open_published(episode: SignalEpisodeV1) -> SignalEpisodeV1:
    return replace(episode, open_published=True)


def mark_silent(episode: SignalEpisodeV1, reason: str) -> SignalEpisodeV1:
    return replace(episode, silent=True, silent_reason=reason)


def needs_open_publish(episode: SignalEpisodeV1 | None) -> bool:
    """An open row is owed: confirmed, not yet published, not silent."""
    return (
        episode is not None
        and episode.confirmed
        and not episode.open_published
        and not episode.silent
    )


# ---------------------------------------------------------------------------
# Trigger payload (MetacogTriggerV1.upstream is a free-form dict; additive).
# ---------------------------------------------------------------------------


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


def episode_upstream(event: EpisodeEvent) -> dict[str, Any]:
    """Everything a reader needs to check the claim against persisted rows."""
    ep, obs = event.episode, event.observation
    refs = list(ep.evidence_refs)[:MAX_EVIDENCE_REFS]
    up: dict[str, Any] = {
        "evidence_source": EVIDENCE_SOURCE,
        "phase": event.phase,
        # The metacog writer drafts prose for the open row only; the close row
        # is a terse deterministic record (cortex-orch skips dispatch for it,
        # orion.metacog.self_signal_episode.trigger_wants_prose).
        "prose": event.phase == "open",
        "signal_id": ep.signal_id,
        "label": obs.label,
        "episode_id": ep.episode_id,
        "opened_at": _iso(ep.opened_at),
        "confirmed_at": _iso(ep.confirmed_at),
        "band_at_open": ep.open_band,
        "percentile_at_open": _r(ep.open_percentile),
        "raw_at_open": _r(ep.open_value),
        "observed_at_open": _iso(ep.open_observed_at),
        "peak_band": ep.peak_band,
        "peak_percentile": _r(ep.peak_percentile),
        "peak_raw": _r(ep.peak_value),
        "peak_observed_at": _iso(ep.peak_observed_at),
        "hot_polls": ep.hot_polls,
        "value_kind": obs.value_kind,
        "polarity": obs.polarity,
        "n_readings_7d": obs.n_readings_7d,
        "p50_7d": obs.p50_7d,
        "p90_7d": obs.p90_7d,
        "trend": obs.trend,
        "source_table": SOURCE_TABLE,
        "evidence_refs": refs,
        "evidence_ref_count": len(refs),
    }
    if event.phase == "close":
        up.update(
            {
                "close_reason": event.close_reason,
                "closed_at": _iso(event.closed_at),
                "duration_sec": event.duration_sec,
                "last_hot_at": _iso(ep.last_hot_at),
                "band_at_close": obs.band,
                "percentile_at_close": _r(obs.percentile),
                "raw_at_close": _r(obs.value),
            }
        )
    return up


def episode_reason(event: EpisodeEvent) -> str:
    """``self_signal_episode:<phase>:<signal>:band=<band>:pct=<p>``. orion-dream
    dedupes metacog rows on this string with digits masked
    (services/orion-dream/app/cycle_store.py METACOG_KEY_RE), so one signal's
    episodes at one band share a novelty key."""
    ep = event.episode
    if event.phase == "open":
        band, pct = ep.open_band, ep.open_percentile
        tail = ""
    else:
        band, pct = ep.peak_band, ep.peak_percentile
        tail = f":closed={event.close_reason}"
    pct_s = "na" if pct is None else f"{pct:.3f}"
    return f"{TRIGGER_KIND}:{event.phase}:{ep.signal_id}:band={band}:pct={pct_s}{tail}"[:500]


def trigger_wants_prose(trigger_kind: Any, upstream: Any) -> bool:
    """False only for a self_signal_episode close row: its row is persisted by
    sql-writer straight off the bus; no LLM draft is dispatched for it. Every
    other trigger keeps today's behaviour (prose)."""
    if str(trigger_kind or "").strip().lower() != TRIGGER_KIND:
        return True
    if not isinstance(upstream, dict):
        return True
    return upstream.get("phase") != "close"


# ---------------------------------------------------------------------------
# Persistence (Redis JSON). Never raises on load.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfSignalEpisodesStateV1:
    episodes: dict[str, SignalEpisodeV1] = field(default_factory=dict)
    # epoch seconds of opens published in the trailing hour (budget).
    recent_opens: tuple[float, ...] = ()


_EP_FIELDS = tuple(SignalEpisodeV1.__dataclass_fields__)
_TUPLE_FIELDS = {"evidence_refs", "recent_hot"}
_OPTIONAL_FLOAT_FIELDS = (
    "confirmed_at", "calm_since", "unknown_since", "open_percentile", "open_value",
    "open_observed_at", "peak_percentile", "peak_value", "peak_observed_at",
)


def state_to_dict(state: SelfSignalEpisodesStateV1) -> dict[str, Any]:
    eps = {}
    for sid, ep in state.episodes.items():
        d = {k: getattr(ep, k) for k in _EP_FIELDS}
        for k in _TUPLE_FIELDS:
            d[k] = list(d[k])
        eps[sid] = d
    return {"version": 1, "episodes": eps, "recent_opens": list(state.recent_opens)}


def state_from_dict(data: Any) -> SelfSignalEpisodesStateV1:
    """Malformed episodes are dropped one by one (never the whole state)."""
    if not isinstance(data, dict):
        return SelfSignalEpisodesStateV1()
    eps: dict[str, SignalEpisodeV1] = {}
    raw_eps = data.get("episodes") if isinstance(data.get("episodes"), dict) else {}
    for sid, d in raw_eps.items():
        if not isinstance(d, dict):
            continue
        try:
            kw = {k: d[k] for k in _EP_FIELDS if k in d and k not in _TUPLE_FIELDS}
            kw["evidence_refs"] = tuple(str(r) for r in (d.get("evidence_refs") or ()))[
                -MAX_EVIDENCE_REFS:
            ]
            kw["recent_hot"] = tuple(float(t) for t in (d.get("recent_hot") or ()))[-_MAX_RECENT_HOT:]
            for k in ("opened_at", "last_hot_at"):
                kw[k] = float(kw[k])
            for k in _OPTIONAL_FLOAT_FIELDS:
                if kw.get(k) is not None:
                    kw[k] = float(kw[k])
            kw["hot_polls"] = int(kw.get("hot_polls", 1))
            ep = SignalEpisodeV1(**kw)
        except (KeyError, TypeError, ValueError):
            continue
        eps[str(sid)] = ep
    try:
        recent = tuple(float(t) for t in (data.get("recent_opens") or ()))
    except (TypeError, ValueError):
        recent = ()
    return SelfSignalEpisodesStateV1(episodes=eps, recent_opens=recent)


def open_budget_ok(state: SelfSignalEpisodesStateV1, now: float, *, max_opens_per_hour: int) -> bool:
    """Global hourly budget on opens (each costs one LLM draft). Over budget,
    the caller marks the episode silent: tracked, no rows. Only PUBLISHED
    opens count (``record_open``), so a failed publish does not eat budget."""
    if max_opens_per_hour <= 0:
        return True
    return sum(1 for t in state.recent_opens if now - t < 3600.0) < max_opens_per_hour


def record_open(state: SelfSignalEpisodesStateV1, now: float) -> SelfSignalEpisodesStateV1:
    recent = tuple(t for t in state.recent_opens if now - t < 3600.0) + (now,)
    return replace(state, recent_opens=recent)
