from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.feedback.builder import build_feedback_frame
from orion.feedback.outcome_resolution import (
    resolve_action_outcomes,
    summarize_control_observations,
)
from orion.feedback.policy import load_feedback_policy
from orion.feedback.world_settlement import CONTROL_ARM, score_world_episode
from orion.autonomy.cabinet_heat import CABINET_HEAT_SIGNAL, cabinet_heat_pressure, minute_mean
from orion.autonomy.contrast import baseline_bin
from orion.schemas.feedback_frame import FeedbackFrameV1

from app.settings import get_settings
from app.store import UNSETTLED_RENDER_STATES, FeedbackRuntimeStore

logger = logging.getLogger("orion.feedback.runtime")

# Parked frames (durable render still settling) are re-read at most this often.
_VISUAL_PARK_RECHECK_SEC = 30.0
# Past this many parked frames the head defers in place instead (still bounded by
# FEEDBACK_VISUAL_SETTLE_MAX_SEC); keeps the exclusion list and the query small.
_VISUAL_PARK_MAX_FRAMES = 256
# A parked entry whose recheck came due this long ago was consumed some other way.
_VISUAL_PARK_FORGET_SEC = 3600.0


class FeedbackRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = FeedbackRuntimeStore(
            self._settings.postgres_uri,
            reconcile_interval_sec=self._settings.feedback_reconcile_interval_sec,
            reconcile_window_sec=self._settings.feedback_reconcile_window_sec,
            reconcile_full_sweep_interval_sec=self._settings.feedback_reconcile_full_sweep_interval_sec,
            reconcile_full_sweep_hour_utc=self._settings.feedback_reconcile_full_sweep_hour_utc,
        )
        self._policy = load_feedback_policy(Path(self._settings.feedback_policy_path))
        self._stop = asyncio.Event()
        self._visual_parked: dict[str, float] = {}
        self._bus = OrionBusAsync(
            self._settings.bus_url,
            enabled=self._settings.bus_enabled,
        )

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self._settings.service_name,
            version=self._settings.service_version,
            node=self._settings.node_name,
        )

    async def start(self) -> None:
        await self._bus.connect()
        asyncio.create_task(self._poll_loop(), name="feedback-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()
        await self._bus.close()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = await asyncio.to_thread(self._tick)
                if frame is not None:
                    await self._publish_feedback_frame(frame)
            except Exception:
                logger.exception("feedback_runtime_tick_failed")
            try:
                await asyncio.to_thread(self._score_world_episodes)
            except Exception:
                logger.exception("feedback_world_settlement_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.feedback_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _publish_feedback_frame(self, frame: FeedbackFrameV1) -> None:
        env = BaseEnvelope(
            kind="feedback.frame.v1",
            source=self._service_ref(),
            payload=frame.model_dump(mode="json"),
        )
        await self._bus.publish(self._settings.feedback_bus_channel, env)
        logger.info(
            "feedback_frame_published frame_id=%s channel=%s",
            frame.frame_id,
            self._settings.feedback_bus_channel,
        )

    def _tick(self) -> Optional[FeedbackFrameV1]:
        if not self._settings.enable_feedback_runtime:
            return None

        # Rate-limited internally (default once per 15 min); safe to call every tick.
        self._store.reconcile_feedback_pending()

        parked = self._visual_parked_frames()
        now_mono = time.monotonic()
        for frame_id, recheck_at in list(parked.items()):
            if now_mono - recheck_at > _VISUAL_PARK_FORGET_SEC:
                # Scored or requeued elsewhere since; never picked back up here.
                parked.pop(frame_id, None)
        excluded = [fid for fid, recheck_at in parked.items() if recheck_at > now_mono]
        if excluded:
            dispatch = self._store.load_latest_dispatch_frame_without_feedback(
                exclude_frame_ids=excluded
            )
        else:
            dispatch = self._store.load_latest_dispatch_frame_without_feedback()
        if dispatch is None:
            return None
        if self._store.load_feedback_frame_for_dispatch(dispatch.frame_id) is not None:
            # Marker was stale-true (e.g. a pre-migration row, or a manual write). Clear it, or
            # this guard returns early every tick on the same row and the FIFO never advances.
            self._store.clear_feedback_pending(dispatch.frame_id)
            return None

        policy_frame = self._store.load_policy_frame(dispatch.source_policy_frame_id)
        proposal_frame = self._store.load_proposal_frame(dispatch.source_proposal_frame_id)
        # Loaded BEFORE the window is chosen, because the window depends on how
        # long these actions actually took. Previously loaded after the frame
        # was built; moving it up is the whole reason the settle can be
        # measured rather than assumed.
        cortex_results = self._store.load_cortex_result_evidence(dispatch)

        pending_renders = _pending_render_dispatch_ids(cortex_results)
        if pending_renders:
            visual_age_sec = (datetime.now(timezone.utc) - dispatch.generated_at).total_seconds()
            max_sec = float(self._settings.feedback_visual_settle_max_sec)
            # Negative age scores as-is, for the same stuck-head reason as the settle defer below.
            if 0.0 <= visual_age_sec < max_sec:
                # A durable render has not settled yet: scoring now would fold
                # "unknown" for an image that may still arrive. Park the frame and
                # let the FIFO move on (utilization is ~95%, so holding the head
                # for up to max_sec would build a backlog); it comes back as the
                # oldest pending frame once its recheck time passes.
                if dispatch.frame_id in parked or len(parked) < _VISUAL_PARK_MAX_FRAMES:
                    parked[dispatch.frame_id] = now_mono + min(
                        _VISUAL_PARK_RECHECK_SEC, max_sec - visual_age_sec
                    )
                logger.info(
                    "feedback_frame_parked_visual_settlement dispatch_frame_id=%s age_sec=%.1f "
                    "max_sec=%.1f pending_dispatch_ids=%s parked=%d",
                    dispatch.frame_id,
                    visual_age_sec,
                    max_sec,
                    sorted(pending_renders),
                    len(parked),
                )
                return None
            (logger.warning if max_sec > 0 else logger.debug)(
                "feedback_visual_settlement_bound_expired dispatch_frame_id=%s age_sec=%.1f "
                "max_sec=%.1f pending_dispatch_ids=%s -- scoring visual as-is (unknown)",
                dispatch.frame_id,
                visual_age_sec,
                max_sec,
                sorted(pending_renders),
            )
        parked.pop(dispatch.frame_id, None)

        # THE WINDOW HAS TO CONTAIN THE ACTION, AND ACTIONS NO LONGER SHARE ONE
        # DURATION. `action_settle_sec` was a single 15s constant sized against
        # a population of 1.2-5.4s actions (see store.load_action_scoring_
        # window). `express` runs ~50s, so its "after" sample was taken 35s
        # BEFORE the action finished: measured live 2026-08-31, three
        # consecutive outcomes with baseline == observed_after to 4dp
        # (0.3525/0.3525, 0.3525/0.3525, 0.3519/0.3519) and latency ~50,000ms.
        # That is not "no effect", it is the same defect that docstring was
        # written to fix, re-created for a slower action by a constant that
        # could not follow it. The action was then retired below the
        # information floor on evidence that was null by construction.
        settle_sec, settle_clamped = self._scoring_settle_sec(
            cortex_results, dispatched_at=dispatch.generated_at
        )
        age_sec = (datetime.now(timezone.utc) - dispatch.generated_at).total_seconds()
        # `0.0 <=` is the retirement path, not a tidiness guard. The defer
        # clears itself only because wall-clock age grows; a NEGATIVE age never
        # reaches the settle, so a future-dated generated_at (a backwards NTP
        # step, a restore, a manual insert, a naive datetime -- the field is a
        # bare `datetime` in the schema, so naive is legal) would park the FIFO
        # head forever, one INFO line per 2s poll with no way out. That is
        # structurally the 2026-07-22 stuck-head incident that
        # store._retire_incompatible_dispatch_frame exists to prevent for the
        # schema case. action_settle_max_sec does NOT bound this: it clamps the
        # settle, not the age.
        if 0.0 <= age_sec < settle_sec:
            # DEFER, do not consume. Scoring now would find no field tick at
            # the closing edge, and `resolve_action_outcomes` would skip every
            # candidate as `missing_field_window` -- while `save_feedback_frame`
            # still clears `feedback_pending`, so the dispatch is never rescored.
            # Silent, permanent loss of the measurement. Real lag measured over
            # 10,261 frames (6h, 2026-08-31): p50 94.5s, p95 172.5s, but
            # min 0.1s -- so the fast tail already loses measurements today at
            # settle=15, and widening the window without this would turn that
            # tail into the common case.
            #
            # Head-of-line blocking here is BOUNDED, not free -- an earlier
            # draft of this comment claimed the latter and was wrong. The lookup
            # is oldest-first (store._PENDING_SQL, ORDER BY generated_at ASC), so
            # the head is the oldest pending frame; when settle was one constant
            # everything behind it was equally unscoreable and deferring really
            # did cost nothing. Per-frame settle breaks that: a younger frame
            # with a SMALLER settle can be scoreable and does get blocked.
            # Measured over 24h: 170 of 857 dispatching frames defer, worst
            # head-block 14.8s, 1,702s total (2.0% of wall time). It stays that
            # small because execution-dispatch-runtime inserts the frame row
            # AFTER its sends, so a frame's age at first visibility already
            # covers its own latency and the wait is ~base, not ~settle.
            logger.info(
                "feedback_frame_deferred dispatch_frame_id=%s age_sec=%.1f settle_sec=%.1f "
                "-- scoring window does not close yet; retrying",
                dispatch.frame_id,
                age_sec,
                settle_sec,
            )
            return None
        # 2026-07-22 (SelfStateV1 burn): field_before is the exact field
        # tick dispatch was built against; field_after is the next real
        # field tick observed within the policy's window, same "did the
        # world move" comparison self-state used to provide a lossy hop for.
        field_before = self._store.load_field_for_tick(dispatch.source_field_tick_id)
        field_after = self._store.load_latest_field_after(
            dispatch.generated_at,
            window_sec=self._policy.windows.field_after_window_sec,
        )
        frame = build_feedback_frame(
            dispatch_frame=dispatch,
            policy_frame=policy_frame,
            proposal_frame=proposal_frame,
            field_before=field_before,
            field_after=field_after,
            cortex_results=cortex_results or None,
            policy=self._policy,
        )
        # Score whatever predictions this tick's dispatched actions made.
        # Never allowed to take the feedback frame down with it: a scoring
        # bug must not stall the pipeline that has been running for months.
        # An empty resolution is a legitimate outcome (a tick can dispatch
        # nothing, or dispatch only actions that declare no signal).
        try:
            # NOT the feedback frame's own field_before/field_after. Those
            # are [t-205s, t+1.1s] and the action has not returned at the
            # closing edge -- see store.load_action_scoring_window for the
            # measurements and why that made every contrast an unbiased
            # estimate of a null quantity.
            if settle_clamped:
                # A window we already know is too short yields a confident wrong
                # posterior, which is strictly worse than a gap. Refuse.
                logger.warning(
                    "feedback_scoring_window_clamped dispatch_frame_id=%s settle_sec=%.1f "
                    "-- action outlasts the ceiling; NOT scoring (a short window would "
                    "fold a belief that is null by construction)",
                    dispatch.frame_id,
                    settle_sec,
                )
                score_before, score_after = None, None
            else:
                score_before, score_after = self._store.load_action_scoring_window(
                    dispatch.generated_at,
                    settle_sec=settle_sec,
                )
            resolution = resolve_action_outcomes(
                dispatch_frame=dispatch,
                feedback_frame_id=frame.frame_id,
                field_before=score_before,
                field_after=score_after,
                priors=self._store.load_effect_posteriors(),
                control_priors=self._store.load_control_posteriors(),
                latency_by_dispatch_id=_latencies(cortex_results),
                cortex_results=cortex_results,
            )
        except Exception:
            logger.exception(
                "action_outcome_resolution_failed dispatch_frame_id=%s", dispatch.frame_id
            )
            resolution = None

        self._store.save_feedback_frame(
            frame,
            outcome_records=resolution.records if resolution else None,
            control_cells=resolution.control_posteriors if resolution else None,
            control_frame_id=dispatch.frame_id,
        )
        logger.info(
            "feedback_frame_saved frame_id=%s dispatch_frame_id=%s outcome_status=%s "
            "observations=%d scored_actions=%d skipped_actions=%d untreated=%s",
            frame.frame_id,
            dispatch.frame_id,
            frame.outcome_status,
            len(frame.observations),
            len(resolution.records) if resolution else 0,
            len(resolution.skipped) if resolution else 0,
            summarize_control_observations(
                resolution.control_observations if resolution else []
            ),
        )
        if resolution and resolution.skipped:
            # Aggregated, not per-dispatch: this fires every tick and the
            # per-reason counts are the part anyone acts on.
            counts: dict[str, int] = {}
            for reason in resolution.skipped.values():
                counts[reason] = counts.get(reason, 0) + 1
            logger.info(
                "action_outcome_skipped dispatch_frame_id=%s reasons=%s",
                dispatch.frame_id,
                sorted(counts.items()),
            )
        return frame

    def _score_world_episodes(self, now: datetime | None = None) -> int:
        """Settle-time scoring for world actions (both arms). Rate limited; returns rows scored."""
        if not getattr(self._settings, "world_settlement_scoring_enabled", False):
            return 0
        mono = time.monotonic()
        next_at = getattr(self, "_world_next_at", None)
        if next_at is not None and mono < next_at:
            return 0
        self._world_next_at = mono + float(self._settings.world_settlement_interval_sec)
        now = now or datetime.now(timezone.utc)
        episodes = self._store.load_world_episodes_due(now=now)
        if not episodes:
            return 0
        posteriors = self._store.load_effect_posteriors()
        controls = self._store.load_control_posteriors()
        scored = 0
        for ep in episodes:
            t0 = ep["decided_at"]
            ttl = float((ep.get("settlement") or {}).get("ttl_sec") or 900.0)
            end = t0 + timedelta(seconds=ttl + 300.0)
            try:
                points = self._store.load_cabinet_points(since=t0 - timedelta(minutes=2), until=end + timedelta(minutes=2))
                incidents = self._store.load_hardware_incidents_opened(since=t0, until=end)
            except Exception:
                logger.warning("world_settlement_inputs_unavailable episode_id=%s", ep.get("episode_id"), exc_info=True)
                continue
            bin_index = baseline_bin(float(cabinet_heat_pressure(minute_mean(points, t0)) or 0.0)) if points else 0
            prior = posteriors.get((ep.get("dispatch_kind"), ep.get("target_id"), CABINET_HEAT_SIGNAL, bin_index))
            control = controls.get((CABINET_HEAT_SIGNAL, CONTROL_ARM, bin_index))
            score = score_world_episode(episode=ep, points=points, incidents=incidents, prior=prior,
                                        control_prior=control, now=now)
            if score is None:
                continue
            if score.loop_outcome is not None:
                try:
                    sal = self._store.load_loop_salience(open_loop_id=score.loop_outcome["loop_id"], since=end)
                except Exception:
                    sal = None
                score.loop_outcome["salience_at_close"] = sal or 0.0
                score.loop_outcome["features_at_close"]["next_tick_salience"] = sal
            if self._store.save_world_score(episode_id=str(ep["episode_id"]), score=score, scored_at=now):
                scored += 1
                logger.info("world_episode_scored episode_id=%s arm=%s terminal=%s delta=%s excluded=%s overlap=%s "
                            "posterior_updated=%s loop_outcome=%s before_c=%s after_c=%s",
                            ep["episode_id"], ep.get("arm"), score.outcome.get("terminal"),
                            score.outcome.get("observed_delta"), score.outcome.get("excluded_reason"),
                            score.outcome.get("overlap"), score.outcome.get("posterior_updated"),
                            bool(score.loop_outcome), score.outcome.get("before_temp_c"),
                            score.outcome.get("after_temp_c"))
        return scored

    def _visual_parked_frames(self) -> dict[str, float]:
        """dispatch frame_id -> monotonic time it may be looked at again."""
        parked = getattr(self, "_visual_parked", None)
        if parked is None:
            parked = self._visual_parked = {}
        return parked

    def _scoring_settle_sec(
        self,
        cortex_results: list[dict[str, object]] | None,
        *,
        dispatched_at: datetime | None = None,
    ) -> tuple[float, bool]:
        """How long to wait before sampling the field, for THIS frame.

        `action_settle_sec` stops being "send offset + action latency + the
        digester's fold" baked into one constant, and becomes only the margin
        on top of a latency that is now MEASURED. Latency was unpopulated on
        every row until 2026-08-21, which is why it was a constant in the first
        place; it is real data now, so the window can follow the action instead
        of assuming its duration.

        Returns (settle_sec, clamped). `clamped` true means the ceiling bound
        and the window is KNOWN to be shorter than the action -- the caller must
        not score that frame.

        Frame-wide max, not per candidate: the field delta is frame-wide (see
        the co-attribution bookkeeping in resolve_action_outcomes), so a window
        that contains only the fastest action in the frame would attribute a
        shared delta from a sample taken while the others were still running.
        Measured cost of that choice over 24h: of 870 dispatching frames, 809
        held a single action; the 60 mixed frames (inspect/maintain/summarize,
        in-frame latency spread 5.4-18.4s) widen from 15s to ~33s, not to 65s.

        Falls back to the bare constant when no latency was reported -- absent
        stays absent, never coerced to 0.0, which would read as "this action
        was free" and silently reproduce the too-narrow window this exists to
        fix. Clamped at `action_settle_max_sec` so one pathological latency
        cannot park the FIFO head for hours.

        A durable render's latency_ms is its GPU seconds, not when the image
        existed: the run may have queued for the diffusion lane first. For those
        the window reaches the run's own finished_at instead, so a render that
        ended minutes after dispatch clamps (and is refused) rather than being
        scored from a sample taken before its image existed.
        """
        base = float(self._settings.action_settle_sec)
        latencies = _window_latencies(cortex_results, dispatched_at)
        if not latencies:
            return base, False
        worst_sec = max(latencies.values()) / 1000.0
        wanted = base + worst_sec
        ceiling = float(self._settings.action_settle_max_sec)
        if wanted > ceiling:
            # Clamped: the window provably does NOT contain the action. Reported
            # so the caller can refuse to score rather than fold a belief it
            # already knows is null by construction.
            return ceiling, True
        return wanted, False


def _pending_render_dispatch_ids(cortex_results: list[dict[str, object]] | None) -> set[str]:
    return {
        str(raw.get("dispatch_id") or "")
        for raw in cortex_results or []
        if raw.get("settlement_state") in UNSETTLED_RENDER_STATES
    }


def _window_latencies(
    cortex_results: list[dict[str, object]] | None, dispatched_at: datetime | None
) -> dict[str, float]:
    """Per-dispatch ms the scoring window must cover: measured latency, widened to
    the durable run's real end (finished_at - dispatch time) for settled renders."""
    out = _latencies(cortex_results)
    if dispatched_at is None:
        return out
    if dispatched_at.tzinfo is None:
        dispatched_at = dispatched_at.replace(tzinfo=timezone.utc)
    for raw in cortex_results or []:
        dispatch_id = str(raw.get("dispatch_id") or "")
        finished_raw = raw.get("settled_finished_at")
        if not dispatch_id or not finished_raw:
            continue
        try:
            finished = datetime.fromisoformat(str(finished_raw))
        except ValueError:
            continue
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
        elapsed_ms = (finished - dispatched_at).total_seconds() * 1000.0
        if elapsed_ms > out.get(dispatch_id, 0.0):
            out[dispatch_id] = elapsed_ms
    return out


def _latencies(cortex_results: list[dict[str, object]] | None) -> dict[str, float]:
    """Real measured cost per dispatch, when the cortex result reported one.

    Absent keys stay absent -- never coerced to 0.0, which would read as
    "this action was free" and quietly bias any cost-weighted comparison
    built on top of this ledger toward whichever executor happens not to
    report timings.
    """
    out: dict[str, float] = {}
    for raw in cortex_results or []:
        dispatch_id = str(raw.get("dispatch_id") or "")
        if not dispatch_id:
            continue
        for key in ("latency_ms", "duration_ms", "elapsed_ms"):
            value = raw.get(key)
            if value is None:
                continue
            try:
                out[dispatch_id] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            break
    return out
