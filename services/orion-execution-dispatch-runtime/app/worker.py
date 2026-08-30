from __future__ import annotations

import asyncio
import logging
from time import monotonic, perf_counter
import math
import random
from collections import deque
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from orion.autonomy.models import ActionOutcomeEmitV1
from orion.bus.ewma import compute_ewma_update
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.autonomy.allocator import Candidate, allocate, candidate_from_dispatch
from orion.autonomy.budget import budget_state, day_elapsed_fraction
from orion.autonomy.contrast import (
    HOLDBACK_BLOCK_REASON,
    TreatedCellKey,
    pooled_treated_mean,
)
from orion.autonomy.prediction import (
    DEFAULT_OBSERVATION_VARIANCE,
    DEFAULT_PRIOR_VARIANCE,
    EffectPosterior,
)
from orion.execution_dispatch.builder import (
    build_execution_dispatch_frame,
    build_stale_discard_execution_dispatch_frame,
    build_unevaluable_execution_dispatch_frame,
)
from orion.execution_dispatch.cortex_client import ExecutionDispatchCortexClient
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.execution_dispatch.result_extraction import (
    RESULT_KIND_EMPTY,
    RESULT_KIND_STRUCTURED,
    extract_final_text,
    is_failed_plan_status,
    parse_structured_observation,
    plan_execution_status,
)
from orion.notify.client import NotifyClient
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.notify import NotificationRequest
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1

from app.settings import get_settings
from app.store import ExecutionDispatchRuntimeStore

logger = logging.getLogger("orion.execution_dispatch.runtime")

THEATER_TRIPWIRE_WINDOW = 10
# Renamed from THEATER_TRIPWIRE_EMPTY_THRESHOLD 2026-08-13: the predicate it
# gates now counts every non-success status, not just literal "empty".
THEATER_TRIPWIRE_UNPRODUCTIVE_THRESHOLD = 0.5

# --- Tripwire recovery (2026-08-25) -------------------------------------
#
# THE INCIDENT THIS EXISTS FOR. 2026-08-23 07:33 UTC -> 2026-08-25 05:20 UTC:
# 45 hours, zero dispatches. A routine redeploy at 06:58 bounced the whole
# stack; cortex-exec came back cold (four dispatches at 06:59 took 61-91s and
# returned plan_status=partial), then six in a row failed fast with
# plan_status=fail at 07:33. That crossed the >5-of-trailing-10 threshold and
# latched the tripwire. Nothing was actually broken by then -- cortex-exec was
# serving other callers normally the whole time.
#
# Three separate defects made a transient into a two-day outage:
#
#   1. NO RE-ARM PATH. `theater_tripwire_active` had exactly one assignment to
#      True and no assignment back to False outside __init__. Recovery required
#      a human noticing and restarting the process.
#   2. SELF-SEALING. Once tripped, _send_prepared_candidates returns before
#      sending, so `_recent_dispatch_statuses` never receives another sample.
#      Even a hypothetical re-arm rule had no evidence it could ever act on --
#      the window is frozen at the moment of the trip, forever.
#   3. SILENT. One warning notification at the moment of the trip, then nothing
#      for 45 hours. Every tick kept logging `motor_budget mode=advisory
#      pace=0.85x` as though the motor were running, because the budget readout
#      sits BELOW the tripwire's early return and never learns it was skipped.
#
# WHY THIS IS NOT SIMPLY REVERSING THE ORIGINAL DECISION. The manual re-arm was
# deliberate, and __init__'s own comment states the reason: "a self-clearing
# tripwire could silently resume sending on a coincidentally-good sample." That
# concern is real and is preserved here. What clears the latch is not a sample,
# it is a probe: at most TRIPWIRE_PROBE_DISPATCHES candidate per attempt, on an
# exponential backoff, and the latch clears only after
# orion_dispatch_tripwire_rearm_successes CONSECUTIVE probe successes -- a
# single lucky result re-arms nothing and resets the run to zero. Probe outcomes
# are tracked separately from `_recent_dispatch_statuses` precisely so the frozen
# pre-trip window cannot vote on the recovery decision.
#
# The cost of being wrong is bounded and directional: a probe is one action, and
# a persistently dead motor settles at one probe per
# orion_dispatch_tripwire_probe_max_cooldown_sec. Compared against 45 hours of
# zero actions, that is the trade this makes on purpose.

# One candidate per probe attempt. Deliberately not "a normal tick, but
# monitored": a probe must not be able to become a de-facto resumption of
# service before the re-arm run is complete.
TRIPWIRE_PROBE_DISPATCHES = 1
# Each failed probe doubles the wait, capped at the configured maximum.
TRIPWIRE_PROBE_BACKOFF_FACTOR = 2.0
# While tripped, say so on this cadence. Defect 3 above: the absence of any
# per-tick record is what made a 45-hour outage indistinguishable from normal
# operation in both the logs and the stored frames.
TRIPWIRE_BLOCKED_LOG_INTERVAL_SEC = 300.0
# And re-notify on this cadence. Fire-once is why nobody acted for two days.
TRIPWIRE_RENOTIFY_INTERVAL_SEC = 3600.0
# Frame-level warning string. Goes into ExecutionDispatchFrameV1.warnings --
# which already exists and is already persisted -- so "was Orion able to act
# at time T" becomes a Postgres question instead of an in-process one that
# dies with the very restart used to recover. No schema change.
TRIPWIRE_BLOCKED_WARNING = "execution_dispatch_theater_tripwire_active"
# Real, disclosed gap closed defensively (review, 2026-07-26): every live
# proposal_kind_to_cortex-routed template today has base_risk >= 0.02 (the
# one base_risk=0.0 template, defer_due_to_low_readiness, has no cortex
# route and never reaches a dispatch candidate) -- so risk_score=0.0 is not
# currently reachable in production. But ExecutionDispatchCandidateV1.
# risk_score only enforces [0,1] at the schema level, nothing prevents a
# future template from shipping one at 0.0, and a genuinely zero-risk
# candidate would make the cumulative risk budget below provide no ceiling
# at all (cumulative_risk + 0.0 > remaining_risk_budget is never true once
# remaining_risk_budget is positive). This floor is a config-independent
# backstop, not a claim any real candidate is actually this risky.
MINIMUM_REAL_RISK_FLOOR = 0.01

# 2026-07-29: self-calibrating daily risk ceiling. Replaces the fixed
# ORION_DISPATCH_MAX_RISK_PER_DAY constant as the primary mechanism (that
# setting is now only the no-history-at-all fallback, see settings.py's own
# comment). Same EWMA-baseline shape already shipped for field-attention's
# recent_perturbation caps (PR #1433) and execution_prediction_error's own
# per-tick baseline (PR #1434), both on orion/bus/ewma.py::
# compute_ewma_update -- new domain, not a new mechanism.
#
# alpha=0.4, not execution_prediction_error's 0.2: that baseline absorbs many
# real samples a day (one execution-trajectory receipt batch per tick), so a
# slower-moving average is appropriate there. This baseline gets AT MOST one
# real sample per UTC calendar day (see _derive_daily_risk_cap) -- a much
# sparser cadence needs faster adaptation to actually track real demand
# instead of staying anchored to stale history for weeks.
DAILY_RISK_BASELINE_ALPHA = 0.4
# Do NOT reuse orion/bus/ewma.py's own default (_MIN_VARIANCE=1e-6) without
# checking it against this domain's real scale -- that floor is calibrated
# for orion-bus-mirror's real-time-gap-in-seconds domain and already
# silently dominated execution_prediction_error's z-scores once before being
# caught (see that function's own _EXECUTION_PREDICTION_ERROR_MIN_VARIANCE
# comment in orion/substrate/prediction_error.py). Real daily uncapped risk
# totals confirmed live in this repo's own Postgres history run O(1)-O(1000)
# (4.4 on 2026-07-26 under the old enforced clamp, 817.65 uncapped on
# 2026-07-28), not O(1e-11) or O(1) time-gaps-in-seconds -- a variance floor
# of 1.0 (a minimum std-dev of 1.0 risk unit) is on the right order of
# magnitude here: small enough that real day-to-day spread (once
# DAILY_RISK_BASELINE_MIN_SAMPLES real samples exist) drives the z-score
# rather than the floor, nonzero enough to guard the first non-cold-start
# day's div-by-zero (prev_variance is exactly 0.0 straight out of a
# single-sample seed). Revisit if real daily totals ever shift by orders of
# magnitude from today's ~800/day.
DAILY_RISK_BASELINE_MIN_VARIANCE = 1.0
# Below this, don't trust the derived mean+variance yet -- a single sample
# gives variance=0.0, so `mean + Z*sqrt(var)` collapses to just `mean`, no
# real margin. Only one real closed day with real *uncapped* demand exists
# as of this patch (2026-07-28: 817.65) -- 2 is the smallest value that
# actually distinguishes "have a real spread" from "have one number".
DAILY_RISK_BASELINE_MIN_SAMPLES = 2
# Reuses this repo's already-established z>=3.0 "anomalous" convention, not
# a fresh number: same constant as _BUS_SYNAPTIC_ZSCORE_SATURATION and
# _EXECUTION_PREDICTION_ERROR_ZSCORE_SATURATION (orion/substrate/
# prediction_error.py) and services/orion-hub/scripts/
# bus_synaptic_graph_routes.py's anomalies() route.
DAILY_RISK_BASELINE_Z = 3.0

# 2026-07-30 (docs/superpowers/specs/2026-07-30-execution-dispatch-
# staleness-discard-design.md): per-tick cap on how many consecutive stale
# policy frames a single _tick() call will fast-drain before yielding back
# to the poll loop's sleep. Without this, the first tick after this patch
# ships would discard the entire pre-existing backlog (46,617 rows measured
# live 2026-07-30) in one pass -- not unsafe, but it would feed the
# staleness-discard EWMA baseline one absurd one-time outlier sample instead
# of several real, representative ones, and would hold the poll loop's
# single worker thread for however long tens of thousands of real DB writes
# take. 200 is a disclosed, uncalibrated starting batch size (no real data
# exists yet to size this against) -- large enough to make real progress
# against a deep backlog across a handful of ticks, small enough that a
# single tick's real wall-clock cost stays bounded.
MAX_STALE_DISCARDS_PER_TICK = 200

# Same disclosure as MAX_STALE_DISCARDS_PER_TICK above: this is the first
# time this metric has ever been measured, so unlike DIMENSION_PRECISION_
# MIN_VARIANCE / DAILY_RISK_BASELINE_MIN_VARIANCE elsewhere in this repo
# (both seeded from a real measured population before shipping), there is no
# real historical variance to calibrate against yet. alpha=0.2 is faster
# than field-digester's per-2s-tick DIMENSION_PRECISION_EWMA_ALPHA (0.02)
# since this updates once per real _tick() call that finds ANY policy frame
# to act on (irregular, can fire far more or less often than every 2s
# depending on backlog depth) -- slower than DAILY_RISK_BASELINE_ALPHA (0.4,
# at most one real sample a day) since this can update many times a minute.
# NOTE (code review, 2026-07-30): a tick where the policy-decision queue is
# fully empty from the start (no frame at all, not even a fresh one) skips
# this update entirely rather than persisting a value=0.0 sample -- there is
# no frame left to carry it on in that exact case. Deliberately not "fixed"
# with a synthetic no-op frame just to carry a metric forward when nothing
# real happened (same "no empty-shell cognition" bias against inventing
# content for its own sake) -- this only under-samples the true-idle case,
# never the backlog case this patch exists for, and under-sampling a rare
# calm tick is the same safe-direction trade DAILY_RISK_BASELINE's own
# multi-day-outage gap comment already accepts elsewhere in this file.
# MIN_VARIANCE=1.0 treats "one
# discard" as the natural unit of spread for a small-non-negative-integer
# count metric. Both are a disclosed, reasonable starting guess, not a
# calibrated value -- revisit once real post-deploy discard-count history
# exists, per this repo's metric-quality-gate discipline (do not treat "the
# EWMA update runs without error" as proof this floor is right).
STALENESS_DISCARD_EWMA_ALPHA = 0.2
STALENESS_DISCARD_EWMA_MIN_VARIANCE = 1.0

# ActionOutcomeEmitV1.summary lands in chat-visible evidence via
# chat_stance.py's _project_recent_dispatch_actions -- bounded here at the
# producer so nothing downstream needs to re-truncate raw model output.
ACTION_OUTCOME_SUMMARY_MAX_CHARS = 280
# Established self-subject convention across orion/autonomy/* (reducer.py,
# substrate_metabolism.py, signal_tension.py all default/use this).
ACTION_OUTCOME_SUBJECT = "orion"


class ExecutionDispatchRuntimeWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = ExecutionDispatchRuntimeStore(
            self._settings.postgres_uri,
            reconcile_interval_sec=getattr(
                self._settings, "dispatch_reconcile_interval_sec", 900.0
            ),
        )
        self._policy = load_execution_dispatch_policy(
            Path(self._settings.execution_dispatch_policy_path)
        )
        self._notify = NotifyClient(
            base_url=self._settings.notify_url,
            api_token=self._settings.notify_api_token,
            timeout=10,
        )
        self._stop = asyncio.Event()
        # In-memory only, by design: the latch never survives a restart, and
        # the tripwire's own judgment is always built from samples this
        # process observed (see _recent_dispatch_statuses below).
        #
        # 2026-08-25: it is no longer true that "once tripped, stays tripped
        # until this process restarts". That rule cost 45 hours of zero
        # dispatches -- see the incident note beside TRIPWIRE_PROBE_DISPATCHES.
        # The original rationale ("a self-clearing tripwire could silently
        # resume sending on a coincidentally-good sample") is preserved rather
        # than discarded: nothing here clears on a sample. Re-arming requires a
        # run of consecutive successful PROBES, each one a single deliberate
        # action taken after a backoff, tracked in state that the frozen
        # pre-trip window cannot influence.
        self.theater_tripwire_active = False
        # Monotonic, not wall-clock: this only ever measures elapsed intervals,
        # and must not be perturbed by an NTP step or a DST boundary.
        self._tripwire_tripped_at: float | None = None
        self._tripwire_next_probe_at: float | None = None
        self._tripwire_probe_cooldown_sec = (
            self._settings.orion_dispatch_tripwire_probe_cooldown_sec
        )
        # Consecutive successes only. Any probe failure resets this to 0 --
        # that reset is what keeps "coincidentally-good sample" from re-arming.
        self._tripwire_probe_successes = 0
        self._tripwire_probe_attempts = 0
        # Consecutive ticks on which the allocator refused every pending
        # candidate. Consecutive-with-reset, not a lifetime total: "refusing
        # everything right now" and "refused everything once last week" are
        # different facts and only the first is an outage.
        self._all_refused_consecutive_ticks = 0
        # Throttle state for the two "still blocked" surfaces. Seeded to None
        # rather than 0.0 so the FIRST blocked tick after a trip always speaks,
        # instead of being swallowed by an interval that appears already-served.
        self._tripwire_last_blocked_log_at: float | None = None
        self._tripwire_last_notify_at: float | None = None
        # True only for the span of a probe tick, so the post-send evaluation
        # knows whether the results it is looking at are probe evidence or an
        # ordinary tick's.
        self._tripwire_probe_in_flight = False
        # Statuses recorded during the CURRENT tick only, in the same
        # vocabulary as _recent_dispatch_statuses. The trailing window is
        # useless for judging a probe: it is maxlen-bounded and, at the moment
        # a probe runs, still full of the pre-trip failures that caused the
        # trip. A probe is judged solely on evidence the probe itself produced.
        self._tick_dispatch_statuses: list[str] = []
        # Per-process window for the tripwire's own judgment. Deliberately
        # NOT sourced from substrate_dispatch_results' full history (that
        # table persists real history for other consumers and is never
        # pruned by this service) -- a restart must give the tripwire a
        # genuinely clean slate. Real incident, 2026-07-25: 10 rows from a
        # cold post-Postgres-rebuild window (all status="empty", the
        # substrate hadn't repopulated yet) were still the newest rows in
        # the table days later, so every restart re-evaluated the SAME
        # stale evidence and re-tripped on the same tick, making "requires a
        # restart to re-arm" false in this case -- not a self-clearing
        # tripwire (still requires a real restart, still can't clear itself
        # mid-process), just a tripwire whose restart-to-re-arm contract
        # actually holds now.
        self._recent_dispatch_statuses: deque[str] = deque(maxlen=THEATER_TRIPWIRE_WINDOW)

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="execution-dispatch-runtime-poll")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("execution_dispatch_runtime_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.execution_dispatch_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _staleness_threshold_sec(self) -> float:
        override = self._settings.execution_dispatch_staleness_override_sec
        if override is not None:
            return float(override)
        return random.uniform(
            self._settings.execution_dispatch_staleness_min_sec,
            self._settings.execution_dispatch_staleness_max_sec,
        )

    @staticmethod
    def _age_sec(frame: PolicyDecisionFrameV1) -> float:
        # Defensive normalize, same idiom as _derive_daily_risk_cap above:
        # PolicyDecisionFrameV1.generated_at is typed as a plain datetime,
        # not enforced tz-aware. A naive value here would raise on the
        # subtraction below -- caught by _poll_loop's broad except, but
        # this exact policy frame would then be stuck unresolved forever
        # in whichever lookup found it, reproducing the same queue-blocking
        # failure shape build_unevaluable_execution_dispatch_frame exists to
        # prevent for a missing proposal. Shared by _drain_stale_policy_
        # frames (FIFO lookup) and _check_freshest_fallback (newest-first
        # lookup) -- both need the identical normalize-then-subtract logic.
        frame_generated_at = frame.generated_at
        if frame_generated_at.tzinfo is None:
            frame_generated_at = frame_generated_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - frame_generated_at).total_seconds()

    def _check_freshest_fallback(self) -> PolicyDecisionFrameV1 | None:
        """2026-07-30 follow-up (docs/superpowers/specs/2026-07-30-execution-
        dispatch-staleness-discard-design.md's own live-verified finding):
        the FIFO staleness drain alone can fully starve real-time dispatch
        when the backlog is deep -- confirmed live, zero real dispatches in
        the 6 minutes following that patch's deploy, because every tick's
        entire MAX_STALE_DISCARDS_PER_TICK budget was spent discarding old
        backlog without ever reaching a frame recent enough to process.

        Called only when _drain_stale_policy_frames() didn't surface a
        candidate this tick (queue exhausted, drain hit the cap, or the
        queue was empty from the start). Checks the single NEWEST unprocessed
        policy frame directly (store.load_freshest_policy_frame_without_
        dispatch, ORDER BY generated_at DESC) -- if it's within a fresh
        staleness threshold, a genuinely current proposal always gets a
        chance to dispatch this tick regardless of how deep the old backlog
        is. Returns None (correctly) if even the newest available frame is
        already stale -- that means production itself has stalled, not that
        backlog depth is hiding something current.
        """
        # Rate-limited internally (default once per 15 min); safe to call every tick.
        self._store.reconcile_dispatch_pending()
        freshest = self._store.load_freshest_policy_frame_without_dispatch()
        if freshest is None:
            return None
        if self._age_sec(freshest) <= self._staleness_threshold_sec():
            return freshest
        return None

    def _drain_stale_policy_frames(
        self, baseline: dict
    ) -> tuple[PolicyDecisionFrameV1 | None, int, ExecutionDispatchFrameV1 | None]:
        """FIFO-drains policy frames older than a randomized staleness
        threshold, discarding (materializing, never silently dropping -- see
        build_stale_discard_execution_dispatch_frame) each one instead of
        dispatching a real cortex action against hours-old field state. Stops
        at the first frame still within its threshold (returned as the real
        candidate for this tick to process normally), an empty queue, or
        MAX_STALE_DISCARDS_PER_TICK, whichever comes first.

        2026-07-30 perf fix (docs/superpowers/specs/2026-07-30-execution-
        dispatch-staleness-discard-design.md's "Part 1c", EXPLAIN-ANALYZE-
        verified live): fetches the whole candidate batch in ONE query
        (store.load_oldest_policy_frames_without_dispatch) instead of a while
        loop calling a LIMIT-1 lookup up to MAX_STALE_DISCARDS_PER_TICK (200)
        times. That old shape cost ~280-300ms PER CALL regardless of LIMIT
        size (Postgres builds the same full hash-joined result before
        sorting either way) -- 200 calls/tick was ~56s of pure query time,
        confirmed live as the real, dominant cause of the ~75s/tick cadence
        observed after this feature and the fresh-priority fallback both
        shipped (adversarially re-tested against stale planner statistics as
        an alternative explanation via a fresh ANALYZE; ruled out, plan and
        cost were unchanged). Fetching the same 200-row batch in one query
        costs ~300ms total, not 200x that -- a ~170x real reduction in
        per-tick SELECT cost, unlocking a tick cadence close to the intended
        ~2s poll interval instead of ~75s.

        Each discard frame saved here carries the CURRENT (not-yet-updated-
        for-this-tick) baseline forward, same as every other saved frame --
        the real update for this tick's own discard count is computed once,
        after this method returns, and re-stamped onto whichever frame ends
        up being the last one saved this tick (see _tick()).

        Returns (fresh_policy_frame_or_none, discarded_count, last_discard_
        frame_or_none).
        """
        batch = self._store.load_oldest_policy_frames_without_dispatch(MAX_STALE_DISCARDS_PER_TICK)
        discarded = 0
        last_discard_frame: ExecutionDispatchFrameV1 | None = None
        for candidate_frame in batch:
            age_sec = self._age_sec(candidate_frame)
            threshold_sec = self._staleness_threshold_sec()
            if age_sec <= threshold_sec:
                return candidate_frame, discarded, last_discard_frame
            discard_frame = build_stale_discard_execution_dispatch_frame(
                policy_frame=candidate_frame,
                policy_id=self._policy.policy_id,
                age_sec=age_sec,
                staleness_threshold_sec=threshold_sec,
            ).model_copy(update=baseline)
            self._store.save_dispatch_frame(discard_frame)
            last_discard_frame = discard_frame
            discarded += 1
            logger.info(
                "execution_dispatch_stale_discard policy_frame_id=%s age_sec=%.1f "
                "threshold_sec=%.1f discarded_this_tick=%d",
                candidate_frame.frame_id,
                age_sec,
                threshold_sec,
                discarded,
            )
        return None, discarded, last_discard_frame

    def _tick(self) -> None:
        if not self._settings.enable_execution_dispatch_runtime:
            return

        baseline = self._store.load_latest_staleness_discard_baseline() or {
            "staleness_discard_count_ewma": 0.0,
            "staleness_discard_count_ewma_var": 0.0,
            "staleness_discard_count_ewma_n": 0,
            "starvation_counts": {},
        }
        # 2026-08-12: starvation counters ride in the SAME carried-forward
        # dict as the EWMA baselines (loaded off the same row -- see
        # store.load_latest_staleness_discard_baseline), which is what puts
        # them on stale-discard frames too. .get() rather than [] because a
        # baseline persisted before this field existed is a real state here,
        # not a bug.
        prev_starvation_counts = baseline.get("starvation_counts") or {}
        policy_frame, discarded_this_tick, last_discard_frame = self._drain_stale_policy_frames(
            baseline
        )
        if policy_frame is None:
            # The FIFO drain didn't surface anything this tick (empty queue,
            # or the cap was hit before reaching a fresh frame) -- check
            # directly whether a genuinely current proposal exists anyway,
            # so real dispatch is never starved by backlog depth alone. See
            # _check_freshest_fallback's own docstring for the live incident
            # this closes (zero real dispatches for the first 6+ minutes
            # after the staleness-discard patch shipped, entirely consumed
            # discarding old backlog).
            policy_frame = self._check_freshest_fallback()

        update = compute_ewma_update(
            prev_ewma=baseline["staleness_discard_count_ewma"],
            prev_variance=baseline["staleness_discard_count_ewma_var"],
            prev_count=baseline["staleness_discard_count_ewma_n"],
            value=float(discarded_this_tick),
            alpha=STALENESS_DISCARD_EWMA_ALPHA,
            min_variance=STALENESS_DISCARD_EWMA_MIN_VARIANCE,
        )
        updated_baseline = {
            "staleness_discard_count_ewma": update.ewma,
            "staleness_discard_count_ewma_var": update.variance,
            "staleness_discard_count_ewma_n": baseline["staleness_discard_count_ewma_n"] + 1,
        }
        # 2026-08-12: the stamp for frames that DO NOT compute their own
        # starvation counters -- the stale-discard and unevaluable paths.
        # Only the real frame built by build_execution_dispatch_frame() gets
        # `updated_baseline` alone, because it produces a fresh map and
        # stamping the previous tick's map over it would freeze every counter.
        #
        # This distinction is the whole mechanism, and getting it wrong is
        # silent in the direction that keeps starved things starved: any saved
        # frame that lands with {} becomes the newest row, and the next tick's
        # load reads {} and erases every accumulated counter. Found by review
        # on exactly the unevaluable path below, which had been missed when
        # the sibling discard path one screen up was handled.
        carry_forward_baseline = {
            **updated_baseline,
            "starvation_counts": prev_starvation_counts,
        }

        if policy_frame is None:
            # Either the queue is fully drained, or this tick hit
            # MAX_STALE_DISCARDS_PER_TICK with real backlog still queued
            # behind it -- the next tick's load_latest_policy_frame_without_
            # dispatch() naturally resumes at the new FIFO head either way,
            # since every discard above was really saved. Nothing fresh to
            # build a real frame for this tick; if at least one discard
            # happened, this tick's real (not-yet-final at save time)
            # baseline update still needs a home -- re-stamp it onto that
            # same last discard frame (save_dispatch_frame's ON CONFLICT
            # upsert makes this idempotent, not a duplicate row).
            if last_discard_frame is not None:
                self._store.save_dispatch_frame(
                    last_discard_frame.model_copy(update=carry_forward_baseline)
                )
            return

        proposal = self._store.load_proposal_frame(policy_frame.source_proposal_frame_id)
        if proposal is None:
            # A naive `return` here would retry this exact same policy frame
            # forever -- it's always "the oldest undispatched" until a
            # dispatch frame exists for it, permanently blocking every policy
            # frame queued behind it. Record an honest "could not evaluate"
            # frame instead so the FIFO queue advances.
            logger.warning(
                "execution_dispatch_proposal_unavailable proposal_frame_id=%s",
                policy_frame.source_proposal_frame_id,
            )
            frame = build_unevaluable_execution_dispatch_frame(
                policy_frame=policy_frame,
                policy_id=self._policy.policy_id,
                reason=f"proposal_frame {policy_frame.source_proposal_frame_id} unavailable",
            ).model_copy(update=carry_forward_baseline)
            self._store.save_dispatch_frame(frame)
            logger.info(
                "execution_dispatch_frame_saved_unevaluable frame_id=%s policy_frame_id=%s",
                frame.frame_id,
                policy_frame.frame_id,
            )
            return

        # 2026-07-22 (SelfStateV1 burn): build_execution_dispatch_frame now
        # takes the field_tick_id straight off policy_frame -- no separate
        # self-state load, and so no "unevaluable" fallback for a load
        # failure that can no longer happen here.
        frame = build_execution_dispatch_frame(
            policy_frame=policy_frame,
            proposal_frame=proposal,
            field_tick_id=policy_frame.source_field_tick_id,
            policy=self._policy,
            override_dispatch_mode=self._settings.execution_dispatch_mode,
            prev_starvation_counts=prev_starvation_counts,
            # Current belief about what each action does to each signal, so
            # every candidate carries a magnitude taken from its own measured
            # history instead of a hand-typed constant. A read failure yields
            # {} -- cold priors, honestly flagged via ExpectedEffectV1.
            # cold_start -- and never stalls a dispatch tick.
            effect_posteriors=self._load_effect_posteriors(),
            # NOTE: updated_baseline deliberately does NOT carry
            # starvation_counts -- this frame computes its own fresh map and
            # model_copy would otherwise stamp the previous tick's map back
            # over it, freezing every counter at its first value.
        ).model_copy(update=updated_baseline)

        if (
            self._settings.execution_dispatch_mode == "dispatch_read_only"
            and self._policy.mode.allow_dispatch_read_only
        ):
            # _tick() runs inside asyncio.to_thread -- this thread has no
            # running event loop of its own, so asyncio.run() here is safe
            # and creates one just for this tick's bus RPCs.
            frame = asyncio.run(self._send_prepared_candidates(frame))

        self._store.save_dispatch_frame(frame)
        logger.info(
            "execution_dispatch_frame_saved frame_id=%s policy_frame_id=%s candidates=%d blocked=%d dispatched=%d",
            frame.frame_id,
            policy_frame.frame_id,
            len(frame.candidates),
            frame.blocked_count,
            frame.dispatch_count,
        )

    def _load_effect_posteriors(self) -> dict[TreatedCellKey, EffectPosterior]:
        try:
            return self._store.load_effect_posteriors()
        except Exception:
            logger.warning("execution_dispatch_effect_posterior_load_failed", exc_info=True)
            return {}

    def _log_allocator_preview(self, frame: ExecutionDispatchFrameV1, motor):
        """Allocate the remaining motor-seconds across this tick's candidates.

        Returns the Allocation so the caller can ACT on it when
        ORION_DISPATCH_ALLOCATOR_ENFORCE is set, or None when there was nothing
        to allocate. Still logs either way: the advisory readout is what the
        enforce flip was decided on, and it stays readable afterwards so a
        regression in the admitted set is visible without redeploying.
        """
        # frame.dispatched_candidates is always empty here -- see the note in
        # the log call below.
        pending = list(frame.candidates)
        if not pending:
            return None

        costs = self._store.load_action_cost_estimates()
        posteriors = self._load_effect_posteriors()

        candidates: list[Candidate] = []
        for item in pending:
            effect = item.expected_effect
            signal = effect.signal_id if effect is not None else None
            # Per-BIN variances, volume weighted. The next observation lands
            # in one baseline bin, so a pooled-across-bins figure is the wrong
            # prior -- and wrong in the direction that reads a well-travelled
            # action as better known than it is.
            cells = (
                [
                    (post.variance, post.n)
                    for key, post in posteriors.items()
                    if key[0] == item.dispatch_kind
                    and key[1] == item.target_id
                    and key[2] == signal
                    and post.n > 0
                ]
                if signal
                else []
            )
            candidates.append(
                candidate_from_dispatch(
                    dispatch_id=item.dispatch_id,
                    dispatch_kind=item.dispatch_kind,
                    target_id=item.target_id,
                    signal_id=signal,
                    claimed_direction=effect.direction if effect is not None else None,
                    cell_variances_by_volume=cells,
                    cost_sec=costs.get((item.dispatch_kind, item.target_id)),
                    cold_variance=DEFAULT_PRIOR_VARIANCE,
                )
            )

        allocation = allocate(
            candidates,
            allowance_sec=motor.remaining_sec,
            min_nats_per_sec=self._settings.orion_dispatch_min_nats_per_sec,
        )
        # `agree=` used to be logged here as if it compared the allocator's
        # choice against what actually dispatched. It could not: the admitted
        # set is a SUBSET of pending by construction, so agree == would_admit
        # on every line, and frame.dispatched_candidates is always empty at
        # this call site (execution_dispatch/builder.py never sets that
        # status; worker._send_prepared_candidates fills it later). It was a
        # tautology dressed as the metric the enforce flip gets decided on.
        # Removed rather than fixed here -- a real comparison has to happen
        # after the send loop, which is a separate change.
        logger.info(
            "motor_allocator_preview pending=%d would_admit=%d would_drop=%d "
            "nats=%.4f cost_sec=%.1f refusals=%s enforcing=%s",
            len(pending),
            len(allocation.admitted),
            len(pending) - len(allocation.admitted),
            allocation.admitted_nats,
            allocation.spent_sec,
            allocation.refusals_by_reason() or "{}",
            self._settings.orion_dispatch_allocator_enforce,
        )
        if pending and not allocation.admitted:
            # "Every candidate was refused" is a decision, not an idle tick, and
            # it must never be indistinguishable from "nothing was pending".
            # This is the state the roadmap's arsonist case warns about: a
            # value-of-information allocator whose action space is fully learned
            # correctly concludes that nothing is worth doing, and then looks
            # perfectly healthy doing nothing forever. Logged at WARNING so the
            # answer to "why is Orion idle" is one grep, and counted so the
            # duration is visible rather than inferred.
            self._all_refused_consecutive_ticks += 1
            logger.warning(
                "motor_allocator_refused_everything pending=%d refusals=%s "
                "consecutive_ticks=%d enforcing=%s -- Orion has nothing worth "
                "spending motor-seconds on; the fix is better actions, not a "
                "lower floor",
                len(pending),
                allocation.refusals_by_reason(),
                self._all_refused_consecutive_ticks,
                self._settings.orion_dispatch_allocator_enforce,
            )
        else:
            self._all_refused_consecutive_ticks = 0
        return allocation

    def _derive_motor_budget(self, frame_generated_at: datetime):
        """The real daily ceiling: motor-seconds spent against an operator-set
        allowance.

        Runs alongside _derive_daily_risk_cap rather than replacing it yet.
        The risk cap stays as the only live enforcement until this one has a
        day of would-refuse counts behind it -- removing the old ceiling while
        the new one is advisory would leave NO ceiling, which is worse than a
        useless one. The exit criterion is in settings.py; when this enforces,
        the risk cap goes, entirely (CLAUDE.md: kill means kill).
        """
        if frame_generated_at.tzinfo is None:
            frame_generated_at = frame_generated_at.replace(tzinfo=timezone.utc)
        else:
            frame_generated_at = frame_generated_at.astimezone(timezone.utc)
        day_start = datetime.combine(frame_generated_at.date(), time.min, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)

        try:
            spent = self._store.sum_motor_seconds_for_day(day_start, day_end)
        except Exception:
            # A budget that cannot read its own spend must not behave as
            # though nothing has been spent. The first version returned None
            # here with a comment saying exactly that -- and the caller's
            # `if motor is not None:` then skipped the whole block INCLUDING
            # the enforce branch, so a transient database error removed the
            # ceiling entirely. The comment asserted the opposite of what the
            # code did.
            #
            # Fail CLOSED when enforcing: an unknown spend is treated as a
            # spent allowance, so a broken meter stops dispatch rather than
            # uncapping it. Advisory keeps returning None, because there is
            # nothing to fail closed on and a fabricated "exhausted" line
            # would poison the very would-refuse counts the flip is decided
            # on.
            logger.warning("motor_budget_spend_read_failed", exc_info=True)
            if self._settings.orion_dispatch_motor_budget_enforce:
                allowance = self._settings.orion_dispatch_motor_budget_sec_per_day
                return budget_state(
                    allowance_sec=allowance,
                    spent_sec=allowance,
                    elapsed_fraction=day_elapsed_fraction(frame_generated_at, day_start),
                    enforcing=True,
                )
            return None

        return budget_state(
            allowance_sec=self._settings.orion_dispatch_motor_budget_sec_per_day,
            spent_sec=spent,
            elapsed_fraction=day_elapsed_fraction(frame_generated_at, day_start),
            enforcing=self._settings.orion_dispatch_motor_budget_enforce,
        )

    def _derive_daily_risk_cap(
        self, frame_generated_at: datetime
    ) -> tuple[float, dict[str, float | int | str | None]]:
        """Real, self-calibrating daily risk ceiling. Returns (effective_cap,
        baseline_fields) where baseline_fields is the (possibly updated)
        daily_risk_baseline_* state to stamp onto the outgoing frame -- every
        saved frame carries this forward (not just the ones that update it)
        so the next call's store.load_latest_daily_risk_baseline() always
        finds the latest state regardless of which tick's frame it reads.

        Feeds on store.sum_uncapped_risk_for_day's *uncapped* real demand,
        never on sum_risk_dispatched_today's actual spend -- see that
        method's own docstring for why: once this cap enforces a real
        ceiling, actual spend is right-censored at whatever the cap was that
        day, so it cannot report true demand back into the thing that sets
        the cap without recreating the exact "clamped value masks true
        magnitude" bug this patch exists to kill, one layer down.

        Day-rollover idempotency: `last_day` always tracks "the UTC calendar
        day as of which this baseline was last updated" -- after ANY update
        (cold-start seed or a real day-boundary rollover), `last_day` is set
        to `today`, the day of the *tick that performed the update*, not the
        day whose data was absorbed. This is deliberate, not a literal reading
        of "set last_day to the seeded/closed day": setting it to the
        absorbed day instead would make the very next tick (same real day,
        ticks run every couple seconds) see `today != last_day` again and
        re-absorb the SAME day's data a second time -- a real double-count,
        not merely a redundant no-op. Setting `last_day = today` after every
        update means the rollover branch only fires again once a tick's
        `frame_generated_at` genuinely lands on a later calendar day than the
        last update, which is the actual "did a day boundary pass" question.
        One consequence: if this service is down across more than one full
        UTC day boundary, only the single most-recently-closed day (the one
        named by `last_day`) gets absorbed on the catch-up tick -- days
        further back are skipped, never double-counted or reconstructed.
        That is a deliberate, safe-direction trade: under-sampling a rare
        multi-day-outage gap is far cheaper than silently corrupting the
        baseline with a duplicated sample.
        """
        if frame_generated_at.tzinfo is None:
            frame_generated_at = frame_generated_at.replace(tzinfo=timezone.utc)
        else:
            frame_generated_at = frame_generated_at.astimezone(timezone.utc)
        today: date = frame_generated_at.date()
        today_str = today.isoformat()
        today_start = datetime.combine(today, time.min, tzinfo=timezone.utc)

        baseline = self._store.load_latest_daily_risk_baseline()
        if baseline is None:
            ewma, variance, n, last_day = 0.0, 0.0, 0, None
        else:
            ewma = float(baseline.get("daily_risk_baseline_ewma") or 0.0)
            variance = float(baseline.get("daily_risk_baseline_ewma_var") or 0.0)
            n = int(baseline.get("daily_risk_baseline_ewma_n") or 0)
            last_day = baseline.get("daily_risk_baseline_last_day")

        if n == 0 and last_day is None:
            seed = self._store.most_recent_closed_day_with_data(today_start)
            if seed is not None:
                _seed_day, seed_total = seed
                update = compute_ewma_update(
                    prev_ewma=0.0,
                    prev_variance=0.0,
                    prev_count=0,
                    value=seed_total,
                    alpha=DAILY_RISK_BASELINE_ALPHA,
                    min_variance=DAILY_RISK_BASELINE_MIN_VARIANCE,
                )
                ewma, variance, n, last_day = update.ewma, update.variance, 1, today_str
            # else: no historical data at all -- carry forward n==0, fall
            # through to the static settings fallback below.
        elif last_day is not None and last_day != today_str:
            last_day_date = date.fromisoformat(last_day)
            last_day_start = datetime.combine(last_day_date, time.min, tzinfo=timezone.utc)
            last_day_end = last_day_start + timedelta(days=1)
            closed_day_total = self._store.sum_uncapped_risk_for_day(last_day_start, last_day_end)
            update = compute_ewma_update(
                prev_ewma=ewma,
                prev_variance=variance,
                prev_count=n,
                value=closed_day_total,
                alpha=DAILY_RISK_BASELINE_ALPHA,
                min_variance=DAILY_RISK_BASELINE_MIN_VARIANCE,
            )
            ewma, variance, n, last_day = update.ewma, update.variance, n + 1, today_str
        # else: same UTC day as the last update (or n==0 with no seed
        # available yet) -- carry forward unchanged.

        if n >= DAILY_RISK_BASELINE_MIN_SAMPLES:
            cap = ewma + DAILY_RISK_BASELINE_Z * math.sqrt(max(variance, DAILY_RISK_BASELINE_MIN_VARIANCE))
        elif n >= 1:
            # Same "2x one observed day" interim-margin convention
            # settings.py's old comment used, now correctly anchored on a
            # real *uncapped* day (817.65) instead of a clamped one (the old
            # enforced cap itself, which is what the pre-2026-07-29 default
            # was actually ~2x of).
            cap = ewma * 2.0
        else:
            # n==0, no seed available -- truly no history anywhere. Should
            # never trigger against this repo's real data (2026-07-28 already
            # has real closed-day history); last-resort fallback only.
            cap = self._settings.orion_dispatch_max_risk_per_day

        baseline_fields: dict[str, float | int | str | None] = {
            "daily_risk_baseline_ewma": ewma,
            "daily_risk_baseline_ewma_var": variance,
            "daily_risk_baseline_ewma_n": n,
            "daily_risk_baseline_last_day": last_day,
        }
        return cap, baseline_fields

    async def _send_prepared_candidates(
        self, frame: ExecutionDispatchFrameV1
    ) -> ExecutionDispatchFrameV1:
        derived_cap, baseline_fields = self._derive_daily_risk_cap(frame.generated_at)

        # Default: no cap beyond the policy's own per-tick burst limit. A probe
        # tick narrows this to TRIPWIRE_PROBE_DISPATCHES.
        send_budget = self._policy.limits.max_dispatches_per_tick
        self._tripwire_probe_in_flight = False
        # Fresh every tick: a probe is judged only on what this tick produced.
        self._tick_dispatch_statuses = []

        if self._check_theater_tripwire():
            # Tripped (this tick or a prior one). Either this tick is a probe --
            # one deliberate action, to find out whether the motor works again --
            # or it sends nothing and SAYS SO, on the frame and in the log.
            #
            # Claim only when something could actually go out. A claim on a tick
            # with nothing preparable is refunded a few lines later, and the
            # refund makes the next tick immediately claimable again -- so the
            # `probe_slots <= 0` branch below would never be reached, and with it
            # neither the frame warning, the throttled log, nor the hourly
            # escalation. Measured on the real code before this guard: 6
            # simulated hours with nothing preparable produced 10,650 claim/
            # refund cycles, 150 frame warnings (all in the first 5 minutes) and
            # ZERO re-notifications -- defect 3 reproduced inside its own fix.
            has_preparable = any(
                c.dispatch_status == "prepared_for_dispatch" for c in frame.candidates
            )
            probe_slots = self._claim_tripwire_probe_slots() if has_preparable else 0
            if probe_slots <= 0:
                return frame.model_copy(
                    update={
                        **baseline_fields,
                        "warnings": self._record_tripwire_blocked_tick(frame),
                    }
                )
            send_budget = min(send_budget, probe_slots)
            self._tripwire_probe_in_flight = True
            logger.warning(
                "execution_dispatch_tripwire_probe attempt=%d slots=%d "
                "consecutive_successes=%d/%d tripped_for_sec=%.0f",
                self._tripwire_probe_attempts,
                send_budget,
                self._tripwire_probe_successes,
                self._settings.orion_dispatch_tripwire_rearm_successes,
                self._tripwire_tripped_for_sec(),
            )

        # THE REAL BUDGET, advisory for now. Denominated in motor-seconds --
        # wall-clock an action actually occupies -- against an operator-set
        # allowance, rather than in risk_score against an EWMA of Orion's own
        # past demand. Reported every tick with what it WOULD have refused,
        # because an advisory budget whose only output is "still fine" is the
        # switch-that-changes-nothing this repo bans; the would-refuse count is
        # the evidence the enforce flip gets decided on.
        motor = self._derive_motor_budget(frame.generated_at)
        if motor is not None:
            pending = list(frame.candidates)
            # The flat-p50 would_refuse projection that used to live here is
            # retired: the allocator three lines below computes the same thing
            # from each action's OWN measured cost, and two estimates that can
            # disagree, logged adjacently, is the defect this repo's contract
            # names directly ("when a metric is replaced by a better one,
            # retire the old one completely").
            logger.info(
                "motor_budget mode=%s spent_sec=%.1f allowance_sec=%.1f "
                "remaining_sec=%.1f pace=%.2fx projected_day_h=%.1f pending=%d",
                motor.mode,
                motor.spent_sec,
                motor.allowance_sec,
                motor.remaining_sec,
                motor.pace,
                motor.projected_day_sec / 3600.0,
                len(pending),
            )
            # STEP 3, ADVISORY: what would an allocator have chosen?
            # Scored on expected information per motor-second -- the epistemic
            # term of expected free energy, which is the only one that
            # discriminates when every action's measured effect sits inside its
            # own error bar. Pragmatic value is a GATE (confidently harmful ->
            # refused), never converted into the score, because inventing that
            # exchange rate is how risk_score happened.
            try:
                allocation = self._log_allocator_preview(frame, motor)
            except Exception:
                # A readout must never be able to stall a real dispatch tick.
                # When the allocator is ENFORCING a failure here means there is
                # no allocation to enforce -- leave it None and fall through to
                # the unallocated path rather than synthesising an empty
                # allocation, which would refuse everything on a transient error.
                logger.warning("motor_allocator_preview_failed", exc_info=True)
                allocation = None

            if motor.mode == "enforcing" and motor.exhausted:
                logger.info("motor_budget_exhausted sending nothing this tick")
                return self._abandon_tick_without_sending(frame, baseline_fields)

        spent_today = self._store.sum_risk_dispatched_today()
        remaining_risk_budget = derived_cap - spent_today
        advisory_only = self._settings.orion_dispatch_risk_cap_advisory_only
        enforced = not advisory_only
        cap_reached = remaining_risk_budget <= 0
        if cap_reached:
            logger.info(
                "execution_dispatch_risk_budget_status spent_today=%.4f derived_max_risk_per_day=%.4f "
                "baseline_n=%d cap_reached=true advisory_only=%s enforced=%s",
                spent_today,
                derived_cap,
                baseline_fields["daily_risk_baseline_ewma_n"],
                advisory_only,
                enforced,
            )
            if enforced:
                return self._abandon_tick_without_sending(frame, baseline_fields)
            # Advisory-only (operator override, no longer the default as of
            # 2026-07-29 -- see settings.py's own comment): log that it would
            # have blocked, but don't actually withhold real dispatches.
            # max_dispatches_per_tick (a real, independent, already-enforced
            # per-tick burst limit) is untouched by this and still applies.
            remaining_risk_budget = float("inf")

        # Real, risk-weighted selection -- not a blind count. `frame.candidates`
        # arrives ordered by builder._admit_candidates' EFFECTIVE priority
        # (2026-08-12: real priority plus the starvation aging bonus), not by
        # build_proposal_frame's raw priority order as this comment used to
        # say -- aging can reorder it, and the sequential take below depends
        # on that order being meaningful. So this takes
        # the highest-priority prepared candidates whose cumulative
        # risk_score fits inside what's left of today's real budget (or all
        # of them, if the cap is advisory and already reached -- see above),
        # stopping at the first one that would exceed it (no skip-ahead to
        # squeeze in a smaller one later, matching the existing simple
        # sequential-take style this replaces). max_dispatches_per_tick
        # stays a separate, independent per-tick burst cap.
        # ALLOCATOR ENFORCEMENT. When on, a candidate the allocator refused is
        # not sent, however high its priority. This is the difference between a
        # ceiling and a choice: the risk cap below can only say "not any more
        # today", while the allocator says "not this one, that one is worth
        # more per second". Priority order still decides ties, because the
        # allocator already sorted by nats-per-second before this point.
        #
        # Refusal here is a real outcome, not a skip: an action Orion declined
        # because it had nothing left to learn from it is exactly what a budget
        # is for. `allocation is None` (no motor budget, or the allocator threw)
        # leaves every candidate eligible -- an absent allocation must never be
        # read as a refusal of everything.
        allocator_admitted_ids: set[str] | None = None
        if self._settings.orion_dispatch_allocator_enforce and allocation is not None:
            allocator_admitted_ids = {c.dispatch_id for c in allocation.admitted}

        to_send: list[ExecutionDispatchCandidateV1] = []
        cumulative_risk = 0.0
        allocator_skipped = 0
        for candidate in frame.candidates:
            if candidate.dispatch_status != "prepared_for_dispatch":
                continue
            if allocator_admitted_ids is not None and candidate.dispatch_id not in allocator_admitted_ids:
                allocator_skipped += 1
                continue
            # send_budget is max_dispatches_per_tick on an ordinary tick, and
            # TRIPWIRE_PROBE_DISPATCHES on a probe tick.
            if len(to_send) >= send_budget:
                break
            # max(..., MINIMUM_REAL_RISK_FLOOR): a real risk_score=0.0 must
            # still cost something against the daily budget, or the budget
            # provides no ceiling at all for that candidate (see the
            # constant's own docstring above).
            spend = max(candidate.risk_score, MINIMUM_REAL_RISK_FLOOR)
            if cumulative_risk + spend > remaining_risk_budget:
                break
            to_send.append(candidate)
            cumulative_risk += spend

        if allocator_skipped:
            logger.info(
                "motor_allocator_enforced skipped=%d sending=%d",
                allocator_skipped,
                len(to_send),
            )

        if not to_send:
            return self._abandon_tick_without_sending(frame, baseline_fields)

        # RANDOMIZED HOLDBACK. Rolled once per TICK, over candidates that have
        # already passed every gate -- so the withheld set is a random sample
        # of what would genuinely have run, which is what makes the resulting
        # arm causal rather than merely matched.
        #
        # Per tick, not per candidate: the field delta is frame-wide, so
        # withholding one candidate while its siblings run yields a control
        # observation contaminated by those siblings. That is precisely the
        # defect that made the capacity-blocked arm unusable, and repeating it
        # here would produce a worse-than-useless arm wearing the word
        # "randomized".
        #
        # A probe tick is never held back. The holdback exists to buy a causal
        # control arm out of ticks that would otherwise have acted; withholding
        # the single action being used to test whether the motor still works
        # would spend a probe to learn nothing and re-arm strictly slower. The
        # arm loses nothing real either -- probe ticks are a deliberately
        # non-random subset of ticks (they only occur while tripped), so they
        # were never eligible to be a uniform sample of ordinary acting ticks.
        holdback_fraction = (
            0.0
            if self._tripwire_probe_in_flight
            else self._settings.orion_dispatch_holdback_fraction
        )
        if holdback_fraction > 0.0 and random.random() < holdback_fraction:
            withheld_ids = {c.dispatch_id for c in to_send}
            withheld = [
                c.model_copy(
                    update={
                        "dispatch_status": "blocked",
                        "blocked_by": [HOLDBACK_BLOCK_REASON],
                        "reasons": list(c.reasons) + [HOLDBACK_BLOCK_REASON],
                    }
                )
                for c in to_send
            ]
            logger.info(
                "execution_dispatch_randomized_holdback withheld=%d fraction=%.3f "
                "-- this tick deliberately does nothing, to buy a causal control arm",
                len(withheld),
                holdback_fraction,
            )
            return frame.model_copy(
                update={
                    **baseline_fields,
                    "candidates": [
                        c for c in frame.candidates if c.dispatch_id not in withheld_ids
                    ],
                    "blocked_candidates": list(frame.blocked_candidates) + withheld,
                    "blocked_count": len(frame.blocked_candidates) + len(withheld),
                }
            )

        sent_ids = {c.dispatch_id for c in to_send}
        remaining_candidates = [c for c in frame.candidates if c.dispatch_id not in sent_ids]

        bus = OrionBusAsync(
            url=self._settings.orion_bus_url, enabled=self._settings.orion_bus_enabled
        )
        await bus.connect()
        client = ExecutionDispatchCortexClient(
            bus,
            request_channel=self._settings.cortex_exec_channel,
            result_prefix=self._settings.cortex_exec_result_prefix,
            timeout_sec=self._settings.execution_dispatch_rpc_timeout_sec,
        )
        # 2026-07-31 (docs/superpowers/specs/2026-07-30-execution-dispatch-
        # staleness-discard-design.md's Part 2): concurrent, not sequential.
        # return_exceptions=True is not optional -- verified empirically,
        # not assumed (missing question 7, two adversarial passes): bare
        # gather() lets one candidate's failure cancel its siblings
        # mid-flight via asyncio.run()'s own shutdown path (this is called
        # from _tick() via asyncio.run(self._send_prepared_candidates(...))),
        # which could cancel a candidate AFTER its real cortex-exec RPC
        # already succeeded but BEFORE save_dispatch_result records it --
        # the exact double-send the idempotency guard above exists to
        # prevent. return_exceptions=True keeps every candidate running to
        # real completion regardless of a batch-mate's failure. _send_one
        # itself (not _send_one_inner) is also now a total function that
        # cannot raise, so in practice no exception object should ever
        # appear in this results list -- return_exceptions=True remains the
        # correct belt-and-suspenders choice regardless, since it removes
        # the cross-candidate cancellation risk structurally rather than
        # relying solely on _send_one never having a bug.
        try:
            newly_dispatched = await asyncio.gather(
                *(self._send_one(client, bus, frame, candidate) for candidate in to_send),
                return_exceptions=True,
            )
        finally:
            await bus.close()

        dispatched_candidates = list(frame.dispatched_candidates) + newly_dispatched
        updated = frame.model_copy(
            update={
                "candidates": remaining_candidates,
                "dispatched_candidates": dispatched_candidates,
                "dispatch_count": len(dispatched_candidates),
                "dispatch_attempted": True,
                **baseline_fields,
            }
        )

        # Re-check after this tick's real sends appended to the in-process
        # window -- lets the tripwire fire the same tick it actually crosses
        # the threshold, not one tick late.
        #
        # Order matters: evaluate the probe FIRST, so a re-arm decided by this
        # tick's probe is applied before the predicate below gets to look at the
        # window. _clear_tripwire empties that window as part of re-arming (see
        # its own comment for why the alternative is a no-op), so the predicate
        # then correctly reads "not enough samples yet" rather than re-tripping
        # on the failures the probes just disproved.
        if self._tripwire_probe_in_flight:
            self._evaluate_tripwire_probe()
        self._check_theater_tripwire()
        return updated

    def _monotonic(self) -> float:
        """Seam for tests. Production always uses the real monotonic clock."""
        return monotonic()

    def _record_dispatch_status(self, status: str, *, live: bool = True) -> None:
        """The single choke point for every recorded dispatch outcome.

        Both the tripwire's trailing window and the current tick's own list are
        written here. They were separate appends at four call sites before
        2026-08-25; a fifth site added later that updated only one of them
        would have produced a probe that could never succeed, or a tripwire
        that could never fire, with nothing failing loudly to say so.

        `live=False` marks a status that did NOT come from contacting the motor
        this tick -- today only the idempotency replay branch, which reads a
        stored result. Such a status still counts toward the trailing window
        (it is real information about a real past attempt) but must never count
        as probe evidence, since a probe's whole job is to establish that the
        motor responds NOW.
        """
        self._recent_dispatch_statuses.append(status)
        if live:
            self._tick_dispatch_statuses.append(status)

    def _tripwire_tripped_for_sec(self) -> float:
        if self._tripwire_tripped_at is None:
            return 0.0
        return max(0.0, self._monotonic() - self._tripwire_tripped_at)

    def _claim_tripwire_probe_slots(self) -> int:
        """How many candidates this tick may send as a recovery probe.

        Claiming is not free: it consumes the current cooldown and counts an
        attempt, so a caller that ends up sending nothing must call
        _refund_tripwire_probe. Returns 0 whenever probing is disabled or the
        backoff has not yet elapsed.
        """
        if not self._settings.orion_dispatch_tripwire_probe_enabled:
            return 0
        now = self._monotonic()
        if self._tripwire_next_probe_at is None:
            # Tripped by a path that did not initialise the schedule (a latch
            # set before this patch's state existed, or a direct test poke).
            # Start the clock now rather than probing instantly.
            self._tripwire_next_probe_at = now + self._tripwire_probe_cooldown_sec
            return 0
        if now < self._tripwire_next_probe_at:
            return 0
        self._tripwire_probe_attempts += 1
        # Schedule the next attempt up front. If this probe succeeds and
        # re-arms, the schedule is discarded by _clear_tripwire anyway; if it
        # fails, _evaluate_tripwire_probe widens the backoff from here.
        self._tripwire_next_probe_at = now + self._tripwire_probe_cooldown_sec
        return TRIPWIRE_PROBE_DISPATCHES

    def _abandon_tick_without_sending(
        self, frame: ExecutionDispatchFrameV1, baseline_fields: dict
    ) -> ExecutionDispatchFrameV1:
        """Single exit for every gate that decides to send nothing after the
        tripwire check has already run.

        There are three such gates (motor budget exhausted, daily risk cap
        reached-and-enforced, and nothing surviving the risk take-loop) and they
        all sit BELOW the probe claim. Review found each one silently consuming
        a probe: the attempt counter climbed, the cooldown was consumed, no
        action went out, no evidence came back, and -- because the blocked-tick
        branch is above them -- no frame warning was written either. The
        "after N recovery probe(s)" line in the escalation then reported probes
        that never happened. Routing all three through here keeps the accounting
        honest and the tick visible.
        """
        if self._tripwire_probe_in_flight:
            self._refund_tripwire_probe()
        update = dict(baseline_fields)
        if self.theater_tripwire_active:
            update["warnings"] = self._record_tripwire_blocked_tick(frame)
        return frame.model_copy(update=update)

    def _refund_tripwire_probe(self) -> None:
        """Undo a claim whose tick sent nothing, so no evidence was gathered.

        Deliberately restores immediate claimability: no action was spent, so no
        backoff is owed. That is only safe because the claim site refuses to
        claim at all unless something is preparable -- without that guard this
        line turns into a per-tick claim/refund spin that starves the blocked-
        tick warning entirely (see the claim site's own comment).

        NOT the right response to a probe that sent a real action and learned
        nothing; that path applies the backoff instead. See
        _evaluate_tripwire_probe.
        """
        self._tripwire_probe_in_flight = False
        self._tripwire_probe_attempts = max(0, self._tripwire_probe_attempts - 1)
        self._tripwire_next_probe_at = self._monotonic()

    def _evaluate_tripwire_probe(self) -> None:
        """Judge this tick's probe and either re-arm dispatch or back off.

        Judged ONLY on statuses recorded during this tick. A probe that somehow
        recorded no status at all is treated as no evidence -- not as a success
        -- because "nothing came back" is the exact condition the tripwire
        exists to catch, and counting it toward re-arming would let a fully
        dead motor talk its way back into service.
        """
        statuses = list(self._tick_dispatch_statuses)
        self._tripwire_probe_in_flight = False
        required = self._settings.orion_dispatch_tripwire_rearm_successes

        if not statuses:
            # A real action WAS sent (this method only runs on a tick that got
            # past every gate and into the send loop); we simply never learned
            # what happened to it -- _send_one swallows any exception from
            # _send_one_inner without recording a status, and
            # save_dispatch_result runs AFTER the cortex-exec RPC has already
            # fired. Refunding here was the first version's mistake: with
            # Postgres unavailable, every recording path raises, every probe
            # comes back inconclusive, every refund makes the next tick
            # immediately claimable, and the 2.0s poll interval turns the
            # documented "one probe per hour" into ~1,800 real dispatches an
            # hour aimed at a dependency that is already failing.
            #
            # So: treat it as a failed probe. No re-arm credit, and the backoff
            # widens. "We spent an action and learned nothing" must cost the
            # same as "we spent an action and it failed" -- it is strictly less
            # informative, and it is the shape a genuinely dead motor produces.
            self._penalise_failed_probe()
            logger.warning(
                "execution_dispatch_tripwire_probe_inconclusive attempt=%d "
                "next_probe_in_sec=%.0f -- an action was sent but no outcome was "
                "recorded; counting it as a failed probe",
                self._tripwire_probe_attempts,
                self._tripwire_probe_cooldown_sec,
            )
            return

        if all(s == "success" for s in statuses):
            self._tripwire_probe_successes += 1
            if self._tripwire_probe_successes >= required:
                self._clear_tripwire()
                return
            logger.warning(
                "execution_dispatch_tripwire_probe_ok %d/%d consecutive -- "
                "still paused until the run completes",
                self._tripwire_probe_successes,
                required,
            )
            return

        self._penalise_failed_probe()
        logger.warning(
            "execution_dispatch_tripwire_probe_failed statuses=%s next_probe_in_sec=%.0f "
            "tripped_for_sec=%.0f",
            ",".join(statuses),
            self._tripwire_probe_cooldown_sec,
            self._tripwire_tripped_for_sec(),
        )

    def _penalise_failed_probe(self) -> None:
        """Reset the re-arm run and widen the backoff.

        The reset is the line that keeps the original design's objection
        answered: a coincidentally-good sample cannot accumulate toward
        re-arming across a flapping motor.
        """
        self._tripwire_probe_successes = 0
        self._tripwire_probe_cooldown_sec = min(
            self._tripwire_probe_cooldown_sec * TRIPWIRE_PROBE_BACKOFF_FACTOR,
            self._settings.orion_dispatch_tripwire_probe_max_cooldown_sec,
        )
        self._tripwire_next_probe_at = self._monotonic() + self._tripwire_probe_cooldown_sec

    def _clear_tripwire(self) -> None:
        tripped_for = self._tripwire_tripped_for_sec()
        attempts = self._tripwire_probe_attempts
        self.theater_tripwire_active = False
        self._tripwire_tripped_at = None
        self._tripwire_next_probe_at = None
        self._tripwire_probe_cooldown_sec = (
            self._settings.orion_dispatch_tripwire_probe_cooldown_sec
        )
        self._tripwire_probe_successes = 0
        self._tripwire_probe_attempts = 0
        self._tripwire_last_blocked_log_at = None
        self._tripwire_last_notify_at = None
        # The trailing window MUST be cleared here, and the first version of
        # this patch got it wrong.
        #
        # It originally left the window alone, reasoning that it would "age out
        # naturally" and that clearing it would blind the tripwire to a motor
        # still mostly failing. Review caught that this made the entire re-arm a
        # no-op: at the moment of clearing, the window still holds the >= 6
        # failures that caused the trip, so the very next _check_theater_
        # tripwire call re-trips instantly. Reproduced against the real code --
        # the default 3-probe run cleared and re-tripped twice before sticking,
        # firing a false "Execution dispatch resumed" notification each time and
        # resetting _tripwire_tripped_at, which would have pinned the "Orion has
        # not acted for Nh" escalation permanently under an hour on a flapping
        # motor. Clearing needs up to 5 appended successes to drag the window
        # under threshold; the default re-arm run is 3.
        #
        # Clearing is also the honest reading: after a run of consecutive
        # successful probes, the pre-trip failures are a resolved incident, not
        # recent evidence. The cost is a real but small blind spot -- the
        # predicate needs THEATER_TRIPWIRE_WINDOW fresh samples before it can
        # trip again (len(recent) < WINDOW returns early), roughly two ticks at
        # max_dispatches_per_tick=5. A motor that passes three spaced probes and
        # then fails re-trips inside that handful of dispatches.
        self._recent_dispatch_statuses.clear()
        logger.warning(
            "execution_dispatch_theater_tripwire_cleared tripped_for_sec=%.0f probes=%d "
            "-- dispatch resuming",
            tripped_for,
            attempts,
        )
        self._notify_tripwire_cleared(tripped_for, attempts)

    def _record_tripwire_blocked_tick(self, frame: ExecutionDispatchFrameV1) -> list[str]:
        """Record, log and escalate the fact that this tick sent nothing.

        Defect 3 of the 2026-08-23 incident: for 45 hours there was no per-tick
        evidence anywhere that Orion had been stopped. The returned warning
        rides on the frame, which is persisted, so the question "could Orion act
        at time T" is answerable from Postgres after the fact -- including after
        the restart that clears the in-process latch.
        """
        now = self._monotonic()
        tripped_for = self._tripwire_tripped_for_sec()
        # Rendered as a word, not a magic negative. This string is persisted on
        # the frame and someone will eventually parse it for "how long until
        # recovery"; "unscheduled" fails loudly there, "-1" quietly becomes a
        # negative duration. Reachable whenever probing is disabled.
        if self._tripwire_next_probe_at is None:
            next_probe_in: float | None = None
        else:
            next_probe_in = max(0.0, self._tripwire_next_probe_at - now)
        next_probe_str = "unscheduled" if next_probe_in is None else f"{next_probe_in:.0f}"

        if (
            self._tripwire_last_blocked_log_at is None
            or now - self._tripwire_last_blocked_log_at >= TRIPWIRE_BLOCKED_LOG_INTERVAL_SEC
        ):
            self._tripwire_last_blocked_log_at = now
            logger.warning(
                "execution_dispatch_theater_tripwire_blocking tripped_for_sec=%.0f "
                "candidates_withheld=%d probes=%d next_probe_in_sec=%s probe_enabled=%s",
                tripped_for,
                len(frame.candidates),
                self._tripwire_probe_attempts,
                next_probe_str,
                self._settings.orion_dispatch_tripwire_probe_enabled,
            )

        if (
            self._tripwire_last_notify_at is None
            or now - self._tripwire_last_notify_at >= TRIPWIRE_RENOTIFY_INTERVAL_SEC
        ):
            # Skip the immediate re-notify on the very first blocked tick --
            # _notify_tripwire already fired at the moment of the trip.
            if self._tripwire_last_notify_at is not None:
                self._notify_tripwire_still_blocked(tripped_for)
            self._tripwire_last_notify_at = now

        return list(frame.warnings) + [
            f"{TRIPWIRE_BLOCKED_WARNING} tripped_for_sec={tripped_for:.0f} "
            f"candidates_withheld={len(frame.candidates)} "
            f"probes={self._tripwire_probe_attempts} "
            f"next_probe_in_sec={next_probe_str}"
        ]

    def _check_theater_tripwire(self) -> bool:
        recent = list(self._recent_dispatch_statuses)
        if len(recent) < THEATER_TRIPWIRE_WINDOW:
            return self.theater_tripwire_active
        # Counts every non-productive result, not just literal "empty".
        #
        # Before the structured-result patch, a `skills.runtime.*` verb that
        # failed was stored as "empty" and therefore counted here. After it,
        # the same failure is stored honestly as "failed" -- which, under the
        # old `s == "empty"` predicate, meant a maintenance verb failing 10
        # times in a row could NEVER trip the one mechanism built to catch a
        # dead motor nerve. This restores that coverage.
        #
        # It also newly counts RPC send failures, which the old predicate
        # ignored even though they were already being appended to this deque.
        # That widening is intended: if half the trailing sends produce no
        # outcome, pausing dispatch is the right response whether the cause is
        # an empty observation or a transport error. Live trip risk measured
        # before shipping, not assumed: 30 failed against 12,782 success over
        # 24h (0.23%) versus a threshold of >50% of the trailing 10.
        unproductive_count = sum(1 for s in recent if s != "success")
        newly_tripped = (
            not self.theater_tripwire_active
            and unproductive_count > THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_UNPRODUCTIVE_THRESHOLD
        )
        if newly_tripped:
            self.theater_tripwire_active = True
            # Start the recovery clock at the moment of the trip. Without this
            # the latch has no schedule and _claim_tripwire_probe_slots has to
            # fall back to seeding one on its first blocked tick.
            now = self._monotonic()
            self._tripwire_tripped_at = now
            self._tripwire_probe_cooldown_sec = (
                self._settings.orion_dispatch_tripwire_probe_cooldown_sec
            )
            self._tripwire_next_probe_at = now + self._tripwire_probe_cooldown_sec
            self._tripwire_probe_successes = 0
            self._tripwire_probe_attempts = 0
            self._tripwire_last_blocked_log_at = None
            self._tripwire_last_notify_at = now
            logger.warning(
                "execution_dispatch_theater_tripwire_active unproductive=%d window=%d",
                unproductive_count,
                len(recent),
            )
            self._notify_tripwire(unproductive_count, len(recent))
        return self.theater_tripwire_active

    async def _send_one(
        self,
        client: ExecutionDispatchCortexClient,
        bus: OrionBusAsync,
        frame: ExecutionDispatchFrameV1,
        candidate: ExecutionDispatchCandidateV1,
    ) -> ExecutionDispatchCandidateV1:
        """Total wrapper around _send_one_inner -- must never raise.

        2026-07-31 (docs/superpowers/specs/2026-07-30-execution-dispatch-
        staleness-discard-design.md's Part 2, missing question 7, verified
        empirically across two adversarial passes before this shipped):
        _send_prepared_candidates now runs concurrent sends via asyncio.
        gather(..., return_exceptions=True). That flag is the real fix for
        the cross-candidate cancellation risk (a sibling's failure can no
        longer cancel this candidate mid-flight, confirmed by direct test,
        not assumed) -- but return_exceptions=True only prevents gather()
        itself from raising; it does nothing to stop a raw exception object
        from landing in the results list if _send_one_inner ever raises for
        its OWN reasons (not a sibling's). ExecutionDispatchFrameV1.
        dispatched_candidates: list[ExecutionDispatchCandidateV1] cannot
        hold a raw exception -- Pydantic would reject the whole frame. This
        wrapper is what keeps that guarantee: every candidate that goes in
        comes back out as a real ExecutionDispatchCandidateV1, recording a
        real dispatch_error if anything unexpected happened, never a bare
        exception escaping upward.
        """
        try:
            return await self._send_one_inner(client, bus, frame, candidate)
        except Exception as exc:
            logger.warning(
                "execution_dispatch_send_one_unexpected_failure dispatch_id=%s error=%s",
                candidate.dispatch_id,
                exc,
                exc_info=True,
            )
            return candidate.model_copy(
                update={
                    "dispatch_status": "dispatched",
                    "dispatched_at": datetime.now(timezone.utc),
                    "dispatch_error": f"unexpected _send_one failure: {str(exc)[:450]}",
                }
            )

    async def _send_one_inner(
        self,
        client: ExecutionDispatchCortexClient,
        bus: OrionBusAsync,
        frame: ExecutionDispatchFrameV1,
        candidate: ExecutionDispatchCandidateV1,
    ) -> ExecutionDispatchCandidateV1:
        now = datetime.now(timezone.utc)
        result_id = f"result:{candidate.dispatch_id}"

        # Idempotency guard: dispatch_id is deterministic (stable_dispatch_id
        # hashes proposal_id+policy_id), so if this process crashed after a
        # prior successful send but before save_dispatch_frame() recorded
        # that this policy frame is now dispatched, the next tick re-selects
        # the same policy frame and rebuilds the identical dispatch_id. A
        # real cortex-exec RPC must never fire twice for the same candidate
        # -- replay the stored result instead of resending.
        existing = self._store.load_dispatch_result_by_dispatch_id(candidate.dispatch_id)
        if existing is not None:
            logger.info(
                "execution_dispatch_result_replayed dispatch_id=%s status=%s",
                candidate.dispatch_id,
                existing["status"],
            )
            # Real information about a real attempt even on replay -- counts
            # toward this process's own tripwire window the same as a fresh
            # send would.
            #
            # But NOT toward a recovery probe. This branch reads a stored result
            # and makes no contact with cortex-exec at all, so a stale
            # "success" here would let a motor that was never touched vote for
            # its own re-arm -- exactly what _evaluate_tripwire_probe's docstring
            # says must not happen. Narrow (it needs a crash between
            # save_dispatch_result and save_dispatch_frame) but real, and a
            # probe is precisely the tick where being wrong matters most.
            self._record_dispatch_status(existing["status"], live=False)
            # Re-emit on replay too: action_outcomes.action_id is the SQL
            # primary key and sql-writer's route upserts by merge(), so a
            # repeat emit for the same dispatch_id idempotently overwrites
            # the same row -- it does not duplicate. Skipping the emit here
            # would risk permanently losing it instead: if the process died
            # between save_dispatch_result (above, on a prior attempt) and
            # the emit, or the emit itself failed transiently, no later tick
            # would ever retry it, since every later tick also hits this
            # replay branch.
            if existing["status"] == "failed":
                error = existing["result_json"].get("error", "previous attempt failed")
                await self._emit_action_outcome(
                    bus,
                    candidate=candidate,
                    summary=f"attempted {candidate.dispatch_kind} on {candidate.target_id}, send failed",
                    success=False,
                    observed_at=now,
                )
                return candidate.model_copy(
                    update={
                        "dispatch_status": "dispatched",
                        "dispatched_at": now,
                        "dispatch_error": str(error)[:500],
                    }
                )
            existing_observation = existing["result_json"].get("observation") or ""
            # Same rule as the fresh-result path above: success is the stored
            # `status`, not the presence of observation prose. Re-deriving it
            # from `bool(observation)` here would have replayed every
            # structured skill result as a failure even after the live path
            # started recording it correctly -- the two-paths-one-patched
            # divergence this arc has already shipped twice.
            existing_kind = existing["result_json"].get("result_kind")
            await self._emit_action_outcome(
                bus,
                candidate=candidate,
                summary=(
                    existing_observation
                    or (
                        f"{candidate.dispatch_kind} on {candidate.target_id} returned a structured result"
                        if existing_kind == RESULT_KIND_STRUCTURED
                        else f"{candidate.dispatch_kind} on {candidate.target_id} returned no observation"
                    )
                ),
                success=existing["status"] == "success",
                observed_at=now,
            )
            return candidate.model_copy(
                update={
                    "dispatch_status": "dispatched",
                    "dispatched_at": now,
                    "result_ref": existing["result_id"],
                }
            )

        context = dict(candidate.request_envelope.get("context") or {})

        # 2026-08-12: per-route RPC budget. None -> the client's global
        # timeout, which is what every read-only route still uses. Only the
        # maintenance route sets one, because a real prune runs for minutes
        # while this consumer is single-threaded -- raising the global instead
        # would let one hung inspect stall all dispatch for the same duration.
        # Looked up by dispatch_kind rather than threaded through the frame so
        # this stays a pure config change for any future long route.
        route = self._policy.proposal_kind_to_cortex.get(candidate.dispatch_kind or "")
        route_timeout = route.rpc_timeout_sec if route is not None else None

        # THE COST MEASUREMENT. Taken here, around the send, rather than read
        # off the verb's own report -- for a budget this is the right quantity
        # and the only reliable one. Right: it is the wall-clock the action
        # actually occupied on the motor path, queueing and transport included,
        # which is what spending it costs. Reliable: it does not depend on a
        # verb choosing to report anything, and skills.runtime.* verbs report
        # nothing.
        #
        # Before this, `latency_ms` existed as a schema field on
        # ActionOutcomeRecordV1, a column on substrate_action_outcomes, and a
        # `_latencies()` reader in the feedback runtime -- and was populated on
        # 0 of 5,739 rows in 6 hours, because nothing ever wrote it. There was
        # no per-action cost anywhere in the system, which is what stopped the
        # decision budget being buildable at all: value with no denominator.
        send_started = perf_counter()
        try:
            payload = await client.dispatch(
                verb=candidate.cortex_verb or "",
                mode=candidate.cortex_mode or "brain",
                context=context,
                dispatch_id=candidate.dispatch_id,
                timeout_sec=route_timeout,
            )
        except Exception as exc:
            # A failed send still consumed real time -- often the whole rpc
            # timeout, which is the most expensive outcome there is. Recording
            # it as absent would make failure look free and bias any
            # cost-weighted comparison toward whatever fails fastest.
            latency_ms = (perf_counter() - send_started) * 1000.0
            logger.warning(
                "execution_dispatch_send_failed dispatch_id=%s error=%s", candidate.dispatch_id, exc
            )
            self._store.save_dispatch_result(
                result_id=result_id,
                dispatch_id=candidate.dispatch_id,
                frame_id=frame.frame_id,
                status="failed",
                result_json={
                    "error": str(exc)[:2000],
                    "evidence_refs": [result_id],
                    "latency_ms": latency_ms,
                },
                raw_len=0,
                latency_ms=latency_ms,
                dispatch_kind=candidate.dispatch_kind,
                target_id=candidate.target_id,
            )
            self._record_dispatch_status("failed")
            await self._emit_action_outcome(
                bus,
                candidate=candidate,
                summary=f"attempted {candidate.dispatch_kind} on {candidate.target_id}, send failed",
                success=False,
                observed_at=now,
            )
            return candidate.model_copy(
                update={
                    "dispatch_status": "dispatched",
                    "dispatched_at": now,
                    "dispatch_error": str(exc)[:500],
                }
            )

        # The plan's own verdict comes FIRST, before any content classification.
        #
        # Nothing upstream surfaces a verb failure: cortex-exec's main.py
        # hardcodes `ok=True` on the result payload, and cortex_client only
        # checks the codec's ok, so `client.dispatch` above does not raise for a
        # failed plan -- it returns a well-formed payload whose `final_text` is
        # the skill's own JSON error report. Classifying that by content alone
        # grades a failure as a success, with a stored summary reading
        # "failed: Command 'docker builder prune' timed out after 600 seconds".
        plan_status, plan_reason = plan_execution_status(payload)
        if is_failed_plan_status(plan_status):
            latency_ms = (perf_counter() - send_started) * 1000.0
            reason = plan_reason or f"plan status={plan_status}"
            logger.warning(
                "execution_dispatch_plan_failed dispatch_id=%s plan_status=%s reason=%s",
                candidate.dispatch_id,
                plan_status,
                reason,
            )
            self._store.save_dispatch_result(
                result_id=result_id,
                dispatch_id=candidate.dispatch_id,
                frame_id=frame.frame_id,
                # Reuses the existing "failed" value rather than minting a new
                # one: orion/feedback/builder.py::_cortex_status_to_outcome maps
                # "failed" to a real failed score, and any unrecognised value to
                # "unknown", which would silently drop these from scoring
                # entirely.
                status="failed",
                result_json={
                    "error": reason[:2000],
                    "plan_status": plan_status,
                    # The verb's own payload is still preserved -- a failed
                    # prune's disk/cache measurements are real observations of
                    # the host and must not be discarded just because the
                    # action did not complete.
                    "structured_result": parse_structured_observation(
                        extract_final_text(payload)
                    )["structured_result"],
                    "evidence_refs": [result_id],
                    "latency_ms": latency_ms,
                },
                raw_len=0,
                latency_ms=latency_ms,
                dispatch_kind=candidate.dispatch_kind,
                target_id=candidate.target_id,
            )
            self._record_dispatch_status("failed")
            await self._emit_action_outcome(
                bus,
                candidate=candidate,
                summary=(
                    f"attempted {candidate.dispatch_kind} on {candidate.target_id}, "
                    f"verb reported {plan_status}: {reason}"
                ),
                success=False,
                observed_at=now,
            )
            return candidate.model_copy(
                update={
                    "dispatch_status": "dispatched",
                    "dispatched_at": now,
                    "result_ref": result_id,
                    "dispatch_error": reason[:500],
                }
            )

        latency_ms = (perf_counter() - send_started) * 1000.0
        final_text = extract_final_text(payload)
        observation_data = parse_structured_observation(final_text)
        raw_len = len(observation_data["observation"])
        # Success is decided by `result_kind`, NOT by `len(observation)`.
        #
        # A `skills.runtime.*` verb returns its own structured result dict with
        # no `observation` key at all, so the old length test recorded every
        # one of them as `empty` -- the same state a dead executor produces.
        # Live, that meant all four builder_prune dispatches (Orion's only
        # mutating action) were stored as empty while cortex-exec had logged
        # `raw_len=739` for the very same correlation ids. A skill that
        # honestly declined to act produced a REAL result and must be
        # recorded as one.
        #
        # `raw_len` deliberately still measures the observation string, so the
        # existing column keeps its existing meaning for existing queries; the
        # structured payload is carried alongside it instead of through it.
        result_kind = observation_data.get("result_kind", RESULT_KIND_EMPTY)
        status = "empty" if result_kind == RESULT_KIND_EMPTY else "success"
        self._store.save_dispatch_result(
            result_id=result_id,
            dispatch_id=candidate.dispatch_id,
            frame_id=frame.frame_id,
            status=status,
            result_json={
                **observation_data,
                "evidence_refs": [result_id],
                "latency_ms": latency_ms,
            },
            raw_len=raw_len,
            latency_ms=latency_ms,
            dispatch_kind=candidate.dispatch_kind,
            target_id=candidate.target_id,
        )
        self._record_dispatch_status(status)
        logger.info(
            "execution_dispatch_result dispatch_id=%s status=%s kind=%s raw_len=%d",
            candidate.dispatch_id,
            status,
            result_kind,
            raw_len,
        )
        await self._emit_action_outcome(
            bus,
            candidate=candidate,
            summary=(
                observation_data["observation"]
                or (
                    # A structured result with no self-summary is still a real
                    # result; say so factually rather than claiming it returned
                    # nothing, which is what the old wording did.
                    f"{candidate.dispatch_kind} on {candidate.target_id} returned a structured result"
                    if result_kind == RESULT_KIND_STRUCTURED
                    else f"{candidate.dispatch_kind} on {candidate.target_id} returned no observation"
                )
            ),
            success=status == "success",
            observed_at=now,
        )
        return candidate.model_copy(
            update={
                "dispatch_status": "dispatched",
                "dispatched_at": now,
                "result_ref": result_id,
            }
        )

    async def _emit_action_outcome(
        self,
        bus: OrionBusAsync,
        *,
        candidate: ExecutionDispatchCandidateV1,
        summary: str,
        success: bool,
        observed_at: datetime,
    ) -> None:
        """Publish onto the same always-on ActionOutcomeEmitV1 route
        orion-spark-concept-induction already uses for curiosity-fetch
        outcomes -- no new pipe, reusing an existing durable sink
        (sql-writer -> action_outcomes) so load_action_outcomes() sees
        Layer 9's real dispatch results the same way it already sees
        curiosity-fetch ones. Never lets a publish failure raise out of
        the tick -- an unreachable bus must not lose a result that's
        already durably recorded via save_dispatch_result above.

        `surprise` was a hardcoded `0.0` placeholder (2026-07-13, honest,
        disclosed, never a real signal). Now reads the real, live
        `bus_synaptic_prediction_error()` value off `substrate_field_state`
        (`self._store.latest_bus_synaptic_prediction_error()`) -- generic
        across the whole bus mesh, already fixed once for a calm-floor bias
        (PR #1391). Falls back to `0.0` if the node isn't written yet or the
        query fails, same as before this patch -- this is a real-signal
        upgrade, not a new hard dependency; a fetch failure here must not
        block emitting the outcome itself.
        """
        try:
            surprise = self._store.latest_bus_synaptic_prediction_error()
        except Exception:
            logger.warning(
                "execution_dispatch_bus_synaptic_surprise_fetch_failed dispatch_id=%s",
                candidate.dispatch_id,
                exc_info=True,
            )
            surprise = None
        try:
            emit = ActionOutcomeEmitV1(
                subject=ACTION_OUTCOME_SUBJECT,
                action_id=candidate.dispatch_id,
                kind=candidate.dispatch_kind,
                summary=summary[:ACTION_OUTCOME_SUMMARY_MAX_CHARS],
                success=success,
                surprise=surprise if surprise is not None else 0.0,
                observed_at=observed_at,
            )
            env = BaseEnvelope(
                kind="action.outcome.emit.v1",
                source=ServiceRef(name=self._settings.service_name),
                correlation_id=str(uuid4()),
                payload=emit.model_dump(mode="json"),
            )
            await bus.publish(self._settings.action_outcome_channel, env)
        except Exception:
            logger.warning(
                "execution_dispatch_action_outcome_emit_failed dispatch_id=%s",
                candidate.dispatch_id,
                exc_info=True,
            )

    def _notify_tripwire(self, empty_count: int, window: int) -> None:
        try:
            self._notify.send(
                NotificationRequest(
                    source_service="orion-execution-dispatch-runtime",
                    event_kind="execution_dispatch_theater_tripwire",
                    severity="warning",
                    title="Execution dispatch theater tripwire tripped",
                    body_text=(
                        f"{empty_count}/{window} of the last real dispatches were "
                        "unproductive. Dispatch sending is paused. Recovery probes will "
                        "retry automatically; this will re-notify hourly until it clears."
                    ),
                    tags=["execution_dispatch", "tripwire"],
                )
            )
        except Exception:
            logger.exception("execution_dispatch_tripwire_notify_failed")

    def _notify_tripwire_still_blocked(self, tripped_for_sec: float) -> None:
        """Hourly escalation while the latch holds.

        The 2026-08-23 outage ran 45 hours on a single fire-once warning. A
        notification that describes a state rather than an instant has to keep
        speaking, or it is indistinguishable from a state that has resolved.
        """
        hours = tripped_for_sec / 3600.0
        probe_enabled = self._settings.orion_dispatch_tripwire_probe_enabled
        try:
            self._notify.send(
                NotificationRequest(
                    source_service="orion-execution-dispatch-runtime",
                    event_kind="execution_dispatch_theater_tripwire_still_blocked",
                    severity="warning",
                    title=f"Orion has not acted for {hours:.1f}h (tripwire still active)",
                    body_text=(
                        f"Dispatch has been paused for {hours:.1f} hours after "
                        f"{self._tripwire_probe_attempts} recovery probe(s). "
                        + (
                            "Probes are retrying on a backoff."
                            if probe_enabled
                            else "Automatic probing is DISABLED; this needs a restart."
                        )
                    ),
                    tags=["execution_dispatch", "tripwire", "autonomy_down"],
                )
            )
        except Exception:
            logger.exception("execution_dispatch_tripwire_notify_failed")

    def _notify_tripwire_cleared(self, tripped_for_sec: float, probes: int) -> None:
        try:
            self._notify.send(
                NotificationRequest(
                    source_service="orion-execution-dispatch-runtime",
                    event_kind="execution_dispatch_theater_tripwire_cleared",
                    severity="info",
                    title="Execution dispatch resumed",
                    body_text=(
                        f"Recovery probes succeeded after {tripped_for_sec / 60.0:.0f} "
                        f"minutes and {probes} attempt(s). Dispatch is sending again."
                    ),
                    tags=["execution_dispatch", "tripwire"],
                )
            )
        except Exception:
            logger.exception("execution_dispatch_tripwire_notify_failed")
