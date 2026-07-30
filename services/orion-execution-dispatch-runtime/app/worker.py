from __future__ import annotations

import asyncio
import logging
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
from orion.execution_dispatch.builder import (
    build_execution_dispatch_frame,
    build_stale_discard_execution_dispatch_frame,
    build_unevaluable_execution_dispatch_frame,
)
from orion.execution_dispatch.cortex_client import ExecutionDispatchCortexClient
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.execution_dispatch.result_extraction import (
    extract_final_text,
    parse_structured_observation,
)
from orion.notify.client import NotifyClient
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.notify import NotificationRequest
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1

from app.settings import get_settings
from app.store import ExecutionDispatchRuntimeStore

logger = logging.getLogger("orion.execution_dispatch.runtime")

THEATER_TRIPWIRE_WINDOW = 10
THEATER_TRIPWIRE_EMPTY_THRESHOLD = 0.5
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
        self._store = ExecutionDispatchRuntimeStore(self._settings.postgres_uri)
        self._policy = load_execution_dispatch_policy(
            Path(self._settings.execution_dispatch_policy_path)
        )
        self._notify = NotifyClient(
            base_url=self._settings.notify_url,
            api_token=self._settings.notify_api_token,
            timeout=10,
        )
        self._stop = asyncio.Event()
        # In-memory only, by design: once tripped, stays tripped until this
        # process restarts (mirrors the "re-arm is manual" rule in the
        # parent spec -- a self-clearing tripwire could silently resume
        # sending on a coincidentally-good sample).
        self.theater_tripwire_active = False
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

        Each discard frame saved here carries the CURRENT (not-yet-updated-
        for-this-tick) baseline forward, same as every other saved frame --
        the real update for this tick's own discard count is computed once,
        after this method returns, and re-stamped onto whichever frame ends
        up being the last one saved this tick (see _tick()).

        Returns (fresh_policy_frame_or_none, discarded_count, last_discard_
        frame_or_none).
        """
        discarded = 0
        last_discard_frame: ExecutionDispatchFrameV1 | None = None
        while discarded < MAX_STALE_DISCARDS_PER_TICK:
            candidate_frame = self._store.load_latest_policy_frame_without_dispatch()
            if candidate_frame is None:
                return None, discarded, last_discard_frame
            # Defensive normalize, same idiom as _derive_daily_risk_cap above:
            # PolicyDecisionFrameV1.generated_at is typed as a plain datetime,
            # not enforced tz-aware. A naive value here would raise on the
            # subtraction below -- caught by _poll_loop's broad except, but
            # this exact policy frame would then be the permanent FIFO head
            # forever (it can never resolve), reproducing the same
            # queue-blocking failure shape build_unevaluable_execution_
            # dispatch_frame exists to prevent for a missing proposal.
            frame_generated_at = candidate_frame.generated_at
            if frame_generated_at.tzinfo is None:
                frame_generated_at = frame_generated_at.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - frame_generated_at).total_seconds()
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
        }
        policy_frame, discarded_this_tick, last_discard_frame = self._drain_stale_policy_frames(
            baseline
        )

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
                    last_discard_frame.model_copy(update=updated_baseline)
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
            ).model_copy(update=updated_baseline)
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

        if self._check_theater_tripwire():
            # tripped (this tick or a prior one) -- send nothing, but still
            # carry the current baseline state forward onto the saved frame.
            return frame.model_copy(update=baseline_fields)

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
                return frame.model_copy(update=baseline_fields)
            # Advisory-only (operator override, no longer the default as of
            # 2026-07-29 -- see settings.py's own comment): log that it would
            # have blocked, but don't actually withhold real dispatches.
            # max_dispatches_per_tick (a real, independent, already-enforced
            # per-tick burst limit) is untouched by this and still applies.
            remaining_risk_budget = float("inf")

        # Real, risk-weighted selection -- not a blind count. `frame.candidates`
        # is already priority-sorted (build_proposal_frame), so this takes
        # the highest-priority prepared candidates whose cumulative
        # risk_score fits inside what's left of today's real budget (or all
        # of them, if the cap is advisory and already reached -- see above),
        # stopping at the first one that would exceed it (no skip-ahead to
        # squeeze in a smaller one later, matching the existing simple
        # sequential-take style this replaces). max_dispatches_per_tick
        # stays a separate, independent per-tick burst cap.
        to_send: list[ExecutionDispatchCandidateV1] = []
        cumulative_risk = 0.0
        for candidate in frame.candidates:
            if candidate.dispatch_status != "prepared_for_dispatch":
                continue
            if len(to_send) >= self._policy.limits.max_dispatches_per_tick:
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

        if not to_send:
            return frame.model_copy(update=baseline_fields)

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
        newly_dispatched: list[ExecutionDispatchCandidateV1] = []
        try:
            for candidate in to_send:
                newly_dispatched.append(await self._send_one(client, bus, frame, candidate))
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
        self._check_theater_tripwire()
        return updated

    def _check_theater_tripwire(self) -> bool:
        recent = list(self._recent_dispatch_statuses)
        if len(recent) < THEATER_TRIPWIRE_WINDOW:
            return self.theater_tripwire_active
        empty_count = sum(1 for s in recent if s == "empty")
        newly_tripped = (
            not self.theater_tripwire_active
            and empty_count > THEATER_TRIPWIRE_WINDOW * THEATER_TRIPWIRE_EMPTY_THRESHOLD
        )
        if newly_tripped:
            self.theater_tripwire_active = True
            logger.warning(
                "execution_dispatch_theater_tripwire_active empty=%d window=%d",
                empty_count,
                len(recent),
            )
            self._notify_tripwire(empty_count, len(recent))
        return self.theater_tripwire_active

    async def _send_one(
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
            self._recent_dispatch_statuses.append(existing["status"])
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
            await self._emit_action_outcome(
                bus,
                candidate=candidate,
                summary=(
                    existing_observation
                    if existing_observation
                    else f"{candidate.dispatch_kind} on {candidate.target_id} returned no observation"
                ),
                success=bool(existing_observation),
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

        try:
            payload = await client.dispatch(
                verb=candidate.cortex_verb or "",
                mode=candidate.cortex_mode or "brain",
                context=context,
                dispatch_id=candidate.dispatch_id,
            )
        except Exception as exc:
            logger.warning(
                "execution_dispatch_send_failed dispatch_id=%s error=%s", candidate.dispatch_id, exc
            )
            self._store.save_dispatch_result(
                result_id=result_id,
                dispatch_id=candidate.dispatch_id,
                frame_id=frame.frame_id,
                status="failed",
                result_json={"error": str(exc)[:2000], "evidence_refs": [result_id]},
                raw_len=0,
            )
            self._recent_dispatch_statuses.append("failed")
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

        final_text = extract_final_text(payload)
        observation_data = parse_structured_observation(final_text)
        raw_len = len(observation_data["observation"])
        status = "success" if raw_len > 0 else "empty"
        self._store.save_dispatch_result(
            result_id=result_id,
            dispatch_id=candidate.dispatch_id,
            frame_id=frame.frame_id,
            status=status,
            result_json={**observation_data, "evidence_refs": [result_id]},
            raw_len=raw_len,
        )
        self._recent_dispatch_statuses.append(status)
        logger.info(
            "execution_dispatch_result dispatch_id=%s status=%s raw_len=%d",
            candidate.dispatch_id,
            status,
            raw_len,
        )
        await self._emit_action_outcome(
            bus,
            candidate=candidate,
            summary=(
                observation_data["observation"]
                if raw_len > 0
                else f"{candidate.dispatch_kind} on {candidate.target_id} returned no observation"
            ),
            success=raw_len > 0,
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
                        f"{empty_count}/{window} of the last real dispatches returned empty "
                        "observations. Dispatch sending is paused until this worker restarts."
                    ),
                    tags=["execution_dispatch", "tripwire"],
                )
            )
        except Exception:
            logger.exception("execution_dispatch_tripwire_notify_failed")
