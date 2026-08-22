from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-execution-dispatch-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    execution_dispatch_policy_path: str = Field(
        "config/execution_dispatch/execution_dispatch_policy.v1.yaml",
        alias="EXECUTION_DISPATCH_POLICY_PATH",
    )
    execution_dispatch_mode: Literal["dry_run", "prepare_only", "dispatch_read_only"] = Field(
        "dry_run",
        alias="EXECUTION_DISPATCH_MODE",
    )
    execution_dispatch_poll_interval_sec: float = Field(
        2.0,
        alias="EXECUTION_DISPATCH_POLL_INTERVAL_SEC",
    )
    enable_execution_dispatch_runtime: bool = Field(
        True,
        alias="ENABLE_EXECUTION_DISPATCH_RUNTIME",
    )
    cortex_exec_channel: str = Field(
        "orion:cortex:exec:request:background", alias="CORTEX_EXEC_CHANNEL"
    )
    cortex_exec_result_prefix: str = Field(
        "orion:exec:result", alias="CORTEX_EXEC_RESULT_PREFIX"
    )
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")
    execution_dispatch_rpc_timeout_sec: float = Field(
        120.0, alias="EXECUTION_DISPATCH_RPC_TIMEOUT_SEC"
    )
    # 2026-07-30 (docs/superpowers/specs/2026-07-30-execution-dispatch-
    # staleness-discard-design.md): a real dispatch is a synchronous
    # cortex-exec RPC, ~7-11s measured live -- this single-threaded FIFO
    # consumer cannot keep pace with real production of new policy_decision_
    # frames (~16/min produced vs ~6.8/min consumed, measured live
    # 2026-07-30), so the oldest-undispatched-first queue grows without
    # bound (46,617 backlogged, oldest 37h old, at the time this was found).
    # A policy frame older than a randomized [min,max] threshold gets
    # discarded (materialized as a real, queryable "stale_discard" frame --
    # see build_stale_discard_execution_dispatch_frame -- never silently
    # dropped) instead of dispatched, so real cortex actions never describe
    # hours-old field state as current. Randomized, not a single fixed
    # constant, so there is no one sharp, predictable cliff every candidate
    # sits at the same distance from.
    execution_dispatch_staleness_min_sec: float = Field(
        120.0, alias="EXECUTION_DISPATCH_STALENESS_MIN_SEC"
    )
    execution_dispatch_staleness_max_sec: float = Field(
        300.0, alias="EXECUTION_DISPATCH_STALENESS_MAX_SEC"
    )
    # Explicit operator override shim: when set, BYPASSES the randomized
    # [min, max] window entirely and uses this fixed value for every tick
    # instead. Same "explicit operator override, no touching the derived
    # machinery itself" shape as orion_dispatch_risk_cap_advisory_only above.
    # Exists because this service's own consumption/production balance is
    # not assumed permanent -- if Orion's attention/dispatch cadence changes
    # later (faster real consumption, fewer but more deliberate proposals,
    # etc.), a deliberate deep-backlog catch-up may become desirable again
    # without a code change: set this very high (or, in the -- currently
    # unimplemented -- limit, disable discarding outright) rather than
    # reverting this patch. None (default) means "use the randomized window."
    execution_dispatch_staleness_override_sec: float | None = Field(
        None, alias="EXECUTION_DISPATCH_STALENESS_OVERRIDE_SEC"
    )
    # 2026-07-29: no longer the primary mechanism -- see
    # orion_dispatch_risk_cap_advisory_only's comment below and
    # app/worker.py::ExecutionDispatchRuntimeWorker._derive_daily_risk_cap
    # for the real, self-calibrating EWMA ceiling that replaced this fixed
    # number as of this patch. This value is now only the last-resort
    # fallback used when the daily-risk EWMA baseline has never been seeded
    # AND no historical closed day with real candidate data exists at all
    # (i.e. a truly first-ever tick against an empty
    # substrate_execution_dispatch_frames table) -- in this repo's real
    # history that never actually triggers (2026-07-28 already has real
    # closed-day data: 817.65), but the fallback still has to be something.
    # Kept at its old value for continuity of what a fresh/empty deployment
    # gets before any real data exists, not because 10.0 means anything
    # about real risk demand -- it never did (see the old comment history in
    # this file's git blame and services/orion-execution-dispatch-runtime/
    # README.md for how this number's original "~2x one day's total"
    # justification was itself later found to be ~2x a *clamped* value, not
    # real demand).
    orion_dispatch_max_risk_per_day: float = Field(10.0, alias="ORION_DISPATCH_MAX_RISK_PER_DAY")

    # The real daily budget, in motor-seconds -- wall-clock an action occupies
    # on the dispatch path. EXOGENOUS on purpose: set by an operator, never
    # derived from usage. _derive_daily_risk_cap sizes its ceiling from an
    # EWMA of Orion's own past demand plus three standard deviations, which
    # cannot bind by construction. An allowance that tracks what you already
    # wanted is a mirror, not a constraint.
    #
    # 129600 = 36 motor-hours. Measured draw the day this shipped was ~40
    # motor-hours/day (p50 5.0s per action, 1.7x concurrency), so this default
    # sits ~10% BELOW current usage -- deliberately, so the mechanism is
    # exercised and its refusals are countable rather than hypothetical.
    orion_dispatch_motor_budget_sec_per_day: float = Field(
        129600.0, alias="ORION_DISPATCH_MOTOR_BUDGET_SEC_PER_DAY"
    )

    # Advisory until proven. OFF means the budget is computed, logged and
    # stamped on every frame but refuses nothing.
    #
    # This is NOT a permanent hedge -- CLAUDE.md 0A bans a switch that reports
    # success while changing nothing. Advisory mode must publish what it WOULD
    # have refused every tick, and the exit criterion is written down: flip it
    # once a full day of `motor_budget_would_refuse` counts exists and the
    # refused set is inspected and judged droppable. If nobody has looked in a
    # week, that is the answer -- either flip it or delete it.
    orion_dispatch_motor_budget_enforce: bool = Field(
        False, alias="ORION_DISPATCH_MOTOR_BUDGET_ENFORCE"
    )

    # What a not-yet-run action is assumed to cost, for the would-refuse
    # projection only. 5.0s is the live p50 measured 2026-08-21 (p95 6.5s).
    # A real allocator will use the action's OWN measured history instead --
    # this is a placeholder for the advisory count, and is deliberately not
    # used for anything that is enforced.
    orion_dispatch_motor_typical_cost_sec: float = Field(
        5.0, alias="ORION_DISPATCH_MOTOR_TYPICAL_COST_SEC"
    )

    # The ABSOLUTE bar: expected information per motor-second below which an
    # action is not worth its seconds, however much allowance is left. This is
    # what makes "none of these were worth doing" expressible -- a relative
    # ranking always crowns a winner however worthless the set.
    #
    # 0.02 nats/sec: a cold, never-measured action costing 5s scores 0.198,
    # ten times over. A thoroughly-measured one (variance 0.001) at the same
    # cost scores 0.0025, ten times under. The bar sits in the gap, so it
    # separates "we have learned what this does" from "we have not" rather
    # than encoding a preference about which actions are nice.
    orion_dispatch_min_nats_per_sec: float = Field(
        0.02, alias="ORION_DISPATCH_MIN_NATS_PER_SEC"
    )

    # Fraction of ACTING TICKS deliberately withheld, to create a genuinely
    # randomized control arm. 0.0 = off.
    #
    # PER TICK, NOT PER CANDIDATE, and the difference is the whole point. The
    # field delta is measured frame-wide, so withholding one candidate while
    # its siblings run gives a "control" observation contaminated by those
    # siblings -- which is exactly the defect that made the capacity-blocked
    # arm unusable (see orion/autonomy/contrast.py). Withholding the entire
    # tick produces a real no-action tick, drawn at random from ticks that
    # WOULD have acted. That is the counterfactual, and it is the only arm
    # in this system that licenses the word "causal": the existing
    # `no_action` arm is quasi-experimental, because ticks where nothing was
    # proposed are systematically calmer ticks and baseline binning absorbs
    # most of that selection but provably not all of it.
    #
    # Cost is a bounded, measurable capability loss: at 0.05, one acting tick
    # in twenty does nothing. Off by default -- this deliberately makes Orion
    # do less, and that is a decision to take on purpose.
    orion_dispatch_holdback_fraction: float = Field(
        0.0, alias="ORION_DISPATCH_HOLDBACK_FRACTION", ge=0.0, le=0.5
    )
    # 2026-07-29: enforcement is back ON (default flipped True -> False).
    # Real sequence, not "we always knew this": ORION_DISPATCH_MAX_RISK_PER_DAY
    # was a fixed 10.0 constant, ENFORCED, from 2026-07-26 through 2026-07-27
    # -- and it worked exactly like a real ceiling, clamping dispatched-risk
    # totals at exactly 10.00/day both days (150 candidates on the 26th, 88
    # on the 27th), which looked like a healthy, working cap. It wasn't --
    # advisory_only=True shipped 2026-07-28 specifically to observe what real
    # *uncapped* demand looked like, and the answer was 817.65/day (15,099
    # candidates) and climbing, ~80x the old enforced number. That one day of
    # advisory-only data is exactly what
    # app/worker.py::ExecutionDispatchRuntimeWorker._derive_daily_risk_cap
    # now feeds into a real EWMA baseline (orion/bus/ewma.py::
    # compute_ewma_update, same mechanism as PR #1433's recent_perturbation
    # fix and PR #1434's execution_prediction_error fix) instead of a
    # hand-picked multiplier -- so re-enforcing now, on a derived ceiling
    # instead of a guessed one, closes the loop advisory-only was opened for.
    # Real per-tick spend still reads from sum_risk_dispatched_today()
    # (right-censored at whatever cap is in force, and correctly so -- that's
    # what "spend" means); the EWMA baseline itself is fed from
    # store.py::sum_uncapped_risk_for_day's *uncapped* demand instead, so the
    # new cap can't recreate the same clamped-value-masks-true-magnitude trap
    # one layer down. Explicit operator override: set this back to True to
    # return to log-only behavior without touching the derived-cap machinery
    # itself.
    orion_dispatch_risk_cap_advisory_only: bool = Field(
        False, alias="ORION_DISPATCH_RISK_CAP_ADVISORY_ONLY"
    )
    action_outcome_channel: str = Field(
        "orion:autonomy:action:outcome", alias="BUS_ACTION_OUTCOME_OUT"
    )
    notify_url: str = Field("http://notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    # ROADMAP D2, 2026-08-19. How often to re-queue policy frames whose `dispatch_pending`
    # marker was cleared without a dispatch frame existing. The marker is cleared
    # transactionally so this should find nothing -- but the failure it guards is SILENT WORK
    # LOSS, and it can only add work back, never remove it. It runs the expensive anti-join the
    # marker exists to avoid, hence once every 15 min rather than on every tick.
    dispatch_reconcile_interval_sec: float = Field(
        900.0, alias="DISPATCH_RECONCILE_INTERVAL_SEC"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
