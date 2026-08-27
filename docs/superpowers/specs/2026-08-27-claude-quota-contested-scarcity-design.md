# Contested scarcity — giving Orion a budget somebody else wants

> **Status:** Design proposal (proposal mode — governs which of Orion's wants get to spend a shared resource). Single scope: make one action cost something real, record the trade-off, randomize the funding.
>
> Companion spec: `2026-08-26-orion-priors-and-worldview-design.md`. That one gives Orion something worth asking about; this one makes asking cost something.

## Arsonist summary

Orion's allocator is good. Its budget is exogenous, denominated in a real resource, and set by an operator instead of derived from demand — the three things that took an arc to learn. And it still cannot produce an opportunity cost, because **motor-seconds are scarce but uncontested.** Nobody else wants Orion's seconds. Refusing an action returns them to a pool no one is drawing from, so the refusal costs nothing anyone would notice, and "was this action worth it" has no counterparty to be worth it *against*.

Claude quota is the first resource in this system with a **second claimant**. Orion and Juniper draw from the same window. Spending it is not a number going down in a dashboard; it is Juniper not being able to code. That is what an opportunity cost is, and no amount of allocator sophistication manufactures one where there is no rival.

Burn the assumption that scarcity is a property of a resource. It is a property of a resource **plus a competitor**.

## Current architecture

Verified live 2026-08-27, not assumed.

### The allocator exists and is better than the spec it replaced

`orion/autonomy/allocator.py`, wired into `services/orion-execution-dispatch-runtime/app/worker.py:17`.

The 2026-07-07 internal-economy spec proposed ranking drives by value-biased priority. That approach is dead and the module says why: every action measures approximately zero (`prune -0.0073 ± 0.0342`, `containers -0.0405 ± 0.0581`, `inspect +0.0386 ± 0.0508`), so ranking by value ranks noise, confidently.

What was built instead splits expected free energy into two terms with different jobs:

- **Epistemic value is the score** — expected information gain, `0.5 * ln(1 + sigma^2 / tau^2)` nats, closed form for the Normal-Normal posterior, verified against 40,000-sample Monte Carlo. Denominated in nats per motor-second. One unit, no conversion.
- **Pragmatic value is a gate** — `confidently_harmful` refuses an action whose measured effect is confidently in the direction it claims to prevent.

Two structural decisions matter enormously for this spec:

1. **`min_nats_per_sec` is an absolute floor, not a rank cut.** The module argues this explicitly: "a relative ranking always crowns a winner and can never say *none of these were worth doing*." Percentages sum to 100% no matter how worthless the set.
2. **No invented exchange rate.** Pragmatic and epistemic value are deliberately never summed, because the conversion constant would recreate `risk_score` — five hand-typed YAML numbers — one layer up.

### Live configuration

```
ORION_DISPATCH_MOTOR_BUDGET_SEC_PER_DAY = 129600.0
ORION_DISPATCH_MOTOR_BUDGET_ENFORCE     = false      <- never binds
ORION_DISPATCH_MIN_NATS_PER_SEC         = 0.02
ORION_DISPATCH_HOLDBACK_FRACTION        = 0.0        <- only randomized arm, off
```

Measured over the trailing 24h at 2026-08-27: **25,152 dispatches, 83,787 motor-seconds** against the 129,600 allowance (**65%**), mean latency 3,331 ms. The allowance has never bound, and with `ENFORCE=false` it could not have.

### The harm gate has never fired

Recorded in the module's own docstring and confirmed: `contrast` and `contrast_sd` have **no producer anywhere in the repository**. `orion.autonomy.contrast.contrast()` is called from exactly one place, `orion/autonomy/evals/eval_action_value_contrast.py`, and nothing persists a `ContrastEstimate`. The dispatch worker passes `None` for both, so `confidently_harmful` returns `False` on every candidate.

This spec does not fix that, but it must not assume the gate protects anything.

### Dev-economics measurement is live

**Each row is a delta since the last tick (~15 min), not a cumulative total.** The producer docstring is explicit: "the real *growth* in token/word/cost totals since the last check." Daily spend is therefore `SUM`, not `max - min` — an early read of this table using `max - min` produced numbers an order of magnitude low, recorded here so the next reader does not repeat it.

```
dev_economics_ledger_log:  1,254 ticks, 2026-08-12 -> 2026-08-27
daily spend (SUM of deltas, notional at API rates):
  2026-08-19   $449.36    1.05B tokens    26/91 ticks active
  2026-08-22   $364.28    1.10B tokens    40/45
  2026-08-23    $50.94    0.22B tokens    11/93
  2026-08-24     NULL     0 tokens         0/95   <- genuine silence, verified
  2026-08-25   $330.17    0.79B tokens    20/94
  2026-08-26   $582.88    1.23B tokens    27/92
unpriced_session_count:    0 on every day -- the pricing table covers the live model mix
```

Two properties this spec depends on, both already implemented:

- **A silent tick writes `total_estimated_cost_usd = NULL`, not `0.0`.** Absence already reads as unknown rather than zero — the property acceptance check 6 asks for exists at the producer.
- **The zeros are real.** 2026-08-26 shows 18 consecutive all-zero ticks from 18:14 to 22:31 UTC, then 4 sessions at 22:47. Cross-checked against transcript mtimes: **zero `~/.claude/projects/*.jsonl` files were modified in that window.** The instrument was reading correctly, not blind.

- `orion/dev_economics/claude_code_ingest.py` reads `~/.claude/projects/*.jsonl` — every Claude Code session on the host, subagents included and deliberately so.
- `orion/dev_economics/pricing.py` — real versioned rate table with effective-date windows; returns `None` rather than a fabricated `$0.00` for an unpriced model.
- `orion/dev_economics/ledger_aggregate.py` — rollups.

**This is already the right denominator.** It measures the whole machine, which is exactly why Orion and Juniper share it.

### The ask-Claude path

- Hub button: `services/orion-hub/templates/index.html:362` -> `static/js/app.js:9220`
- Relay: `services/orion-hub/scripts/room_claude_relay.py`, schema `orion/schemas/room_claude.py`
- Session: `services/orion-room-companion/app/claude_session.py` — `claude -p <prompt> --output-format json --model <m>`, `--session-id`/`--resume` for continuity, `--tools ""` (companion, not agent). Already parses `total_cost_usd` and `modelUsage` off the response.

**Per-turn cost is already measured at the point of spend.** No new instrumentation needed for the numerator.

### What is NOT readable

Checked, negative result:

- No `claude usage` subcommand (`claude --help`: `agents auth auto-mode doctor gateway import install mcp plugin project setup-token ultrareview update`).
- `~/.claude/policy-limits.json` is enterprise policy config (`restrictions.enforce_web_search_mcp_isolation`, `defaults.remote_control_at_startup`) — not rate limits.
- Nothing rate-limit-shaped in `orion/dev_economics/` or `services/orion-room-companion/`.

**"% of quota remaining" cannot be read. It has to be calibrated.** Numerator is real and continuously measured; the denominator is discovered by hitting the limit once and recording cumulative spend at that moment. Any UI that renders a percentage must not imply it was read from an authority.

## The core problem

Add a second currency to an allocator that was built specifically to avoid having one.

`allocate()` takes `cost_sec`, `allowance_sec`, `min_nats_per_sec` — a single-constraint fractional knapsack. A Claude action costs dollars against a contested quota *and* wall-clock on the dispatch path. Ranking it by nats-per-motor-second prices it as though the dollars were free: a $2 / 30-second ask looks cheap, wins, and eats the quota. That is the exchange-rate problem sneaking back in through the ranking function.

## The resolution: two knapsacks, one doctrine

**Because the floor is absolute rather than relative, floors compose across currencies without a conversion rate.**

A relative ranking cannot span two currencies — you must convert to compare. An absolute floor can, because each floor is a statement *within* its own units: "is this worth its seconds?" and "is this worth its dollars?" are separately answerable, and an action that consumes both must clear both. No constant converts nats-per-second to nats-per-dollar, and none is needed.

```
motor knapsack:   score = nats / cost_sec    floor = min_nats_per_sec    allowance = motor_sec_per_day
quota knapsack:   score = nats / cost_usd    floor = min_nats_per_usd    allowance = usd_per_window
```

Two calls to the same `allocate()` shape, different denominators. An action declaring both costs runs both gates and is admitted only if both admit it.

**What this deliberately gives up:** the cross-currency trade-off. Orion cannot ask "is this Claude conversation worth more than four docker prunes?" That question is unanswerable without inventing the rate this codebase spent an arc deleting, and it is not the interesting question anyway. The interesting one — *is this worth spending Juniper's window* — is entirely inside the quota knapsack.

**What it costs in optimality:** greedy is exactly optimal for the divisible single-constraint case and near-optimal as used today. Two independent greedy passes are not jointly optimal over the two-constraint problem. Accepted, and cheaper than false precision over cost estimates that carry their own error bars.

## Hazards, named before they ship

### 1. Cold start hands Orion's first scarce decision to an action with no evidence

`Candidate.posterior_variance` cold-starts at maximum, scoring ~0.9905 nats — maximally informative by construction. This is correct for a free resource: an unmeasured action is worth measuring. Under a **contested** budget it means Orion's first act under scarcity is to spend Juniper's quota on the one action nobody has any evidence for, repeatedly, until the posterior tightens.

The module already carries the scar: across 57 consecutive previews every admitted candidate scored exactly `0.9905007` — the cold-start default — because unmeasurable actions were being scored as maximally informative.

**Mitigation:** cap the fraction of a window any single cold cell may consume, and require `cost_usd` to be a measured value from that action's own history (mirroring the existing `no_cost_estimate` refusal) rather than an estimate.

### 2. `cost_usd` is notional on a subscription token

The subscription's real currency is the rate-limit window unit, not dollars. `total_cost_usd` is API-rate pricing applied to a model-weighted token count, which is *probably* monotonic with the limit unit — a decent proxy. **"Probably" is a metric-gate claim, not a finding.** See the gate section below.

### 3. The cap cannot be enforced, only observed

Resolved direction (Juniper, 2026-08-14, agent board): Hub holds `/var/run/docker.sock`, so a Hub-resident agent is root-equivalent on the host and no software cap is enforceable wherever the logic lives. **Advisory cap plus reconciliation. Detect, do not pretend to prevent.**

Keep it advisory and label it so. A budget Orion could break and does not is a stronger signal about Orion than one it physically cannot break — but only if nobody dresses the advisory number up as a ceiling.

### 4. The percentage is decoration unless a deferral is written

"Orion sees 34% remaining" changes nothing by itself. The artifact of record is the refusal: what Orion wanted, what it would have cost, what it was worth in nats, and which floor or allowance turned it down. `Allocation.refused` already carries exactly this shape and `refusals_by_reason()` already aggregates it. Without persisting it, the percentage is a keyword cathedral with a number in it.

## Metric quality gate (CLAUDE.md §0A)

Run for `quota_fraction_remaining`, the one genuinely new signal.

1. **Provenance.** Numerator: `dev_economics_ledger_log.total_estimated_cost_usd`, produced by `claude_code_ingest.py` parsing real transcript token counts through `pricing.py`'s dated table. Denominator: an operator constant, calibrated. Traced to producing code, not schema comment.
2. **Independence.** Not a transform of anything in the model today. The allocator's existing inputs are `posterior_variance` (from outcome resolution) and `cost_sec` (from `substrate_dispatch_results.latency_ms`). Dollar spend on Claude sessions shares no sensor and no upstream computation with either. **Independent.**
3. **Theory anchor.** Opportunity cost requires a rival claimant on a finite pool. This is the only resource in the system that has one. Not "seems related."
4. **Live-data sanity.** 1,254 ticks over 15 days; $51-583/day, real variance, no saturation. The rest-state check was done the hard way rather than by eyeballing variance: 2026-08-24 reads 0 across all 95 ticks, and that was confirmed as **genuine silence** by cross-checking host transcript mtimes (zero files modified), not inferred from the metric's own zero. A silent tick nulls rather than zeroes the cost, so a decayed-to-zero or blind-instrument artifact would be distinguishable from calm. **Passes** — this is the check the `bus_synaptic_prediction_error` and `node:substrate.route` incidents exist to force, run in both directions.
5. **Existing mechanism.** `orion/dev_economics/` already does the measurement; this consumes it rather than rebuilding. `orion/autonomy/budget.py` already establishes the exogenous-allowance pattern to copy.
6. **Reversibility.** One flag restores today's behavior. No schema default, no manifest, no training input. Cheap to unwind.

**Open item failing the gate until measured:** the dollars-to-window-unit monotonicity in Hazard 2. Until it is checked, the budget is honestly denominated in *dollars at API rates*, which is a real quantity, rather than claimed to be *quota*, which it is not yet shown to track.

## Proposed schema / API changes

```python
# orion/autonomy/allocator.py -- additive, both default None
@dataclass(frozen=True)
class Candidate:
    ...
    cost_usd: float | None = None          # measured, from this action's own history
    quota_fraction_cap: float | None = None  # max share of one window a single cell may take

RefusalReason = Literal[
    ..., "quota_exhausted", "below_value_floor_usd", "no_usd_cost_estimate",
]
```

```python
# orion/autonomy/quota_budget.py -- new, mirrors budget.py
def quota_state(now) -> QuotaState:
    """Rolling-window spend from dev_economics_ledger_log against an operator allowance.

    ADVISORY. Cannot be enforced (see Hazard 3). `fraction_remaining` is
    computed against a CALIBRATED denominator, never a read one.
    """
```

New channel `orion:autonomy:quota:allocation` carrying an allocation record with its refusals; register in `orion/bus/channels.yaml` and `orion/schemas/registry.py`.

## Env/config changes

```
ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW=      # operator-set, calibrated; empty = disabled
ORION_QUOTA_WINDOW_HOURS=5
ORION_QUOTA_MIN_NATS_PER_USD=
ORION_QUOTA_COLD_CELL_MAX_FRACTION=0.15
ORION_QUOTA_ENFORCE=false                  # advisory by default, per Hazard 3
ORION_QUOTA_HOLDBACK_FRACTION=0.0
```

`.env_example` for the owning service plus `python scripts/sync_local_env_from_example.py`.

## Files likely to touch

- `orion/autonomy/allocator.py` — two optional fields, three refusal reasons
- `orion/autonomy/quota_budget.py` — new
- `orion/dev_economics/ledger_aggregate.py` — rolling-window read
- `services/orion-room-companion/app/claude_session.py` — quote cost before the call, not only after
- `services/orion-hub/scripts/room_claude_relay.py` — carry the quote
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`
- Tests + an eval that replays a real spend week against the allocator

## Non-goals

- Enforcing the cap. Impossible here; do not pretend (Hazard 3).
- Converting between motor-seconds and dollars.
- Giving Claude tools. `--tools ""` stays.
- Making every action scarce. Exactly one action gains a real price. The rest stay free, so the quota knapsack binds only where Claude is in the running. Honest limitation, and the right place to start, because it is the only place the scarcity is not invented.
- Fixing the inert harm gate.

## The experiment this buys

The reason `HOLDBACK_FRACTION` sits at 0.0 is that withholding actions costs capability. Here the contrast is naturally paired and nearly free:

**Orion forms a question worth asking -> randomize whether it may spend -> compare the journals.** Same question, funded versus deferred.

Same-question pairing removes the confound that defeated every observational contrast in this arc: matched on the thing that generated the decision, differing only in whether it was funded. This is the cleanest causal design available anywhere in the system and it falls out of the scarcity mechanism at no extra cost.

## Acceptance checks

1. `quota_state()` sums per-tick deltas over a real rolling window (never `max - min`), returns a fraction that moved across the last 7 days, and reports 2026-08-24 as *unknown/rest*, not as `$0.00` spent.
2. A candidate with `cost_usd` above the remaining allowance is refused `quota_exhausted`, and the refusal is persisted and readable.
3. A cold cell cannot consume more than `COLD_CELL_MAX_FRACTION` of one window — mutation-tested against the real allocator, not a synthetic copy.
4. With `ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW` empty, `allocate()` behavior is byte-identical to today. Pinned by a test.
5. An eval replays a real spend week and reports how many asks would have been refused, by reason.
6. Reconciliation: total observed spend from `claude_code_ingest` minus metered companion spend surfaces a gap rather than silently reading zero. **Absence reads as unknown, never as zero.**
7. Monotonicity of `cost_usd` against the real limit unit is either measured or the doc says `UNVERIFIED`.

## Recommended next patch

Smallest slice that binds:

1. `quota_budget.py` + its test. Read-only, no allocator change, no behavior change. Prints the number.
2. Run it for a week against live spend. Confirm it would have refused something real.
3. Only then wire `cost_usd` into `Candidate` behind the flag.

Step 2 is the gate. If a week of real spend never produces a refusal, the allowance is set too high or the action is too cheap to matter, and wiring it in would ship the same ornamental scarcity the 2026-07-07 spec was correctly refused for.

## Rollback

`ORION_QUOTA_ALLOWANCE_USD_PER_WINDOW=` (empty) restores today's single-knapsack behavior exactly, pinned by acceptance check 4. No data migration, no schema default.

## Privacy boundary

`claude_code_ingest.py` reads transcript *token counts and timestamps*, not content, for this path. The quota reader must not surface prompt or response text. Existing dev-economics privacy behavior is unchanged; this spec adds no new read of transcript bodies.
