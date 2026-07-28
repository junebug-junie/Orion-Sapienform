# Precision-weighted surprise gating for `capability_policy.py` (Sentience Striving Program, §9a item 3)

Status: design/proposal mode per root `CLAUDE.md` §0A. Nothing implemented. This does not
authorize any change to `capability_policy.py`'s live gating behavior. Converged 2026-07-25
after two rounds of independent verification (code re-tracing + literature search, both
run by separate agents instructed to find where this design was wrong, not confirm it) and
one arsonist pass on the resulting "big swing" proposal that found real overreach.

**Re-scoped 2026-07-28, materially, not a footnote.** The 2026-07-25 version below built
`domain_surprise_score` from a new `NormalizationContext` call site over `prediction_error.py`'s
four naive-delta domains. That plan is superseded: `bus_synaptic_prediction_error()` (PR #1377,
calm-floor fixed PR #1391, now wired live into real `action_outcomes` rows via PR #1400/#1403)
is a **better** signal than what this design set out to build — grounded in directly-measured
bus traffic (`orion-bus-mirror`'s real per-edge EWMA/z-score, `compute_ewma_update()`), not a
one-step-removed delta on an already-processed projection the way the four `prediction_error.py`
domains are. It requires **zero new wiring**: already continuously computed, already durably
published to `substrate_field_state`, already validated live (see Live-data sanity check,
2026-07-28, below). Missing Question 1 (where does the tracker live) doesn't get answered by
this update — it dissolves, because no new tracker is needed at all. See "Re-scope, 2026-07-28"
for the full reasoning and what's still open.

## Arsonist summary

`orion/sentience_striving_program/README.md` §9a item 3 names "Free-energy/active-inference
reframing — `capability_policy` as literal expected-free-energy action selection" as an
unbuilt blue-sky direction. Read against the real code
(`orion/autonomy/capability_policy.py::evaluate_capability()`), that framing overclaims what
this function is: it's called once per capability_id, from four fixed call sites, never a
competition among several candidates. This design stays scoped to that reality —
**single-candidate admission gating**, not multi-candidate selection (that's §9a item 2,
Society-of-Mind, a separate, already-partially-built direction —
`orion/attention/field_attention/candidate_society_of_mind.py` — deliberately not touched
here; see Non-goals).

**Three real, load-bearing findings this design is built on, each independently verified:**

1. `ActionOutcomeRefV1.surprise` is not usable as an epistemic-value signal for most emitters
   — every call site that existed at the time this finding was written computes it as a
   binary success/fail proxy, and live data confirmed it read exactly `0.0` across all 12
   real rows since the Postgres rebuild. Ruled out for those emitters; flagged in the
   field's own docstring (`orion/autonomy/models.py`) so nobody reuses it under the wrong
   assumption. **Update, 2026-07-28**: `services/orion-execution-dispatch-runtime` is now a
   real exception — its rows carry a genuine `bus_synaptic_prediction_error()` value, not
   the proxy (see that service's README, "Experience loop" section). This does not change
   this design's own approach (still building `domain_surprise_score` on
   `NormalizationContext` over `prediction_error.py`'s domains, per Missing Question 1) —
   noted here so this finding doesn't read as still-universally-true.
2. **This design never reads, writes, persists, or validates against `GoalProposalV1`,
   `drive_origin`, or `required_drive_origins`, anywhere, for any reason.** An earlier draft
   proposed coexistence with and validation against the halted drives gate — both wrong, both
   corrected. The Sentience Striving Program exists to kill that apparatus; this design does
   not invest in it even transitionally.
3. **The originally-proposed epistemic-value substrate (`prediction_error_confidence`,
   `orion/substrate/attention_self_model.py`) is itself built on a naive, zero-order signal.**
   `orion/substrate/prediction_error.py`'s five domain functions compute `abs(current -
   previous_tick)` with no learned baseline anywhere in the file — independently confirmed by
   a code-verification pass that read every line. This is the real, corrected foundation of
   this document: the fix isn't a threshold on that scalar, it's replacing what feeds it.

**What the literature actually says, independently verified by a second agent instructed to
find where this reasoning breaks:**

- Predictive coding (Rao & Ballard 1999; Friston's free-energy principle) defines prediction
  error against a **learned expectation**, and combines errors via **precision-weighting**
  (error scaled by inverse variance) — not a naive lag-1 delta. The naive/persistence
  forecast (`x_t - x_{t-1}`) is a real, named forecasting benchmark (Hyndman &
  Athanasopoulos), and it is only the theoretically optimal predictor when the underlying
  process is a true random walk — not when a signal has a stable operating baseline to
  revert to, which is the more honest claim to make about our telemetry.
- An EWMA-tracked mean is literally the steady-state special case of a Kalman filter (the
  canonical recursive Bayesian estimator predictive coding is built on), and a z-scored
  deviation (error divided by an estimated scale) is structurally the same operation as
  precision-weighting. This is the best-supported claim checked — a real, specific,
  citable correspondence, not a loose analogy. **This is what this design builds.**
- **Naming honesty, not yet fully resolved**: Sajid, Parr, Da Costa & Friston (2020,
  *J. Mathematical Psychology*) formally distinguish "retrospective surprise" (how surprising
  was an observation that already happened — a real, derived lower bound on expected free
  energy) from Active Inference's actual "epistemic value" (a *prospective*, policy-
  conditioned expected information gain about hidden states — requires simulating outcomes
  under a not-yet-taken action). What this design computes is closer to the former. **This
  document does not call the resulting field "epistemic value"** — see Proposed schema. It
  is also not verified that a raw z-score is precisely Sajid et al.'s formal "retrospective
  surprise" quantity (their derivation wasn't independently re-derived here) — so the field
  is named descriptively (`domain_surprise_score`), not by borrowing their term either,
  until that correspondence is actually checked.
- No literature (Active Inference or hierarchical-RL/options) addresses retrofitting a
  pre-existing action/capability set onto an independently-designed state-factor space — the
  field's working assumption is co-design. **The capability→domain mapping in this design is
  therefore an explicit, disclosed, first-party judgment call, not a derivation** — see
  Missing Question 2.

**One more real finding, from an arsonist pass on an earlier "big swing" version of this
document that proposed writing a new tracker module**: no new tracking primitive is needed.
`orion/signals/normalization.py`'s `EwmaBand`/`InductionTracker`/`NormalizationContext` are
already real, live (wired into `orion/signals/adapters/biometrics.py` and `equilibrium.py`
today, via `services/orion-signal-gateway`), completely generic (`get_band(organ_id,
metric_key)`/`get_tracker(organ_id, metric_key)`, keyed by arbitrary strings — confirmed by
reading the file: zero biometrics-specific logic anywhere), and structurally uninvolved with
drives (`DeviationGate`, the other real EWMA implementation in this codebase, has exactly two
real callers, both inside the halted drives system — confirmed by exhaustive grep — so this
design deliberately does not build on it, even though the class itself carries no
drive-specific state). `EwmaBand.normalize()` already computes a bounded, baseline-relative
score: `clamp01((value - (mean - 2*dev)) / ((mean + 2*dev) - (mean - 2*dev)))` — real
deviation-relative normalization, already written, already tested, already running in
production for an unrelated purpose. Reusing it here is the "reuse the live pipeline, don't
parallel it" move (§7), not a new build.

## Re-scope, 2026-07-28

**Why bus_synaptic instead of NormalizationContext over prediction_error.py's domains:**

- `orion-bus-mirror`'s `compute_ewma_update()` (`services/orion-bus-mirror/app/graph_writer.py`)
  is a real, live, per-edge EWMA mean/variance + z-score mechanism, continuously updated on
  **directly observed** inter-service bus traffic as messages actually flow. `bus_synaptic_
  prediction_error()` (`orion/substrate/prediction_error.py`) aggregates those real z-scores
  into a bounded `[0,1]` score — the exact shape `NormalizationContext`/`EwmaBand` would have
  produced, but measuring the real thing directly rather than diffing an already-processed
  projection one derivation step removed from the raw event (which is what `execution`/
  `biometrics`/`chat`/`route` all do).
- It's already continuously computed in one canonical process (`orion-substrate-runtime`'s
  `_bus_synaptic_tick`) and already durably published to `substrate_field_state`
  (`node:substrate.bus_synaptic`'s `prediction_error` field) — the same pipe this session built,
  validated, and shipped three times over (`services/orion-execution-dispatch-runtime`'s store
  method, PR #1400; extracted to a shared `orion/substrate/bus_synaptic_surprise.py`, PR #1403;
  `orion-spark-concept-induction`'s three emitters, also PR #1403).
- **`NormalizationContext` itself is a plain in-memory dict with zero persistence**
  (`services/orion-signal-gateway/app/normalization_state.py::NormalizationStateRegistry`), and
  `evaluate_capability()` is called from at least two separate services/processes
  (`orion-spark-concept-induction` via `orion/autonomy/policy_act.py`, and
  `services/orion-world-pulse/app/services/curiosity.py`). Wiring `NormalizationContext` inline
  at the callsite (one half of the original Missing Question 1) would have given each calling
  process its own uncoordinated, silently-diverging copy of the same domain's baseline — a real
  architectural problem the original design didn't fully surface. Reusing `bus_synaptic_
  prediction_error()`'s existing durable-publish pattern sidesteps this entirely: any process
  reads the same real value via a plain SQL query, the way `execution-dispatch-runtime` already
  does.

**Live-data sanity check, 2026-07-28** (the charter's metric-quality-gate step 4, done before
committing to this re-scope, not after): pulled the full retained `substrate_field_state`
history for `node:substrate.bus_synaptic`'s `prediction_error` field (2026-07-25 through
2026-07-28, ~128k rows) and checked it against both known failure modes this codebase has
actually hit before (permanent floor bias, decay-to-zero-forever):

- **Real distributional spread, not degenerate at either extreme**: post-calm-floor-fix
  (after PR #1391, 2026-07-26 21:08 UTC), 40.5% of ticks read near-zero (genuinely calm),
  56.4% mid-range, 3.1% saturated at the ceiling — a real distribution, not stuck high or
  stuck at zero.
- **Genuinely event-driven, not clock-driven**: updates irregularly (roughly every 30s–3min,
  not a fixed cadence), with real magnitude variance (0.0003 to 1.0) tracking real bus-mirror
  edge z-scores, not a synthetic oscillation.
- **Mean-reverting, not sticky**: one real spike to 0.90 decayed back through 0.27 → 0.06 →
  0.0003 within 8 minutes — recovers to calm rather than pinning high, the exact failure mode
  `node:substrate.route` had (48h of unopposed decay, a different bug, but the same class of
  "looks alive, isn't actually responsive" risk this check exists to catch).
- **Gap, disclosed**: could not independently corroborate any *specific* spike against an
  external anomaly (e.g., confirm a real incident happened at a given spike's timestamp) —
  container logs had already rotated past the checked window. This verifies the instrument's
  statistical health, not a causal story behind any one reading. Acceptance check #1 below
  should still be re-run against a live capability-gating call site once one exists, per its
  original text — this check verifies the source signal, not the eventual consumer.

**What's still genuinely open, not solved by this re-scope**: `journal.compose.episode` (the
third of the four capabilities named in the original Missing Question 2) has no direct
`bus_synaptic`-backed surprise the way `web.fetch.readonly`/`recall.query.readonly` now do
(their own `action_outcomes` rows carry it, via PR #1403) — it isn't a fetch/recall action, so
there's no natural per-outcome-row signal to point at. **Proposed, not yet built**: give
`journal.compose.episode` the exact same ambient `domain_surprise_score` the other two get,
read directly off `node:substrate.bus_synaptic` at gate-evaluation time rather than off its own
outcome row — the signal is mesh-wide by construction, not action-specific, so there's no
theoretical reason it needs a per-capability outcome row to attach to. This means Missing
Question 2 (the original "which of four domains maps to which capability" judgment call)
dissolves for all three real capabilities at once: there's only one domain now
(`bus_synaptic`), so there's nothing left to map.

## Current architecture

- **`evaluate_capability(capability_id, ctx)`** (`orion/autonomy/capability_policy.py:109-159`):
  a strictly sequential chain of hard boolean gates — budget exhaustion, `requires_goal_
  status`, `required_drive_origins` (untouched by this design, see above), `required_signal_
  kinds`, goal-status sufficiency, `side_effect_class` promotion, and two env-gated
  auto-execute threshold checks. **Hard safety/workflow gates stay untouched**: budget,
  goal-status/promotion, `required_signal_kinds` (already field-native — e.g.
  `world_coverage_gap`). This design adds one new, independent condition; it does not touch,
  replace, or reason about any existing gate, including `required_drive_origins`.
- **The real surprise substrate this design builds on, as of the 2026-07-28 re-scope**:
  `bus_synaptic_prediction_error()` (`orion/substrate/prediction_error.py`), already live,
  already generic across the whole bus mesh, already unrelated to drives, already durably
  published to `substrate_field_state`'s `node:substrate.bus_synaptic` node by
  `orion-substrate-runtime`'s `_bus_synaptic_tick`. No new call site, no new tracker — see
  Re-scope, 2026-07-28. (`NormalizationContext`/`EwmaBand`, the original 2026-07-25 substrate,
  remains real and reusable for other purposes — see the Arsonist summary's own finding on it
  — just not needed here now that a better-grounded signal already exists live.)
- **Pragmatic-value substrate, real, thin**: `action_outcomes` (Postgres,
  `orion/autonomy/action_outcomes.py`), `success: bool | None` per row. **Real volume is
  thin**: 12 rows total since the Postgres rebuild, 2 distinct `kind` values. Not enough
  history for a trustworthy per-capability success-rate prior — an honest data-volume gap,
  named as a real precondition in Acceptance checks, not routed around.

## Missing questions

1. **Resolved and dissolved, 2026-07-28.** Originally: does a live baseline-tracking
   mechanism exist, and where does its call site live? Superseded — `bus_synaptic_
   prediction_error()` is already computed and durably published; there is no new tracker or
   call site to place anywhere. See Re-scope, 2026-07-28.
2. **Mostly dissolved, 2026-07-28, one real gap remains.** Originally: which of four
   `prediction_error.py` domains maps to which capability rule — a disclosed, no-literature-
   precedent judgment call. With one mesh-wide domain (`bus_synaptic`) instead of four, there
   is no mapping left to make for `web.fetch.readonly`/`recall.query.readonly` (same ambient
   signal, already on their own `action_outcomes` rows since PR #1403). **`journal.compose.
   episode` remains genuinely open** — proposed treatment (not yet built): same ambient
   `domain_surprise_score`, read directly off `node:substrate.bus_synaptic` at gate-evaluation
   time rather than off an outcome row (it has none of the fetch/recall shape to attach one
   to). Not solved in this patch; named here as the one real remaining gap.
3. **Is `action_outcomes` volume enough for a `success`-rate pragmatic-value prior?** Better,
   not yet sufficient. Volume grew substantially since 2026-07-25 (12 rows → 450+ as of
   2026-07-28, per this session's own dispatch-pipeline fixes), but still concentrated in one
   `kind` (`inspect`) — re-check the real distinct-`kind` count before building a per-capability
   prior, don't assume raw row count alone clears this.
4. **Hard gate or soft advisory for first deployment?** Unchanged: soft/advisory
   (`CapabilityDecisionV1.notes`, gating nothing) — given `capability_policy.py`'s existing
   gates are all hard, adding a new hard gate before its own signal quality is validated
   would repeat the "wire it live before proving it" mistake this session's `predicted_
   shift` reversion-formula fix (2026-07-23) was built to correct.

## Proposed schema / API changes

**New, additive-only context field** on `CapabilityEvaluationContext`
(`orion/autonomy/capability_policy.py:26-32`) — the pre-existing `goal: GoalProposalV1 |
None` field stays exactly as-is and is never read by any new code here:

```python
@dataclass
class CapabilityEvaluationContext:
    predictive_pressure: float
    curiosity_strength: float
    signal_kinds: list[str]
    goal: GoalProposalV1 | None
    budget_used: dict[str, int] = field(default_factory=dict)
    # New, additive. Named descriptively, not as "epistemic_value" -- see Arsonist summary
    # on why that term isn't earned yet, and not as "prediction_error_confidence" -- this
    # is a genuinely different (baseline-relative, not naive-delta) computation.
    domain_surprise_score: float | None = None
    domain_surprise_source: str | None = None  # which real domain this score came from
```

**New rule field** on `CapabilityPolicyRuleV1` (`orion/autonomy/models.py`), standalone:

```python
required_domain_surprise_below: float | None = None  # threshold on the new score
required_min_success_rate: float | None = None         # pragmatic-value threshold, needs history
```

**The actual computation, 2026-07-28 re-scope — no new tracking primitive, no new call site,
reuses the already-durable `bus_synaptic_prediction_error()` publish path as-is:**

```python
def latest_domain_surprise(engine: Engine) -> float | None:
    """Read the current mesh-wide surprise value for capability gating -- same function
    services/orion-execution-dispatch-runtime already uses for ActionOutcomeEmitV1.surprise,
    no new query or staleness logic to write.
    """
    from orion.substrate.bus_synaptic_surprise import latest_bus_synaptic_prediction_error
    return latest_bus_synaptic_prediction_error(engine)
```

The `domain_surprise_source` field below is now always `"bus_synaptic"` when the score is
present -- kept on the schema for forward-compatibility (e.g., if `journal.compose.episode`
or a future capability ever gets its own distinct signal), not because there is a mapping to
disclose today.

**New pure gate function**, mirroring the existing style, reading only the new context
fields — never `ctx.goal`:

```python
def _domain_surprise_gate(ctx: CapabilityEvaluationContext, rule: CapabilityPolicyRuleV1) -> tuple[bool, str]:
    if rule.required_domain_surprise_below is None:
        return True, "no_surprise_gate"
    if ctx.domain_surprise_score is None:
        return False, "domain_surprise_unavailable"  # honest absence, not a silent pass
    if ctx.domain_surprise_score >= rule.required_domain_surprise_below:
        return False, "domain_surprise_insufficient"
    return True, "domain_surprise_satisfied"
```

Wired into `evaluate_capability()` as one more independent link in the existing chain — its
own condition, not paired with, compared against, or gated on `required_drive_origins`.

**No change** to the hard safety/workflow gates — budget, goal-status, promotion,
`required_signal_kinds` all untouched.

## Files likely to touch (if/when this proceeds past design)

```
orion/autonomy/capability_policy.py          # new gate function, wired independently
orion/autonomy/models.py                     # CapabilityPolicyRuleV1 new fields
config/autonomy/capability_policy.v1.yaml    # new fields on whichever rules adopt the new gate
orion/substrate/bus_synaptic_surprise.py     # reused as-is -- no change expected
scripts/analysis/                            # new shadow-measurement/replay script
orion/autonomy/tests/test_capability_policy.py
```

(2026-07-28: `orion/signals/normalization.py` and `services/orion-substrate-runtime/app/` no
longer appear here -- no new tracker or call site is needed, see Re-scope above.)

## Non-goals

- Not implementing anything in this patch — design/proposal mode only, per `CLAUDE.md` §0A.
- Not building multi-candidate action selection (§9a item 2, Society-of-Mind) — single-
  candidate admission gating only.
- **Not integrating with `candidate_society_of_mind.py`'s Borda-count competition.**
  Considered and rejected in an arsonist pass on an earlier draft: that mechanism's own
  docstring discloses its three-scorer combination has never been validated against real
  data (only `magnitude_scorer` has) and its magnitude/novelty scorer target universes don't
  currently overlap in live data at all. Bolting a fourth, also-unvalidated scorer onto an
  already-incomplete validation is stacking unmeasured complexity on unmeasured complexity —
  the opposite of measure-before-minting. If Candidate B's own acceptance check closes later,
  revisit; not before.
- **Not touching `orion-field-digester`'s ingestion pipeline.** Also considered and rejected:
  the charter's own §2 states this program does not govern the field substrate itself
  ("this program *consumes* and *wires to* those, it does not own them"). Any fix to
  field-digester's node-ingestion is a separate, separately-gated proposal, not folded into
  this design.
- Not touching the hard safety/workflow gates (budget, goal-status, promotion,
  `required_signal_kinds`).
- **Never reading, writing, persisting, or validating against `GoalProposalV1`, `drive_
  origin`, or `required_drive_origins`, anywhere, for any reason** — including as a
  validation baseline.
- Not using `action_outcomes.surprise` generically across every emitter — still a fake
  success/fail proxy for `episode_fetch.py`/`policy_act.py`/`curiosity_reuse.py`'s *other*
  fields (`success`, etc.); this design reads `bus_synaptic_prediction_error()` directly, not
  through `action_outcomes` at all.
- Not building on `DeviationGate` or `NormalizationContext`/`EwmaBand` — both real, both
  reusable elsewhere, neither needed here now that `bus_synaptic_prediction_error()` already
  exists live (see Re-scope, 2026-07-28).
- **Not building `journal.compose.episode`'s wiring in this patch** — proposed (ambient
  `domain_surprise_score`, same as the other two capabilities) but not implemented; the real
  remaining piece of the original Missing Question 2.
- Not calling the new field "epistemic value" — naming honesty per the literature check;
  not calling it "retrospective surprise" either, since that correspondence hasn't been
  independently verified against Sajid et al.'s actual formula.

## Acceptance checks (all required before anything here goes live; none run in this patch)

1. **Live-data sanity check**: done at the source-signal level 2026-07-28 (see Re-scope) --
   real spread, event-driven, mean-reverting. Still needs re-running specifically at the
   moments a real capability gate is actually evaluated once that call site exists, not just
   confirmed non-degenerate in general.
2. **Outcome-grounded validation, never drives-agreement validation.** For real historical
   moments where a gated capability was actually exercised (`action_outcomes`), check whether
   `bus_synaptic_prediction_error()`'s value at that moment correlates with `success`. Report
   the correlation and whether the gate's hypothetical allow/deny split is non-degenerate.
   "Compiles" and "schema-valid" are not acceptance criteria, per root `CLAUDE.md`'s
   metric-quality-gate. No longer needs a `substrate_field_state`-to-`action_outcomes`
   timestamp join for `web.fetch.readonly`/`recall.query.readonly` -- their own outcome rows
   already carry the real surprise value directly (PR #1403); still needs the join for
   `journal.compose.episode` once that capability has its own signal wired.
3. **Capability→domain mapping validation**: dissolved for `web.fetch.readonly`/
   `recall.query.readonly` (one shared domain, nothing to map). Still applies, narrowly, to
   confirming `journal.compose.episode`'s proposed ambient-signal treatment actually tracks
   its real invocations meaningfully once built.
4. **Pragmatic-value prior needs real volume first**: `action_outcomes` volume grew
   substantially (12 → 450+ rows) but re-check distinct-`kind` coverage before building a
   per-capability `success`-rate gate — don't assume raw row count alone clears this.
5. **Ship as soft/advisory first**: first real deployment surfaces the score in
   `CapabilityDecisionV1.notes` without gating anything, observed against real traffic,
   before flipping to a hard gate.
6. **Reversibility**: every schema change is additive; the new gate function is a pure,
   isolated addition — removable by deleting one `if` block and two config fields, no data
   migration, no consumer breakage.

## Recommended next patch

Not implementation — still design/proposal mode. Two things, in order, now that the
re-scope removes the architecture question that used to block both:

1. Build the read-only outcome-correlation replay script (Acceptance check #2) for
   `web.fetch.readonly`/`recall.query.readonly` — no longer needs a `substrate_field_state`
   timestamp-join for these two; it's a direct `corr(action_outcomes.surprise,
   action_outcomes.success)` query against their own rows (PR #1403's real surprise values).
   No longer blocked on data volume the way it was at 12 rows — re-check the real distinct-row
   count for these two `kind`s specifically before running it.
2. Decide `journal.compose.episode`'s treatment concretely (the one real remaining gap from
   the original Missing Question 2) — wire it the ambient way proposed above, or leave it
   ungated for now and revisit once its own real invocation volume exists. Either is
   acceptable; leaving it silently unaddressed is not.

## Source material

- `orion/sentience_striving_program/README.md` §9a item 3 — the named blue-sky direction.
- `orion/autonomy/capability_policy.py`, `orion/autonomy/models.py` — the real gate chain.
- `orion/autonomy/models.py:167-206`, `orion/autonomy/episode_fetch.py:73-107`,
  `orion/autonomy/policy_act.py` — `surprise`/`success` call sites, traced for degeneracy.
- `orion/substrate/prediction_error.py` — the five domain functions, confirmed naive-delta.
- `orion/autonomy/deviation_gate.py` — confirmed drives-only (2 real callers, both halted).
- `orion/signals/normalization.py` — `EwmaBand`/`InductionTracker`/`NormalizationContext`,
  confirmed generic and live, the actual substrate this design reuses.
- `orion/attention/field_attention/candidate_society_of_mind.py` — Candidate B, real,
  shadow-only, its own disclosed validation gaps named in Non-goals.
- Literature: Rao & Ballard 1999 (*Nature Neuroscience*); Friston, "The free-energy
  principle: a unified brain theory?" (*Nature Reviews Neuroscience* 2010); Friston et al.
  2015, "Active inference and epistemic value"; Parr & Friston 2019, "Generalised free
  energy and active inference" (*Biological Cybernetics*); Sajid, Parr, Da Costa & Friston
  2020, "Retrospective surprise" (*J. Mathematical Psychology*); Millidge, Tschantz &
  Buckley 2021, "Whence the Expected Free Energy?" (*Neural Computation*); Hyndman &
  Athanasopoulos, *Forecasting: Principles and Practice* §5.2.
- Live Postgres checks (2026-07-24/25): `action_outcomes` row count/degeneracy,
  `evaluate_capability()` call-site enumeration, `DeviationGate`/`NormalizationContext`
  real-caller enumeration.
- `services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update()` — real per-edge
  EWMA/z-score on directly-measured bus traffic, the actual source `bus_synaptic_prediction_
  error()` aggregates.
- `services/orion-signal-gateway/app/normalization_state.py` — confirms `NormalizationContext`
  is a plain in-process dict, zero persistence, one instance per process.
- `orion/substrate/bus_synaptic_surprise.py` — the shared, already-live query+staleness
  function this re-scope reuses directly (PR #1400/#1403).
- Live Postgres/log checks (2026-07-28): `substrate_field_state` full-history temporal
  analysis of `node:substrate.bus_synaptic` (distribution, event-driven update cadence,
  mean-reversion after a real spike), `evaluate_capability()` call-site re-enumeration
  (confirmed 2+ separate services).
