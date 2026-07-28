# Precision-weighted surprise gating for `capability_policy.py` (Sentience Striving Program, §9a item 3)

Status: design/proposal mode per root `CLAUDE.md` §0A. Nothing implemented. This does not
authorize any change to `capability_policy.py`'s live gating behavior. Converged 2026-07-25
after two rounds of independent verification (code re-tracing + literature search, both
run by separate agents instructed to find where this design was wrong, not confirm it) and
one arsonist pass on the resulting "big swing" proposal that found real overreach — this
version is the corrected, right-sized result, not another option to weigh.

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

## Current architecture

- **`evaluate_capability(capability_id, ctx)`** (`orion/autonomy/capability_policy.py:109-159`):
  a strictly sequential chain of hard boolean gates — budget exhaustion, `requires_goal_
  status`, `required_drive_origins` (untouched by this design, see above), `required_signal_
  kinds`, goal-status sufficiency, `side_effect_class` promotion, and two env-gated
  auto-execute threshold checks. **Hard safety/workflow gates stay untouched**: budget,
  goal-status/promotion, `required_signal_kinds` (already field-native — e.g.
  `world_coverage_gap`). This design adds one new, independent condition; it does not touch,
  replace, or reason about any existing gate, including `required_drive_origins`.
- **The real surprise substrate this design builds on**: `orion/signals/normalization.py`'s
  `NormalizationContext`, already live, already generic, already unrelated to drives (see
  Arsonist summary). Not yet applied to the four real `prediction_error.py` domains
  (execution/biometrics/chat/route — `transport` confirmed dead, per PR #1329, excluded).
  Applying it requires only new call sites feeding it real per-domain raw values (the same
  `pressure_hints`/projection deltas `prediction_error.py` already reads) — no new tracking
  code.
- **Pragmatic-value substrate, real, thin**: `action_outcomes` (Postgres,
  `orion/autonomy/action_outcomes.py`), `success: bool | None` per row. **Real volume is
  thin**: 12 rows total since the Postgres rebuild, 2 distinct `kind` values. Not enough
  history for a trustworthy per-capability success-rate prior — an honest data-volume gap,
  named as a real precondition in Acceptance checks, not routed around.

## Missing questions

1. **Resolved, 2026-07-25**: does a live baseline-tracking mechanism already exist, or does
   one need building? It already exists (`NormalizationContext`) — no new tracker needed.
   What's still open: **where does the new call site live** — inside
   `services/orion-substrate-runtime`'s existing per-domain tick loops (natural home, same
   process already computing the raw `prediction_error.py` deltas), or at the
   `capability_policy.py` callsite itself (pulls raw history and calls `NormalizationContext`
   inline)? Real architecture decision, not resolved here — the former keeps the tracker
   state co-located with the data it tracks and matches how `NormalizationContext` is already
   used by `orion-signal-gateway`; the latter avoids touching `orion-substrate-runtime` at
   all but means capability-gating owns tracker state, a genuinely different tradeoff.
2. **Still open, and confirmed to have no literature precedent either way (Arsonist
   summary)**: which of the four capability rules maps to which of the four real domains?
   `web.fetch.readonly`/`recall.query.readonly`/`journal.compose.episode` don't obviously
   correspond to `execution`/`biometrics`/`chat`/`route` the way a co-designed system's
   actions would. **This is not solvable by more research — it's a first-party judgment
   call that has to be made explicitly, disclosed as a judgment call, and validated
   empirically once real data exists** (Acceptance check #3) — not guessed and left silent.
   Empirical derivation (measure which domain's surprise moves most after each real
   capability invocation) is the theoretically honest path per O4 ("named categories...
   re-derivable, not a constant") but is **currently blocked on data volume** (12 rows) —
   named as a deferred workstream, not an active one, in Recommended next patch.
3. **Is 12 rows of `action_outcomes` enough for a `success`-rate pragmatic-value prior?** No.
   Same "insufficient accumulated history" pattern hit repeatedly this session post-rebuild.
   Needs real time, not a design fix.
4. **Hard gate or soft advisory for first deployment?** Recommend soft/advisory
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

**The actual computation — no new tracking primitive, reuses `NormalizationContext` as-is:**

```python
def compute_domain_surprise(
    ctx: NormalizationContext, domain: str, raw_value: float
) -> float:
    """Baseline-relative surprise for one real prediction_error domain, reusing the
    already-live, already-generic EwmaBand mechanism (orion/signals/normalization.py) --
    not a new tracker. `domain` is a NormalizationContext organ_id/metric_key pair, e.g.
    ("substrate", "biometrics"). Returns EwmaBand.normalize()'s [0,1] deviation-relative
    score directly -- update() must be called by the same caller feeding it real ticks
    over time; this function does not itself own persistence.
    """
    band = ctx.get_band("substrate", domain)
    band.update(raw_value)
    return band.normalize(raw_value)
```

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
orion/signals/normalization.py               # reused as-is -- no change expected
services/orion-substrate-runtime/app/        # new call site feeding NormalizationContext
                                              # real per-domain values (Missing Q1's open half)
scripts/analysis/                            # new shadow-measurement/replay script
orion/autonomy/tests/test_capability_policy.py
```

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
- Not using `action_outcomes.surprise` for anything — confirmed degenerate/mislabeled.
- Not building on `DeviationGate` — its class carries no drive-specific state, but its only
  real callers today are inside the halted system; `NormalizationContext` achieves the same
  mechanism with zero adjacency.
- **Not empirically deriving the capability→domain mapping in this patch** (Missing
  Question 2) — blocked on `action_outcomes` data volume, named as deferred, not attempted
  with a synthetic or guessed substitute.
- Not calling the new field "epistemic value" — naming honesty per the literature check;
  not calling it "retrospective surprise" either, since that correspondence hasn't been
  independently verified against Sajid et al.'s actual formula.

## Acceptance checks (all required before anything here goes live; none run in this patch)

1. **Live-data sanity check**: once a live `NormalizationContext` call site exists (Missing
   Question 1), confirm `domain_surprise_score` is non-degenerate specifically at the moments
   gated capabilities are actually evaluated — not just non-degenerate in general.
2. **Outcome-grounded validation, never drives-agreement validation.** For real historical
   moments where a gated capability was actually exercised (`action_outcomes`), check whether
   `domain_surprise_score` at that moment correlates with `success`. Report the correlation
   and whether the gate's hypothetical allow/deny split is non-degenerate. "Compiles" and
   "schema-valid" are not acceptance criteria, per root `CLAUDE.md`'s metric-quality-gate.
3. **Capability→domain mapping validation, once data volume allows**: for each real
   capability invocation, check which domain's `domain_surprise_score` moved most
   immediately after — does the disclosed judgment-call mapping (Missing Question 2) survive
   contact with real data, or does a different domain actually track each capability better.
4. **Pragmatic-value prior needs real volume first**: do not build a `success`-rate gate
   until `action_outcomes` has accumulated meaningfully more than 12 rows across enough
   distinct capability kinds.
5. **Ship as soft/advisory first**: first real deployment surfaces the score in
   `CapabilityDecisionV1.notes` without gating anything, observed against real traffic,
   before flipping to a hard gate.
6. **Reversibility**: every schema change is additive; the new gate function is a pure,
   isolated addition — removable by deleting one `if` block and two config fields, no data
   migration, no consumer breakage.

## Recommended next patch

Not implementation — the two things every acceptance check is blocked on, in order:

1. Resolve Missing Question 1's remaining half concretely: decide where the new
   `NormalizationContext` call site actually lives (`orion-substrate-runtime`'s existing tick
   loops vs. an inline pull at the `capability_policy.py` callsite) — real architecture work,
   not a brainstorm answer.
2. Build the read-only outcome-correlation replay script (Acceptance check #2) — joins real
   `substrate_field_state`/raw domain values (fed through `EwmaBand` offline, same math, same
   pattern as every other replay script this session) to real `action_outcomes.success` by
   nearest-preceding timestamp. Blocked today on `action_outcomes`' thin volume (12 rows) —
   re-check row count before building, not a reason to skip building it once volume grows.

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
