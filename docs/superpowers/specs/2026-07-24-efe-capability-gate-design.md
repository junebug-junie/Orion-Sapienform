# Expected-free-energy capability gating — design (Sentience Striving Program, §9a item 3)

Status: design/proposal mode per root `CLAUDE.md` §0A. Nothing implemented. This does not
authorize any change to `capability_policy.py`'s live gating behavior — it names what
would change, what data it touches, what could go wrong, and what would have to be measured
before any of it goes live.

## Arsonist summary

`orion/sentience_striving_program/README.md` §9a item 3 names "Free-energy/active-inference
reframing — `capability_policy` as literal expected-free-energy action selection" as an
unbuilt blue-sky direction. Read literally against real Active Inference theory, that would
mean: given several candidate actions, pick the one minimizing expected free energy
(epistemic value + pragmatic value). **Read against the real code**
(`orion/autonomy/capability_policy.py::evaluate_capability()`), that framing overclaims what
this function actually does today: it is called once per capability_id, from four fixed
call sites (`orion/autonomy/policy_act.py` x3,
`services/orion-world-pulse/app/services/curiosity.py` x1), each hardcoded to a single
`_READONLY_CAPABILITY`/`_RECALL_CAPABILITY`/`_EPISODE_JOURNAL_CAPABILITY` constant — never a
competition among several candidates. There is no "selection" happening anywhere in this
codebase today; there is single-candidate admission control. This design is scoped
honestly to that reality: **EFE-based admission gating for one candidate at a time**, not
EFE-based selection among several. Real multi-candidate competition is a different,
separate direction (§9a item 2, Society-of-Mind) that could sit on top of this later — not
conflated with it here.

**Real, load-bearing finding from grounding this in actual code (2026-07-24), not assumed:**
`ActionOutcomeRefV1.surprise` (`orion/autonomy/models.py:174`) looked like the obvious
epistemic-value signal to reuse — a `float` field named exactly what Active Inference calls
its own epistemic term. It is not usable as one. Every real call site that sets it
(`orion/autonomy/episode_fetch.py:78-97`, `orion/autonomy/policy_act.py` lines 150/175/187/
200/217) computes it as a **binary success/fail proxy** — `surprise = 0.0 if success else
1.0`, or a hardcoded `1.0` for "something notable happened." It carries zero continuous
uncertainty information; it is a relabeled inverse of the `success` field sitting right next
to it in the same schema. Confirmed against live data: 12 real rows in `action_outcomes`
since the Postgres rebuild, `surprise` reads exactly `0.0` for all 12 (both real `kind`
values present, `inspect` and `web.fetch.readonly`) — completely degenerate, exactly the
kind of thing CLAUDE.md's metric-quality-gate's live-data-sanity-check exists to catch
before treating a field as real signal. Using it here would have repeated, in a brand-new
patch, the identical mistake (trust the field name, skip the live-data check) this whole
session has been correcting in other people's code.

**What this means for the design**: epistemic value cannot come from `action_outcomes`. It
has to come from the substrate this program already spent real effort validating —
`orion/substrate/prediction_error.py`'s five domains, surfaced through
`orion/substrate/attention_self_model.py`'s `AttentionSelfModelV1.prediction_error_
confidence`/`predicted_shift` (real, live, non-degenerate — `prediction_error_confidence`
populates ~99.9% of ticks per the 2026-07-24 branch-starvation fix, PR #1329). Pragmatic
value can still come from `action_outcomes` — just from the real field, `success`, not the
degenerate one.

## Current architecture

- **`evaluate_capability(capability_id, ctx)`** (`orion/autonomy/capability_policy.py:109-159`):
  a strictly sequential chain of hard boolean gates, each an immediate `deny`/`requires_
  promote` on failure, falling through to `allowed` only if every gate passes. In order:
  budget exhaustion, `requires_goal_status`, `required_drive_origins` (the halted-taxonomy
  coupling this whole program exists to remove), `required_signal_kinds`, goal-status
  sufficiency, `side_effect_class` promotion requirement, and (for auto-execute readonly/
  episode-journal capabilities only) two env-gated pressure/curiosity threshold checks
  (`_layer_a_readonly_auto_enabled`/`_layer_a_episode_journal_enabled`).
- **Correction (2026-07-25): this design does not touch, coexist with, replace, compare
  against, or in any way reference `required_drive_origins`, `GoalProposalV1`, or
  `drive_origin` — not transitionally, not for validation, not ever.** An earlier version
  of this document proposed the new EFE gate live "additive next to" `required_drive_origins`
  and validated against real historical `drive_origin_mismatch` decisions. Both ideas were
  wrong: the Sentience Striving Program's whole purpose is killing the drives/goals
  apparatus, and using it — even read-only, even just as a comparison baseline — is new
  investment in the thing being retired, and would have made this design's own validation
  circular (using the system being replaced as the ground truth for its replacement). The
  halted `required_drive_origins` gate's eventual removal from `capability_policy.v1.yaml`
  is a separate decision (Objective 3 Phase 5's retirement, once every producer has
  migrated) — this design does not depend on that timeline and does not need to know
  `required_drive_origins` exists at all.
- **These gates are not all the same kind of thing** — this matters for what EFE reframing
  should and shouldn't touch:
  - **Hard safety/workflow gates, untouched**: `budget_per_cycle` (rate limiting),
    `requires_goal_status`/promotion checks (workflow-stage gating), `required_signal_kinds`
    (a real, already-field-native precondition — e.g. `world_coverage_gap`). None of these
    are "is this theoretically the right kind of goal" questions; they're "is it
    safe/appropriate to act right now" questions, and an EFE score should not override them.
  - **This design adds a new, independent condition** — not a replacement for, extension of,
    or comparison against `required_drive_origins`. Whether that gate is later deleted from
    the YAML is out of scope here (Phase 5's job); this design's own code has zero reason to
    read `ctx.goal` or anything on it.
- **Epistemic-value substrate, real, already validated**:
  `orion/substrate/attention_self_model.py::reduce_attention_self_model()` — pure function,
  no I/O (confirmed by its own module docstring and by
  `orion/substrate/tests/test_attention_self_model.py`'s DB-free fixtures). Given
  `prediction_error_by_domain` (execution/transport/biometrics/chat/route — transport
  confirmed dead, excluded from confidence per PR #1329) and
  `prediction_error_trend_by_domain`, it derives `prediction_error_confidence` (a
  population-wide `1 - mean(prediction_error)` scalar, additive, unconditional — the field
  this design should read) and `predicted_shift` (a narrative naming whichever domain is
  currently trending fastest). Both are computed by
  `scripts/analysis/measure_ast_hot_reducer.py`'s replay layer today (offline replay only,
  no live bus wiring) — the reducer itself takes these as caller-supplied inputs, it does
  not compute them. **A live capability-gating consumer would need this same trend/
  confidence computation to exist somewhere live**, not just in the offline replay script —
  see Missing Questions.
- **Pragmatic-value substrate, real, thin**: `action_outcomes` (Postgres,
  `orion/autonomy/action_outcomes.py`), `success: bool | None` per row, keyed by `kind` (a
  free-text action-kind label, e.g. `web.fetch.readonly`, `inspect`) and `subject` (always
  `"orion"` — self-directed only, confirmed in the channel registry comment). **Real
  volume is thin**: only 12 rows total since the Postgres rebuild (2026-07-23), 2 distinct
  `kind` values. Not enough real history yet to compute a trustworthy empirical
  success-rate-per-capability prior — this is an honest data-volume gap, not a design flaw,
  and the acceptance check below names it as a real precondition, not something to route
  around with a smaller sample or a synthetic prior.

## Missing questions

1. **Is there a live-wired place that computes `prediction_error_confidence`/`predicted_
   shift` today, or does one need to be built first?** Confirmed: no. Today this
   computation exists only inside `scripts/analysis/measure_ast_hot_reducer.py`'s offline
   replay layer, reading historical Postgres rows. `reduce_attention_self_model()` itself
   takes these as pre-computed arguments — nothing in a running service computes them from
   live `substrate_field_state` ticks and calls the reducer in real time. Before any EFE
   gate can read a live `prediction_error_confidence`, either (a) a new lightweight live
   consumer (in `orion-substrate-runtime` or a new small service) replicates the replay
   script's trend computation on the live tick stream and calls the reducer per-tick, or (b)
   the capability-gating callsite pulls the most recent N `substrate_field_state` rows
   itself and computes the trend inline. Neither exists today — this is real, non-trivial
   scope, not a config flip.
2. **What does "epistemic value of THIS candidate action" mean concretely**, given
   `prediction_error_confidence` is a population-wide scalar (one number across all live
   domains), not a per-action, per-capability quantity? A capability request (e.g.
   `web.fetch.readonly`) doesn't obviously map to "reduces domain X's prediction error" the
   way, e.g., a literal Active-Inference agent's discrete action space would. The most
   honest mapping found in this pass: `required_signal_kinds` already ties some rules to a
   specific real-world condition (`world_coverage_gap`) — epistemic value could be scoped as
   "is the domain this capability is meant to address currently showing elevated/rising
   prediction error" (using `predicted_shift`'s domain-trend narrative), rather than a
   generic population-wide score applied identically to every capability. This needs a real
   per-capability-to-domain mapping decision, not assumed here.
3. **Is 12 rows of `action_outcomes` history enough to trust a `success`-rate pragmatic-value
   prior for any capability?** No — not close. This is the same "insufficient accumulated
   history" pattern this whole session has hit repeatedly post-rebuild (predicted_shift's
   TEST validation, the vol-trigger acceptance check). Needs real time to pass, not a
   design fix.
4. **Should the EFE score be a hard additional gate (deny below threshold) or a soft
   `notes`-carried advisory** (visible in `CapabilityDecisionV1.notes`, influencing nothing
   yet) for its first live deployment? Given `capability_policy.py`'s existing gates are all
   hard, adding a new hard gate before its own signal quality is validated would repeat the
   "wire it live before proving it" mistake `predicted_shift`'s own reversion-formula fix
   (2026-07-23) was built to correct. Recommend soft/advisory first — named in Acceptance
   checks below, not decided here.
5. **Retracted (2026-07-25).** This question assumed the new gate needed to coordinate with
   `required_drive_origins`'s replacement. It doesn't — this design never reads or reasons
   about `drive_origin` at all, so there is no shared code path to coordinate. The sibling
   goal-provenance brainstorm's own fate is unrelated to whether this design proceeds.

## Proposed schema / API changes

**New, additive-only context field** on `CapabilityEvaluationContext`
(`orion/autonomy/capability_policy.py:26-32`) — note the pre-existing `goal: GoalProposalV1
| None` field stays exactly as-is (it's real, unrelated production code this design doesn't
touch); the new fields below are the only addition, and no new code reads `.goal` at all:

```python
@dataclass
class CapabilityEvaluationContext:
    predictive_pressure: float
    curiosity_strength: float
    signal_kinds: list[str]
    goal: GoalProposalV1 | None
    budget_used: dict[str, int] = field(default_factory=dict)
    # New, additive:
    prediction_error_confidence: float | None = None
    predicted_shift_domain: str | None = None
```

**New rule field** on `CapabilityPolicyRuleV1` (`orion/autonomy/models.py`) — standalone,
not paired with or contingent on `required_drive_origins` in any way:

```python
required_prediction_error_confidence_below: float | None = None  # epistemic-value threshold
required_min_success_rate: float | None = None                    # pragmatic-value threshold, needs history
```

**New pure function**, mirroring the existing gate style exactly (no new abstraction layer),
reading only the new context fields — never `ctx.goal`:

```python
def _efe_epistemic_gate(ctx: CapabilityEvaluationContext, rule: CapabilityPolicyRuleV1) -> tuple[bool, str]:
    if rule.required_prediction_error_confidence_below is None:
        return True, "no_efe_gate"
    if ctx.prediction_error_confidence is None:
        return False, "prediction_error_confidence_unavailable"  # honest absence, not a silent pass
    if ctx.prediction_error_confidence >= rule.required_prediction_error_confidence_below:
        return False, "efe_epistemic_value_insufficient"
    return True, "efe_epistemic_satisfied"
```

Wired into `evaluate_capability()` as one more independent link in the existing sequential
chain — its own condition, standing alone, not paired with, compared against, or gated on
`required_drive_origins`'s presence, absence, or eventual removal.

**No change** to the hard safety/workflow gates named in Current Architecture — budget,
goal-status, promotion, `required_signal_kinds` all untouched.

## Files likely to touch (if/when this proceeds past design)

```
orion/autonomy/capability_policy.py          # new gate function, wired independently
orion/autonomy/models.py                     # CapabilityPolicyRuleV1 new fields
config/autonomy/capability_policy.v1.yaml    # new fields on whichever rules adopt the new gate
orion/substrate/attention_self_model.py      # no change expected -- reused as-is
services/orion-substrate-runtime/app/        # new live consumer computing prediction_error_confidence in real time (Missing Q1) -- the actual non-trivial build
scripts/analysis/                            # new shadow-measurement/replay script -- required before any live wiring
orion/autonomy/tests/test_capability_policy.py
```

## Non-goals

- Not implementing anything in this patch — design/proposal mode only, per `CLAUDE.md` §0A.
- Not building multi-candidate action selection (that's §9a item 2, Society-of-Mind) — this
  is single-candidate admission gating only, scoped honestly against what `evaluate_
  capability()` actually is today.
- Not touching the hard safety/workflow gates (budget, goal-status, promotion,
  `required_signal_kinds`).
- **Never reading, writing, persisting, or validating against `GoalProposalV1`, `drive_
  origin`, or `required_drive_origins`, anywhere in this design, for any reason** — including
  as a validation baseline. The Sentience Striving Program exists to kill that apparatus;
  this design does not invest in it even transitionally or for comparison purposes.
- Not using `action_outcomes.surprise` for anything — confirmed degenerate/mislabeled, a
  real finding of this pass, not a stylistic choice.
- Not committing to a specific per-capability epistemic-domain mapping (Missing Question 2)
  — named as the real open design fork, not guessed.

## Acceptance checks (all required before anything here goes live; none run in this patch)

1. **Live-data sanity check on the new signal in its actual gating context**: once a live
   `prediction_error_confidence` consumer exists (Missing Question 1), confirm it is
   non-degenerate specifically at the moments the three `[predictive]`-gated capabilities are
   actually evaluated (not just non-degenerate in general, already shown by PR #1329's own
   work) — a signal can be real in aggregate and still happen to be flat at the particular
   ticks that matter here.
2. **Outcome-grounded validation, not drives-agreement validation.** The new gate is judged
   against real outcomes, never against what the halted drives system would have decided.
   Concretely: for real historical moments where a gated capability was actually exercised
   (`action_outcomes`), check whether `prediction_error_confidence` at that moment
   correlates with `success` — does the epistemic-value signal actually carry real
   information about whether the action worked, on its own terms. Report the correlation
   and whether the gate's own hypothetical allow/deny split (at a candidate threshold) is
   non-degenerate (not always-allow, not always-deny). Must be genuinely convincing per
   program §7 — "compiles" and "schema-valid" are not acceptance criteria, per root
   `CLAUDE.md`'s metric-quality-gate.
3. **Pragmatic-value prior needs real volume first**: do not build a `success`-rate gate
   until `action_outcomes` has accumulated meaningfully more than 12 rows across enough
   distinct capability kinds to compute a non-noisy per-capability rate — a real, dated
   precondition, re-check the row count before building this half.
4. **Ship as soft/advisory first** (Missing Question 4): first real deployment surfaces the
   EFE score in `CapabilityDecisionV1.notes` without gating anything, observed against real
   traffic for a real window, before flipping it to a hard gate.
5. **Reversibility**: every schema change above is additive; the new gate function is a
   pure, isolated addition to an existing sequential chain — removable by deleting one
   `if` block and two config fields, no data migration, no consumer breakage.

## Recommended next patch

Not implementation — the two things every acceptance check above is blocked on:

1. Resolve Missing Question 1 concretely: decide where a live `prediction_error_confidence`
   computation would actually run (new small consumer in `orion-substrate-runtime`, vs. an
   inline pull-and-compute at the `capability_policy.py` callsite) — this is real
   architecture work, not a brainstorm answer, and should get its own focused design pass
   given it's the one piece every other option in this document depends on.
2. Build the read-only outcome-correlation replay script (Acceptance check #2) — joins real
   `substrate_field_state`/`prediction_error` history to real `action_outcomes.success` by
   nearest-preceding timestamp, same pattern as every other replay script this session, and
   answers whether the epistemic-gate idea is even worth the live-wiring investment in #1
   before that investment is made. Blocked today on `action_outcomes`' thin volume (12 rows)
   — re-check row count before building, not a reason to skip building it once volume grows.

## Source material

- `orion/sentience_striving_program/README.md` §9a item 3 — the named blue-sky direction
  this design scopes.
- `orion/autonomy/capability_policy.py`, `orion/autonomy/models.py` — the real current gate
  chain, read in full before writing this document.
- `orion/autonomy/models.py:167-206`, `orion/autonomy/episode_fetch.py:73-107`,
  `orion/autonomy/policy_act.py` — the real `surprise`/`success` call sites, traced to
  confirm `surprise`'s degeneracy before ruling it out.
- `orion/substrate/attention_self_model.py`, `orion/schemas/attention_self_model.py` — the
  real epistemic-value substrate this design reuses.
- Live Postgres checks (2026-07-24): `action_outcomes` row count/degeneracy,
  `evaluate_capability()` call-site enumeration.
