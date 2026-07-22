# Attention/salience cathedral replacement — tentative plan

Status: **design/proposal mode, tentative, not sign-off-approved for implementation.**
Per root `CLAUDE.md` §0A this is cognition-loop-adjacent (it changes what shapes what
Orion attends to and what it is permitted to do) and needs explicit Juniper sign-off
before any live-behavior-affecting patch. This document records the plan discussed and
corrected live in conversation on 2026-07-21 — including two real course-corrections
mid-discussion — so the reasoning survives past the conversation, not just the
conclusion.

## Origin

`scripts/analysis/measure_emergent_clustering_probe.py` (Sentience Striving Program §6
item 5, PR #1230) found `field:recent_perturbations` winning attention 99.98% of
127,936 real ticks — at or above the drives system's own pre-fix 96% monoculture, the
exact pathology the whole program exists to escape. Root-caused to
`select_system_targets()` (`orion/attention/field_attention/selectors.py:128-153`)
bypassing the shared salience formula entirely: `salience = min(1.0,
recent_perturbation_count / 10.0)`, novelty/urgency/confidence hardcoded to `0.0`. A
near-constant real perturbation rate saturates this to ~1.0 almost permanently — not
because the signal is genuinely most important, but because its scoring math never had
to prove it, unlike everything else it competes against.

## Finding: the disease is systemic, not one bug

Investigating whether `compute_salience()` (the formula `select_system_targets` *should*
be competing against) is itself trustworthy surfaced three independent instances of the
same shape — hand-typed linear-weighted-sum, zero citation, zero calibration, never
outcome-validated:

1. `orion.autonomy.endogenous_origination`'s D/W/A composite (`P = w_drift*D + w_dwell*W
   + w_agency*A`) — already proven dead: measured against 84,511 real ticks (PR #1156),
   never fired once.
2. `orion/attention/field_attention/scoring.py::compute_salience()` — `pressure*0.45 +
   novelty*0.20 + urgency*0.25 + confidence*0.10` (`AttentionWeightsV1`,
   `config/attention/field_attention_policy.v1.yaml`). Authored 2026-05-25, two months
   before this program existed, touched twice since (both feature additions, never a
   calibration pass), zero comment justifying the split or the ~25 per-channel weights
   (`node_channel_weights`/`capability_channel_weights`) sitting alongside it.
3. `orion/substrate/attention/salience.py::LinearSalienceCombiner` — `SEED_WEIGHTS`,
   explicitly self-labeled `WEIGHTS_VERSION = "seed-v1"` — an acknowledged placeholder
   that never got a v2, governing the GWT-dispatch (Rung-3) coalition scorer.

None of the three has ever been checked against real outcomes. All three were kept
because they looked better than the drives system by comparison, which is exactly the
"math got fixed, taxonomy didn't" trap named in the charter's own §1 historical account
as the original mistake this whole program exists to correct — just never turned
backward onto pre-existing infrastructure until this thread.

## Finding: delivery is unverified — signal may die before it reaches consequence

Traced the real downstream path from `compute_salience()`'s output, in code, not
documentation:

- `orion/self_state/builder.py:213-442` takes the full `FieldAttentionFrameV1` and does
  exactly one thing with it: `dominant_attention_targets = [t.target_id for t in
  attention.dominant_targets[:5]]`. `pressure_score`/`novelty_score`/`urgency_score`/
  `confidence_score`/`dominant_channels`/`reasons` are computed and **discarded** —
  confirmed live in source, matching a design-doc finding from 2026-07-12 that was never
  acted on.
- The five surviving IDs do have a real consumer — `orion/proposals/builder.py:112`
  (`motivating_targets`) and `builder.py:30`
  (`ATTENTION_FIRST_TARGET_BINDING = "self_state.dominant_attention_targets[0]"`).
- But the resulting `ProposalCandidateV1` carries `execution_intent = {"mode":
  "descriptive_only", ...}` for at least some templates.
- `orion/execution_dispatch/builder.py` has a live `"dry_run"` dispatch-status branch
  (lines 60, 190, 202) — real, not hypothetical, but **the real fraction of dry-run vs.
  actually-executed dispatch driven by attention signal is not yet measured.**

This is a real, open, unresolved question, not yet answered: does any of this reach
consequence, or is Layer 5 attention itself another instance of the theater the drives
system was killed for, just with better-looking math underneath. **No further
architecture work should be treated as load-bearing until this is measured** — but per
Juniper's direction, that measurement is a separate, parallel thread (see Open threads
below), not a blocker on writing down the architecture itself.

## Course correction: single-theory overreach

First draft of this plan converged unilaterally on Free Energy Principle / Active
Inference (Friston) as *the* replacement architecture. This directly violates charter §7:
*"Multi-theory, not single-theory... run in parallel as measurements, not as competing
final answers. Integration is decided from data, later, not from a Design Mode debate
now."* Friston was picked because it was verifiable against a primary source in-thread
and already has real shipped infrastructure (the five prediction-error instruments,
PRs #1205/#1210/#1218/#1222) — momentum, not a reasoned exclusion of alternatives.

The charter itself already ran the broader theory survey (§1 point 8): IIT, Attention
Schema Theory, Predictive Processing/Active Inference, Higher-Order Theories, and
Recurrent Processing Theory were traced for real existing substrate in this codebase
specifically, out of the full space of consciousness/attention theories. Applying that
existing survey rather than re-litigating it:

- **RPT (Lamme)** — already live, already an *ingredient* inside the current (broken)
  formula's novelty term (`novelty_for_target()` diffing against the prior frame). Not a
  separate candidate architecture; already folded in.
- **HOT** — already served by item 2's AST/HOT reducer (a meta-representational layer,
  not a competing scoring formula).
- **IIT / φ** — orphaned Causal Geometry v1 φ, scoped in §9a item 4 as a *meta-regulator*
  (widen/narrow competition breadth), not a base scoring formula. Complementary, not
  competing.
- **Core-affect/circumplex (Russell)** — a legibility readout (§9a item 7), orthogonal to
  the competition mechanism itself.
- **Predictive Processing/Active Inference (Friston)** and **Global Workspace/
  Society-of-Mind (Baars, Dehaene, Minsky)** are the two genuinely competing,
  structurally different candidates for *replacing the base scoring mechanism itself* —
  precision-weighted magnitude (still one scalar per target, via different math) vs.
  rank-aggregated independent scorers (no single formula at all). Both real, both
  charter-named, neither yet built.

## Tentative plan: two parallel candidate instruments, not one chosen architecture

Per §7's own discipline — same shape as §9b's "each instrument gets built as read-only
measurement first, replayed against real historical data... before any of them gate
anything live or get compared against each other."

### Candidate A — Precision-weighted prediction error (Friston; Feldman & Friston 2010,
"Attention, Uncertainty, and Free-Energy")

**Arsonist summary**: `compute_salience()` is a hand-typed 4-weight formula with zero
citation, authored two months before this program existed, never calibrated. Replace it,
for whichever domain has real live data, with one real theoretical quantity instead of
four guessed ones.

**Current architecture**: `orion/attention/field_attention/scoring.py::compute_salience()`
— `pressure*0.45 + novelty*0.20 + urgency*0.25 + confidence*0.10`, `AttentionWeightsV1`
from `config/attention/field_attention_policy.v1.yaml`. Five real prediction-error
instruments already exist (`orion/substrate/prediction_error.py`) but write only
current-value snapshots to FalkorDB nodes — no history there. Real history lives in
Postgres `substrate_reduction_receipts`.

`salience(target) = precision(target) × |prediction_error(target)|`. Precision =
inverse variance of that target's real historical error, read from
`substrate_reduction_receipts` (the only place real history exists — FalkorDB nodes hold
current-value-only, no history; confirmed live via direct query 2026-07-21).

**Real data check, done, not assumed** (corrects an earlier property-name query bug this
same day — `n.node_id`, not `n.id`):

```
node:substrate.harness_closure   prediction_error=0.65      observed ~18.5h stale
node:substrate.transport         prediction_error=0.000922  observed ~3.6d stale
node:substrate.execution         prediction_error=0.000154  observed ~3.7d stale
node:substrate.biometrics        prediction_error=0.0429    observed seconds-fresh, live
node:substrate.chat / .route     do not exist yet
```

Only biometrics has a real, currently-refreshing series today (192 real receipts,
non-degenerate spread 0.000-0.119). Execution/transport are real but currently dormant,
not "ready" as earlier assumed from the charter's status note rather than checked
directly. Any first real replay of Candidate A has to be honest about this, not silently
scoped to all five domains.

**Correction (2026-07-21, later same day): the above table measured the wrong thing.**
Re-traced end to end, not re-asserted. `substrate_reduction_receipts` itself is healthy
for every domain — confirmed live: `transport_bus_reducer` was firing every ~10-12s and
`execution_trajectory_reducer` every ~1-2min at the moment this correction was written,
both with real, distinct payload content. **"Only biometrics is real" is false as a
claim about the receipts.** The staleness numbers above are real, but they measure
`node:substrate.*`'s `observed_at`, which is a *derived* FalkorDB write, not receipt
health — and that write is gated by `app/worker.py::_write_prediction_error_node()`,
called only when that tick's computed error is `> 0.0` (line 653/846 and the
execution/transport call sites). Two different, unrelated mechanisms were hiding behind
one misleading proxy:

- **Transport is genuinely idle, not broken.** Sampled the last 500 live
  `transport_bus_reducer` receipts (~80 continuous minutes): `bus_health` and
  `delivery_confidence` each took exactly **1** distinct value across all 500 rows;
  `transport_pressure` took 2 (`0` / `0.0009`). `transport_prediction_error()`
  (`orion/substrate/prediction_error.py:34`) is diffing a bus that has had nothing
  happen — `error = 0.0` is the *correct* score here, not a failure. Matches this
  program's own `orion-field-digester` README verdict on the same underlying
  channels ("one-way ratchet... currently benign since the bus is genuinely stable").
- **Execution's instrument has a structural bug, unrelated to volume.** Sampled all 26
  live `execution_trajectory_reducer` receipts: 100% `operation: "create"`, and every
  `target_id` (trace_id) is unique — zero repeats. `execution_prediction_error()`
  (`prediction_error.py:17`) only scores a delta when `prev.runs.get(trace_id)` finds
  the *same* trace_id in both the previous and current projection snapshot. Real
  cortex-exec runs observed here (`reverie_narrate` calls) are single-shot: created
  once, never revised. There is structurally never a "prev" of the same run to diff
  against — `error` returns exactly `0.0` in perpetuity, independent of how much real
  execution activity exists. `route_prediction_error()` shares the identical
  trace_id-diff design and likely has the same defect (only 9-10 real receipts ever,
  consistent with one-shot arbitration runs, not yet independently confirmed).

Net effect: the receipts pipeline needs no fix. The FalkorDB `observed_at` snapshot used
above is not a valid proxy for "does real history exist" and should not be used as one
again. Transport's apparent dormancy is real signal (a quiet bus, correctly scored);
execution's is an instrument defect that will not resolve with more data or more uptime
— it needs `execution_prediction_error()`'s diff key redesigned (e.g. compare against
the most recent prior run regardless of trace_id, not require the same trace_id to
recur) before it can ever register a nonzero value. See board finding `92c15e6c` for the
full trace.

**Missing questions**:
- ~~Does real receipt history actually exist in enough volume, right now, for more than
  just biometrics~~ — answered above: yes for transport (idle, not absent) and no for
  execution, but for a code-defect reason, not a volume reason. Chat/route not yet
  independently re-checked against live data past this correction.
- Is variance-based precision numerically stable at the magnitude these real signals
  actually operate at (0.0-0.12 range confirmed for biometrics), or does it need a
  floor/epsilon to avoid blowing up on a near-zero-variance stretch?
- Windowed (last N receipts) or exponentially-weighted precision — not yet decided.

**Proposed schema/API change**: new pure function, new module
(`orion/attention/field_attention/candidate_precision_weighted.py`), no schema registry
change yet — this is a shadow instrument, not a live-consumed artifact.

**Files likely to touch**: `orion/attention/field_attention/candidate_precision_weighted.py`
(new), `scripts/analysis/` (new replay script), tests.

**Non-goals**: no wiring into `compute_salience()`/`select_system_targets`/
`self_state/builder.py` in this patch. No claim about execution/transport/chat/route
until real data confirms readiness. **Blocked from any live-wiring phase by the
SelfStateV1 hard gate below, same as Candidate B.**

**Acceptance checks**: non-degenerate score distribution over real history; precision
doesn't blow up/degenerate at low-variance edges; real numbers reported, not asserted.

### Candidate B — Rank-aggregated independent scorers (Global Workspace/Society-of-Mind)

**Arsonist summary**: instead of guessing better weights for one formula, remove the
weight-guessing problem structurally — three independently real scorers, combined by
rank, no formula left to calibrate at all.

**Current architecture**: no rank-aggregation mechanism exists anywhere in this codebase
today — confirmed, this is genuine new invention, not reuse. The GWT-dispatch layer's own
`LinearSalienceCombiner`/`seed-v1` is *also* a hand-weighted formula, not real precedent
for this approach.

N independently-sourced real scorers (not weighted-summed — ranked, e.g. Borda count),
eliminating the "what weight per factor" problem structurally rather than calibrating it.
The baseline design doc's own smallest-probe language names the starting triad exactly:
magnitude-based, novelty-based, unresolved-duration-based.

1. **Magnitude scorer** — real prediction-error magnitude, same source and same
   real-data-freshness caveat as Candidate A (biometrics-only until proven otherwise).
2. **Novelty scorer** — `novelty_for_target()` (`orion/attention/field_attention/
   scoring.py:83-91`), already real, already live, diffs current salience against the
   same target's salience in the previous `FieldAttentionFrameV1`. Reused here as an
   **independent rank scorer**, not as a weighted-summed component the way the current
   (broken) `compute_salience()` uses it — a genuinely different structural role for the
   same real signal.
3. **Unresolved-duration (dwell) scorer** — real, live,
   `substrate_coalition_dwell_log` / `orion/substrate/attention_broadcast.py`'s dwell
   mechanism (already had a prior bug fixed in this program — loop-scoped, not a global
   scalar).

Combined by Borda count (or a better-justified standard social-choice method) — each
scorer independently ranks all real targets present in a tick; lower summed rank wins.
**Zero existing infrastructure for the rank-aggregation mechanism itself** — this is
real invention, not reuse, unlike Candidate A.

**Missing questions**:
- Does dwell data actually have enough real volume/diversity to be a meaningful
  independent scorer, or is it as thin as execution/transport turned out to be?
- Borda count vs. another aggregation method — a real design decision, not mechanical;
  worth deciding with real data in hand rather than picking in the abstract.
- Tie-handling: what happens on a 2-2 tie, or when a target is missing from one scorer's
  ranking entirely (e.g. present in the field but never dwelt on)?
- Independence check: are magnitude/novelty/dwell actually measuring different things,
  or secretly correlated in practice (the same class of bug review already caught once
  this program, in `chat_prediction_error`'s `topic_coherence`/`repair_pressure`
  redundancy)?

**Proposed schema/API change**: three pure scorer functions + one combiner, new module
(`orion/attention/field_attention/candidate_society_of_mind.py`), no live schema change.

**Files likely to touch**: `orion/attention/field_attention/candidate_society_of_mind.py`
(new), replay script, tests.

**Non-goals**: same as Candidate A — shadow-only, no live wiring. **Blocked from any
live-wiring phase by the SelfStateV1 hard gate below, same as Candidate A.**

**Acceptance checks**: does rank-aggregation actually break the 99.98% monoculture in
replay; do the three scorers ever disagree, and is that disagreement itself informative
(direct baseline-design question) — report real examples, not a summary claim.

### What ships in the same patch as either candidate, per "no keyword cathedral"

- Formal retirement of the three theater instances (D/W/A corpse, `AttentionWeightsV1`'s
  four weights + ~25 channel weights, `seed-v1`) — not left to rot alongside a fourth,
  newer mechanism.
- A real, named, in-the-same-changeset consumer past `self_state/builder.py`'s current
  5-ID truncation — this candidate does not ship as a scoring change alone if the score
  still dies at the same discard point.

## Build sequencing (added 2026-07-21, post-correction)

Two separate things were being conflated as one build. Splitting them into phases, per
Juniper's direction:

**Phase 1 — build first. Fix the live "bus version" — the existing, already-deployed,
event-driven reducers in `services/orion-substrate-runtime/app/worker.py` /
`orion/substrate/prediction_error.py`.** These already run continuously inside the real,
bus-consuming substrate-runtime service (biometrics/execution/transport/chat/route poll
loops, `worker.py:213-218`); they are not new instruments, they are the existing
prediction-error producers with a confirmed defect. Concretely: redesign
`execution_prediction_error()`'s (and check `route_prediction_error()`'s) diff key so it
does not require the exact same `trace_id` to recur across two consecutive polls —
compare against the most recent prior run instead. This is a bug fix to code that
already runs in production, touches no live consumer of the score (nothing downstream of
`node:substrate.execution` exists yet), and is **not** blocked by the SelfStateV1 hard
gate below — that gate is about wiring a *candidate salience formula* into
`self_state/builder.py`/`orion/proposals/`/`capability_policy.py`, not about correcting
an already-shipped diagnostic instrument that nothing live consumes. Also correct this
document's own now-superseded "only biometrics is real" framing (done above) so Candidate
A's real-data assumptions start from the corrected picture, not the stale one.

**Phase 2 — subsequent phase. Build the standalone "non-bus" versions — Candidate A's
and Candidate B's new modules** (`candidate_precision_weighted.py`,
`candidate_society_of_mind.py`) **and their replay scripts under `scripts/analysis/`.**
These are pure, offline analysis code with no dependency on the live bus-driven worker
loops — they read `substrate_reduction_receipts` directly and can be run, tested, and
iterated on without touching the running services at all. Deliberately sequenced after
Phase 1: replaying Candidate A's precision math against execution history before the
diff-key fix would replay against data that structurally can never show variance,
producing a misleading "execution never varies" result that is actually just the
unfixed bug restated as a finding.

Neither phase touches the SelfStateV1 hard gate or any live consumer — both stay
shadow/diagnostic-only, consistent with the rest of this document's non-goals.

## Hard gate — RESOLVED (2026-07-22), via kill not fix

**Original gate (2026-07-21, preserved below for the record):** `SelfStateV1` data-quality/
theory/logic audit, full upstream trace, is a hard gate, not a parallel nice-to-have.
Neither Candidate A nor Candidate B may be wired into any live consumer
(`self_state/builder.py`, `orion/proposals/`, `capability_policy.py`, or anything
downstream of them) until this is done. Scope: not just the
`dominant_attention_targets[:5]` discard point already found, but `SelfStateV1`'s full
construction — its dimensions, their scoring, what else besides attention gets
lossy-compressed on the way in.

**Resolution:** the audit ran (2026-07-22, see
`docs/superpowers/specs/2026-07-22-self-state-phi-endo-origination-burn-spec.md` for the
full trace). It found the gate's suspicion confirmed and worse: `SelfStateV1`'s dimension
weights/thresholds are hand-picked with zero calibration (traced to the introducing
commit, no design doc, no justification anywhere), and the signal itself is empirically
dead independent of the weights — a live replay found 12/12 dimensions pinned/flat
(`scripts/analysis/measure_self_state_signal_quality.py`, up from 8/12 four days prior).
Decision: kill `SelfStateV1` outright rather than fix the compression — full burn list,
rollout order, and per-consumer disposition in the linked spec.

**Implication for this document's live-wiring phase**: the gate's original framing
("wired into `self_state/builder.py`...") is now moot — that file is being deleted, not
fixed. The live-wiring target for whichever candidate gets chosen is no longer
`SelfStateV1` at all; it moves to a `FieldStateV1`-native consumer, consistent with the
already-established field-native-over-self-state direction. `orion/proposals/` (Layer 7)
is being repointed at `FieldStateV1` directly in the same burn, for the same reason. Any
future live-wiring decision for Candidate A/B should target that field-native path, not a
`SelfStateV1` dimension that will no longer exist.

**This gate was not satisfied by either candidate's own acceptance checks** — it required
its own separate investigation, and got one, not assumed clear by omission.

## Open threads, tracked, not gating

1. **Delivery measurement** — real dry-run vs. actually-executed dispatch ratio for
   attention-sourced proposals. Not yet measured. Location of
   `ExecutionDispatchFrameV1`'s real persistence table not yet confirmed.
2. **`capability_policy.py`'s live gating**, confirmed directly rather than cited from
   the charter's prior status note (same discipline failure already caught once this
   session with the FalkorDB query bug — don't repeat it here).

## Non-goals (this document)

- Not choosing between Candidate A and B here — that's exactly the single-theory
  overreach this document exists to correct. Both get built as real, replayed,
  comparable instruments; integration is decided from that data.
- Not implementing anything — proposal mode, sign-off required per `CLAUDE.md` §0A.
- Not re-deriving the charter's own theory survey (§1 point 8) — applied here, not
  redone.
