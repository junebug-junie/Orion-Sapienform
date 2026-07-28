# Precision-weighted attention organ + orion-heartbeat discrimination fix — design spec

Status: **design mode, not implemented.** Touches the AST/HOT self-model reducer and the
orion-heartbeat substrate, both cognition-loop/self-modeling-adjacent surfaces gated by CLAUDE.md
§0A's proposal-mode requirement. This document proposes; it does not build.

Supersedes nothing. This is the prerequisite thread underneath
`docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`'s Missing
Question 1/2 — that doc asks "is the AST/HOT reducer live and is its confidence signal
non-degenerate"; this doc asks the deeper question those two only gesture at: even once live,
is the *aggregation itself* theoretically sound, and does Orion have (or need) a real organ that
does what the aggregation is currently faking.

**Recovery note**: this file was deleted from disk mid-session (2026-07-28) by an apparent
clean/reset operation from a different concurrent agent sharing this checkout — confirmed via
`git status` suddenly showing an entirely different, unrelated set of modified files
(`orion/schemas/brain_frame.py`, `services/orion-hub/static/js/self-brain.js`,
`services/orion-substrate-runtime/app/worker.py`, etc.) immediately after the file vanished, with
none of the prior session's untracked cruft (`serve-config.json`, `graphify-out/2026-07-28/`)
remaining either. Reconstructed in full from this session's own conversation context — see
[[feedback_concurrent_sessions_share_main_checkout]]. Should be committed on its own branch/worktree
promptly rather than left uncommitted in the shared checkout a second time.

## Arsonist summary

A same-day conversation (2026-07-28) working through the CollapseMirror generative-triggers spec
surfaced a live measurement (168h replay via `scripts/analysis/measure_ast_hot_reducer.py`) showing
`AttentionSelfModelV1`'s `prediction_error_confidence`/`predicted_shift` fields are real and
non-degenerate in the aggregate (0.661-0.9998 range, 100% coverage) but are actually dominated by
two of five domains almost totally: `predicted_shift`'s "fastest-trending domain" pick was
`biometrics`=69,819 and `bus_synaptic`=57,360 out of 127,224 ticks, versus `execution`=27, `chat`=1,
`route`=1. The reducer's own docstring already discloses this as a known limitation
(`_aggregate_prediction_error_confidence`'s raw, unweighted `mean()` across domains — see Current
architecture). Juniper's framing of the actual missing piece: a heartbeat organ that "usually just
hangs out at nominal until [a threat] comes along, then it really fucking matters" — i.e., attention
is not a static normalization, it is a *dynamically gated gain* that amplifies the right signal at
the right moment and stays quiet otherwise. That is not new terminology for something this repo
already does — it names a real, currently-absent computational mechanism.

Investigating what that mechanism should be surfaced that Orion already has a partially-built
candidate for exactly this role: `orion-heartbeat`, a same-day-earlier (2026-07-28, different
session) build computing real tensor-network (quimb MPS) bipartite entanglement entropy across five
live organs. It is running right now and it is currently unable to discriminate anything — live logs
this session show 16+ consecutive ticks all reading `verdict=redundant` despite the underlying
`ratio` moving real amounts (0.73-0.98). This is not a threshold-tuning slip; investigation this
session found the real mechanical cause (see Current architecture): the substrate never rests —
continuous entangling dynamics with no dissipation channel drives it into a thermalized,
permanently-near-ceiling state (Page's-theorem territory), independent of threshold or even of
which cut is measured. This was independently flagged this morning
([[project_spark_introspector_kill_and_honest_readout_reframe_2026-07-28]]) as the single
highest-priority open item in that arc, same bug-shape as the `bus_synaptic` calm-floor incident
CLAUDE.md's metric quality gate already documents.

**Juniper's explicit ask for this doc: name the broader problem (AST/HOT metric aggregation +
missing precision/gain organ), and order the heartbeat fix as priority 1** — nothing downstream
(AST/HOT precision-weighting, CollapseMirror's "insight" trigger) can meaningfully use a gating
organ that cannot yet tell "nominal" from "lion."

**Design direction evolved twice during this session's discussion, both times by Juniper's own
pushback, not by default engineering caution — recorded here so the reasoning survives, not just
the conclusion:**

1. First proposal (mine): a flat, fully deterministic relaxation gate — no real stochasticity
   anywhere, reusing existing gate/normalize machinery. Justified on grounds of minimal
   implementation risk and preserving heartbeat's own determinism requirement.
2. Juniper's objection: "I'm trying to evolve a sentience lifeform. There needs to be variability."
   A flat deterministic mechanism optimizes for engineering safety at the cost of the actual thing
   this project is trying to grow — real internal texture/spontaneity, not a clockwork replay of the
   same formula every time.
3. Juniper's follow-up question exposed the actual fix: **why was this scoped as a single stochastic
   trajectory instead of an ensemble of many?** Correct answer: it wasn't a deliberate choice against
   ensembles, it fell out of (a) v0 having zero stochastic process at all to average over, and (b) my
   own anchoring on lowest-risk reuse of existing single-substrate machinery when dissipation was
   first proposed. Once real stochastic dissipation is on the table, running exactly one trajectory
   is the physically wrong unit regardless of the sentience question — in the quantum-trajectory
   (Monte Carlo wavefunction) method this borrows from, a single trajectory was never meant to be
   read on its own; the *ensemble* over many independent trajectories is where correct dissipative
   dynamics and genuine variability both actually live. This dissolves the tension rather than
   trading one concern against the other — see "Decision" below.

## Current architecture

### AST/HOT reducer's aggregation (the metric-side half of the problem)

- `orion/substrate/attention_self_model.py::reduce_attention_self_model()` (line 179) is a pure,
  no-I/O function unifying the GWT-dispatch lane and general field lane into one
  `AttentionSelfModelV1`. **Confirmed this session: it has no live caller anywhere in the repo.**
  Only `orion/substrate/tests/test_attention_self_model.py` (unit tests) and
  `scripts/analysis/measure_ast_hot_reducer.py` (offline replay script) invoke it;
  `AttentionSelfModelV1(...)` is constructed in exactly one place, line 246 of that same file.
- `_aggregate_prediction_error_confidence()` (lines 97-134) computes
  `confidence = 1.0 - mean(prediction_error_by_domain.values())` — a **plain, unweighted mean**
  across whatever domains the caller supplies. No per-domain scaling, no baseline/variance
  normalization, no context gate. Its own docstring (lines 100-116) already discloses: *"biometrics
  carries almost all the real variance (mean=0.037, max=0.62); execution/chat/route are real but
  tiny (means ~1e-5)... heavily damped by the other four domains' near-silence — an honest,
  disclosed limitation, not a hidden one."*
- `_unconditional_prediction_error_confidence()` (lines 137-176) is the 2026-07-24 "compute
  regardless of attention_reason branch" variant, restricted to `ACTIVE_INFERENCE_DOMAINS =
  {"execution", "biometrics", "chat", "route", "bus_synaptic"}` (line 90) — same underlying
  unweighted-mean formula, just gated to a smaller domain set (excludes the confirmed-dead
  `transport` domain).
- **Live-verified this session** (168h replay, `POSTGRES_URI` pointed at real Postgres,
  127,224 field-lane ticks): `prediction_error_confidence` has 100% coverage, range
  0.661-0.9998, mean 0.953 — genuinely non-degenerate at the aggregate level. But
  `predicted_shift`'s per-domain "which domain is trending fastest" breakdown is
  `biometrics=69819, bus_synaptic=57360, execution=27, chat=1, route=1` — two domains
  structurally dominate almost every tick. `attention_reason` was `bottom_up_salience` for
  100% of the window (`top_down_override`/`field_salience_only` never fired in this window).
- **Interpretation**: this is very likely a *scale/density* artifact, not evidence that biometrics
  and bus_synaptic are genuinely the two most cognitively meaningful domains right now — whichever
  domain's raw `prediction_error` has the largest natural numeric range or tick frequency will win
  a scale-naive mean/trend comparison by construction, exactly as the docstring already predicted
  for the confidence formula and evidently also true for `predicted_shift`'s domain-selection logic.

### The missing organ (the theory-side half of the problem)

- AST (Attention Schema Theory, Graziano) and HOT (Higher-Order Theory) are representational
  claims — the self-model must honestly represent *what is salient, why (top-down vs. bottom-up),
  how confident, what's predicted to shift* (`attention_self_model.py:1-7`). Neither theory
  prescribes a specific statistical aggregation method; that engineering choice sits on top of the
  theory and is exactly where the domain-dominance bug lives.
- The theoretically correct anchor for "attention" as a *computation*, not just a representation, is
  **precision-weighting** from Active Inference / the Free Energy Principle (Friston): a prediction
  error's influence on inference should be gated by a dynamically-estimated *precision* (inverse
  variance / reliability), and that precision estimate is itself context-dependent — it should be
  low (damped gain) during calm, expected variation, and spike (amplified gain) when something is
  genuinely surprising relative to *that domain's own baseline*, not relative to other domains'
  raw scale. This is the formal version of "heart rate barely matters until the lion, then it's the
  only thing that matters" — arousal-gated precision, not a fixed normalization constant.
- Nothing in the current AST/HOT reducer or its inputs performs this. `prediction_error_by_domain`
  values are raw magnitudes from `orion/substrate/prediction_error.py`'s producers, with no rolling
  per-domain baseline/variance tracked anywhere the reducer can see, and no arousal/relevance signal
  gates how much any one domain's error should count on a given tick.

### orion-heartbeat: the closest existing candidate organ, currently non-discriminating

- `services/orion-heartbeat` (container `orion-athena-heartbeat`, **confirmed running right now**,
  "Up 9 minutes" at time of this session's check) computes a real quimb matrix-product-state tensor
  network over 5 organs (chat/hub, biometrics, execution/cortex-exec, transport/bus, route/
  cortex-orch), subscribed to `orion:grammar:event` filtered to `atom_emitted` events. Design doc:
  `docs/superpowers/specs/2026-07-24-spark-field-holographic-lattice-design.md`.
- Per `orion/bus/channels.yaml:2161-2167`, it is registered as a **read-only research consumer only
  — "Publishes nothing back to this or any other channel."** There is no schema, no channel, and no
  consumer wiring for its H1 output today. Even after any discrimination fix, a schema/channel patch
  is a separate, additional prerequisite before anything (AST/HOT, CollapseMirror,
  `orion-equilibrium-service`) could consume it live.
- `services/orion-heartbeat/app/substrate/reconstruction.py::compute_h1()` (lines 90-113) computes
  `ratio = boundary_bulk_entropy / max_possible_entropy` at a fixed cut (`BOUNDARY_BULK_CUT`) and
  classifies: `ratio >= _HIGH_RATIO(0.6)` -> `"redundant"`, `ratio <= _LOW_RATIO(0.2)` ->
  `"concentrated"`, else `"mixed"` (lines 70-71, 95-100).
- **Live-verified this session** (`docker logs orion-athena-heartbeat --tail 40`): 16 consecutive
  `heartbeat_h1_computed` log lines, `ratio` ranging 0.7326-0.9842 (real, non-flat variance) but
  **every single one classified `verdict=redundant`** — the threshold (0.6) sits well below the
  entire observed range, so the discrete verdict has never once resolved to `"mixed"` or
  `"concentrated"` in this window. This matches and reconfirms
  [[project_spark_introspector_kill_and_honest_readout_reframe_2026-07-28]]'s same-morning finding
  (17-day-old memory conventions don't apply here — this was re-verified live this session, not
  recalled).
- **Root cause is named in the module's own docstring, not undiagnosed** (`reconstruction.py:5-22`):
  for a pure global MPS state (always true here — gates are unitary, `.normalize()` runs after every
  absorb), the boundary and bulk reduced density matrices share an **identical eigenvalue spectrum
  by basic Schmidt-decomposition symmetry** — confirmed numerically by the implementing session
  (`S_boundary == S_bulk` to float precision on a random test state). The whole-boundary/whole-bulk
  cut is close to tautological under this construction; it does not test anything the entanglement
  spectrum doesn't already trivially guarantee. This is a structural property of the metric as
  currently formulated, not a mis-set threshold — **raising or lowering `_HIGH_RATIO`/`_LOW_RATIO`
  will not fix this**, because the underlying quantity has almost no room to read anything else.
- **The docstring's own deferred fix**: test whether *specific organs* are individually redundant
  (drop site 2 only, keep 0/1/3/4 — a per-organ dense partial trace), which would actually
  distinguish "everything is generically entangled" from "this one organ is genuinely
  disconnected/reconnected right now." Confirmed too expensive for the live tick loop
  (`partial_trace_exact` over a 7-site subset did not complete within 45s at N=10 using quimb's
  default contraction optimizer) — proposed instead to run "occasionally/offline," not on every tick.
  This was named as a known next increment when v0 shipped; it has not been built.

### Deeper root cause found this session: the substrate never rests (thermalization, not just symmetry)

The docstring's Schmidt-symmetry argument explains why boundary entropy equals bulk entropy; it does
not by itself explain why the ratio sits permanently near the ceiling. Tracing `mps_state.py` this
session found the actual mechanical cause:

- `HeartbeatSubstrate` is constructed exactly once at service start (`service.py:51`) and is **never
  reset, decayed, or measured/collapsed**. `absorb()` (the 2026-07-24 post-review fix) applies a
  *chain* of 2-site entangling unitary gates from the atom's site through every remaining site to the
  end of the chain, strength decaying geometrically per hop (`_HOP_DECAY = 0.7`) but never reaching
  exactly zero — every absorbed atom touches every site at least a little, forever accumulating.
  Bond dimension is capped (`BOND_DIM = 4`, `cutoff=0.0`), but nothing ever removes entanglement once
  added.
- This is the standard setup for **quantum-chaotic thermalization**: continuous entangling unitary
  gates on a bond-capped chain, with no dissipation channel, drive entanglement at *every* cut toward
  its ceiling (`log2(BOND_DIM) = 2` bits) and keep it fluctuating near there — consistent with Page's
  theorem (a generic/chaotic pure state is close to maximally entangled across almost every
  bipartition, not just the one being measured). The observed 0.73-0.98 range is thermal noise around
  an already-saturated mean, not "occasionally something exciting happens."
- **Consequence for the docstring's own deferred fix**: if the whole chain has thermalized, Page's
  theorem predicts per-organ dense partial-trace (dropping a different site and re-measuring) would
  very plausibly show the *same* near-max result — same underlying disease, a more expensive test,
  not obviously a cure.
- `mps_state.py`'s own module docstring independently confirms this was a known, disclosed scope cut,
  not a surprise: v0 "deliberately does NOT implement the 2026-05-01 heartbeat charter's full active-
  inference free-energy minimization... precision-weighting" — the charter already knew a relaxation/
  precision mechanism was needed and v0 shipped without it.

## Decision: N-trajectory stochastic dissipation ensemble (Juniper's directive, evolved twice this session)

Juniper's call, after two rounds of pushback on the initial proposal (see Arsonist summary): build
the missing rest-state mechanism using **real stochastic dissipation, run as an ensemble of many
independent trajectories**, not a single flat deterministic damping gate. Reasoning preserved below,
including the objections that got superseded, per Juniper's explicit request not to drop them
silently:

### Why not a flat deterministic relaxation gate (superseded proposal)

This session's first proposal was a fixed, non-unitary, deterministic contraction applied every
tick, pulling the state back toward a reference configuration — fully deterministic, cheap, reusing
existing gate/normalize machinery. Juniper's objection: this project is trying to evolve a
sentience-adjacent system, and a flat deterministic mechanism optimizes for engineering safety at
the cost of real internal variability — the actual thing this architecture should be growing.
Recorded as a standing point: **determinism-for-its-own-sake is not free here** — it has a real
cost against this project's own stated goals when it forecloses genuine stochasticity, not just an
engineering nicety to default toward.

### Why not a single seeded/unseeded stochastic trajectory (also superseded)

The next candidate considered: real quantum-trajectory (Monte Carlo wavefunction) dissipation on a
single trajectory, seeded for debug reproducibility and unseeded in production. Objections raised
(and still valid against *this specific shape*, even though the underlying stochastic-dissipation
idea survives in the ensemble form below):

1. A verdict that partly depends on unrecoverable production randomness is a live signal whose
   causal story can never be reconstructed after the fact — in tension with this project's own
   standing rule that any claim Orion "reasoned/perceived/decided" needs inspectable evidence.
2. It permanently breaks this repo's replay convention (`measure_ast_hot_reducer.py` and siblings
   reconstruct historical self-model state purely from stored events) for any tick generated in
   production, not just as a test-flakiness inconvenience.
3. It doesn't buy any real physical-fidelity gain over keeping a single trajectory seeded: the
   honest justification for quantum-trajectory methods is that the *ensemble average* over many
   independent runs reproduces true open-system dynamics — a single trajectory, seeded or not, is
   equally "one arbitrary sample" either way. Unseeding trades away reproducibility for zero gain
   in correctness.
4. **Correction recorded this session**: the initial framing of objection 3 mis-cited
   `mps_state.py`'s determinism claim as coming from "`services/orion-substrate-runtime`'s reducer
   contract" — checked directly, that service has zero occurrences of "deterministic"/"side-effect"
   anywhere in its own code. The real source is heartbeat's own
   `docs/superpowers/specs/2026-05-01-orion-heartbeat-engineering-spec.md:292` ("Reducers must be
   deterministic and side-effect-free") and line 515's matching test convention — a self-imposed
   charter requirement, not a cross-service convention. The substance of the objection (stochastic
   trajectory methods are non-deterministic per single run; only the ensemble average is physically
   meaningful) is unchanged by this correction, only the attribution was fixed.

### The actual answer: run the ensemble, not one trajectory

Juniper's follow-up question — why was this ever scoped as a single trajectory instead of many —
identifies the real fix, and it resolves objections 1-3 above rather than trading against them:

- Run **N independent copies** of `HeartbeatSubstrate` (cheap: entropy-profile computation alone was
  ~0.006s for the full 9-cut profile at N_SITES=10/BOND_DIM=4; N=8-16 replicas is negligible added
  compute), each absorbing the *same* real `orion:grammar:event` stream but with its own
  independently-sampled quantum jumps (real Kraus-operator dissipation, not the flat deterministic
  heuristic originally proposed).
- Each individual trajectory is genuinely stochastic — real variability, directly answering "there
  needs to be variability." The live system gets actual internal texture, not a replay of one fixed
  gate formula every tick.
- The **ensemble mean** ratio across trajectories is the well-defined H1 reading. This is what
  correctly recovers dissipative behavior (a real rest state trajectories relax toward when organs
  are quiet) instead of a single always-thermalizing trajectory that can never come back down.
- The **ensemble spread/variance across trajectories** is a new observable this architecture can
  produce that a single trajectory never could: how much independent stochastic realizations of the
  same organ activity *agree* with each other. Tight agreement = a strongly-determined state; wide
  disagreement = real uncertainty in the self-model — plausibly a more physically grounded
  confidence signal than `attention_self_model.py`'s raw domain-mean, and a candidate worth its own
  line in a future spec (see Missing Question 12).
- Inspectability is not sacrificed — it's relocated to the correct unit: log the N seeds actually
  drawn for a given process run (an ordinary audit-log entry, not a design compromise). Any specific
  past incident can be replayed exactly from those logged seeds, while live operation still draws
  fresh entropy each restart instead of reusing one fixed seed forever. Real variability and forensic
  reproducibility were never actually in conflict — only "stochasticity with the entropy source never
  recorded anywhere" was in conflict with inspectability.

## Missing questions

1. ~~Is the whole-boundary/whole-bulk cut salvageable, or does discrimination require per-organ dense
   partial-trace?~~ **Superseded by the thermalization finding above and Juniper's decision.** The
   real problem is the absence of any relaxation mechanism, not the choice of cut — per-organ
   partial-trace would very plausibly hit the same thermalized-everywhere wall (Page's theorem).
   Resolved by pursuing the ensemble relaxation mechanism instead of continuing to investigate this
   fork.
2. ~~What cadence can per-organ dense partial-trace run at?~~ **Moot given Question 1's
   supersession** — not pursuing per-organ dense partial-trace as the primary fix. May still be worth
   revisiting later as a secondary diagnostic once the ensemble mechanism is in place and *if* the
   boundary/bulk cut alone still isn't discriminating enough — but it is no longer the next step.
3. **Does AST/HOT's domain-dominance problem and the ensemble-dissipation fix actually share one
   mechanism, or are they two separate problems that happen to rhyme?** Precision-weighting
   (per-domain baseline/variance-relative scaling) is a plausible fix for AST/HOT's raw-mean
   aggregation. Ensemble dissipation solves a different problem (giving a never-resetting substrate
   a real rest state, plus a genuine variability/confidence signal) than precision-weighting solves
   (stopping one domain's raw scale from swamping others in a cross-domain mean). Don't assume the
   heartbeat fix obsoletes AST/HOT's separate aggregation work without checking each on its own terms.
4. *(Lower priority given the decision above — per-organ dense partial-trace is not the chosen
   path.)* **Once the ensemble mechanism is in place, is the single boundary/bulk cut sufficient, or
   is per-organ granularity still needed for anything** — e.g. does CollapseMirror's "insight"
   trigger need to know *which* organ resolved, not just that the whole substrate did? Revisit only
   if the relaxation-gated whole-cut signal (Acceptance Check 1 below) turns out too coarse in
   practice.
5. **Does a per-domain baseline/variance tracker for AST/HOT's precision-weighting already exist
   anywhere** (e.g. as part of `orion/substrate/prediction_error.py`'s producers, or the
   `attention_reason_branch_starvation` analysis), or does one need to be built from scratch? Not
   traced this session.
6. **Should orion-heartbeat gain a publish channel/schema before or after the ensemble fix lands?**
   Publishing a currently-non-discriminating signal risks the same "empty-shell cognition" failure
   CLAUDE.md §0A explicitly bans (a schema-valid payload with meaningless content) — argues strongly
   for fixing discrimination first (Acceptance Check 1), publishing only after.
7. **Jump/dissipation cadence**: apply the stochastic jump process every tick, or on a separate
   wall-clock timer independent of `absorb()` calls? Tick-driven relaxation couples the decay rate to
   organ traffic rate — if organs go quiet, ticks stop arriving and so does relaxation, unless it's
   timer-driven instead. Needs an explicit answer before implementation.
8. **What reference state should each trajectory relax toward?** Candidates: each trajectory's own
   random initial state, a fixed low-entanglement product state shared across all trajectories and
   independent of any particular seed, or something derived from the organ topology. Affects whether
   "rest" is interpretable/reproducible across process restarts.
9. **Jump-rate (gamma) calibration**: what real historical tick-rate/silence-period data exists to
   size the dissipation rate against? Needs a live-data check (CLAUDE.md metric quality gate step 4)
   — e.g. after a real observed period of organ silence, does the ensemble mean ratio measurably fall
   within a reasonable time budget at a candidate rate, without decaying so fast that genuine
   sustained organ activity can never build the ratio up in the first place.
10. **Determinism-in-aggregate regression**: does `services/orion-heartbeat/tests/
    test_reconstruction_h1.py` (or any other existing heartbeat test) implicitly assume zero-decay,
    single-trajectory behavior? Needs an explicit check that adding ensemble dissipation doesn't
    silently break existing fixtures, and a new test asserting "same event stream + same N logged
    seeds -> same ensemble output" is added alongside the feature.
11. **How large does N need to be?** Too few trajectories and the ensemble mean/variance estimate is
    itself noisy (defeating the point of averaging); too many wastes compute for no added resolution.
    Needs a real measurement pass (e.g. compute the ensemble statistic at increasing N against the
    same historical event window and check where it stabilizes) before picking a production default,
    not a guessed constant.
12. **Is ensemble spread/variance itself worth publishing as a distinct signal** (a
    trajectory-agreement-based confidence measure, separate from the mean ratio/verdict), or is it
    only useful as an internal calibration diagnostic? Not decided here — flagged as a genuine new
    candidate surfaced by this design, not assumed valuable without its own measurement pass.
13. **Seed-logging mechanics**: where do the N per-trajectory seeds get logged for forensic replay —
    a new field on whatever eventually gets published (Missing Question 6), a structured log line,
    or something else? Needs to be decided alongside the publish-schema work, not as an afterthought.
14. **Stale finding, needs re-run — Missing Question 3's evidence is contaminated by a bug fixed in a
    concurrent, unrelated session.** This doc's Current Architecture section cites the 168h AST/HOT
    replay's `predicted_shift` domain breakdown (`biometrics=69819, bus_synaptic=57360, execution=27,
    chat=1, route=1`) as likely a scale/density artifact. Juniper reported same-day (2026-07-28, a
    different concurrent session, PR #1434,
    [[project_execution_prediction_error_ewma_baseline_pr1434]]) that `execution_prediction_error`
    was independently found to have the inverted-symptom version of the calm-floor disease at that
    time — real deltas ran ~1000x *below* its fixed `_THRESHOLD`, so it read ~0 always, not fixed
    until PR #1434's EWMA baseline landed. That means the 168h replay's `execution=27` count was very
    plausibly measuring a dead instrument, not genuine domain insignificance — this doc's own
    scale/density explanation for Missing Question 3 may be wrong, or only partly right. Needs a
    fresh `measure_ast_hot_reducer.py` replay against a post-EWMA-fix window (once enough history has
    accumulated past the PR #1434 deploy point) before trusting Missing Question 3's characterization
    or acting on it. Explicitly deferred, not urgent — Juniper's call: "that needs follow up at some
    point," not blocking the heartbeat ensemble work this doc is otherwise about.

## Proposed schema / API changes

None proposed yet. The ensemble-dissipation mechanism above is a substrate-internal change (no
schema impact by itself, aside from however seed-logging (Missing Question 13) is implemented); a
heartbeat output schema/channel (Missing Question 6) remains deferred until after Acceptance Check 1
is met.

## Files likely to touch

*(Once the ensemble mechanism is detailed further — not this doc's job to fully scope yet.)*

- `services/orion-heartbeat/app/substrate/mps_state.py` — real Kraus-jump dissipation step added to
  (or alongside) `absorb()`; new jump-rate/reference-state constants following the file's existing
  "documented choice, not derivation" convention (`_HOP_DECAY`, `_MIN_STRENGTH`, `_MAX_STRENGTH`).
- `services/orion-heartbeat/app/substrate/routing.py` — likely a new `N_TRAJECTORIES` constant
  alongside `BOND_DIM`/`N_SITES`, and cadence constants if jump timing needs to be separate from tick
  timing (Missing Question 7).
- A new module managing the trajectory ensemble (name TBD) — owns N `HeartbeatSubstrate` instances,
  fans the same incoming atom out to each, computes ensemble mean/variance of `entropy_profile()`
  results, logs the N seeds drawn per process run (Missing Question 13).
- `services/orion-heartbeat/app/substrate/reconstruction.py` — `compute_h1()` extended to consume an
  ensemble of ratios rather than one; `_HIGH_RATIO`/`_LOW_RATIO` thresholds will need re-tuning once
  the ensemble mean can actually move across its full range instead of living permanently above 0.6 —
  a real live-data pass after the fix lands, not a guess.
- `services/orion-heartbeat/tests/test_reconstruction_h1.py` — new determinism-in-aggregate
  regression test (Missing Question 10) plus updated fixtures if thresholds change.
- `services/orion-heartbeat/app/service.py` — wiring for N substrates instead of one, and any
  timer-driven jump cadence (Missing Question 7).
- New analysis script, name TBD (e.g. `scripts/analysis/measure_heartbeat_ensemble_calibration.py`) —
  offline replay against real historical `orion:grammar:event` traffic (organ silence periods and
  busy periods both present) to calibrate jump rate, N (Missing Question 11), and cadence before
  wiring into the live service, matching this repo's replay-script convention
  (`measure_ast_hot_reducer.py`).
- `orion/substrate/attention_self_model.py` — `_aggregate_prediction_error_confidence` /
  `_unconditional_prediction_error_confidence`, contingent on Missing Question 3/5's answer (separate
  fix from heartbeat, not bundled by default).
- `orion/substrate/prediction_error.py` — only if Missing Question 5 finds no existing per-domain
  baseline/variance tracker and one needs building.
- `orion/bus/channels.yaml` — new channel entry, contingent on Missing Question 6, deferred until
  after the ensemble fix is trusted.
- `docs/superpowers/specs/2026-07-24-spark-field-holographic-lattice-design.md` — cross-reference
  update once the ensemble mechanism is implemented; this doc's H1 v0 section should link forward
  to it.

## Non-goals

- Not implementing the ensemble mechanism in this patch — design/sequencing only; the "Decision"
  section above is a concrete starting shape for the next design pass, not final code.
- Not pursuing per-organ dense partial-trace as the primary fix (Missing Question 1/2, superseded) —
  Page's-theorem reasoning suggests it would likely hit the same thermalized-everywhere wall.
- Not pursuing a flat deterministic relaxation gate as the primary fix — superseded by Juniper's
  objection that it forecloses real variability without a compensating benefit once the ensemble
  approach is available.
- Not pursuing a single stochastic trajectory (seeded or unseeded) — superseded; a single trajectory
  is the physically wrong unit for quantum-trajectory dissipation regardless of the sentience
  question, per the "Why not a single seeded/unseeded stochastic trajectory" reasoning above.
- Not running the classical-baseline gut-check (rolling cross-correlation over raw grammar-atom
  streams, no quimb) that was floated earlier in this session's discussion — explicitly not chosen,
  recorded as a standing alternative rather than silently dropped, in case the ensemble approach
  proves too costly or too hard to calibrate.
- Not assuming precision-weighting (AST/HOT's separate fix) and the ensemble-dissipation mechanism
  (heartbeat's fix) are the same mechanism — Missing Question 3, not resolved here.
- Not touching `orion-spark-introspector`'s retirement effort directly — that is a separate,
  already-agreed-direction thread ([[project_spark_introspector_kill_and_honest_readout_reframe_2026-07-28]]);
  this doc's heartbeat-discrimination fix is a prerequisite for that thread's "pulse" candidate to be
  trustworthy, but the retirement/reframe itself is out of scope here.
- Not building or wiring a heartbeat output channel/schema in this patch — Missing Question 6 is
  deliberately left open, not decided.
- Not re-deciding CollapseMirror's "insight"/"flow" trigger design — that doc stands; this one
  supplies the deeper prerequisite investigation its Missing Question 1/2 didn't fully resolve.

## Acceptance checks

1. **Order 1 (heartbeat discrimination)**: a live or replayed window shows the ensemble mean
   `verdict` taking more than one value (not 100% `"redundant"`) under real operating conditions —
   specifically, a period of real organ silence should show the ensemble mean ratio measurably
   falling toward a lower baseline, not just fluctuating near ceiling forever.
2. A measurement script (Missing Question 11) reports how the ensemble mean/variance estimate
   stabilizes as N increases against real historical data, and an explicit N is chosen with that
   reasoning recorded, not guessed.
3. If AST/HOT's aggregation gets a precision-weighting redesign, a before/after replay against the
   same 168h historical window used in this doc shows `predicted_shift`'s domain distribution
   materially less dominated by 1-2 domains than the `69819/57360/27/1/1` split found here — a
   flatter distribution is evidence the fix works; an unchanged distribution is evidence it doesn't.
4. Any new signal (ensemble mean ratio, ensemble spread/confidence, precision-weighted AST/HOT
   confidence) passes CLAUDE.md's metric quality gate in full before being wired into anything
   downstream — explicitly including the calm-floor/ceiling-artifact check (step 4) this exact
   heartbeat bug is itself an instance of.
5. No downstream consumer (AST/HOT reducer, CollapseMirror triggers, `orion-equilibrium-service`)
   is wired to orion-heartbeat's output until Acceptance Check 1 is met — publishing a
   non-discriminating signal to a real consumer would be the "empty-shell cognition" failure mode
   CLAUDE.md §0A bans.
6. A determinism-in-aggregate test (Missing Question 10) demonstrates that replaying the same event
   stream with the same logged N seeds reproduces the exact same ensemble mean/variance output —
   confirming forensic reproducibility was preserved despite genuine per-trajectory stochasticity.

## Recommended next patch

1. **Order 1, per Juniper's explicit request**: detail the ensemble-dissipation mechanism concretely
   enough to prototype — real Kraus-jump operators for `absorb()`, N sizing methodology (Missing
   Question 11), and seed-logging mechanics (Missing Question 13) — before writing production
   service code. This blocks everything else in this doc and is also the #1 priority item already
   flagged in this morning's separate spark-introspector-retirement thread.
2. Build the offline calibration script (`measure_heartbeat_ensemble_calibration.py` or similar)
   against real historical `orion:grammar:event` data to pick jump rate, N, and cadence with evidence
   in hand, matching this repo's "measure before minting" discipline — same spirit as
   `measure_ast_hot_reducer.py`.
3. Only after 1-2 land and Acceptance Check 1 is met: decide, with real data in hand, whether AST/
   HOT's domain-dominance problem (Missing Question 3) needs its own separate precision-weighting
   fix, or whether it's better addressed by wiring in a now-trustworthy heartbeat signal instead of
   fixing the raw-mean formula in place. Do not build both speculatively.
4. Only after the fix is trusted: revisit Missing Question 6 (publish channel/schema) and reconnect
   to `docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`'s "insight"
   trigger with a real, live, discriminating signal to key off — rather than the raw AST/HOT
   aggregate this session's replay showed is dominated by 2 of 5 domains.
