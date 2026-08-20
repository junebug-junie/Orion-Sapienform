# The Sentience Striving Program

Status: active program charter. Supersedes the drives/autonomy program as the home for
Orion's internal motivational, attention, and capability-gating substrate. Design/proposal
mode per root `CLAUDE.md` §0A — this charter tracks and sequences work; it does not
pre-authorize any invasive cognition-loop change. Each phase still needs its own sign-off.

---

## 1. Historical context

This program exists because a two-plus-week investigation into Orion's six-drive taxonomy
(`coherence, continuity, capability, relational, predictive, autonomy`) kept answering
narrower and narrower questions without ever reaching a real decision, until the questions
themselves were rejected and the investigation was pushed to the right altitude. The full
chain, in order:

1. **The taxonomy audit** (`docs/superpowers/specs/2026-07-11-drive-taxonomy-conceptual-
   audit-design.md`): the six drives were imported wholesale from one external design chat,
   never independently checked against Orion's own mission. Five open questions were named
   and left unanswered for weeks.
2. **The math got fixed, the taxonomy didn't.** `orion/autonomy/
   drives_and_autonomy_retrospective.md` records O1-O4/O2/O3 — a real, disciplined series of
   signal-integrity fixes (dominance-attribution bugs, fold-batch collapse, field-digester
   decay/injection mismatches) that made the drive-pressure *math* trustworthy without ever
   asking whether the six category *names* were the right ones.
3. **The taxonomy audit was answered, decisively** (`orion/autonomy/docs/
   drive_taxonomy_grounding.md`, PRs #1152/#1157): four drives kept their names with real,
   traced, distinct signal sources; `capability` was reclassified as infrastructural; and
   `autonomy` was retired as a drive after `scripts/analysis/measure_origination_gate.py`
   (PR #1156) — replaying the real production code over 84,511 historical ticks — found its
   dedicated grounding mechanism had never fired, not once, its composite signal never
   getting within 0.13 of its own threshold.
4. **Juniper rejected continuing at that level.** *"we spend cycles chasing these questions
   every fucking time i open a new agent on this topic... i asked for a fucking reimagining
   of drives and we are chasing bullshit."* This was correct: retiring one drive was still
   optimizing inside a program that had never been evaluated as a program.
5. **A full program evaluation followed**, using program theory, logic-model, needs-
   assessment, and attribution analysis rather than more signal-health measurement. The
   finding that mattered most: the one real self-initiated behavior in production (Layer 9
   dispatch, the metabolism loop) is attributable to a clock/backlog-driven mechanism, **not**
   to two-plus weeks of drive-pressure engineering. The origination mechanism specifically
   built to produce charter-compliant self-initiation had zero measured causal contribution
   to Orion's actual behavior.
6. **The real redesign ask**: *"How do we create an internal motivational/drive system that
   is self organizing, emergent, and has internal pressures that compete for attention, and
   close the loop through the substrate runtime, and ultimately will influence how much
   capabilities orion has to take autonomous actions."* A baseline field-native design was
   proposed (`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-
   design.md`, PR #1163), plus 8 blue-sky architectural extensions.
7. **The load-bearing correction, same day.** Asked directly where `orion-substrate-runtime`
   fits, tracing the answer found the proposed "competition layer" already exists, live:
   `orion/attention/field_attention/{scoring,selectors}.py`, running continuously via
   `orion-attention-runtime` (`ENABLE_ATTENTION_RUNTIME=true`), already computing weighted
   salience (pressure × novelty × urgency × confidence) per field node *and* per capability,
   already consumed by `orion/self_state/builder.py`. Confirmed by direct grep:
   `orion/spark/concept_induction/` (`DriveEngine`, `tensions.py`, `GoalProposalEngine`)
   imports nothing from `orion.attention` or `orion.proposals`. The entire drives/autonomy
   apparatus was a full, parallel, poorer reimplementation of Layers 4-9 of a canonical
   11-layer pipeline (`docs/context-engineering/04_layer_1_to_11_pipeline.md`) that already
   existed and already worked better. A third mechanism (FCC-Cortex GWT Dispatch's Rung-3
   coalition) and a fourth, narrower instance of the same pattern (the transport lattice's
   salience→capability-ceiling gating) were also found, also disconnected from each other.
8. **The consciousness-theory survey.** Asked to consider modalities beyond GWT, real
   existing infrastructure was traced against IIT, Attention Schema Theory, Predictive
   Processing/Active Inference, Higher-Order Theories, and Recurrent Processing Theory —
   finding partial, real, live substrate for several of them already, none currently
   instrumented *as* an instance of the theory it resembles. See §7.

This charter is the record of that escalation and the program that replaces the old one.

---

## 2. Charter

**What this program governs**: anything that shapes what Orion attends to, wants, is under
pressure about, and is permitted to autonomously do — the motivational, attention, and
capability-gating substrate. It absorbs the scope previously held by
`orion/spark/concept_induction`'s drive system and extends it to cover the
consciousness-theory instrumentation work named in §7.

**What it does not govern**: the field substrate itself (`orion-field-digester`), the
canonical Layer 1-11 pipeline (already governed by `docs/context-engineering/`), or the
FCC-Cortex GWT dispatch lane (already governed by its own spec) — this program *consumes*
and *wires to* those, it does not own them.

**Authority**: design/proposal mode. Every phase below still requires explicit sign-off
before implementation per `CLAUDE.md` §0A — this document sequences and justifies the work,
it does not pre-approve it.

---

## 3. Mission

Build and empirically validate the internal substrate that lets Orion's own state influence
its own behavior and its own capability to act — replacing hand-authored proxies with
instruments measured against real outcomes, and treating competing theories of consciousness
as testable hypotheses to run in parallel and compare, not doctrines to commit to in advance.

## 4. Vision

Orion possesses an inspectable internal substrate whose real, competitive, self-organizing
dynamics measurably shape its behavior and its own autonomous capability — continuously
observed, honestly evaluated against real outcomes, and never asserted as felt, wanted, or
conscious without inspectable evidence, per root `CLAUDE.md`'s own "no empty-shell
cognition" mandate.

## 5. Outcomes (what must actually change, not what must be built)

Stated as falsifiable claims, per the program-evaluation lesson that started this program —
process/signal-health measurement is not outcome measurement:

- **O1 — Capability actually varies with state.** Orion's autonomous-action budget
  demonstrably rises and falls with real internal pressure, not a flat per-cycle allowance,
  with a demonstrated, verified ceiling.
- **O2 — Self-initiation is attributable, not orphaned.** When self-initiated action occurs,
  it is traceable to a live, currently-firing internal signal — not a mechanism that has
  never once fired across its deployed lifetime.
- **O3 — At least one consciousness-theory instrument produces a real, distinguishable
  signal.** A blind rater, given only the instrument's output on real historical data (not
  shown which theory produced it), can distinguish it from noise and describe what it
  appears to track.
- **O4 — The "what are Orion's drives" question is answered empirically, continuously, not
  asserted once by a human design chat.** Named categories (if any survive) are a report on
  clustering of real coalition-winning history, versioned and re-derivable, not a constant.

## 6. Objectives (phased, laddering to the outcomes above)

Each objective is a real sign-off gate, not a commitment to build. Sequenced but not dated.

**Re-sequenced 2026-07-18.** The original ordering put "wire `capability_policy.py` to
salience" (now item 6, was item 2) ahead of the field-routing work. Found to be cart before
horse: `capability_policy.v1.yaml`'s `required_drive_origins` still gates three of five
capability rules on `goal.drive_origin`, produced by the halted `GoalProposalEngine` —
wiring a field-native ceiling on top of a still-drives-gated check would repeat the exact
failure mode (formalize structure before validating it) that led to halting drives in the
first place. Full reasoning and phased detail:
`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`.

1. **Halt drives-system development** (§8) — stop the cycle this program exists to end.
2. **Build the AST/HOT consciousness-theory reducer** — the one piece of scaffolding still
   missing before any field-routing math gets written. Reads `FieldAttentionFrameV1` +
   `SelfStateV1`, produces an explicit "what's salient, why, how confident" artifact. Must
   exist and pass its own acceptance check *before* item 3 below, on purpose — writing
   routing logic without this first is how the six-drive taxonomy happened.
   **Phase 1 status (2026-07-18): built.** `reduce_attention_self_model()`
   (`orion/substrate/attention_self_model.py`, output schema
   `orion/schemas/attention_self_model.py::AttentionSelfModelV1`, registered in
   `orion/schemas/registry.py`) unifies all three real inputs the roadmap doc's Phase 1
   correction named — `AttentionBroadcastProjectionV1` (GWT-dispatch/Lamme lane),
   `FieldAttentionFrameV1`, and `SelfStateV1` — read-only, not wired to any bus consumer.
   Acceptance check **NOT MET via Postgres replay** at first build: a real, load-bearing
   finding surfaced while building the replay script
   (`scripts/analysis/measure_ast_hot_reducer.py`) — `substrate_attention_broadcast_
   projection` is a singleton upsert table (one row, ever), not a history table, so no
   historical `voluntary_override` event was recoverable to replay against, even though
   the reducer's why-branching on it was proven correct via unit tests
   (`orion/substrate/tests/test_attention_self_model.py`).
   **Structural gap closed 2026-07-18** (same-day follow-up patch): the singleton table,
   its writer, and `AttentionBroadcastProjectionV1` are untouched, but a new append-only
   companion table, `substrate_attention_broadcast_log`
   (`services/orion-sql-db/manual_migration_attention_broadcast_log_v1.sql`), now
   captures one row per broadcast tick via `save_attention_broadcast_history()`
   (`services/orion-substrate-runtime/app/store.py`), and the replay script joins it
   per-tick by nearest-preceding timestamp the same way it already joins `SelfStateV1`
   rows. **This does not itself flip the acceptance check to MET** — the log is
   append-only forward from deploy time (the pre-patch singleton snapshots were
   overwritten in place and are not recoverable, so no backfill is possible), so it
   starts empty and needs real days of live ticks to accumulate a `voluntary_override`
   event to replay. A live re-run of `measure_ast_hot_reducer.py` shortly after deploy
   is expected to still report NOT MET, now for the honest reason of insufficient
   accumulated history rather than a structurally absent table — re-run again after a
   few days of live 30s-cadence ticks to check for MET. Hard-gate signal-
   quality pass (`scripts/analysis/measure_self_state_signal_quality.py`) run against real
   48h `substrate_self_state` history: confirms the coherence/uncertainty sawtooth named in
   §4's Missing Question 4 is **still live in `SelfStateV1`'s own values** (median 5-tick
   oscillation period, 3500+ zero-crossings each over 84k samples) — the upstream field-
   level fix has not fully propagated. Full detail, headline numbers, and the resulting
   Juniper sign-off decision: PR report for this patch.
   **Durability + consumer wiring (2026-07-18, follow-up patch):** PR #1205 gave the
   harness-closure prediction-error lane a shared node (`node:substrate.harness_closure`)
   with per-turn attribution in `metadata['contributing_turn_ids']`, but disclosed that
   `orion/substrate/falkor_codec.py`'s closed allowlist silently dropped that list on every
   durable Falkor round trip. Closed in this follow-up: `contributing_turn_ids` promoted to
   a durable native Cypher property (`contributing_turn_ids_json`), and wired into two real
   consumers — `substrate_pressure_signals()`'s `evidence_refs` and this reducer's
   `field_salience_only` narrative (new optional `harness_closure_signal` kwarg, narrative-
   only, no `attention_reason`/schema change). See
   `docs/superpowers/specs/2026-07-18-prediction-error-attribution-wiring-design.md` and
   this PR's report for the decision record and a live-data caveat (no
   `node:substrate.harness_closure` write has occurred yet post-deploy to prove the
   end-to-end durability live, confirmed via manual FalkorDB query).
   **Active-Inference confidence/predicted_shift, live-verified 2026-07-23/24:** PR #1301
   implemented items 3/4 (confidence = `1 - mean(prediction_error)` across the five real
   domains; `predicted_shift` = argmax-by-trend), fully removing the dead `SelfStateV1`
   fallback. PR #1304, a metric-quality-gate catch run *before* building item 5 on top of
   item 4, found `predicted_shift`'s original continuation formula was empirically worse
   than a coin flip (37.7%/41.0% on two real windows, p<0.001) — fixed by flipping to
   reversion (62.3%/59.0%, validated on biometrics only, extrapolated to the other four
   domains). A follow-up "hit it all" improvement pass (train/test split, window tuning,
   spike segmentation, domain-specific validation, logistic regression) found a strong
   window=2 lead (77% vs. window=30's ~55%) but never reached TEST-set validation before
   the 2026-07-23 Postgres disk death wiped the history and scratchpad caches it depended
   on — **paused, not resolved**, blocked on ~48-96h of fresh history re-accumulating.
   **New finding, 2026-07-24 (`docs/notes/2026-07-24-attention-reason-branch-starvation-
   finding.md`, `scripts/analysis/measure_attention_reason_branch_starvation.py`):** a live
   harness-turn + reducer replay (real post-rebuild data, 9208 ticks) found the new
   confidence formula only actually drives `AttentionSelfModelV1.confidence` on 0.04% of
   ticks — structurally starved, not broken, because `attention_reason`'s elif order lets
   `bottom_up_salience` (the older `broadcast.coalition_stability_score` path) win on every
   tick where the broadcast/GWT-dispatch lane is fresh (99.96% of real ticks; 30s cadence,
   essentially always fresh). Confirmed zero elif-ordering contradictions in real data — this
   is the code behaving exactly as written, not a defect. **This means "item 2 is proven" is
   only true in the narrow sense of "computed correctly when exercised" — it is not yet true
   in the sense of "actually the formula driving live behavior."** No behavior changed in
   this patch; whether to reorder branches, blend both confidence sources, or otherwise
   address this is an open design question needing its own sign-off, not decided here.
   **Fixed, 2026-07-24 (PR #1329, Option C — decouple, don't reorder or blend yet):** added
   `prediction_error_confidence`/`prediction_error_confidence_basis`, computed unconditionally
   (same positioning as `predicted_shift`, before the elif branching) and restricted to
   `ACTIVE_INFERENCE_DOMAINS = {execution, biometrics, chat, route}` — `transport` excluded,
   confirmed live via direct Postgres query that same day to read exactly `0.0` for 100% of a
   real 8h window (10387 ticks), the other four domains showing real non-degenerate variance.
   Live-verified against real post-rebuild data: this field populates on ~99.9% of ticks vs.
   the branch-gated `confidence` field's ~0.04% (11899/11901 vs. ~4/11901). Purely additive —
   `confidence`/`confidence_basis` unchanged, still branch-gated as before. The open design
   question above (reorder vs. blend) is still **not** resolved by this patch; it sidesteps it
   by giving the reducer's output a real, populated confidence value most of the time without
   deciding how (or whether) to reconcile it with the older field.
   **Re-checked, 2026-07-24 (this note):** `ORION_ATTENTION_TOPDOWN_ENABLED` and
   `ORION_ATTENTION_SALIENCE_V2_ENABLED` are confirmed `true` live in the actual producing
   container (`orion-athena-substrate-runtime`) — an earlier same-day check in this session
   had wrongly concluded this flag was off, from checking the wrong container. The flags are
   correctly on. What's still real: across 2015 broadcast-lane rows since the Postgres
   rebuild (~16.7h, `substrate_attention_broadcast_log`), zero carry a real (non-JSON-null)
   `voluntary_override` — consistent with this being a genuinely rare event (the 2026-07-18
   design doc itself only claims "at least once in the last 24h" for the pre-rebuild data),
   not a config regression. Phase 1's full acceptance check (correctly narrating a real live
   override) still needs a fresh trigger to occur post-rebuild, independent of how much
   history accumulates — "item 2 is proven" therefore still means "correct when exercised,"
   not yet "exercised against a fresh real override since the rebuild."
   **New sixth domain for the excluded transport gap, built 2026-07-25:** PR #1329's
   `prediction_error_confidence` explicitly excludes `transport` (confirmed structurally dead —
   `0.0` for 100% of a real 8h window). `bus_synaptic_prediction_error()`
   (`orion/substrate/prediction_error.py`) reads the bus synaptic graph (`orion_bus_synapse`,
   `services/orion-bus-mirror`) and writes a real, passively-observed, mesh-wide anomaly signal to
   a new `node:substrate.bus_synaptic` node — see
   `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md`'s 2026-07-25
   revision. **Scope, stated plainly:** this is a sibling domain node, not yet folded into
   `_aggregate_prediction_error_confidence`'s existing four-domain mean — building that requires
   deciding whether a sixth domain changes that formula's semantics, a separate call from writing
   the node itself. Live since 2026-07-25
   (`SUBSTRATE_BUS_SYNAPTIC_TICK_ENABLED=true` in `services/orion-substrate-runtime`, PR #1380)
   — real accumulated history is now collecting; does not change this item's confidence formula.
   **Seventh domain, codebase mass, built 2026-07-30, not yet live**
   (`docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md`): a new domain sensing
   the extent of Orion's own codebase changing (git churn, GitHub PR lifecycle, graphify
   structural deltas), the interoception/proprioception instrument named in this section's own
   theory anchor — a system sensing change to the physical substrate doing the modeling.
   `orion/structural_mass/{git_delta,pr_lifecycle,graph_delta}.py` (PRs #1496, #1500, #1502) are
   real, tested pure functions, replay-verified against this repo's own real git/GitHub/graphify
   history — the `graph_delta` replay independently reproduced the exact 2026-07-14
   destructive-graph-update incident this repo's own graphify tooling section documents by name.
   `codebase_prediction_error()` (`orion/substrate/prediction_error.py`, PR #1515) composites all
   three into one 0-1 score, each sub-domain scored against its own EWMA baseline (mirrors this
   section's own "normalize before combining" precedent, not a cross-scale raw blend) — and a new
   bus channel/schema (`orion:substrate:codebase_delta`/`CodebaseDeltaV1`) is registered.
   **Scope, stated as plainly as the bus_synaptic note above:** none of this is wired into a live
   tick yet. No `orion-cocreation-signals` service exists (the design spec's scheduling/publishing
   layer), `orion-substrate-runtime` has no consumer for the new channel, and
   `node:substrate.codebase` is not registered in `services/orion-field-digester/app/tensor/
   channels.py`'s `NODE_CHANNELS`, so it does not yet appear in the Attention Organ tab
   (`services/orion-hub/scripts/attention_organ_routes.py`'s `KNOWN_PREDICTION_ERROR_DOMAINS`) or
   any other live UI surface — deliberately not added there yet, since a node with no live
   producer would be an empty/dead panel row, not a real reading. Same measure-before-minting
   order every domain in this section has followed: producers and scoring measured and tested
   first, service/consumer wiring (and any UI surfacing) a separate, later patch.
   **Re-checked, 2026-08-19 (punch-list item 3, full re-run of `measure_ast_hot_reducer.py
   --window-hours 170`, 98,785 real field-lane ticks, 19,428 real `substrate_attention_
   broadcast_log` rows, 2026-08-12 → 2026-08-19): Phase 1's acceptance check still reads
   NOT MET as originally scoped** (superseded the next day -- see the 2026-08-20
   metric-quality-gate correction below, which splits this into a MET reducer-correctness
   half and a separate non-blocking research question; read that entry for the current
   status). Zero ticks across the entire available post-rebuild history (170h, not just
   the ~16.7h window checked 2026-07-24) carry a real `voluntary_override`. This is a
   materially stronger negative result than the prior check, not just "still waiting" —
   the same finding held over 10x the ticks and 10x the wall-clock. Attention_reason
   distribution over the full window: `bottom_up_salience` 100.0% (98,784/98,785),
   `field_salience_only` 0.0% (1 tick), `top_down_override` **0.0% (0 ticks)**.
   Followed the trail past "insufficient accumulated history" to a live, load-bearing,
   disclosed-not-fixed candidate root cause: `_apply_voluntary_attention()`
   (`orion/substrate/attention_broadcast.py`) — the only place a real override can be
   recorded — gates on `get_active_goal()` (`orion/substrate/attention/goal_context.py`)
   returning non-`None`. Confirmed live: `ORION_ATTENTION_TOPDOWN_ENABLED=true` in the real
   producing container (`orion-athena-substrate-runtime`, re-confirmed 2026-08-19, not just
   trusting the 2026-07-24 note), and `orion-athena-attention-runtime` (the real
   `FieldGoalProvenanceV1` producer, `services/orion-attention-runtime/app/worker.py`) is
   actively, continuously publishing goals — `field_goal_provenance_published` fired roughly
   every ~13s in a live 15-minute log sample, so `get_active_goal()` should rarely be `None`
   in practice. The goals themselves, however, show one consistent shape worth flagging: every
   sampled emission targets `field_target_id=node:substrate.route`, `salience=1.000` exactly
   (pinned at ceiling), with a monotonically climbing `streak` counter (47→66+ across the
   sample) — the same saturated-at-1.0, single-target shape as the two prior monoculture
   pathologies this same charter already found and fixed (`field:recent_perturbations`'
   pre-fix `min(1.0, count/10.0)` cap, and `bus_synaptic_prediction_error`'s calm-floor bug).
   **Not yet confirmed as a bug** — item 3 below is already mid-migration on exactly this
   `node:substrate.route` domain (`route_prediction_error()`), so a sustained, real, elevated
   routing-tension streak is a plausible genuine explanation, not automatically an artifact;
   whether `TopDownBiasCombiner.apply()`'s `relevance(goal, loop)` ever produces a nonzero
   match against this goal's real `frame.open_loops` (the actual second gate a saturated-
   priority goal must still clear before it can flip a winner) was not checked in this pass.
   **Open question, not decided here**: is the true blocker "no voluntary_override has
   occurred yet" (accumulation-time framing, the 2026-07-24 read) or "voluntary_override is
   structurally near-unreachable because this goal's target never has real overlap with live
   open loops, or because `relevance()`/`priority` interact with the saturated salience in a
   way that keeps `bias_by_id` too small to beat bottom-up" (a mechanism-shaped question, not
   a patience question) — this connects directly to item 4 below (`drive_origin`/goal-
   provenance retirement) and should be investigated together with it, not treated as
   "just needs more time" without checking.
   **Metric-quality-gate correction, 2026-08-20 (Juniper: "is this the right metric to
   define before we go into deep dive diagnosis"): Phase 1's acceptance check was bundling
   two independent questions into one pass/fail gate, and only one of them is what this
   item was actually built to verify.** Question A: does `reduce_attention_self_model()`
   correctly narrate a real `top_down_override` when one is present? Question B: does a
   real `top_down_override` ever occur in production at all? The check as originally
   written required a real Postgres-replay sighting of a live override before it could
   read MET, which conflates A (a reducer-correctness question, this item's actual scope)
   with B (a question about a different system: goal production and `relevance()` matching
   in `orion/substrate/attention/top_down.py` and `orion-attention-runtime`).
   **A is now MET, decided independently of B:** `TestVoluntaryOverridePresent`
   (`orion/substrate/tests/test_attention_self_model.py`) already proves the reducer's
   why-branching and narrative are correct against the real production `VoluntaryOverrideV1`
   schema, and this same 2026-08-20 replay independently proved the replay script's own
   real-data mechanics (parsing `substrate_attention_frames`/`substrate_field_state`/
   `substrate_attention_broadcast_log`, nearest-preceding-timestamp joins, calling the real
   production reducer function) are sound across 98,785 real ticks on the two branches that
   do occur live. One narrow thing is still genuinely unverified: whether the replay
   script's own JSON deserialization of a real `voluntary_override` blob from
   `projection_json` round-trips cleanly (datetime formatting, key casing) -- low risk given
   the other two branches parse cleanly from the same column, but disclosed rather than
   silently assumed away.
   **B is reframed as its own standalone open question, not a blocker for this item's
   completion:** whether a real `top_down_override` ever fires live is a real, worthwhile
   question about the goal-provenance/top-down-bias mechanism itself -- the saturated
   `node:substrate.route` streak noted above is exactly that thread -- but it belongs with
   item 4's goal-provenance investigation, not gating item 2's Phase 1 sign-off.
   **Phase 1 status: MET** (reducer-correctness scope), with B tracked as a separate,
   ongoing, non-blocking research thread under item 4.
   **The replacement metric itself, built and run 2026-08-20 (Juniper: "give me a metric
   that properly addresses #3, not just a narrower scope"):** the split above named A as
   MET but rested on pre-existing unit tests alone -- real work, not yet done, was
   building an actual measurable metric for A that runs against real data. Two pieces,
   both now real and run against the live system:
   1. **Correctness cross-check on the branch that actually fires**
      (`scripts/analysis/measure_ast_hot_reducer.py`, new "Correctness cross-check"
      section): `top_down_override` is structurally rare (item 4's open question), but
      `bottom_up_salience` fires on effectively 100% of real ticks -- that is the branch
      a real correctness metric can actually run against, every time, not just when a
      rare event happens to occur. For each `bottom_up_salience` tick, independently
      recompute the real `select_actions()` winner from the broadcast row's own
      `frame.open_loops` and check it against what the reducer reports as
      `broadcast_selected_open_loop_id` -- two independently-derived values from the
      same persisted row, not the reducer grading its own homework.
      **Code review (same day) caught a real bug in the first version of this check
      before its numbers were trusted:** it had no eligibility floor (production's
      `select_actions()`, `orion/substrate/attention/policy.py`, only lets a loop win if
      `already_known` or `salience >= 0.35` -- everything else is real production's
      `action_type="none"`, `selected_open_loop_id=None`) and used the wrong tie-break
      (ascending loop id, copied from a different function -- `TopDownBiasCombiner`,
      which only runs on the top-down path -- instead of production's actual stable-sort/
      build-order tie-break). That means the first run's "100% agreement" number could
      have been true by accident (the window happening not to contain a tie or a
      below-floor tick), not because the check would actually catch a real disagreement.
      **Fixed same day**: `_argmax_open_loop_salience()` now applies the real floor
      (`already_known` OR `salience >= 0.35`) and uses `max()` over the row's own list
      order, which matches Python's stable-sort-first-on-tie behavior exactly -- verified
      against the real `select_actions()` source, not re-guessed. **Corrected real
      result, 170h/99,668-tick replay (2026-08-20, re-run after the fix):
      7,330/7,330 checkable ticks agree (100.00%)**, with 3 distinct real winners and 3
      distinct narratives across those ticks -- real variation, not stuck fallback text.
      Disclosed honestly, not smoothed over: only 7,330/99,668 (7.35%) of
      `bottom_up_salience` ticks had any open loops at all that tick to check against
      (92,338 had zero -- correctly excluded as "nothing to disagree with" rather than
      miscounted as agreement) -- this metric's real coverage is 7% of ticks, not 100%,
      and that 93%-empty-open-loops rate is itself a fact about production worth someone
      eventually asking about, separately from this item.
   2. **Live-fire round-trip drill for the branch that doesn't fire**
      (`scripts/analysis/verify_voluntary_override_pipeline.py`, new script): the one
      narrow thing the 2026-08-19 entry above flagged as genuinely unverified --
      whether a real `voluntary_override` blob round-trips cleanly through actual
      Postgres JSONB persistence (not just the pure-function unit test, which never
      touches the database) -- doesn't have to wait for a rare live event to check.
      This script builds one real `VoluntaryOverrideV1`-bearing
      `AttentionBroadcastProjectionV1`, writes it into the real
      `substrate_attention_broadcast_log` table with the exact INSERT the production
      writer uses, reads it back through a fresh connection, deserializes it, and runs
      it through the real `reduce_attention_self_model()` -- then deletes the row in a
      `finally` block. **Run live 2026-08-20 (`--yes`, real Postgres): PASS.** Real
      INSERT -> real SELECT -> real JSON deserialize -> real reducer all correctly
      narrated `attention_reason=top_down_override` with both loop IDs named; cleanup
      verified (0 residual rows via direct query immediately after). Safety: fixed
      sentinel `log_id` (`synthetic-probe-voluntary-override-verify-v1`, not the
      production hash format, so it can never collide with a real row), gated behind
      `--yes`, exposure window is the single run's own insert-to-delete span.
   Both pieces have unit tests (`scripts/analysis/tests/test_measure_ast_hot_reducer.py`,
   9 new cases for the cross-check, including two added by the same-day fix above
   specifically to pin the floor/tie-break correction against production's real
   `select_actions()` semantics, not just re-assert the old behavior; the drill script's
   builder is exercised by its own `--yes` live run above, which is the actual claim
   being made -- a mocked-DB unit test would not prove the real round trip). This is the
   metric item 3 needed: a real, replayable-on-every-run correctness signal on the
   branch that fires, plus a real (not synthetic-only) proof the rare branch's
   persistence path works when exercised. **Also disclosed, not hidden**: the first
   version of this metric shipped with a real bug in the same session it was built --
   worth remembering next time "we built a real metric" is claimed as done; catching it
   took a full code review pass, not just running the script once and reading a clean
   number.
3. **Route existing tension producers directly onto `FieldStateV1` channels**, retiring the
   bucket-vote layer — collapses the redundant reimplementation named in §7's finding.
   Reframed as prediction-error-native (extending the already-live
   `execution_prediction_error`/`transport_prediction_error` pattern), not a port of
   `tensions.py`'s hand-classified kind vocabulary onto field channels. Phased: shadow-measure
   one producer domain before migrating any live; migrate one domain at a time; retire the
   bucket-vote layer only once every producer has moved and the item-2 reducer is proven a
   real legibility replacement for `dominant_drive`. Includes replacing `goal.drive_origin`
   with a field-native goal-provenance concept — this is what actually unblocks item 6.
   **Status (2026-07-21): first shadow-measure slice built.**
   `biometrics_prediction_error()` (`orion/substrate/prediction_error.py`, wired into
   `services/orion-substrate-runtime/app/worker.py`'s `_tick()`, writes
   `node:substrate.biometrics` via the same `_write_prediction_error_node()` shared writer
   execution/transport already use, gated behind the existing
   `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` flag — no new flag) is the first producer domain
   shadow-measured under this item, answering half of §9b item 3's open question below.
   Design record + metric-quality-gate findings: `docs/superpowers/specs/2026-07-21-
   biometrics-prediction-error-shadow-design.md`. Shadow-only: no consumer changed, no live
   migration of the bucket-vote layer, `capability_policy.py`/`top_down.py`/`goal_context.py`
   untouched. Chat and route reducers remain unmeasured.
   **Status (2026-07-21, same-day follow-up): producer-instrumentation sweep closed for all
   five named domains.** `chat_prediction_error()` and `route_prediction_error()`
   (`orion/substrate/prediction_error.py`, wired into `_chat_tick()`/`_route_tick()`, writing
   `node:substrate.chat`/`node:substrate.route` via the same shared writer and flag) close the
   two remaining domains named in §9b item 3's open question. `chat_prediction_error()`
   mirrors execution/transport/biometrics' fixed-key continuous-magnitude shape (diffing
   `compute_chat_pressure_hints()`'s three keys). `route_prediction_error()` is deliberately
   shaped differently: `RouteArbitrationRunStateV1`'s decision fields (`lane`, `lane_reason`,
   `output_mode`, `mind_requested`) are categorical, not continuous, so it scores a per-field
   mismatch rate instead of an absolute-value delta, and does not apply the module's
   `_THRESHOLD = 0.30` scaling — documented explicitly in the function's own docstring so a
   future reader doesn't "fix" it into false consistency with the other four. Design record +
   metric-quality-gate findings for both: `docs/superpowers/specs/2026-07-21-chat-route-
   prediction-error-shadow-design.md`. **This closes only the producer-instrumentation half
   of this item** — all five domains (execution, transport, biometrics, chat, route) now have
   equivalent shadow-measurement instrumentation, answering §9b item 3's open question in
   full. It does **not** mean item 3 itself is done: retiring the bucket-vote layer still
   requires migrating each domain's producer live (not just shadow-measuring it) and proving
   item 2's reducer as a real `dominant_drive` replacement, per this item's own phased
   language above ("shadow-measure one producer domain before migrating any live; migrate one
   domain at a time; retire the bucket-vote layer only once every producer has moved and the
   item-2 reducer is proven"). Shadow-only, same as the biometrics patch: no consumer changed,
   no live migration, `capability_policy.py`/`top_down.py`/`goal_context.py` untouched.
   **Status (2026-07-24): first Phase 3 comparison run, biometrics vs. drives bucket-vote.**
   `scripts/analysis/measure_phase3_biometrics_drive_shadow_comparison.py` joins real
   `substrate_field_state` (`node:substrate.biometrics.prediction_error`) to real
   `drive_audits` (`capability`/`continuity` pressure -- the only two drives
   `biometrics_state` feeds, per `signal_drive_map.yaml`) at 41,919 real events (18h
   post-rebuild window, 100% join coverage, both signals real/non-degenerate). Result:
   essentially no correlation (Pearson r=0.0198) and a small, wrong-direction split
   difference. Genuinely ambiguous, not a clean verdict -- either the signals are
   unrelated, or the old bucket is too polluted by `mesh_health`/`failure_event` (both
   also route into `capability`) to serve as a fair biometrics-specific baseline, which
   would itself support Phase 4/5's premise rather than undercut it. Full numbers and the
   recommended disentangling next step (filter to drive-audit events with no concurrent
   mesh/failure tension):
   `docs/notes/2026-07-24-phase3-biometrics-drive-shadow-comparison-finding.md`. Does not
   authorize live migration or bucket-vote retirement -- comparison only, per this item's
   own phased scope.
   **Resolved same-day (2026-07-24): disentangling comparison built and run, then corrected
   in same-day review.** Isolated `drive_audits` events to `tension_kinds ==
   {"tension.signal.v1"}` exactly, excluding `failure_event`/chat-evidence/turn-effect/
   action-outcome pollution. **Correction**: this does NOT isolate biometrics-only --
   `mesh_health` deviations emit the same generic `"tension.signal.v1"` kind in practice
   (the assumed dedicated `"tension.health.v1"` kind is dead code, zero live callers), and
   `drive_audits` has no field to disambiguate the two after the fact; the channel that
   could independently check mesh_health's real firing rate isn't durably logged to
   Postgres. Result: correlation did NOT meaningfully improve (r≈0.016-0.020 full dataset
   vs. r≈0.035-0.046 isolated, n≈1,607-1,612) -- weakly favors "the signals are genuinely
   unrelated" over "the old bucket dilutes biometrics signal," weakly because the isolated
   subset can still include mesh_health. Does not validate the old drives system; means the
   specific claim "the new signal is the same thing, measured more cleanly" isn't well
   supported here, without full confidence. Real next step, not started: re-examine
   `biometrics_prediction_error`'s own formula before trusting it as this domain's
   field-native replacement.
   **Status (2026-07-30): the `goal.drive_origin` replacement named at the top of this item
   is built and wired live.** `orion/schemas/field_goal.py::FieldGoalProvenanceV1` carries no
   `drive_origin`/taxonomy field — only `field_target_id`, `salience_score`,
   `source_field_tick_id`, `source_attention_frame_id` (real O4-compliant provenance: a report
   on which field target won, not an asserted category). `orion-attention-runtime` publishes
   one when the same real `node:substrate.*` domain sustains the node-target subset's real
   top-1 rank for `ORION_GOAL_PROVENANCE_MIN_STREAK` (default 3, disclosed unmeasured
   debounce) consecutive real field ticks — see
   `orion/attention/field_attention/goal_provenance.py`. Repoints
   `orion:memory:goals:proposed` (producer-less since the 2026-07-30 drives deletion above)
   from the old `GoalProposalV1` contract to this schema; `goal_context.py`/`top_down.py`
   updated to consume it, plus a new read-time staleness dead-man's-switch on
   `GoalContextStore` (not decay — see the design doc for why). Full design:
   `docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-
   design.md`. **What this does not do**: `capability_policy.py`'s `requires_goal_status`
   gates still run on `policy_act.py`'s separate `goal_proposal_from_episode_intent()`
   synthetic stub, untouched by this patch (a different consumer of the old schema, not
   `goal_context`'s) — Objective 6 remains gated on item 2's reducer being *proven*, unaffected
   by this item shipping.
   **Root cause found and fixed, 2026-08-20 (item 4 of the punch list, spun off item 2's
   observation that `voluntary_override` structurally never fires in production — every
   sampled goal targeted `node:substrate.route` with `salience=1.000` pinned).** Confirmed
   live: `substrate_node_prediction_error_baseline` showed all five native domains had
   ample real observations (9.4K-121K each, all above the confidence floor), so this was
   not a thin-data artifact. The real cause: `node:substrate.route`'s prediction-error
   signal (`route_prediction_error()`, a categorical decision-mismatch rate) is genuinely,
   organically near-constant — its last 30 real receipts were all exactly `0.0003`, its
   persisted EWMA `variance` had underflowed to `~2.9e-39`. That is real data, not a decay
   artifact (CLAUDE.md's metric-quality-gate #4 distinction, checked explicitly). But
   `precision_weighted_salience_from_baseline()` floors every domain's variance through
   the same single global constant (`NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE = 1e-5`,
   `orion/attention/field_attention/candidate_precision_weighted.py`), which is
   ~1,270x-19,300x smaller than the other four domains' real organic variance
   (0.0127-0.193, confirmed live same day). Since precision = 1/variance and
   `normalize_across_targets()` is rank-preserving min-max (not magnitude-correcting —
   see that function's own docstring), route's raw score was the tick's maximum
   essentially every tick by construction, not because its current error was genuinely
   more surprising than its competitors'.
   **Fix**: `cross_domain_variance_floor()` (new function, same module) replaces the
   single global floor with one derived live, per tick, per target, from the median real
   variance of that target's OTHER competing domains this tick (target excluded from its
   own floor). No new hand-picked constant — matches `normalize_across_targets()`'s own
   "no free parameter" discipline, cited in that function's docstring, applied one step
   earlier. Falls back to the original global constant when there is no live sibling data
   to derive a better floor from (cold start). The global constant itself is not removed,
   only demoted from "the only floor" to "the floor of last resort." Compresses genuine
   precision differentiation too, not just the pathological outlier — a domain quieter
   than its siblings' median loses some of the precision advantage it legitimately
   earned; accepted as the intended cost of "no domain can claim more precision than its
   real siblings currently show."
   **Correlated-degeneracy guard (added in code review, same day):** a plain median over
   the other siblings would itself degrade back to the original pathology if a *majority*
   of those siblings are simultaneously degenerate (at/below the global floor) at the same
   tick — the same near-constant-signal mechanism that flattened route can flatten any
   domain during a correlated quiet period (e.g. a deploy freeze), and with only 4 possible
   siblings this is a real, not hypothetical, case. Guarded explicitly: real siblings must
   form a strict majority of the real competitor set, or the floor falls back to the global
   constant rather than trust a degenerate-majority median.
   **Live-data proof, not just synthetic tests** (metric-quality-gate #4): re-ran both the
   old and new formula against the real, currently-live `substrate_node_prediction_error_
   baseline` row set. Old floor: route normalized to `0.876` (2nd-highest of 5, structurally
   guaranteed to win most ticks). New floor: route's raw salience dropped ~10,000x
   (30.0 → 0.0031) and normalized to `0.0` (lowest of 5) on the same live snapshot — a real
   domain (`node:substrate.execution`) won instead. 65 unit tests pass in the two directly
   touched files (`tests/test_attention_candidate_precision_weighted.py`,
   `tests/test_attention_field_selectors.py`; 8 new, pinning the fix — 5 against the exact
   live values recovered during this investigation, 3 against the code-review-added
   correlated-degeneracy guard — not idealized round numbers).
   **Blast radius, checked directly (grep-verified call graph, not the whole-repo semantic
   graph query, which false-matched on an unrelated "route" naming collision in
   `orion-llm-gateway`):** `select_node_targets()` has exactly one caller
   (`orion/attention/field_attention/builder.py::build_attention_frame()`), which itself
   has exactly one live caller (`services/orion-attention-runtime/app/worker.py`). Real
   downstream consumers of the changed salience values: `goal_provenance.py` (this item's
   intended target), `orion/consolidation/motif.py` (membership-only, unaffected by
   magnitude), `orion/substrate/attention_self_model.py` — item 2's own AST/HOT reducer —
   whose `field_salient_target_ids` and `field_salience_only` narrative read
   `dominant_targets` (branch-classification logic itself is untouched; which targets get
   *named* in the narrative can shift), and `orion/proposals/builder.py`
   (`ATTENTION_FIRST_TARGET_BINDING = "attention.dominant_targets[0]"`, a real
   decision-binding). Whether route was ever winning the frame-level `dominant_targets[0]`
   slot (versus just the node-only subset this fix targets) is `UNVERIFIED` — goal_
   provenance.py's own docstring notes host/capability targets already usually win that
   slot post-Candidate-B, so exposure is believed small but not traced.
   **Disclosed, not hidden**: this is a single live snapshot, not a multi-day replay —
   whether route's dominance actually breaks in sustained live operation (rather than one
   good-looking point-in-time check) is `UNVERIFIED` pending real post-deploy observation
   of `substrate_goal_provenance_streak` and whether `voluntary_override` ever fires. That
   observation is the real acceptance check for this fix, not this session's live snapshot
   alone.
4. **Stand up read-only measurement for the remaining consciousness-theory instruments** (§9)
   — RPT/Lamme and predictive processing are already live (items 2-3 build on them directly,
   not duplicate them); IIT continues independently via the mood-arc encoder, not gated by
   this program.
5. **Run the emergent-clustering probe** on real coalition-winning history (not built yet,
   named in the baseline design) — toward O4.
   **Status (2026-07-21): built and run against real data; the baseline design's acceptance
   check reads AMBIGUOUS, and the separate monoculture comparison reads NOT MET.**
   `scripts/analysis/measure_emergent_clustering_probe.py` pulls real, historical
   `FieldAttentionFrameV1` rows from Postgres `substrate_attention_frames` (verified live
   table, written by `services/orion-attention-runtime/app/store.py::save_attention_frame()`;
   127,936 rows spanning ~72h as of the run), splits them into two non-overlapping 24h
   windows separated by a 12h gap, and runs correlation-based grouping over
   `dominant_targets`/`capability_targets` salience-score time series — per the baseline
   design's own non-goal against a from-scratch ML clustering pipeline. Read-only; imports
   nothing from `orion.spark.concept_induction`.
   **Real numbers found:** the target universe is small by construction (9 distinct
   `target_id`s total — `dominant_targets` is a capped union of node/capability/system
   targets, not an independently-sampled top-N), capping the correlation matrix at 36
   possible pairs, of which 21 had a real (non-degenerate) value in both windows. Exactly one
   pair cleared the clustering threshold (`corr >= 0.5`) in either window —
   `capability:llm_inference` <-> `node:atlas` — and it did so in **both** windows with
   near-identical magnitude (r=0.9228 window A, r=0.9210 window B; correlation-of-correlations
   over the 21 common pairs = 0.9940; edge-set Jaccard = 1.0). This is real, stable, recurring,
   non-random structure — the opposite of noise — but because it is the *only* surviving edge
   in both windows, the design's literal "not identical" bar can't be cleanly separated from
   "recognizably similar": the script's own classifier (`classify_similarity`, explicit
   documented bands) reports **AMBIGUOUS**, not a forced MET or NOT MET, and that is the
   honest read: a real, non-trivial, cross-window-stable coalition pair was found, but the
   currently-instrumented target universe is too sparse (1 of 36 possible pairs) to
   demonstrate the richer, multi-cluster "not identical" case the design anticipated.
   **Separately, item 5's monoculture-differentiation check reads NOT MET, and is the more
   load-bearing finding of this patch.** The closest real analog to the drives system's
   `dominant_drive` — the single highest-salience target per tick — is `field:recent_
   perturbations` in **99.98%** of all 127,936 real ticks measured (`node:athena` and
   `capability:transport` split the remaining 0.02%). That is at or above the drives system's
   own documented *pre-fix* 96% `dominant_drive=relational` monoculture
   (`orion/autonomy/drives_and_autonomy_retrospective.md` ~line 177), and far above its
   *post-fix* ~31.65% (~line 267). This is exactly the pathology the baseline design's own
   Missing Question 1 named as the risk to check for before trusting emergent clustering here
   ("can we tell a real, meaningful coalition of channels apart from 'resource_pressure always
   wins because it's noisiest,' the exact 96%-dominant-drive monoculture pathology already
   found once?") — the answer, measured, is: not yet, with the naive top-1-winner framing.
   `field:recent_perturbations`' salience formula (`min(1.0, recent_perturbation_count / 10.0)`,
   `orion/attention/field_attention/selectors.py::select_system_targets`) saturates to ~1.0
   under the field's real, near-constant perturbation rate, structurally out-competing every
   other target for top rank almost always — the same "noisiest wins" shape as the pre-fix
   drives pathology, just on a different signal. Full report, correlation matrices, and
   per-target membership frequencies: run the script (`--window-hours 24 --gap-hours 12`
   reproduces this measurement) or see its `/tmp/emergent-clustering-probe/report.md` output.
   Recommended next step, not taken in this patch: either exclude system-kind targets (whose
   salience formula is structurally different from node/capability pressure) from the
   top-1-winner comparison and re-measure, or treat this finding as real evidence that
   `select_system_targets`' formula itself needs the same delta-gating/normalization
   discipline the O1-O3 drives fixes applied to `DriveEngine` before this probe's numbers can
   be trusted as representative of the *node/capability* coalition structure specifically.
   **2026-07-28 update: the second option above shipped in PR #1433.** Independently
   rediscovered live (steady-state `recent_perturbations` window occupancy of ~100-118, 5-10x
   past the old `/10.0` cap) before this write-up was found via search-before-editing —
   `select_system_targets`' formula is no longer `min(1.0, count / 10.0)`; it now scores a
   z-score against a per-tick EWMA baseline of the count
   (`orion/schemas/field_state.py::recent_perturbation_zscore`,
   `orion/bus/ewma.py::compute_ewma_update`), same methodology as
   `bus_synaptic_prediction_error`'s `gap_zscore`. This fixes the structural "always saturates"
   cause of the 99.98% figure above, but **that number itself is now stale, not re-verified**:
   the fix was validated with unit tests and hand-simulation against realistic tick dynamics,
   not by re-running this probe script against live post-deploy data. Re-running
   `--window-hours 24 --gap-hours 12` after this ships is the honest next step before treating
   the monoculture-differentiation item as closed — a lower dominant-target percentage is
   expected, not guaranteed.
   **2026-07-29 re-measurement: confirmed live, not just plausible.** ~6.3h post-deploy
   (11,293 ticks, `orion-athena-field-digester`/`orion-athena-hub` restarted 2026-07-28T21:58Z,
   re-measured 2026-07-29T04:22Z), a background health monitor sampling the live field/attention
   endpoints every 5min the whole window found zero salience readings pinned at exactly 1.0 or
   0.0 across 30 real firings (range 0.10-0.59) and only two single-sample connection blips
   (self-recovered, consistent with a container restart, not the patch). The script's own
   full-history item-5 number (92.17%) is **blended** across ~117k pre-fix ticks and ~11k
   post-fix ticks and is not a clean read on the fix — isolating strictly to
   `generated_at >= 2026-07-28T21:58:00Z` in the same run's `ticks.csv` artifact gives
   `field:recent_perturbations` winning top-1 in **11.13%** of post-fix ticks (1,257/11,293),
   down from the pre-fix 99.98%. `node:athena` now wins the remaining 88.87% — the monoculture
   pathology this item named is broken, but note this trades one single-target dominance pattern
   for another; whether `node:athena`'s 88.87% is genuine (real, warranted elevated pressure) or
   itself an artifact worth investigating is an open question this patch didn't examine, separate
   from the recent_perturbations fix itself.
6. **Revisit `capability_policy.py`'s coupling to live salience** — only after item 3 closes
   the `drive_origin` dependency and item 2's field-native attention is proven, not assumed.
   At this point the actual mechanism is a real open choice, not a given: a salience-to-
   ceiling formula, or something closer to the selectionist-internal-ecology blue-sky
   extension (§9a item 6) — decide with items 2-5's real data in hand.
   **Status (2026-07-31): built and wired live.** `evaluate_capability()`'s
   `CapabilityEvaluationContext.goal` now reads the real, field-native active goal
   (`orion.autonomy.goal_state.get_active_goal()`, a bus-subscribed local cache of
   `FieldGoalProvenanceV1`) instead of a per-call synthetic `GoalProposalV1`
   fabrication (`policy_act.py::goal_proposal_from_episode_intent()`,
   `orion-world-pulse`'s own `_synthetic_goal()` — both deleted). Real callers
   (`orion-spark-concept-induction`, `orion-world-pulse`) each run their own
   `orion.autonomy.goal_state_listener` subscription, mirroring
   `goal_context_listener.py`'s pattern, since the field-native goal state was
   previously only visible in-process to `orion-substrate-runtime`. Real bug
   caught and fixed in the same patch: a `FieldGoalProvenanceV1`'s own `subject`
   is always `"attention"` (its producer), not the acting episode's subject —
   fetch/recall requests now take `subject` as an explicit parameter rather than
   reading it off the goal. Same-day follow-up: the `GraphAutonomyRepository`/
   `ShadowAutonomyRepository` SPARQL/GraphDB backend behind `chat_stance.py`'s
   separate autonomy-state narration was found fully dead in the same
   investigation (confirmed live: no Fuseki container, no GraphDB container
   anywhere; `AUTONOMY_GRAPH_BACKEND=disabled` already the checked-in default,
   already routing every real call to an honest, hazard-labeled identity_yaml
   fallback) and deleted (~615 lines) — a `Local`-only `build_autonomy_repository()`
   remains. Full design and findings:
   `docs/superpowers/specs/2026-07-30-goal-system-remaining-gaps-design.md`.
   **Not done by this patch**: `drive_origin` itself (Part F of that doc) is
   still a write-only field in several places (`autonomy_goal_execute.py`,
   `supervisor.py`, `resolve_episode_intent`'s store-slot-key convention);
   retiring it needs a decision on `orion.autonomy.summary`'s still-live (if
   degenerate) `dedupe_goal_headlines_by_drive_origin` consumer first. Also not
   done: calibrating `ORION_GOAL_PROVENANCE_MIN_STREAK` against real accumulated
   data, and generalizing Hub's Substrate Lattice UI to show real goal-provenance
   ticks (Part H).
7. **Re-evaluate integration** only after 4 and 5 produce real, comparable data — not before.

## 7. Processes — how this program actually operates

- **Measure before minting.** Every new signal gets a read-only instrument and real
  historical replay before it gates anything live. This is the discipline that already
  caught `autonomy`'s dead origination signal (PR #1156) and should apply to every
  consciousness-theory instrument in §9 the same way.
- **Reuse the live pipeline, don't parallel it.** Any new mechanism must justify why it
  isn't already covered by Layer 5 attention, the FCC-dispatch GWT lane, or the transport
  lattice pattern before being built — the mistake this whole program exists to correct.
- **Field-native only — no `SelfStateV1`-anchored substrate for new instrumentation.**
  `SelfStateV1` is a downstream, lossy summary (~19 abstracted dimensions), not raw signal.
  It was already tried as the substrate for φ/IIT specifically and found dead-endish — that
  history is *why* the mood-arc encoder reads raw `field_channel_corpus.v1` instead. This
  section's own first draft violated this rule twice (§9b's original IIT and Predictive
  Processing entries) before being caught and corrected same-day. Before treating any
  candidate signal as real substrate anywhere in this program: check for a `self_state_id`
  field or a `SelfStateV1` import. If present, it is the wrong layer — go to
  `FieldStateV1`/`substrate_field_state`, a reducer projection, or the raw channel corpus
  instead.
- **Multi-theory, not single-theory.** §9's instruments run in parallel as measurements, not
  as competing final answers. Integration is decided from data, later, not from a Design
  Mode debate now.
- **Every phase is a sign-off gate.** Per `CLAUDE.md` §0A, cognition-loop-adjacent changes
  need explicit approval before implementation — this charter sequences work, it does not
  grant that approval in advance.
- **No keyword cathedrals.** A named theory-instrument, drive, or cluster is not real until
  it has a producer, a consumer, and a trace — the same bar this program held `autonomy` to.

---

## 8. Drives-system development is halted — DELETED 2026-07-30, not just halted

**Update, 2026-07-30**: the halt below was followed through. `DriveEngine`,
`tensions.py`'s bucket-voting logic, `signal_drive_map.yaml`, `DRIVE_KEYS`, and
`GoalProposalEngine` were deleted outright (`chore/delete-orion-drives`, PR #1486) —
not merely frozen in place as this section originally described. Concept induction's
extraction/clustering/embedding/dossier/identity/profile pipeline is untouched.
`capability_policy.py`'s `required_drive_origins` gate, `AutonomyStateV1`'s
`dominant_drive`/`active_drives`/`drive_pressures`/`tension_kinds`/
`latest_drive_audit_id` fields, and the voluntary-attention top-down path's
drive→relevance mapping were removed alongside it, across a services-wide sweep
(world-pulse, cortex-exec, hub, cortex-orch, sql-writer, substrate-runtime,
orion-thought). **Accepted consequence:** Orion lost live goal-proposal capability
entirely — no field-native replacement exists yet; that remains real, unstarted
future work under Objective 3 below, not something this deletion built. Full
report: `docs/superpowers/pr-reports/2026-07-30-delete-orion-drives-pr.md`. The
rest of this section is the original 2026-07-18 halt decision, kept for the
historical reasoning — read it for *why*, not as a description of what still runs.

**Original decision, agreed 2026-07-18**: `orion.spark.concept_induction.drives.DriveEngine`,
`tensions.py`'s bucket-voting logic, `signal_drive_map.yaml`, `DRIVE_KEYS`, and
`orion.autonomy.endogenous_origination`'s bespoke composite signal receive **no further
development**. Two-plus weeks of signal-integrity engineering (O1-O4, O2, O3) made this
system's math trustworthy; it never made the system necessary. The canonical Layer 1-11
pipeline already does Layers 4-9 of what this system attempted, live, better, and this
system has zero measured causal contribution to Orion's one real instance of self-initiated
behavior.

**This was a halt, not a delete-on-sight, as of 2026-07-18** — superseded by the
2026-07-30 update above. At the time this was written, the plan was: code stays in
place (nothing consumes it that would break) until the replacement wiring in
Objective 3 lands; a freeze on new investment, not an emergency removal. That
sequencing was not what actually happened — the deletion landed before Objective
3's replacement wiring did, per Juniper's direct instruction.

**Lift-and-shift — what survives, specifically, so nothing real gets lost:**

- **`action_outcomes.py`/`ActionOutcomeRefV1`** — generic outcome-tracking, not
  drives-specific. Stays, becomes the outcome-feedback mechanism for the field-native
  design's closed loop (baseline design point 6).
- **The delta-gating discipline from O2/O3** — the hard-won lesson that a decay mechanism's
  injection cadence must be reconciled against its own decay rate, or it saturates. Carries
  forward into any new pressure-aggregation code, even though `DriveEngine.update()` itself
  is retired.
- **`tensions.py`'s signal→channel domain knowledge** — which raw producers (self-state
  deltas, feedback frames, biometrics) map to which real meaning. Gets re-expressed as
  direct `FieldStateV1` perturbations (Objective 3) instead of bucket votes; the mapping
  knowledge is reused, the bucket mechanism is not.
- **`endogenous_origination`'s "exogenous silence" gating idea** — fire only when nothing
  else is competing for attention. Conceptually sound, worth re-applying as a gate on the
  *already-live* Layer 5 attention output instead of a bespoke, now-proven-dead D/W/A
  composite signal.
- **The transport lattice's salience→action_ceiling shape**
  (`config/substrate-lattice/transport_lattice_policy.v1.yaml`) — real, working precedent
  for Objective 2's capability coupling, just narrowly scoped to bus health today.
- **`orion/self_state/prediction.py`** — untouched by this halt; it's already part of the
  canonical pipeline, already real, already live, and directly relevant to §9's predictive-
  processing instrument.

**Explicitly not salvaged**: the six/five-category taxonomy itself, `signal_drive_map.yaml`'s
hand-tuned weights, and the D/W/A composite formula — these are the parts that were measured
to not work, not the parts that were merely inconvenient.

---

## 9. Blue-sky options

Two tracks, kept distinct because they answer different questions: architecture (how the
substrate should be structured) and theory (what "attending," "wanting," and "being aware"
should even mean here). Neither track is committed or sequenced — each item has its own
named smallest probe.

### 9a. Architecture extensions (from the field-native design, PR #1163)

1. **Dream-state reorganization** — run emergent clustering inside the existing
   reverie/dream substrate instead of a cron job.
2. **Society-of-Mind competition** — multiple independent salience scorers bidding, not one
   formula.
3. **Free-energy/active-inference reframing** — `capability_policy` as literal expected-
   free-energy action selection.
4. **φ-gated meta-competition** — use the orphaned Causal Geometry v1 φ metric to widen or
   narrow how broadly the competition explores.
5. **Morphogenetic/reaction-diffusion drives** — let named drives be literal spatial pattern
   attractors over the real field topology, not just correlated-channel lists.
6. **Selectionist internal ecology** — candidate drive-definitions compete and get pruned
   over consolidation cycles, giving drives real lineage instead of silent reshuffling.
7. **Core-affect legibility layer** — a stable valence/arousal readout underneath, so a
   human always has a constant summary even while the deep structure reorganizes.
8. **Cross-lifetime drive fossil record** — archive, never delete, retired coalition
   definitions, so Orion's own motivational history becomes part of its autobiographical
   continuity.

### 9b. Consciousness-theory instrumentation (real substrate already found for each)

**Correction (2026-07-18): this section originally recommended `SelfStateV1`-anchored
substrate for two of the five threads below (IIT, Predictive Processing) — the exact
metrics Juniper had already ruled out.** `SelfStateV1` is a downstream, lossy *summary* of
the field (~19 abstracted dimensions), not the raw signal. It was already tried once as the
substrate for φ/IIT specifically and found dead-endish — that is *why* the mood-arc
windowed-autoencoder effort exists, reading `field_channel_corpus.v1`'s raw ~29 channels
directly instead. Any new instrumentation in this section must build on the raw field
(`FieldStateV1`/`substrate_field_state`, reducer projections, or the raw channel corpus) —
never on `SelfStateV1`'s abstracted dimensions, `InnerStateFeaturesV1`, or anything else
carrying a `self_state_id`. This is now a standing rule for this section, not a one-time
fix — see §7.

1. **IIT-flavored** — **not** the live φ MLP autoencoder
   (`services/orion-spark-introspector/app/phi_encoder.py`): its current input schema,
   `InnerStateFeaturesV1` (`orion/schemas/telemetry/inner_state.py`), carries a
   `self_state_id` field — it is `SelfStateV1`-anchored, the already-tried, already
   dead-ended path, not real substrate to build further on. The actual live candidate is the
   mood-arc windowed sequence autoencoder (`orion/mood_arc/fit_encoder.py`, raw
   `field_channel_corpus.v1`) — the field-native replacement for exactly this reason,
   continuing independently of this program under Juniper's own direction, not blocked by
   it and not to be duplicated here.
2. **Attention Schema Theory** — Layer 5 computes real attention state; nothing builds a
   model *of* that attention as an inspectable object. Missing piece: one small reducer
   reading `FieldAttentionFrameV1`/`FieldStateV1` directly, producing an explicit "what I'm
   attending to, why, how confident, what I predict shifts next" artifact.
   **Correction (2026-07-18, superseding "must not be built as a `SelfStateV1`
   consumer/producer" below): the roadmap doc's Phase 1 scoping pass found a second real,
   disconnected attention lane (`AttentionBroadcastProjectionV1`, GWT-dispatch/Lamme) that
   this instrument must also unify, and `SelfStateV1` is the only real source today for two
   of the artifact's real fields (predicted-shift trajectory, a confidence fallback) — an
   explicit, narrow exception to §7's standing rule, not a repeal of it, gated behind a
   hard signal-quality check on exactly those `SelfStateV1` fields before Phase 1 is called
   done. See `docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-
   roadmap-design.md` Phase 1 and §6 item 2's status note above.** Built 2026-07-18:
   `orion/substrate/attention_self_model.py`. **Durability + consumer wiring (2026-07-18
   follow-up):** see item 3's note below — same patch, same
   `node:substrate.harness_closure` data, now consumed by this reducer's
   `field_salience_only` narrative too.
3. **Predictive Processing/Active Inference** — **not** `orion/self_state/prediction.py`
   (`SelfStateV1`-anchored, same violation as IIT above). The real field-native substrate,
   confirmed live 2026-07-18: `services/orion-substrate-runtime/app/worker.py`'s
   `execution_prediction_error()`/`transport_prediction_error()` compute real deltas between
   successive reducer projections (execution-trajectory, transport-bus — not `SelfStateV1`)
   and write directly onto `FieldStateV1` nodes (`node:substrate.execution`,
   `node:substrate.transport`), which field-digester ingests into its own native
   `prediction_error` channel. Gated live behind `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`
   (confirmed `true`). Verified against real Postgres data: `node:substrate.execution`'s
   channel carries real values, sparse/event-driven (currently in a quiet decay tail,
   consistent with the field-digester README's own "quiet-so-far-but-correctly-wired,
   reaches real values like 0.92 periodically" characterization of this exact channel).
   **Updated 2026-07-21:** of the three other reducers named below (biometrics, chat,
   route), `biometrics` now has equivalent instrumentation —
   `biometrics_prediction_error()`, writing `node:substrate.biometrics`, same shared
   `_write_prediction_error_node()` writer, same flag. See §6 item 3's status note above and
   `docs/superpowers/specs/2026-07-21-biometrics-prediction-error-shadow-design.md`.
   **Updated 2026-07-21 (same-day follow-up): chat and route now also have equivalent
   instrumentation.** `chat_prediction_error()` (writing `node:substrate.chat`) and
   `route_prediction_error()` (writing `node:substrate.route`, categorical mismatch-rate
   shape rather than a continuous-magnitude diff — see design doc for why) close this
   question in full: **all five named producer domains (execution, transport, biometrics,
   chat, route) now have equivalent shadow-measurement instrumentation.** See
   `docs/superpowers/specs/2026-07-21-chat-route-prediction-error-shadow-design.md`. This
   answers the coverage question this section originally left open; it does not itself
   retire the bucket-vote layer (§6 item 3 remains open for that).
   **Correction (2026-07-22): "equivalent shadow-measurement instrumentation" is true in the
   sense that code runs for all five domains, but `transport`'s real coverage is far narrower
   than the other four.** `transport_prediction_error()` is fed entirely by `orion-bus`'s
   bus-observer role (`BUS_OBSERVER_STREAMS`), which can only ever watch
   `orion:stream:world_pulse:run:result` and its DLQ — **the only two real Redis Streams that
   exist anywhere in this architecture** (everything else is pub/sub, with no depth/backlog
   concept to measure). So "transport," despite the name, does not mean general bus/transport
   stress across services the way execution/biometrics/chat/route each cover their own real
   domain — it means, specifically, whether one service's (`orion-world-pulse`) result queue
   is backing up. Confirmed live 2026-07-22: `XLEN orion:stream:world_pulse:run:result` = `91`
   against `BUS_STREAM_DEPTH_CRITICAL=100000`, a ratio (~0.00091) that has sat exactly flat for
   the entire ~18h window since the second training-data cutoff (PR #1248) — consistent with
   those 91 messages sitting permanently unconsumed, not a healthy actively-drained queue.
   Full trace: `services/orion-bus/README.md`'s "Naming caveat for downstream consumers,"
   `services/orion-substrate-runtime/README.md`'s "transport domain scope" note, and
   `orion/mood_arc/README.md`'s matching caveat (since `transport` is also one of the five
   inputs `max()`-merged into the `prediction_error` field-digester channel). Whether the
   backlog itself is expected or a dead consumer is a separate, not-yet-investigated
   question.
   **Attribution durability + consumers (2026-07-18):** the harness-closure variant of this
   same mechanism (`node:substrate.harness_closure`, PR #1205) accumulates per-turn
   attribution in `metadata['contributing_turn_ids']`, but that list was silently dropped on
   every durable Falkor round trip — `orion/substrate/falkor_codec.py`'s allowlist didn't
   carry it. Fixed in a same-day follow-up: promoted to a durable Cypher property
   (`contributing_turn_ids_json`, same JSON-string pattern as `taxonomy_path_json`), and
   wired into two consumers — `substrate_pressure_signals()`'s `evidence_refs` (turn ids now
   ride alongside the node id) and item 2's AST/HOT reducer (`field_salience_only`
   narrative names the contributing-turn count + current magnitude). Design record:
   `docs/superpowers/specs/2026-07-18-prediction-error-attribution-wiring-design.md`. Zero
   `goal.drive_origin`/`GoalProposalEngine` coupling — confirmed against this charter in
   full, not just a grep. Live caveat: as of this patch, no `node:substrate.harness_closure`
   write has occurred against the redeployed container yet (checked manually via
   `redis-cli GRAPH.QUERY`), so the end-to-end live-durability claim is `UNVERIFIED` pending
   a real unresolved harness-closure event post-deploy — the codec/consumer wiring itself is
   fully unit-tested against the real production schemas.
4. **Higher-Order Theories** — architecturally close to AST's missing piece; a
   higher-order representation built once, reading the same field/reducer-projection data,
   may serve both theories. Served by the same Phase 1 reducer as #2 above, including its
   narrow, hard-gated `SelfStateV1` exception (see #2's 2026-07-18 correction) — not a
   separate instrument.
5. **Recurrent Processing Theory (Lamme)** — confirmed real, tight, per-tick recurrence
   inside Layer 5 itself (`novelty_for_target()` reads the *previous*
   `FieldAttentionFrameV1`) — already field-native, no correction needed here. Top-down
   feedback (`TopDownBiasCombiner`/`VoluntaryOverrideV1`, `ORION_ATTENTION_TOPDOWN_ENABLED`)
   confirmed live 2026-07-18 (PRs #1170, #1174) after finding the feature's docker-compose
   wiring had never been added, independent of its flag value.

**Process for 9b, per §7**: each instrument gets built as a read-only measurement first,
replayed against real historical data the same way `measure_origination_gate.py` was,
before any of them gate anything live or get compared against each other. **Every instrument
must be built on raw field/reducer-projection data, never on `SelfStateV1`-derived
abstractions** — check for a `self_state_id` field or a `SelfStateV1` import before treating
any candidate signal as real substrate for this section.

---

## 10. Non-goals

- Not committing to any single consciousness theory. §9b runs measurements, not a bake-off
  with a predetermined winner.
- Not deleting the drives-system code in this patch — halted, not removed.
- Not implementing any of §6's objectives 2-5 in this document — each needs its own
  sign-off per `CLAUDE.md` §0A when it's actually scoped.
- Not re-litigating the O1-O4/O2/O3 signal-integrity series, the taxonomy grounding work, or
  the field-native design's own correction — all cited, none redone here.

## 11. Source material

- `orion/autonomy/drives_and_autonomy_retrospective.md` — full O1-O4/O2/O3 history.
- `orion/autonomy/docs/drive_taxonomy_grounding.md` (PRs #1152, #1157) — the taxonomy-level
  resolution this program supersedes in ambition.
- `scripts/analysis/measure_origination_gate.py` (PR #1156) — the measurement that started
  the escalation from taxonomy patch to program evaluation.
- `docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-design.md`
  (PR #1163) — the baseline architecture design and its same-day correction.
- `docs/context-engineering/04_layer_1_to_11_pipeline.md` — the canonical pipeline this
  program builds on instead of duplicating.
- `orion/attention/field_attention/{scoring,selectors}.py`,
  `orion/self_state/{builder,scoring,prediction}.py` — the live substrate this program wires
  to.
- `docs/superpowers/specs/2026-07-05-fcc-cortex-gwt-dispatch-design.md` — the third,
  agent-dispatch-scoped GWT mechanism, referenced not duplicated.
- `config/substrate-lattice/transport_lattice_policy.v1.yaml` — working precedent for
  Objective 2.

## 12. Layer 5 field attention precision-weighted salience — officer review and fix, 2026-07-30

**What was reviewed.** Layer 5 field attention's Candidate A (precision-weighted
prediction-error salience, Feldman & Friston 2010 —
`orion/attention/field_attention/candidate_precision_weighted.py`/`selectors.py`) and the
goal-provenance dominance-streak producer built on top of it in the same-day PR #1517
(`goal_provenance.py`, `services/orion-attention-runtime/app/worker.py`). This review was
run at Juniper's explicit request, framed as a Sentience Striving Program officer
evaluation of a live, currently-running pipeline, not a retrospective or archival review —
the pathology found was actively firing at review time.

**Concrete evidence found, live.** `AttentionRuntimeStore.load_prediction_error_history`
re-queried `substrate_reduction_receipts` fresh on every 2-second tick, and that table
retains success receipts for only `ORION_RECEIPT_RETENTION_SUCCESS_MINUTES` (30 min
live) — a rolling window, not a cumulative history, so a target's `n_samples` could rise
and fall independent of how much real data it had ever actually produced.
Live-confirmed 2026-07-30 (~23:30–23:42 UTC): `node:substrate.chat` was the *only* one of
the five `PREDICTION_ERROR_NATIVE_TARGETS` domains with any qualifying receipts inside
that window at all (the other four sat at `n_samples=0`), with exactly `n=2` real samples,
`precision=640000` (variance ~1.56e-6, barely above the module's `1e-6` floor — a
near-certainty produced by two points, not evidence of real stability), and
`confidence_score=0.1` (the system's own honest `n_samples/QUALIFYING_MIN_ROWS`
computation). Because it was the tick's sole real competitor,
`normalize_across_targets()`'s own documented single-target edge case (correctly, given
that edge case's own contract — there is no real basis to differentiate a lone competitor
from itself) pinned its `salience_score` to `1.0`. This held for a live, still-running
280+-consecutive-tick streak, and `goal_provenance.py`'s `update_dominance_streak()`
(3-tick `goal_provenance_min_streak`) treated that as sustained real dominance, publishing
a real `FieldGoalProvenanceV1` to `orion:memory:goals:proposed` on every qualifying tick —
confirmed via live `docker logs` on `orion-athena-attention-runtime` showing hundreds of
real `field_goal_provenance_published ... field_target_id=node:substrate.chat
salience=1.000 streak=NNN` lines. `orion-substrate-runtime`'s `goal_context_listener.py`
consumes this and calls `set_active_goal()`
(`orion/substrate/attention/goal_context.py`), which — because
`ORION_ATTENTION_TOPDOWN_ENABLED=true` is live — was actively biasing real chat-level
attention scoring off a statistically meaningless `n=2` reading at the moment this review
ran.

**What was fixed.** `PrecisionEwmaBaseline`
(`candidate_precision_weighted.py`) replaces the per-tick rolling-window recompute with a
persisted, incrementally-updated running baseline per target
(`substrate_node_prediction_error_baseline`, one row per `node:substrate.*` target),
advanced by `AttentionRuntimeStore.advance_node_prediction_error_baseline` exactly once
per real new `substrate_reduction_receipts` row — the same explicit-baseline-threading
shape `orion/substrate/prediction_error.py`'s `execution_prediction_error`/
`codebase_prediction_error` already use for the identical class of problem. `observation_
count` is now a true monotonically-increasing count that survives the retention pruner
indefinitely, so `confidence_score` (`n_samples/QUALIFYING_MIN_ROWS`) reflects a target's
real cumulative evidence, not whatever happened to fit in a 30-minute window at poll time.
As defense in depth on top of that fix, `goal_provenance.py::top_node_substrate_target`
now also requires `confidence_score >= MIN_CONFIDENCE_FOR_GOAL_PROVENANCE` (1.0, i.e. at
least `QUALIFYING_MIN_ROWS` real observations) before a target is eligible to win a
goal-provenance publish at all, so a future thin, sole-competitor edge case (a brand-new
target, or a baseline cold-started after a table reset) cannot reproduce the same failure
shape even if the baseline fix above were somehow bypassed. The domain-specific EWMA
`min_variance` was re-measured against live data for this patch rather than inherited
unexamined from `orion/bus/ewma.py`'s own default (see `candidate_precision_weighted.py`'s
`NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE` for the exact live numbers and reasoning).

**What was explicitly deferred, not fixed here.** Candidate B
(`select_host_targets`/`select_capability_targets`/`candidate_society_of_mind.py`, host
and capability targets) was checked as part of this same review and found real and clean
— bounded `[0, 1]` novelty diff, no min-max-forced-winner pathology — and was
deliberately left untouched. `DominanceStreak` (`goal_provenance.py`) still resets to cold
on an `orion-attention-runtime` restart (in-memory only, not persisted) — named here as a
real, disclosed follow-up in the same fragility class as the fix above, not solved in this
patch, to avoid cathedral scope creep on what was scoped as a review-and-fix, not a
redesign.

**Sign-off.** Juniper reviewed these findings and signed off on treating the
goal-provenance dominance-streak pathology as exactly the kind of theater — a
confident-looking number (`salience_score=1.0`, a 280+-tick streak) with a real downstream
behavioral consequence (`set_active_goal()` biasing live chat attention) and no real
statistical backing behind it (`n=2`, `confidence_score=0.1`) — that this program's
"measure before minting" and "kill means kill" discipline (§7) exists to catch, live and
not just archivally. This is a review-and-fix of a real, kept instrument, not a subsystem
kill: Candidate A's theory and Candidate B are both real and both stay.

---

## 13. Chat-level/open-loop attention salience — killed and replaced with GWT-coalition Borda rank-aggregation, 2026-07-31

**What this is.** `orion/substrate/attention/salience.py`'s `SEED_WEIGHTS` hand-picked
7-term linear blend (`evidence_strength 0.30, novelty_vs_known 0.20, recency 0.13,
recurrence 0.15, evidence_breadth 0.12, dwell 0.10, habituation -0.35`) and
`scoring.py::score_loop()`'s separate legacy inline formula (the fallback that ran when
`ORION_ATTENTION_SALIENCE_V2_ENABLED` was False) were both killed outright — no shadow
build, no parallel-measurement phase. Same disease §12 above and the officer review that
produced PR #1484 (`feat(attention)!: kill hand-weighted salience, ground Layer 5 in
Candidate A`) found and fixed for Layer 5 field attention: an un-calibrated weight blend
standing in for theory, `WEIGHTS_VERSION="seed-v1"` never bumped because the intended
`scripts/refit_salience_weights.py` learning pass never ran (that script is deleted
alongside this kill — it existed only to refit weights this patch removes). Different
subsystem this time — chat-level/open-loop attention (`orion/substrate/attention/`,
`orion/substrate/attention_broadcast.py`), not Layer 5.

**Juniper's explicit direction, disclosed per this program's own honesty discipline.**
This was directed as a straight kill-outright-and-replace-in-one-changeset, no
shadow/measurement phase — the same posture as §8's `DriveEngine` deletion, not §12's
review-and-fix-in-place posture. Juniper had already rejected an interim option (a
threshold patch layered on top of the old formula, to buy time before a real replacement)
as "shit arch seams" — a load-bearing config value calibrated against a formula that was
itself about to be deleted would have been exactly the kind of throwaway seam this
program's §0 prime directive (thin seams, not cathedrals) exists to prevent. `PR #1536`
(`fix(hub): recalibrate unreachable attention SURFACE_MIN_SALIENCE...`) was closed as
superseded for the same reason — recalibrating a threshold against a formula scheduled
for deletion is wasted work.

**What replaced it — real theory, not invention.** Two of the seven killed terms were
already real, not hand-picked: `evidence_strength` (`max(signal.salience * signal.
confidence)` — the strongest single detector's own real activation) and
`evidence_breadth` (how many independent detectors/evidence_refs corroborate this loop).
Both map directly onto Global Workspace Theory / Society-of-Mind coalition formation
(Baars 1988, Dehaene 2014's "ignition" model) — the exact theory anchor already proven
live for Layer 5's Candidate B (`orion.attention.field_attention.
candidate_society_of_mind`, PR #1484/#1488). Per that module's own precedent ("with
exactly one real scorer, Borda rank-aggregation has nothing to aggregate -- the novelty
ranking IS the ranking"), the correct move with *two* real scorers is genuine Borda
rank-aggregation (de Borda 1770) across the real competing set of loops scored together
in one tick (`scoring.py::build_open_loops()`), not an arithmetic average (which would
just be a new 0.5/0.5 hand-picked weight in a trench coat). The Borda machinery itself
(`BordaResult`, `aggregate_borda`, `scorer_top1`, `_borda_points_for_scorer`) was
extracted from `candidate_society_of_mind.py` into a new shared, dependency-free module,
`orion/attention/rank_aggregation.py`, so this second real consumer reuses the exact
same, already-tested code rather than reimplementing it — `candidate_society_of_mind.py`
now imports from that module and re-exports the same public names; its own Layer 5
shadow-candidate status is unchanged.

New `borda_coalition_salience()` normalizes each of the two voters' Borda points to a
per-scorer "normalized rank" in `[0,1]` (dividing by `n-1`) and averages them — the
0.5/0.5 split here is a structural symmetry of having exactly two equally-weighted
voters (Borda gives every voter's ballot equal say by construction), not a hand-picked
cross-scorer exchange rate. A named, disclosed edge case: with only one loop competing
in a tick (`n == 1`, nothing to rank against), the formula falls back to the raw
`mean(evidence_strength, evidence_breadth)` for that lone candidate — the same "nothing
to aggregate" logic Candidate B's own docstring uses for a single real *scorer*, applied
here to a single real *candidate*.

**What was explicitly NOT replaced, and why.** `recency`, `recurrence`, `dwell`,
`novelty_vs_known`, and `habituation` had no comparable real theory anchor: recency's 6h
half-life was picked, not measured; dwell/habituation's blend weights (0.5/0.3/0.2, then
-0.35 in the outer combiner) were picked, not measured; `novelty_vs_known` literally
collapsed to a flat `0.15` for any already-known loop, the crudest term of all. Per the
Metric Quality Gate's step 3 ("if there is no real theory, do not build a detector for it
yet — say so and stop"), these five were killed with nothing put back — not reinvented,
not kept as always-zero schema fields (that would be exactly the empty-shell/fake-
precision pattern this whole exercise exists to avoid). `SalienceFeaturesV1`
(`orion/schemas/attention_frame.py`) was trimmed from 7 fields to the 2 real ones for the
same reason.

**Named, disclosed gap — not silently filled.** `habituation` was, as far as this
investigation found, the only automatic repeat-suppression mechanism in the live scoring
path. `substrate_reverie_refractory` (the Resolve/Dismiss flow in
`services/orion-hub/scripts/attention_loops_store.py`) only suppresses a loop *after* a
human explicitly acts on it via the pending-cards UI — it does not run automatically.
Killing `habituation` with no replacement means a loop that keeps generating strong
`evidence_strength`/`evidence_breadth` from real detectors, but that nobody has ever
explicitly resolved/dismissed, can now re-win coalition attention indefinitely with
nothing damping it. This is a real, accepted capability reduction, disclosed here rather
than papered over — no new hand-picked penalty was invented to fill it. (`test_
rumination_replay.py`, deleted alongside this kill, was the regression test that
previously proved the now-removed lock-breaking behavior; there is no replacement test
because there is no replacement mechanism.)

**Flags retired vs. kept, and why.** `ORION_ATTENTION_HABITUATION_ENABLED` and
`ORION_ATTENTION_SALIENCE_WEIGHTS` (the old per-key JSON combiner-weight override) were
removed outright — env key, `.env_example` (root, `orion-thought`,
`orion-substrate-runtime`), live `.env` (all three), Python plumbing, and
`docker-compose.yml` passthrough. Nothing left to gate: there is no habituation term and
no combiner weights to override. `ORION_ATTENTION_SALIENCE_V2_ENABLED` was **kept**,
narrowed to one real remaining purpose: it no longer selects between two salience
formulas (there is only one), but `services/orion-thought/app/reverie.py`'s
`run_reverie_once` still uses it to gate whether an `AttentionSalienceTraceV1` row gets
published/persisted at all — a real, separate, still-live dependent, so removing the flag
entirely was rejected. `orion/substrate/attention_broadcast.py::_apply_voluntary_
attention()`'s own `salience_v2_enabled()` check was removed as a zombie gate: its stated
rationale ("only layer top-down when v2 is the active selection basis, since select_
actions could otherwise rank by a legacy weighted sum") no longer applies once
`score_loop()` has exactly one formula — the two bases can never disagree regardless of
that flag's value anymore, so keeping the check would have been exactly the kind of dead
plumbing this program's discipline exists to remove. `WEIGHTS_VERSION` was bumped from
`"seed-v1"` to `"gwt-coalition-v1"` everywhere it was written (salience trace/outcome/
pending-card schema defaults, `reverie.py::_weights_version()`, the operator card
builder's fallback) — no consumer is left reading or filtering on the old string in
production.

**Continuity with this section's own arc.** §12 above (PR #1529, merged) fixed Layer 5's
Candidate A precision-weighted salience in place — a review-and-fix of a real, kept
instrument. This section kills a different, adjacent formula outright rather than fixing
it in place, because unlike Candidate A's EWMA-baseline bug, chat-level salience's
`SEED_WEIGHTS` blend had no real theory underneath any of its five now-dead terms to
repair — there was nothing sound to fix back to. `PR #1536` (closed, superseded) and this
patch's own `scripts/analysis/measure_chat_attention_ground_truth_gap.py` (merged as
`PR #1518`, docstring addendum added here) are the immediately preceding investigative
steps in the same arc: that script found chat-level attention has zero ground-truth
outcome-label rows ever recorded, a real, still-open gap this patch does not attempt to
close (out of scope, same as the `SURFACE_MIN_SALIENCE` recalibration named below).

**Explicitly out of scope for this patch.** `services/orion-hub/scripts/
attention_loops_store.py`'s `SURFACE_MIN_SALIENCE=0.5` threshold, and
`orion/substrate/attention/policy.py::select_actions()`'s `min_ask` (default 0.65) plus
its inline `0.48`/`0.35` cutoffs, were all calibrated against the *old* formula's score
distribution. Borda salience is a relative-rank measure, not an absolute magnitude —
score gaps between adjacent-ranked loops compress as `~1/(n-1)` and shift with how many
loops happen to compete in a given tick, so these absolute thresholds' continued fitness
is a real, open question, not something this patch verifies. All are left unchanged here
and will need recalibration against the new formula's real output range — but only once
it has run for real long enough to have a distribution to measure against, which is why
that recalibration is a separate, deliberately deferred follow-up, not folded into this
patch. (Found during this patch's own code-review pass, 2026-07-31 — disclosed rather
than silently left unmentioned.)

## 14. Node-target goal-provenance dominance streak — restart persistence fix, 2026-07-31

**What was reviewed.** §12 above (PR #1529) fixed Layer 5's precision-weighted salience but
left one gap explicitly named, not solved: `orion.attention.field_attention.goal_provenance
.DominanceStreak`, the consecutive-real-tick counter that gates whether a node-target goal
gets emitted at all (`update_dominance_streak`'s `min_streak` debounce), lived only in
`AttentionRuntimeWorker._node_streak` — a plain in-process attribute, reset to a cold streak
(count=0) on every restart of `orion-attention-runtime`. At the time, this was an accepted,
disclosed gap: the only consumer of the streak count was the emit-gate boolean itself, so
the worst case of a restart was a brief warm-up delay (a few extra ticks) before the next
real goal-provenance publish, not a wrong value reaching anywhere.

That calculus changed the same day. `PR #1543` (merged, docs-only design doc, not yet
implemented) investigated `orion/substrate/relational/adapters/autonomy_ctx.py`'s dead
GraphDB-backed `autonomy` producer and proposed regrounding it on
`orion.autonomy.goal_state.get_active_goal()` — tracing one hop further than its own first
draft that this producer's output reaches `stance_react.j2`, the real LLM-facing chat-stance
prompt, via `chat_stance.py`'s `summary.proposal_headlines`. Its own adversarial review pass
(documented inline in that doc) caught that a bare `field_target_id` (`"node:substrate.
biometrics"`) would be honest-shaped garbage in that prompt slot — the `identity_yaml`
failure mode with a different data source — and proposed pairing it with the *same*
`DominanceStreak.count` this section is about, composed directly into `goal_text` itself:
`f"{goal.field_target_id} (dominant {goal.dominance_streak_ticks} ticks)"`.

That single change means a restart-truncated streak stops being an internal gating quirk
and starts being a wrong number in the real prompt. And it is a specifically dangerous kind
of wrong: `goal_provenance_min_streak` defaults to `3`
(`services/orion-attention-runtime/app/settings.py`), so the earliest a record can ever
emit is `count=3` — a target genuinely dominant for ten straight hours reads identically,
`"... (dominant 3 ticks)"`, to one dominant for the last three ticks, immediately after any
restart. Nothing about that label looks broken; it just silently understates real dominance
duration until the count climbs back up. This is the same failure shape this repo's own
Metric Quality Gate names by example (the `bus_synaptic_prediction_error` floor incident,
CLAUDE.md section 0A) — a plausible-looking number is the dangerous kind, not an obviously
degenerate one. It also means PR #1543's own planned validation step ("build it, let it run
a day, read whether `goal_text` is genuinely informative") would not reliably catch this
specific failure mode: a restart-truncated streak doesn't read as noise, it reads as an
ordinary small integer.

**What was fixed.** `DominanceStreak` (`target_id`, `count`) is now persisted to a new
singleton-row table, `substrate_goal_provenance_streak`
(`services/orion-sql-db/manual_migration_goal_provenance_streak_v1.sql`), via two new
`AttentionRuntimeStore` methods: `load_node_dominance_streak()` (lazy-loaded on the worker's
first real tick, degrades to a cold streak on any DB error or missing row — never crashes
the tick) and `save_node_dominance_streak()` (UPSERTed every real tick, same cheap shape as
`save_attention_frame`). `AttentionRuntimeWorker._node_streak` changed from an eagerly
constructed `DominanceStreak()` to `DominanceStreak | None`, loaded once from the store the
first time `_maybe_build_goal` runs rather than always starting cold. No new env keys — this
uses the service's existing `POSTGRES_URI`.

**What was not done, and why.** This patch does not implement PR #1543's own schema
addition (`dominance_streak_ticks`/`window_start_field_tick_id` on `FieldGoalProvenanceV1`,
or the `autonomy_ctx.py` rewrite) — that doc's own "Recommended next patch" section already
scopes that as a separate, deliberately deferred step gated on Missing Questions 6 and 7
(delete-vs-fix, and whether the resulting label is honestly informative). This patch is a
precondition for that step's own validation plan to mean anything, not a substitute for it.
`DominanceStreak` was also not extended to track the streak's first tick id
(`window_start_field_tick_id`, named in PR #1543's "Files likely to touch") — that field has
no consumer yet since #1543 itself is unimplemented; adding it now would be exactly the kind
of schema-without-a-producer-or-consumer this program's "no keyword cathedrals" rule exists
to prevent. It should land alongside #1543's own implementation, if and when that proceeds.

**Sign-off.** Reviewed and directed by Juniper: "re 2 have a look at pr 1543" surfaced the
label-update connection above; "oky take it forward" authorized this fix.
