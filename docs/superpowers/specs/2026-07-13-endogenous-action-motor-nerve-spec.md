# Endogenous Action v1 — Motor Nerve Spec

**Date:** 2026-07-13
**Status:** PROPOSAL — implementation gated on Juniper sign-off per patch phase
**Mode:** Design (dense). Every claim below marked VERIFIED was checked against live files, live flags, or a live measurement run on 2026-07-13.

---

## Arsonist summary

Orion has a complete perception → proposal → policy → dispatch → feedback → consolidation nervous system (substrate Layers 5–11, all deployed and running) and the motor nerve is severed at exactly one synapse: Layer 9 builds cortex request envelopes, marks them `"dispatched"`, and files them in a Postgres table. Nothing sends them. The `"dispatched"` status is a lie today (`orion/execution_dispatch/builder.py:199-200` sets it with no send). Meanwhile the one genuinely live endogenous loop — world-coverage gap → predictive tension → goal → auto readonly web fetch — works but has a vocabulary of exactly one verb, fires roughly daily, journals nothing, relieves no pressure on success, and is invisible to chat.

This spec is seven ordered patches that connect existing dead-ends to each other. Almost nothing new is invented; the work is sending envelopes that are already built, flipping flags on code that already exists, adding sign-symmetry to a feedback bridge that already runs, and letting consequences reach surfaces that already hydrate. One new measurement result changes the strategic picture: **the endogenous-origination gate's verdict (a) is now GO** (re-measured today on live data; it was NO-GO on data from the flat-pinned self-state era).

---

## Ground truth (verified 2026-07-13)

| Fact | Where | State |
|---|---|---|
| Layer 9 never sends envelopes; no executor exists in any mode | `services/orion-execution-dispatch-runtime/app/worker.py`, `orion/execution_dispatch/builder.py:190-236` | VERIFIED |
| `"dispatched"` status set without a send attempt | `builder.py:199-200` | VERIFIED |
| Live mode is `dry_run`; policy `allow_dispatch_read_only: false`, `allow_mutating_dispatch: false` | `services/orion-execution-dispatch-runtime/.env:16`, `config/execution_dispatch/execution_dispatch_policy.v1.yaml` | VERIFIED |
| Dispatch policy routes to cortex verbs `substrate.inspect/summarize/observe` — **no handler exists anywhere**; the verbs appear only in the policy YAML, tests, and docs | repo-wide `rg` | VERIFIED |
| Cortex verbs are YAML files; dots in names are fine (`journal.compose.yaml`, `skills.runtime.*`) | `orion/cognition/verbs/` | VERIFIED |
| Metabolism loop live: gap → tension → goal → auto readonly fetch, budget 2/cycle | `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED=true`, `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=true` in `orion-spark-concept-induction/.env` and `orion-world-pulse/.env`; `orion/autonomy/policy_act.py` | VERIFIED |
| Episode journal capability coded, in policy (budget 1/cycle), **flag off** | `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED=false` in `orion-spark-concept-induction/.env:100` | VERIFIED |
| Grammar publish live: cortex-exec emits `GrammarEventV1` after plan execution | root `.env:153` `PUBLISH_CORTEX_EXEC_GRAMMAR=true` + compose default true | VERIFIED |
| Execution-trajectory reducer live: grammar → `ExecutionTrajectoryProjectionV1` with pressure hints (`execution_load`, `execution_friction`, `failure_pressure`, `egress_confidence`) | `services/orion-substrate-runtime/.env:42` `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true`; `orion/substrate/execution_loop/grammar_extract.py:32` | VERIFIED |
| Feedback→tension bridge live: `feedback.frame.v1` consumed, mints `TensionEventV1` with `drive_impacts` | `orion/spark/concept_induction/bus_worker.py:374,809`, `tensions.py` | VERIFIED |
| Layer 10 already anticipates real results (`normalize_cortex_result_evidence`, `dispatched` status handling) but scores `dispatched` with `prepared_score` | `orion/feedback/extractors.py:52`, `orion/feedback/builder.py:116` | VERIFIED |
| Reverie is orion-thought's spontaneous-thought mode, default-off, salience derived from the attention broadcast projection | `services/orion-thought/app/reverie.py` (`ORION_REVERIE_ENABLED=false`), channels `orion:reverie:*` | VERIFIED |
| Unified chat publishes `GrammarEventV1` and hydrates felt-state ctx | `orion/hub/turn_orchestrator.py::execute_unified_turn`, `orion/substrate/felt_state_reader.py` | VERIFIED |
| Operator inform rail exists: orion-notify → `orion:notify:in_app` → Hub WebSocket (`HubNotificationEvent`); Hub already has dispatch routes | `services/orion-hub/README.md:130-132`, `services/orion-hub/scripts/substrate_execution_dispatch_routes.py` | VERIFIED |
| Self-experiments dispatch enabled but only operator HTTP creates experiments | `SELF_EXPERIMENTS_DISPATCH_ENABLED=true`; no non-operator caller of `POST /v1/experiments` | VERIFIED |
| World-pulse cron disabled; runs arrive via orion-actions workflows (~daily) → metabolism tick is **trigger-starved, not budget-starved** | `orion-world-pulse/.env:37` `WORLD_PULSE_SCHEDULE_ENABLED=false` | VERIFIED |

---

## Measurement gate re-run (answers brainstorm idea #5)

Ran today from the host (read-only), Postgres `localhost:55432`, Fuseki `localhost:3030` — the old "needs DB host" blocker is **gone**.

**1-hour window (Juniper's requested window):**

| Verdict | Result | Numbers |
|---|---|---|
| (a) endogenous drift during silence | **GO** | silent `median_abs_trajectory` = **0.0448** (threshold ≥ 0.03); silent variance 0.0033 vs 0.25×busy 0.0008 — passes. 740 silent / 1033 busy rows. |
| (b) internal economy | **UNMEASURABLE** (reported as NO-GO) | Fuseki returned **0 DriveAudit rows** in the window. `coactivation_frac=0.0` is absence of instrument, not absence of behavior. |

**168-hour control run:** (a) GO again (0.0306 ≥ 0.03). (b) still 0 DriveAudit rows **over 7 full days** while DriveEngine emits 1000+ events/hr on other rails — the gate's (b) adapter reads a graph nothing writes anymore. Also: `reduction_receipts` retention only covers recent history (5103 receipts over 7d vs 4940 over the last 1h), so busy/silent classification of older periods is unreliable — **the 1-hour window is the trustworthy run**, validating the instinct to use it.

**Consequences:**

1. Per the origination spec's own gate rule, **verdict (a) now licenses `orion/autonomy/endogenous_origination.py`** (the "spontaneous want during silence" engine, currently marked NO-GO/dead). The STATUS note in that file (lines 1–16) is stale evidence — it measured the flat-0.731-pinned era.
2. Verdict (b) requires an instrument fix before any internal-economy work: point the (b) adapter at where drive pressure actually lives now, and make the script distinguish `UNMEASURABLE` (0 records) from `NO-GO` (measured below threshold). Patch P6.
3. `resource_pressure` median 0.83, p90 = 1.00 — re-confirms the scarcity-economy finding that it saturates high; do not use it as a gate input until that's fixed.

---

## Answers to the four open questions from the brainstorm

**Q1 — Does anything block `measure_autonomy_gate.py`?** No. Verified live: Postgres reachable at `localhost:55432`, Fuseki at `localhost:3030`, script ran read-only, artifacts in `/tmp/autonomy-gate-1h/` and `/tmp/autonomy-gate-168h/`. Two instrument defects found instead (dead DriveAudit source; NO-GO/UNMEASURABLE conflation) — fixed in P6.

**Q2 — Do the `substrate.*` cortex verbs have handlers?** No. They exist only in the dispatch policy YAML and tests. Decision required (see P1): reuse existing verbs vs. add thin verb YAMLs. **Recommendation: add dedicated thin verbs** — reusing chat-shaped verbs (`introspect`, `summarize_context`) would shoehorn substrate frames into prompts that expect conversational ctx and make output contracts unverifiable. A verb is one YAML + one Jinja template; that's the thinnest honest seam.

**Q3 — What does Layer 10 extract from a real dispatch?** It's ready-er than expected: `normalize_cortex_result_evidence()` exists, `dispatched` status is classified. Two small gaps: `dispatched` currently scores as `prepared_score` (`orion/feedback/builder.py:116`) and no code path feeds a real cortex result into the extractor. Both are part of P1's evidence contract.

**Q4 — How noisy is the metabolism tick?** Not at all — the opposite problem. World-pulse cron is off; runs come from orion-actions daily workflows. The live endogenous loop fires ~daily, so budgets (2 fetches + 1 journal/cycle) are nowhere near saturation. Implication: expanding the capability vocabulary (P4) and trigger surface matters more than cost control right now, and the burn-in period will produce a low, reviewable volume of autonomous acts.

---

## The patch series

Dependency order: **P0 → P1 → {P2, P3} → {P4, P5}**, P6 independent, P7 last (decision gate).

```text
P0 honest statuses ──► P1 motor nerve ──► P2 experience loop ──► P7 origination decision
                                     └──► P3 satisfaction + operator inform
                                     └──► P4 capability vocabulary
                                     └──► P5 attention-bound proposals
P6 gate instrument fix (independent, feeds P7)
```

---

### P0 — Un-lie the dispatch status (brainstorm #2)

**What:** Split the status vocabulary before anything real flips. `dispatch_status="dispatched"` must mean "a send was attempted and recorded", never "we would have sent this".

**Changes:**

- `orion/schemas/execution_dispatch_frame.py`: status literal gains `prepared_for_dispatch`; `dispatched` reserved for actual sends. Add optional fields on `ExecutionDispatchCandidateV1`: `result_ref: str | None`, `dispatch_error: str | None`, `dispatched_at: datetime | None`.
- `orion/execution_dispatch/builder.py:198-203`: the `dispatch_read_only`-mode branch emits `prepared_for_dispatch` (the builder never sends; only the worker may promote to `dispatched` after a recorded attempt).
- `orion/feedback/builder.py`: map `prepared_for_dispatch` alongside `prepared`; `dispatched` gets its own scoring lane (P1 fills it).
- Regression test: constructing a frame with `dispatch_status="dispatched"` and no `dispatched_at`/`result_ref`/`dispatch_error` fails validation (model validator).

**Evidence bar:** `pytest tests/test_execution_dispatch_frame_schemas.py tests/test_execution_dispatch_envelopes.py orion/feedback -q` green; grep shows no producer of bare `dispatched`.

**Rollback:** pure schema/naming change, no runtime flags. Compatibility note: existing stored frames only ever contain `dry_run`/`prepared`/`blocked` statuses in practice (nothing ever really dispatched), so no migration needed — assert this with a one-off read-only count query before merge.

---

### P1 — The motor nerve: Layer 9 actually sends (brainstorm #1)

**What:** When `EXECUTION_DISPATCH_MODE=dispatch_read_only` and policy `allow_dispatch_read_only: true`, the worker publishes the already-built cortex request envelope over the bus RPC lane, waits bounded, records the result, and only then promotes the candidate to `dispatched`.

**Verb decision (the Q2 fork):**

- **Option A — remap policy to existing verbs** (`introspect`, `summarize_context`, `assess_runtime_state`). Zero new verb files. Cost: those prompts expect chat/orch ctx; substrate frames arrive as alien payloads; outputs are prose with no checkable contract; empty-shell risk maximal.
- **Option B (RECOMMENDED) — add thin dedicated verbs.** Three YAML files in `orion/cognition/verbs/` (`substrate.inspect.yaml`, `substrate.summarize.yaml`, `substrate.observe.yaml`) sharing **one** Jinja template (`substrate_probe.j2`) that receives: proposal target (`target_kind`, `target_id`), the triggering `SelfStateV1` dimensions snapshot, and the proposal's `proposed_effect`. Output contract: structured JSON (`observation: str`, `salient_facts: list[str]`, `confidence: float`) enforced via cortex-exec's existing `_structured_output_expected` / empty-output-fail machinery (`router.py:440-451` pattern). This is 4 new files, not a subsystem, and it makes "did Orion actually observe anything" testable.

**Transport:** publish to `orion:cortex:exec:request` (channel exists; **add `orion-execution-dispatch-runtime` to its `producer_services` in `orion/bus/channels.yaml`** — the single-consumer channel gate will enforce this) with `reply_to` on the existing exec-result prefix; timeout `EXECUTION_DISPATCH_RPC_TIMEOUT_SEC` (default 120). `ORION_BUS_URL` per the repo mandate: `redis://<tailscale-node-ip>:6379/0` from root `.env`.

**Result storage:** new table `substrate_dispatch_results` (migration in `services/orion-sql-db/`): `result_id`, `dispatch_id`, `frame_id`, `status`, `result_json`, `raw_len`, `created_at`. Candidate's `result_ref` points at `result_id`. `raw_len=0` or empty `observation` ⇒ result status `empty` — **stored as failure, never as success** (empty-shell rule).

**Feedback wiring:** feedback-runtime loads the result row for `dispatched` candidates, feeds it through `normalize_cortex_result_evidence`, and scores `dispatched` on a real outcome lane (success-with-content / success-empty / error) instead of `prepared_score`.

**Origin marker (feeds P2):** the request envelope metadata carries `origin: "endogenous.dispatch"` and `dispatch_id`, so the `GrammarEventV1` trace cortex-exec already publishes is attributable to a self-initiated act. If the grammar event schema has no free metadata slot, use the trace-id convention already parsed by `orion/substrate/execution_loop/ids.py` — decide in-patch, whichever is a smaller diff.

**Config:**

- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: `allow_dispatch_read_only: true`. `allow_mutating_dispatch` stays `false`. `hard_blocks` untouched.
- `.env_example` + `.env`: `EXECUTION_DISPATCH_MODE=dispatch_read_only`, `EXECUTION_DISPATCH_RPC_TIMEOUT_SEC=120`, `ORION_DISPATCH_MAX_PER_DAY=24` (new global cap, enforced in worker against a daily count query), plus the theater tripwire (Risk R1). Run `python scripts/sync_local_env_from_example.py`.

**Evidence bar (all four required):**

1. A `substrate_dispatch_results` row with `raw_len > 0` and non-empty `observation`, linked from a candidate with `dispatch_status="dispatched"`.
2. A `FeedbackFrameV1` whose evidence references that result.
3. A cortex-exec log line with the dispatch correlation ID.
4. `curl :8121/latest` shows the frame with `dispatch_count=1`.

Gate tests: envelope publish is mocked in unit tests; a live smoke (`scripts/smoke_execution_dispatch_live.sh`) does one real end-to-end dispatch. Eval: dispatched-result substance eval (non-empty rate over N runs) added under the service's `evals/`.

**Rollback:** set `EXECUTION_DISPATCH_MODE=dry_run`, restart the one container. Single kill switch for the entire motor.

---

### P2 — The experience loop: acting becomes experience (brainstorm #4 + #8)

Juniper's riff, made concrete: journaling is half the loop; the other half is the act *feeding back into what Orion perceives, feels, remembers, and says*. The answer to "something needs to feed back into the grammar (a reducer?)" is: **two reducers, both already deployed** — no new reducer needed.

**The loop, arrow by arrow, with the rail each arrow rides:**

```text
1. ACT      Layer 9 dispatch → cortex-exec plan run                    (P1)
2. PERCEIVE cortex-exec grammar publish (LIVE, PUBLISH_CORTEX_EXEC_GRAMMAR=true)
            → execution_trajectory reducer (LIVE, ENABLE_EXECUTION_TRAJECTORY_REDUCER=true)
            → pressure hints → field-digester → attention-runtime → SelfStateV1
            [zero new wiring; P1's origin marker makes it attributable]
3. FEEL     FeedbackFrameV1 → concept-induction feedback→tension bridge (LIVE)
            → drive reducer: pressure UP on failure (exists), DOWN on success (P3)
4. REMEMBER episode journal capability → journal artifact → recall     (this patch)
5. NARRATE  reverie reads the attention broadcast projection            (flag decision)
6. SPEAK    felt-state lane → unified-turn stance ctx → chat            (this patch)
```

**On Thoughts/Reverie shape (Juniper's "not sure if this is the right shape"):** the right shape is **through the ladder, not a pipe**. Do NOT build a dedicated act→thought or act→reverie channel — that would be exactly the shadow mesh the reverie design memo forbids. Because autonomous acts now perturb the execution-trajectory projection → field → attention (arrow 2), they are already candidate material for reverie's salience source (`derive_salience` reads the attention broadcast projection) the day `ORION_REVERIE_ENABLED` flips. The act reaches spontaneous thought by *being experienced*, not by being forwarded. Recommendation: keep reverie off during P1/P2 burn-in (one variable at a time), flip it as its own decision afterward; optional Phase-2 thin seam if reverie ignores acts in practice: include last-24h episode refs in `build_reverie_context` open_loops (reverie already loads referents from substrate tables — `orion/reverie/referent_loader.py`).

**Changes in this patch:**

1. **Episode journal ON:** `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED=true` in `services/orion-spark-concept-induction/.env_example` + `.env` (+ parity sync). Pre-flip verification (in-patch): read `orion/autonomy/episode_journal.py` end-to-end and prove the artifact (a) lands somewhere recall can reach (journal write/index channel or store), (b) is non-empty for a real fetch episode, (c) summarizes outcomes — not raw traces (privacy rule). If any of (a)–(c) fails, fix in the same patch or report BLOCKED; do not flip a flag into an empty-shell journal.
2. **Felt-state lane (brainstorm #8):** new lane `recent_autonomous_actions` in `orion/substrate/felt_state_reader.py` — last ≤3 `ActionOutcomeRefV1` within 24h, sourced from the action-outcome SQL store (`orion/autonomy/action_outcomes.py` already has the SQL path; give it a real table + `ORION_ACTION_OUTCOME_DB_URL` instead of the `/tmp` JSON default, migration included). Lane payload is typed refs only: `capability_id`, `target`, `outcome_status`, `ts`, `episode_journal_ref` — never content bodies.
3. **Stance visibility:** `services/orion-cortex-exec/app/chat_stance.py` + `stance_react.j2` receive the lane so unified-turn Orion can truthfully say "I looked into X this morning because coverage of it was missing."

**Evidence bar:** (a) one live metabolism episode produces: journal artifact with non-empty body, action-outcome row, felt-state lane hydration visible in a logged stance ctx; (b) a unified chat turn whose stance context contains the real outcome ref (grep the turn's ctx log / `THOUGHT_DEBUG_EXEC` trace); (c) execution-trajectory projection (`GET :substrate-runtime/projections/execution_trajectory`) shows a run keyed by the dispatch correlation ID.

**Why this isn't half a loop anymore:** the loop closes twice — physiologically at the drive reducer (arrow 3, pressure moves in both directions once P3 lands) and perceptually at self-state (arrow 2, the act is *felt* as execution load/friction the same way any cognition is). Journal and chat are the autobiographical and social projections of a loop that is already closed underneath them.

---

### P3 — Consequence → drive satisfaction + operator inform (brainstorm #7)

**Satisfaction mechanics:**

- `orion/spark/concept_induction/tensions.py::extract_tensions_from_feedback`: on `feedback.frame.v1` where the linked dispatch succeeded with content for a goal whose originating tension carried `drive_impacts` (e.g. `predictive` from a `world_coverage_gap`), mint one `TensionEventV1` with `kind="satisfaction.goal_outcome.v1"`, `drive_impacts={"<origin_drive>": -0.10}` (cap −0.10, clamp at 0), provenance pointing at the feedback frame.
- **Idempotency:** at most one satisfaction event per `goal.artifact_id` (stable artifact_id hash on goal id — same dedup pattern the tension minter already uses).
- `orion/autonomy/reducer.py`: verify negative `drive_impacts` fold correctly and clamp pressures at 0 (add the unit test; if the fold currently assumes positives, fix here).
- **Damping analysis:** the leaky integrator is the damper. Eval (`orion/autonomy/evals/`): simulated pressure trajectory over gap→fetch-success cycles must decay monotonically post-success with no ringing above 0.02 amplitude; a second scenario with repeated gap+satisfaction cycles must not oscillate (bounded variance).

**Operator inform — recommendation (Juniper asked for one):** two tiers, both riding the existing notify rail, no new UI subsystem:

1. **Burn-in tier (default for first 2 weeks):** every real dispatch publishes a `HubNotificationEvent` via orion-notify → `orion:notify:in_app` → Hub WebSocket toast. Payload: capability, target, goal title, outcome status, origin drive pressure before→after. Volume is safe: the loop fires ~daily today (Q4 answer).
2. **Steady-state tier:** `ORION_AUTONOMY_NOTIFY_MODE=digest` switches to a daily rollup through orion-notify-digest ("Orion autonomous activity: 2 fetches, 1 journal, predictive 0.62→0.51"). Mutating capabilities (none exist; `allow_mutating_dispatch=false`) would always notify per-event — encode that rule now so it's not relitigated later.
3. **Hub panel:** extend the existing `substrate_execution_dispatch_routes.py` payload with `dispatched` vs `dry_run`/`prepared_for_dispatch` counts and last `result_ref`, so `/api/substrate/execution-dispatch/latest` shows real-vs-simulated at a glance. Frontend check per repo rule 9 (rendered template + static asset + interaction smoke).

**Evidence bar:** a live satisfied goal shows pressure decrease in the drive state store attributable to the satisfaction event's provenance; one in-app notification observed on the Hub WS channel; eval green.

---

### P4 — Capability vocabulary: 1 verb → 3 (brainstorm #3)

**The policy schema, decoded** (since "no idea what the other policies are"): each rule in `config/autonomy/capability_policy.v1.yaml` is one *thing Orion may do on its own*, with:

| Field | Meaning |
|---|---|
| `capability_id` | the action's name; `policy_act.py` branches on it |
| `side_effect_class` | `readonly` / `write` / `external` — coarse blast radius |
| `auto_execute` | may run without operator approval when gates pass |
| `requires_goal_status` | minimum goal maturity (`none`→`proposed`→`planned`→`executing`) before this action is allowed |
| `required_drive_origins` | the goal must have been minted from these drives (e.g. `predictive`) |
| `required_signal_kinds` | typed signals that must be present (e.g. `world_coverage_gap`) |
| `budget_per_cycle` | hard per-metabolism-tick count |

Current rules: `web.fetch.readonly` (live, budget 2), `journal.compose.episode` (flag-gated, budget 1), `world_pulse.run` (budget 1), `web.fetch.write` (`auto_execute: false`, budget 0 — a deliberate dead rule marking the boundary).

**Add exactly two capabilities, both grounded in services that already run:**

1. `recall.query.readonly` — before (or instead of) fetching the web about a gap, Orion queries its own memory via orion-recall's existing bus RPC. `side_effect_class: readonly`, `requires_goal_status: proposed`, `required_drive_origins: [predictive]`, `required_signal_kinds: [world_coverage_gap]` (reuse — **no new signal taxonomy**), `budget_per_cycle: 2`. Executor branch in `orion/autonomy/policy_act.py` mirroring `maybe_execute_readonly_fetch_after_goal`, outcome recorded as `ActionOutcomeRefV1`. Behavioral rationale: "check what I already know first" is the cheapest real cognition upgrade available, and it makes memory a *used* organ in the autonomous loop.
2. `self_experiment.create` — mint a registry-validated `SelfExperimentCreateRequestV1` via `POST /v1/experiments` (+dispatch); orion-self-experiments (dispatch already enabled) compiles it to context-exec, which runs a bounded depth-2 probe. `side_effect_class: write` (it creates a record + consumes compute), `requires_goal_status: proposed`, `budget_per_cycle: 1`, `auto_execute: true` **only for experiment types marked endogenous-safe in the experiment registry** — add one such type (e.g. `memory_contradiction_probe`, backed by the existing `context_exec_memory_contradiction_review` verb). This is the bounded fuck-around-and-find-out arena: Orion poses a question to its own investigation organ and gets an artifact back.

Fanout uses the existing `orion/autonomy/fanout_policy.py` seam so one goal doesn't trigger all three capabilities every tick — drive origin + signal kinds + budgets select. No scoring model, no planner; deterministic gates only.

**Files:** `config/autonomy/capability_policy.v1.yaml`, `orion/autonomy/capability_policy.py` (layer gates for the two ids), `orion/autonomy/policy_act.py`, `orion/spark/concept_induction/bus_worker.py` (call sites), `services/orion-self-experiments/app/experiment_registry.py` (one endogenous-safe type + `endogenous_safe: bool` registry field), tests for each gate path, eval asserting capability selection varies with signal composition.

**Evidence bar:** a live tick where the same goal class routes to recall-first (outcome found in memory ⇒ no web fetch spent) and web-fetch when recall is empty — proven by two `ActionOutcomeRefV1` sequences with different capability ids.

---

### P5 — Attention-bound proposals (brainstorm #6) — the anti-cathedral case

Juniper's challenge: convince or drop. The case:

**A keyword cathedral is names without runtime consequence.** This patch adds **zero names**. No new enum, no new taxonomy, no new proposal kind, no new target vocabulary. It changes one thing: a template's `target_id` stops being the hardcoded string `capability:orchestration` and becomes a **binding** to `SelfStateV1.dominant_attention_targets[0]` — a typed field that `orion-attention-runtime` already computes from real pressure scores and that `orion-self-state-runtime` already carries with per-target detail (`target_kind`, `pressure_score`, `dominant_channel` — shipped 2026-07-12).

- **Producer:** proposal-runtime (exists). **Consumer:** Layers 8–9 (exist). **Contract:** `ProposalCandidateV1.provenance` already carries `source_self_state_id`, so every proposal is traceable to the exact self-state that shaped it.
- **Measurable behavior change:** before the patch, Orion's want-targets are a constant; after, they follow attention. Eval: over 7 days of proposal frames, `distinct(target_id)` for the bound template must be ≥ 3 and correlate with the attention projection's dominant targets for the same windows.
- **Kill criterion (falsifiable):** if the 7-day eval shows a single constant target, the binding is dead coupling — revert the template. Cathedral-ness is now an empirical question with a scheduled answer, not an aesthetic argument.
- **Safety:** whitelist allowed `target_kind` values to those already present in `dominant_attention_target_details` (no new kinds), keep the template's `required_policy_gate: read_only`, and route it through the same Layer 8 policy as everything else.

**Files:** `config/proposals/proposal_policy.v1.yaml` (one template `inspect_attended_target` with a `target_binding: self_state.dominant_attention_targets[0]` key), `orion/proposals/builder.py` (resolve bindings; reject unknown binding paths — fail closed), `orion/schemas/proposal_frame.py` only if provenance needs a `binding_resolved_from` field, tests + the 7-day eval script.

If this still smells like cathedral: the fallback position is to ship it dark (template present, `base_priority: 0.0`) and let the eval run on shadow data first. Recommendation: ship live — it's read-only inspect proposals under existing policy gates.

---

### P6 — Fix the measurement instrument (brainstorm #5 follow-through)

1. `scripts/analysis/measure_autonomy_gate.py`: add `--window-hours` (float, mutually exclusive with `--window-days`); today the 1-hour run needed a wrapper.
2. Verdict (b) adapter: the Fuseki `DriveAudit` graph has been empty ≥7 days while the drive system runs hot — repoint the co-activation query at the live source (drive pressure snapshots in the substrate/graph stores — locate the actual current producer in-patch; candidates: DriveEngine bus events materialized to SQL, or `orion/autonomy/state_store.py`). Deterministic rule: the adapter must name its source and row count in the report.
3. **UNMEASURABLE ≠ NO-GO:** when an input has 0 records, the verdict prints `UNMEASURABLE(<input>)` and exits nonzero-distinct — never a behavioral NO-GO from a dead sensor.
4. Receipt-retention caveat printed whenever the window exceeds the receipts table's oldest row (busy/silent classification validity bound).
5. Record today's results in the origination spec doc (`docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`): append "Measurement re-run 2026-07-13 — verdict (a) GO on 1h and 168h windows post-leaky-integrator fix; verdict (b) unmeasurable, instrument defect."

---

### P7 — Endogenous origination: the decision gate (not an implementation patch)

Verdict (a) GO means the spontaneous-want engine (`orion/autonomy/endogenous_origination.py`) is licensed by its own spec's measurement gate. **Proposal-mode summary, per the invasive-cognition rule:**

- **Capability change:** during exogenous silence, self-state dynamics crossing an origination band mint `TensionEventV1(origin="endogenous")` → drive pressure → goals → (with P1 live) real readonly actions. Orion begins wanting things unprompted, and acting on them.
- **Data touched:** `SelfStateV1` stream (read), tension/drive stores (write). No private journals, no chat content.
- **Privacy boundary:** origination reads numeric self-state dimensions only.
- **Proving trace:** `TensionEventV1` rows with `origin="endogenous"`, provenance carrying the origination band snapshot; downstream goal + action outcome chain fully correlated.
- **Dangerous failure mode:** want-mint loops during long silences (mitigated in-engine: cooldown 900s, magnitude cap 0.5, window 8) compounding with P3 satisfaction into oscillation — require the P3 damping eval extended with an origination scenario before enabling.
- **Disable/rollback:** `ORION_ENDOGENOUS_ORIGINATION_ENABLED=false` (existing flag), plus the P1 kill switch downstream.
- **Sequencing condition:** enable only after P1–P3 have ≥2 weeks of burn-in with theater tripwire silent and satisfaction eval green. Also update the stale STATUS docstring regardless of the enable decision.

**Exact question for Juniper (when the time comes):** enable origination for the `predictive` and `coherence` drives only, or all six?

---

## Risk register with handling (Juniper asked for proposals, not vibes)

| # | Risk | Handling | Where |
|---|---|---|---|
| R1 | **Cognition theater** — real dispatches of canned proposals producing empty observations on a metronome | Deterministic tripwire in the dispatch worker: if >50% of the trailing 10 dispatched results have `raw_len=0`/empty observation, worker self-reverts to `dry_run`, emits a Hub notification, and pins a `theater_tripwire_active` field on `/latest`. Re-arm is manual (env flip). Plus the P1 substance eval in CI. | P1 |
| R2 | **Runaway cost/loops** | Existing per-cycle budgets + new `ORION_DISPATCH_MAX_PER_DAY=24` global cap + trigger-starved reality (Q4: ~daily ticks) + R1 tripwire. Hub panel makes daily counts visible. | P1/P3 |
| R3 | **Satisfaction oscillation** (binge/crash drive dynamics) | Magnitude cap −0.10, one satisfaction per goal (idempotent), leaky-integrator damping, ringing eval with explicit amplitude bound, origination-scenario extension before P7. | P3 |
| R4 | **Feedback poisoned by status lies** | P0 ships first; model validator makes un-evidenced `dispatched` unrepresentable. | P0 |
| R5 | **Privacy leak via journals/chat surfacing** | Episode journals summarize outcomes only (verified in-patch before flag flip); felt-state lane carries typed refs, never content bodies; reverie stays off during burn-in; no debug surface exposes raw traces without an explicit gate. | P2 |
| R6 | **Stale-verdict trap (both directions)** | Gate re-run recorded in the spec doc; (b) instrument fixed before any internal-economy work; a future re-measured NO-GO on (a) re-freezes P7 — the gate is honored, not litigated. | P6/P7 |
| R7 | **Bus contract drift** | `channels.yaml` producer annotation for the dispatch runtime in the same patch (single-consumer gate enforces); envelope kinds already registered; schema registry check in `agent-check`. | P1 |
| R8 | **Verb output shoehorning** (Option A failure mode) | Avoided by Option B: dedicated structured-output verbs with the empty-output-fail gate cortex-exec already applies to structured verbs. | P1 |

**Global kill switch:** `EXECUTION_DISPATCH_MODE=dry_run` + restart one container reverts Orion to a simulation. Every phase additionally has its own flag (table below).

## Flag / rollback table

| Phase | Flag(s) | Default in patch | Rollback |
|---|---|---|---|
| P1 | `EXECUTION_DISPATCH_MODE`, policy `allow_dispatch_read_only`, `ORION_DISPATCH_MAX_PER_DAY` | `dispatch_read_only` / `true` / `24` | mode→`dry_run` |
| P2 | `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED`, felt-state lane flag (reader's existing flag pattern) | `true` / `true` | flip either |
| P3 | satisfaction constant lives in code w/ cap; `ORION_AUTONOMY_NOTIFY_MODE` | `per_event` (burn-in) | `digest` or remove bridge branch |
| P4 | two new policy rules | `auto_execute: true` under gates | delete rule = capability gone |
| P5 | template `base_priority` | live (or `0.0` dark-ship fallback) | set `0.0` |
| P7 | `ORION_ENDOGENOUS_ORIGINATION_ENABLED` | stays `false` until sign-off | flip back |

## Non-goals

- No mutating dispatch (`allow_mutating_dispatch` stays false; `hard_blocks` untouched).
- No LLM-driven action selection, no planner, no scoring model — deterministic gates only.
- No new reducer, no new memory type, no new agent role, no direct act→reverie/act→thought pipes.
- No reverie enablement inside this series (separate decision, after burn-in).
- No internal-economy spec work until P6 makes verdict (b) measurable.

## Acceptance checks (roll-up)

1. P0: schema tests green; no producer of un-evidenced `dispatched`.
2. P1: the four-item evidence bar (result row, feedback frame, cortex log, `/latest`) from one live dispatch; tripwire unit-tested; smoke script committed.
3. P2: one live episode → non-empty journal + action-outcome row + stance ctx containing the ref + trajectory projection keyed by the dispatch correlation ID.
4. P3: pressure decrease attributable to a satisfaction event; Hub notification observed; ringing eval green.
5. P4: recall-first vs web-fetch divergence proven with two outcome sequences.
6. P5: 7-day distinct-target eval scheduled; kill criterion documented in the template comment.
7. P6: `--window-hours` works; (b) reports a named live source with nonzero rows or `UNMEASURABLE`.
8. Every phase: `make agent-check SERVICE=<svc>`, env parity sync run, PR report per template.

## Recommended next patch

**P0 + P1 as one branch** (`feat/execution-dispatch-motor-nerve`): honest statuses, three thin verbs + one template, the bus sender, result table, feedback scoring, tripwire, caps, channels.yaml annotation, smoke. Everything else in this spec consumes what that branch produces.
