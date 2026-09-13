# Visual reverie as a recurring baseline process

**Status:** revised proposal; implementation and production verification pending.  
**Date:** 2026-09-13  
**Source inspected:** `4ce78f63377d7998cb552fa37da449105be64844`, worktree `docs/visual-reverie-homeostasis`.

## Arsonist summary

Visual reverie should recur like a heartbeat. Attention/autonomy can influence its content and request additional runs; an individual image does not need to demonstrate information gain to justify the baseline continuing. Heat, unavailable capacity, and explicit policy can defer execution without extinguishing that need.

Carry one history-derived baseline eligibility record through the existing proposal → policy → dispatch path. Change both the proposal builder's tick-wide action-warrant veto and the allocator's unmeasurable/information-floor veto for this narrowly validated case. Keep resource accounting and the existing visual execution thermal gate. Preserve the actual attention-to-content path and save the selected source identity beside the image.

This replaces the previous draft's accumulating pressure/debt, speculative drive taxonomy, and direct thought-worker watchdog. Neither raising priority nor removing `expected_signal` alone fixes admission.

## Current architecture

### Source-confirmed behavior

Paths and symbols below refer to the inspected revision; they establish code behavior, not a claim that the production path moved.

| Seam | Confirmed behavior |
|---|---|
| `services/orion-attention-runtime/app/worker.py:185`, `AttentionRuntimeWorker._maybe_build_goal()` | Selects a qualified real node target using dominance streak and optional competition read, then creates `FieldGoalProvenanceV1` with field tick and attention-frame provenance. |
| `orion/substrate/attention/top_down.py`, `relevance()` / `TopDownBiasCombiner.apply()` | Goal relevance is binary membership of `goal.target_id` in an open loop's actual `source_refs`. Bias uses existing priority, relevance, effort budget, and agency readiness. An override is recorded only when the winner changes. A fabricated visual goal target would not create relevance. |
| `services/orion-thought/app/reverie.py:531`, `run_reverie_once()` | Reads the attention broadcast, builds its coalition snapshot, narrates, validates substance/grounding, and persists/publishes `SpontaneousThoughtV1`. No coalition or hollow narration does not produce a valid thought. |
| `services/orion-thought/app/store.py:585`, `load_latest_reverie_interpretation()` | Selects a fresh thought already listed in a persisted text chain's `thought_ids`; checks stored hollow flag and current `is_hollow()`; word-truncates the interpretation. Returning only text discards thought identity, correlation, and coalition. |
| `services/orion-thought/app/visual_chain.py:874`, `_run_visual_chain_body()` / `select_context_slot()` | Loads text reverie, self-study, memory, and prior visual continuity. Round-robin selects one available context slot. Optional interpretation makes a visual metaphor; prompt building incorporates it and effective prior continuity. This rotation is deterministic source selection, not an autonomous attention decision. |
| `orion/proposals/builder.py:190`, `build_proposal_frame()` | Merges external candidates before `action_warrant` evaluation. An insufficient or unavailable warrant suppresses **all** built candidates; ordinary priority thresholds and candidate limit also apply. An externally injected heartbeat currently cannot escape. |
| `orion/autonomy/allocator.py:275`, `allocate()` / `candidate_from_dispatch()` | Missing signal yields unmeasurable variance and refusal. Measured candidates below information floor are refused; cold-start exemption is temporary. Harm, cost estimate, and allowance are separate checks. Removing `expected_signal` makes the first refusal worse. |
| `services/orion-cortex-exec/app/verb_adapters.py:1618`, `RenderSceneVerb.execute()` | Posts `{}` to thought's run-once endpoint. Its result allowlist forwards `ran`, refusal, chain ID, terminal reason, held time, reason, and detail. It does not select content. |
| `services/orion-thought/app/main.py:150`, `visual_chain_run_once()` | Accepts no content-selection input. `ran=True` means a chain result was returned, including thermal refusal and generation failure; it is not image-production evidence. |
| `visual_chain.py`, `run_visual_chain_once()` / `evaluate_thermal_gate()`; `orion/autonomy/thermal_gate.py:70`, `thermal_state()` | Checks busy state and then thermal state before prompt work. Hot refuses with hysteresis. Missing/stale readings allow GPU work with a degraded verdict; unknown readings do not replace the last real hysteresis state. Thermal refusal persists a chain row. |
| `store.py:461`, `visual_chain_age_minutes()` | Ages the newest chain row, including thermal refusals and failures. It cannot measure successful visual activity. |

The content path to preserve is:

```text
Field attention goal selection
  → substrate voluntary attention (actual source_refs matching)
  → attention broadcast
  → run_reverie_once()
  → non-hollow thought in a settled text-reverie chain
  → visual context loader
  → select_context_slot() → prompt → image
```

The execution path to extend is:

```text
proposal-runtime worker → build_proposal_frame()
  → policy evaluator/builder → execution-dispatch builder
  → dispatch-runtime motor allocator / send gates
  → RenderSceneVerb → POST /visual-chain/run-once
  → busy / thermal gate → visual body → persisted chain + artifact
```

Existing registered contracts include `proposal.frame.v1`, `policy.decision.frame.v1`, `execution.dispatch.frame.v1`, `reverie.thought.v1`, and `reverie.visual.chain.v1`. Keep their existing transport routes; this proposal adds no independent execution rail. Services retain their existing settings, `.env_example`, compose files and tests. Relevant tests are listed below; runtime implementation must inspect each affected service's config and eval surfaces before editing.

### Prior production report: UNVERIFIED in this revision

The earlier draft reported disabled visual cron, proposals without recent express successes, information-floor/unmeasurable refusals, and `resource_pressure` as the wrong learned outcome. It included a September 8 last-success date and posterior estimates without attached query output or correlation IDs. Treat those as investigation leads, not newly verified evidence. No production query or visual execution was performed for this revision.

## Missing questions

- Cadence remains an operator choice. Proposed initial interval is 90 minutes, with a shared 10-minute minimum gap between attempts; these are scheduling parameters, not measured physiological constants. Validate service cost/allowance against these values before enabling.
- Can production motor allowance accommodate one visual run at the desired cadence? If not, report resource deferral; no design can guarantee a heartbeat while denying it the resources to execute.
- Does production retain enough attention broadcast/goal evidence to join a selected thought back to a voluntary override? `CoalitionSnapshotV1` has node/loop IDs and time, not a complete goal/broadcast identity. Do not claim an override from a coalition alone.
- Live artifact integrity, service replica count, and dispatch retry behavior require verification before baseline activation. Source inspection confirms artifact inserts deduplicate on `sha256`; production receipt behavior needs the explicit fix below. A process-local lock is not proof of exclusion across replicas or timed-out thread work.

## Proposed schema / API changes

All connections in this section are **proposed**, not existing production behavior.

### 1. Preserve selected content and its provenance

Change `load_latest_reverie_interpretation()` to return a typed `ReverieVisualContextV1 | None` instead of `str | None`. Define it in `orion/schemas/reverie_visual.py`. Fields: selected `text`, `thought_id`, `thought_correlation_id`, `thought_created_at`, enclosing `text_chain_id`, `coalition: CoalitionSnapshotV1`, and bounded `evidence_refs`. Resolve the enclosing chain from the same settled-membership query, not merely an unchecked thought `chain_id`. Revalidate the original full thought before truncating text; preserve the existing candidate bound, freshness, stored-hollow, current-hollow, and settled-chain membership requirements.

Adapt `_run_visual_chain_body()` to give only `.text` to the existing slot selector/interpreter. Add an optional typed `context_selection` field to `ReverieVisualChainV1`, serialized explicitly into `chain_json.context_selection` by the store writer and restored by readers. The current writer persists only `chain.chain_json`, so merely adding a top-level model field would lose it. It records:

- `selection_method="round_robin"` for the existing automatic selector; `source_kind` is `reverie`, `self_study`, `memory`, `prior_visual`, or `default_seed` according to what actually seeded this run.
- The selected reverie context record only when the reverie slot won. A loaded-but-unused thought is not the source of the generated image.
- For other sources, their real row identity and selected text. Extend `load_latest_self_study_reflection()`, `load_latest_memory_crystallization()`, and `load_latest_visual_chain_continuity_state()` to preserve the selected journal entry, crystallization, and prior visual chain identities from their existing queries. Keep their privacy/freshness filters; use actual primary keys from inspected table definitions during implementation.
- Prior visual continuity as a separate contributing reference when combined with a selected slot; distinguish continuity-only and fixed-seed fallback. Preserve reset behavior and `context_slot_used`, rotation, interpreted text, prompt, and `description` for existing readers.

Persist the selection on success and generation failure. On thermal/busy deferral it is absent with `source_selection_not_reached`; thermal happens before source selection. Do not move expensive content work ahead of that gate merely to fill a trace.

A selected reverie with validated coalition is **attention-grounded**. It is **voluntary-attention-influenced** only if a retained broadcast/goal trace proves that stronger claim. If existing retained traces cannot support the join, add optional attention provenance to `SpontaneousThoughtV1` at `run_reverie_once()` from the actual broadcast: broadcast identity, selected loop, and existing goal/override identity when available. Carry it through the context record; absent historical provenance stays absent. Do not add another attention score, synthetic goal, or drive classification.

**Proposal-selected content is not required for this patch.** An ordinary voluntary proposal requests *when* to render; it does not choose the slot. The HTTP addition below carries execution identity only. If a later patch requires a proposal's particular thought, it must add typed `source_thought_id` selection through proposal → dispatch request args → `RenderSceneVerb` body → endpoint → a by-ID settled/fresh/non-hollow loader, with explicit rejection on invalid/stale selection rather than silent round-robin substitution. Arbitrary caller-supplied prompt text is outside this design.

### 2. Define visual activity from real production

Add a thought-owned read API `GET /visual-chain/activity`, backed by a typed `VisualActivityV1` in `reverie_visual.py`, and a targeted store query. It returns `observed_at`, `history_status` (`ok` or `unavailable`), latest qualifying artifact/chain identity and production time, last attempt time/outcome, and active attempt identity when known. A successful empty query means no previous production; a database error means unavailable, not an overdue heartbeat.

For historical rows, qualifying production requires a join between `reverie_visual_artifact` and its chain: matching `chain_id`, artifact `sha256` matching `chain_json.artifact_sha256`, positive `bytes`, nonempty `path`, and chain `terminal_reason="max_steps"`. Use artifact `created_at` for these historical rows, not newest chain time. Count distinct completed chains, not captions or attempts. Verify stored bytes/path/hash against the actual storage rail in smoke checks; SQL alone is not disk integrity proof.

Caption/upload failure does not negate an image durably produced: `description` may be absent. A carried-forward `prior_description`, HTTP `ran=True`, a chain row without its artifact row, thermal refusal, generation/storage failure, or a deadline-only row does not satisfy the heartbeat. Where a deadline records `abandoned_chain_id`, reconcile any actually committed matching artifact once; an abandoned attempt is not automatically proof that no image exists. `persist_reverie_visual_artifact()` uses `ON CONFLICT (sha256) DO NOTHING` and returns true even when no new row was inserted. Therefore a chain-ID join alone would miss subsequent production of identical bytes; image novelty must not become an accidental heartbeat requirement.

Make persistence acknowledgement explicit: `_run_visual_chain_body()` currently returns a success-shaped chain even if chain persistence fails and does not propagate artifact persistence acknowledgement. The proposed result reports `artifact_persisted`, `artifact_sha256`, and production time only after both records are confirmed. Add a per-run `chain_json.production_receipt` with chain/attempt ID, sha, positive byte count, path and `produced_at`. A store transaction inserts or verifies the content-addressed artifact row and writes this receipt to the current chain after successful byte storage; a hash conflict verifies the existing object instead of pretending a new artifact row was inserted. For new rows, activity joins this acknowledged receipt to the artifact by sha and uses receipt `produced_at`; the artifact may belong to an earlier chain. A unique chain/attempt receipt counts once, even when bytes repeat. Retain the historical predicate only for pre-receipt rows, never as a fallback for a new failed acknowledgement. This is necessary before activity can safely drive scheduling.

### 3. One baseline need, carried through proposal and policy

Add `VisualBaselineEligibilityV1` to `reverie_visual.py`: `need_id`, `observed_at`, `due_at`, `last_success_at`, `last_success_chain_id`, `last_success_sha256`, and scheduling `policy_id`. No pressure/debt scalar and no image-quality reward. Its producer is proposal-runtime, reading the thought-owned activity API; its consumers are proposal builder, policy, dispatch, and the thought endpoint.

Persist one small singleton scheduling checkpoint in proposal-runtime's store: policy start anchor, current need identity, last attempt/dispatch identity, and `next_attempt_not_before`. No independent scheduler service. Derive due time as `last_success_at + interval`; with confirmed empty history use the persisted activation anchor for one initial due need. Unavailable/stale activity blocks new baseline authorization with an explicit reason. Use a bounded API timeout and policy-configured freshness limit.

The worker computes eligibility each existing proposal tick. Once due, keep the same `need_id` until a qualifying production satisfies it; never accumulate missed intervals. Failure, heat, busy, or policy/capacity denial preserves the need with bounded retry spacing. Any qualifying voluntary run also satisfies it. Restart recovers the checkpoint and re-reads activity. A policy revision recomputes due time but does not create a second pending need.

Add optional `visual_baseline: VisualBaselineEligibilityV1 = None` to **all three** `ProposalCandidateV1`, `PolicyDecisionV1`, and `ExecutionDispatchCandidateV1`. The only accepted route is existing `express` / template `render_scene` / `skills.imagination.render_scene.v1` and its configured execution target. That target identifies execution, not an attention goal. LLM/external candidate input cannot mint eligibility: the deterministic worker attaches it after inspecting activity, and consumers validate route, policy revision, freshness, due time and matching provenance. Invalid assertions get a recorded denial, never a global bypass.

In `build_proposal_frame()`, partition validated due baseline candidates from ordinary candidates **before** applying the tick-wide veto. Keep the real `action_warrant` and its original gate result unchanged. Suppress ordinary candidates on insufficient/unavailable warrant as today. Admit at most one valid baseline candidate through explicit baseline eligibility, including during calm, without applying ordinary priority thresholds to it. Reserve its slot within `max_candidates`; do not append it after truncation where it can disappear. If a voluntary `render_scene` already exists, coalesce the execution request while preserving its provenance and attaching the same due need.

In `orion/policy/evaluator.py` and `builder.py`, validate/copy eligibility into `PolicyDecisionV1`; retain ordinary express approval, risk, scope, operator controls, and rejected/deferred decisions. Eligibility is an admission reason, not policy approval. In `orion/execution_dispatch/builder.py`, copy it onto prepared **and blocked** dispatch candidates, and reserve one otherwise-approved baseline slot within existing kind/per-tick capacity limits (`_admit_candidates()`). Keep all hard route/scope gates and stale-frame handling. Record a reason whenever another gate defers the pending need.

### 4. Allocator admission with real accounting

Extend `orion/autonomy/allocator.py`'s internal `Candidate` and `candidate_from_dispatch()` with validated baseline eligibility; adapt dispatch-runtime `_log_allocator_preview()` to pass it. Add a narrow baseline branch in `allocate()` **before** `unmeasurable` and `below_information_floor` rejection. It requires finite positive estimated cost independently of `nats_per_sec`; it does not fabricate variance, signal, or information gain.

Allocate at most one approved due baseline candidate first, charging its full estimated motor seconds to the same allowance. Then run the existing information-based allocation for ordinary candidates on the remainder. Preserve applicable harm and resource/policy gates; the retired resource-pressure reward is not evidence of harmfulness. Insufficient allowance returns `allowance_exhausted`; missing cost returns `no_cost_estimate`. Baseline does not get free work, negative spend, or an unlimited bypass.

Dispatch-runtime `_send_prepared_candidates()` must honor this ordering within send limits and retain daily motor/risk accounting, tripwire, mode, staleness, and per-tick limits. Record measured duration on success, failure and deferral; a deferral can still consume HTTP/worker time. Baseline authorization requires a valid allocation even if the current advisory/error path would otherwise fall through without one. Resource unavailability remains a visible deferral, not evidence that reverie lacks value.

Remove `resource_pressure` and its direction from the `render_scene` template in `config/proposals/proposal_policy.v1.yaml`. Baseline uses `expected_effect=None`; no new learned continuity metric. Ordinary voluntary extras retain normal admission: without a justified measurable signal they may still be unmeasurable. That cannot stop the baseline and must be described honestly; this patch does not invent a signal to guarantee extras.

### 5. Dispatch identity, execution outcomes, and thermal authority

Add typed optional `VisualRunRequestV1` to `POST /visual-chain/run-once`: `dispatch_id`, `proposal_id`, `decision_id`, `correlation_id`, and `visual_baseline`. Empty body remains compatible. Thread these fields through dispatch builder request args and `RenderSceneVerb.execute()` instead of losing them in its empty POST. The endpoint validates baseline policy/freshness and rechecks activity under a durable per-need claim before starting work. Proposal-runtime's checkpoint is scheduling state; thought owns the atomic execution claim keyed by `need_id`, shared across replicas. Replays return the existing attempt/receipt, not another generation. A completed voluntary artifact invalidates an obsolete due need at this final check.

Persist a typed execution receipt in the visual chain JSON and dispatch result, including gate decisions, source selection, thermal verdict and artifact acknowledgement. Use explicit result `outcome`: `produced`, `deferred_thermal`, `deferred_busy`, `already_satisfied`, `failed`, or `unknown` for ambiguous timeout. Preserve legacy fields but never infer production from `ok`/`ran`. Forward these additions through the verb's result allowlist and dispatch result persistence.

Keep `evaluate_thermal_gate()` / `thermal_state()` as execution authority. No thermal score or new upstream sensor veto. Hot refuses until rearm; missing/stale allows degraded; disabled thermal gate is recorded as disabled. **No thermal-policy change is proposed.** Save successful/degraded verdicts too, rather than only refusal details. A busy check that precedes thermal evaluation records `thermal_not_evaluated`.

Retain the existing single-flight lock and deadline, plus durable claim/reconciliation for replay. Timeout cannot cancel an already-running blocking diffusion thread; therefore do not promise physical non-overlap solely from lock release. Hold ambiguous attempts for reconciliation/retry cooldown, and prove retry behavior against the diffusion rail before enabling. Resume with one attempt after deferral, then compute the next due time from actual production; no burst to repay elapsed intervals.

Trace the new `outcome` through dispatch-runtime `_send_one_inner()` / `_emit_action_outcome()`, feedback normalization (`orion/feedback/builder.py`), and feedback-runtime's effect resolution entry point. Deferred, already-satisfied and unknown outcomes must not become zero/negative effect observations. Baseline produces no effect posterior even on success (`expected_effect=None`); failures remain operational errors with measured cost. Keep old posterior rows as history but retire every active `render_scene` resource-pressure claim/learning path. The verb's current `ok=True` on refusal is not proof that downstream learning already excludes it.

All additions need schema/registry and producer/consumer compatibility tests. Closed `extra="forbid"` models require consumer-first rollout (including policy, dispatch and persisted-frame readers) before emitting new fields; do not rely on optional defaults making old readers tolerant. Keep existing channel names and update their documented payload contracts in `orion/bus/channels.yaml` as needed.

### Scheduling-state quality gate and rollback

This is an elapsed-time scheduling contract, not a detector of distress or consciousness. Its provenance is the acknowledged artifact/chain join above; age and any diagnostic daily count share that producer and are **not independent signals**. The justification is periodic scheduling with a single outstanding need, not an information-theoretic or biological measurement claim. Calm is represented by not-due after production, with no synthetic field pressure.

Existing-mechanism check found newest-row age unsuitable and the existing proposal/dispatch/thermal machinery reusable. **Live-data gate remains UNVERIFIED**: before activation inspect raw successful artifacts, failed/refused chains, duplicate-byte behavior and reset-after-success timestamps; confirm ongoing refusals cannot appear as activity and successful production really returns eligibility to not-due. Do not wire an unverified activity metric into any cognition model. Reversibility is a baseline scheduling policy switch plus optional fields; no training default, ontology node, or learned reward depends on it.

Privacy boundary stays at already-settled text reverie and existing self-study/memory source filters. Do not add raw chat/private recall or export source text into broad dispatch telemetry. Dispatch carries identities; selected excerpts stay in the existing visual artifact's access boundary. Dangerous failures are duplicate GPU work, stale/private content selection, false production acknowledgements, and false attention-attribution. Disable baseline policy to stop new baseline proposals; keep activity/provenance readers and thermal gating available. Disabling does not revert to the old resource-pressure reward or start the retired cron.

## Files likely to touch

| Files | Intended change |
|---|---|
| `orion/schemas/reverie_visual.py`, `reverie.py` if trace join requires it | Typed context, activity, eligibility, request and receipt; optional real broadcast provenance. |
| `orion/schemas/{proposal_frame,policy_decision_frame,execution_dispatch_frame}.py` | Carry optional validated eligibility through every decision boundary. |
| `services/orion-thought/app/{store,visual_chain,main}.py` | Identity-preserving reads, acknowledged production, activity API, durable claim, request/receipt handling; preserve thermal path. |
| `services/orion-proposal-runtime/app/{worker,store}.py`, `orion/proposals/{builder,policy}.py` | Activity client/checkpoint, narrow eligibility gate and bounded scheduling policy. |
| `orion/policy/{evaluator,builder}.py`, `orion/execution_dispatch/builder.py` | Validate/copy eligibility; retain approvals and reserve a bounded baseline dispatch slot. |
| `orion/autonomy/allocator.py`, `services/orion-execution-dispatch-runtime/app/worker.py` and result store | Costed baseline allocation, dispatch identity, replay/outcome propagation. |
| `services/orion-cortex-exec/app/verb_adapters.py` | Send typed execution metadata and retain receipt fields. |
| `orion/feedback/builder.py`, feedback-runtime effect resolution and shared outcome schema as inspection requires | Explicit non-observation semantics for deferral/unknown; consumer tests. |
| `config/proposals/proposal_policy.v1.yaml`, relevant service settings/examples/README, schema registry/channel documentation | Scheduling contract and removal of wrong reward; consumer-first rollout instructions. Sync local `.env` if templates change. |
| `services/orion-sql-db/manual_migration_reverie_visual_chain.sql` and new migrations following that service's conventions | Small checkpoint/claim tables and indexes only after inspecting current schema conventions; no graph changes. |

Attention scoring/top-down selection and thermal arithmetic need no redesign. No code, runtime config, environment template, bus data or graph was changed by this document revision.

## Non-goals

- Proving sentience or that every image is useful/novel.
- New attention scores, synthetic goal targets, drive taxonomy, debt pressure, or rolling-density admission.
- A separate visual cron/watchdog or Curiosity durable-admission broker.
- Globally lowering the information floor or waiving thermal/resource/policy constraints.
- Calling rotation an autonomous choice or promising proposal-selected content without the missing handoff.
- Changing missing/stale thermal readings to fail closed.

## Acceptance checks

### Deterministic tests and behavioral eval

1. Extend `tests/test_proposal_frame_builder.py`: a due baseline survives low and unavailable warrant; ordinary external candidates remain suppressed; original warrant is unchanged; invalid/forged eligibility fails; baseline survives bounded candidate truncation.
2. Extend proposal/policy/dispatch schema and builder tests: eligibility round-trips through approved, blocked, stale and deferred decisions; wrong route, stale snapshot and disabled policy do not dispatch; no duplicate candidate for a voluntary proposal plus due need.
3. Extend `tests/test_motor_allocator.py` and dispatch-runtime tests: baseline with no signal and baseline below information floor can be admitted with known cost; missing cost/insufficient allowance cannot; one baseline consumes allowance before ordinary allocations; send limits and error/advisory paths cannot lose resource accounting.
4. Extend thought `test_store.py`, `test_visual_chain.py`, thermal and deadline tests: fresh settled non-hollow thought identity survives selection; unselected thoughts are not credited; self-study/memory/continuity/default seed have honest provenance; old/private/hollow thoughts remain excluded.
5. Activity fixtures: thermal/failure rows newer than a real artifact do not reset due time; caption failure with stored image does; missing artifact/persistence failure does not; deadline/artifact reconciliation and duplicate-byte storage are explicit; unavailable database differs from confirmed empty history.
6. Fake-clock replay eval: calm → due → resource defer → thermal defer → cooldown → one production → not due. Include many missed intervals, restart, concurrent/replayed dispatches and a voluntary success immediately before baseline dispatch. Assert one pending need, bounded attempts, no catch-up burst, and no effect-posterior update on deferral or baseline production.
7. Verb/HTTP/feedback integration tests retain exact dispatch/proposal/need/source/chain IDs, outcome and thermal verdict; replay does not regenerate. Test actual result-normalization path, not just verb `ok`.

Use the existing focused test suites above and add the smallest replay eval under the affected service's `evals/` if absent. Implementation must report test results separately from eval results and run affected Docker/service smoke through `scripts/safe_docker_build.sh` from a worktree. No runtime tests/evals were run for this documentation-only revision.

### Correlated evidence required before production claims

Capture one successful baseline trace during a genuinely calm tick and one voluntary trace, plus a deferred-and-resumed baseline. Export bounded receipt/query evidence with these joins:

```text
activity observed_at + last_success_chain_id/sha → need_id + due_at
  → proposal frame/id + unchanged action_warrant + baseline gate reason
  → policy decision/id + approval or deferral reasons
  → dispatch_id + allocation reason + estimated cost/allowance + send result
  → HTTP attempt/correlation_id + need claim/replay decision
  → thermal verdict (reading age, state, degraded, allows_gpu_work)
  → selected source ID + selection method + optional verified attention trace
  → visual chain_id + artifact sha/path/bytes + persistence acknowledgement
  → activity last_success advance + same need satisfied + actual motor cost
```

For reverie-selected content, join selected thought to settled text chain and coalition, then to retained broadcast and field-goal evidence where available. A goal that fails real `source_refs` matching must not be credited with influencing selection. For another selected slot, report its own identity and do not label the run attention-grounded merely because a reverie was also loaded. A thermal refusal should end before source selection, with no image and the same pending need; resume must complete the source/artifact join.

These are required evidence fields and test scenarios, **not fabricated successful trace output**. Production remains **UNVERIFIED** until actual records demonstrate the complete path, including source selection and an accessible generated image. Daily counts alone cannot demonstrate recurrence or causality.

## Recommended next patch

First implement the inspectable source/activity boundary: preserve source identity, acknowledge actual persisted production, add the read-only activity projection, and test against failure/refusal/settled-source fixtures. Verify its real-data gate before scheduling relies on it.

Then ship the smallest complete baseline vertical slice: eligibility schema and policy → both admission exceptions → bounded dispatch and resource accounting → request/receipt and deduplication → outcome exclusion and correlated replay eval. Remove the wrong template signal in that same slice, not as a standalone fix that leaves every candidate unmeasurable. Enable only after consumer compatibility and the activity gate pass; retain explicit production `UNVERIFIED` until the live trace is collected.
