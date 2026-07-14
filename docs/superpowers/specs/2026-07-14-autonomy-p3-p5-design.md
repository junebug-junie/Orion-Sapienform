# Autonomy P3–P5 (motor-nerve series) — design

**Date:** 2026-07-14
**Status:** Implementation, this session
**Mode:** Three thin patches, re-grounded against live code (main now has P0/P1/P2/P6 merged). All three original brainstorm-level plans needed real correction after grounding — documented per-section below.

## P3 — Consequence → drive satisfaction + operator inform

### Ground truth (corrects the original brainstorm)

The tension→goal pipeline (`orion/spark/concept_induction`, `DriveEngine`) and the P1/P2 Layer-9 dispatch pipeline (`orion/execution_dispatch`, `orion/proposals`) are **structurally separate** — no `drive_origin` field survives from a proposal candidate through to a completed dispatch's outcome. Bridging that fully (threading `drive_origin` through `ProposalCandidateV1` → `ExecutionDispatchCandidateV1` → `FeedbackFrameV1`) is out of scope here — too invasive for a thin patch, and not needed: a smaller, real seam already exists.

`ConceptWorker` (`orion-spark-concept-induction`) already subscribes to `orion:feedback:frame` and mints tensions from `FeedbackFrameV1` — but only on failure (`extract_tensions_from_feedback`, `tensions.py:399-465`, fires on `outcome_status in {"failed","absent","blocked"}` or low score; never on success). Separately, P2 just made `orion:autonomy:action:outcome` (schema `ActionOutcomeEmitV1`) carry every real Layer-9 dispatch outcome — success and failure both — with `orion-execution-dispatch-runtime` already a registered producer and `orion-spark-concept-induction` already covered by that channel's `consumer_services: ["orion-sql-writer", "*"]` wildcard. `ConceptWorker` does not yet subscribe to it.

**Chosen seam**: subscribe `ConceptWorker` to `orion:autonomy:action:outcome`, mint a small relief tension locally from `ActionOutcomeEmitV1.success`, fold through the existing per-tick `DriveEngine.update()` call (`bus_worker.py:866-872`) — no new call site, no new channel, no proposal/candidate schema changes. Rejected alternatives: a new dedicated tension-inject channel (strictly larger, no functional gain — `TensionEventV1.magnitude` is schema-clamped non-negative regardless of channel, so the real work is the DriveEngine fix either way); riding the existing `FeedbackFrameV1` path (depends on a second service's 2s-poll/Postgres-join correctness that the P2 commit's own history suggests wasn't trusted for this purpose).

**Mandatory companion fix, more load-bearing than the wiring itself**: `DriveEngine.update()` (`orion/spark/concept_induction/drives.py`) cannot represent relief *at all* today — `impact_sum[drive] += mag * self._clamp01(weight)` clamps `weight` to `[0,1]`, silently zeroing any negative `drive_impacts` value, and the leaky-math branch's `impulse = self._clamp01(impact_sum[drive])` clamps again. A negative-weight tension is a complete no-op under current code, not a partially-working relief — this needs a real fix, not "confirm it handles negatives," and it's the single most sensitive part of this patch: `DriveEngine.update()` is live, running on every accepted bus event (1000+/hr), already fixed once before for the flat-0.731 pin regression.

### Proposed changes

**`orion/spark/concept_induction/drives.py`**:
- `impact_sum[drive] += mag * self._clamp01(weight)` → clamp `weight` to `[-1.0, 1.0]` instead of `[0.0, 1.0]` (new small `_clamp_signed` helper, or inline).
- Leaky branch: `impulse = self._clamp01(impact_sum[drive])` → clamp to `[-1.0, 1.0]`. Then branch the pressure update on sign:
  - `impulse >= 0`: unchanged — `base + impulse * (1.0 - base)` (headroom-toward-ceiling).
  - `impulse < 0`: `base + impulse * base` (headroom-toward-floor — symmetric design: relief has diminishing effect as pressure approaches 0, exactly mirroring how growth has diminishing effect approaching 1; also means `base=0` naturally yields zero relief without depending on the outer clamp to save it).
  - Both branches still pass through the existing final `self._clamp01(...)`.
- Legacy `leaky_math_enabled=False` branch: left as-is (not live/configured default; `_soft_saturate`'s internal `_clamp01` will floor a negative `raw` at 0 — acceptably inert for a non-live path, not optimized further).
- This is the highest-risk part of the whole P3-P5 sprint (live numerical core). Existing tests for `DriveEngine.update()` must all still pass unchanged — every existing producer only ever emits non-negative `drive_impacts`, so this change must be provably a no-op for all of them (only reachable via a genuinely negative weight, which nothing produces today except the new satisfaction tension).

**`orion/autonomy/tension_ratelimit.py`**: `TensionRateLimiter._signature()` currently only includes drives with `weight > 0.0` in its signature tuple — a negative-weight-only tension collides into an empty-tuple bucket shared with any other zero-positive-weight tension of the same `kind`. Change the filter to `weight != 0.0` (or `abs(weight) > 1e-9`) — safe, backward-compatible (every existing producer only emits positive weights, so this is a no-op for all of them) and correctly captures which drives a satisfaction tension targets for rate-limiting purposes.

**`orion/spark/concept_induction/tensions.py`**: new `extract_tensions_from_action_outcome(payload: dict, ...) -> list[TensionEventV1]` mirroring `extract_tensions_from_feedback`'s shape (same `subject="orion"`/`model_layer="self-model"`/`entity_id="self:orion"` convention, same deterministic-artifact-id pattern via `_artifact_id`). Fires only on `success=True` (relief on success; `success=False`/`None` mints nothing here — failures are already covered by the existing `extract_tensions_from_feedback` failure path, not duplicated). Maps `ActionOutcomeEmitV1.kind` (`inspect`/`summarize`/`observe`) to a small, explicit, closed drive-impact table — not a fuzzy inference:
  - `inspect` → `{"coherence": -0.10}` (a targeted look reduces uncertainty/incoherence)
  - `summarize` → `{"predictive": -0.10}` (a wide-angle picture reduces predictive gap)
  - `observe` → `{"continuity": -0.05}` (a lighter, lower-magnitude relief matching `observe`'s own lower-stakes design intent from P1)
  - Magnitude: fixed `0.3` (not derived from `surprise`/anything else — a simple, legible constant, matching the cap language from the original brainstorm's "−0.10 cap" intent, expressed here as `magnitude * weight` staying within a small bound).

**`orion/spark/concept_induction/bus_worker.py`**:
- `settings.py`'s `intake_channels` (or wherever the subscription list lives) gains `"orion:autonomy:action:outcome"`.
- `handle_envelope` gains an `elif env.kind == "action.outcome.emit.v1":` branch mirroring the existing `feedback.frame.v1` branch (`bus_worker.py:809-813`), calling the new extractor, folding results into the same `spark_tensions`/`all_spark_tensions` list that already reaches `bus_worker.py:866-872`'s `drive_engine.update()` call.
- **Self-publish filter, mandatory**: `ConceptWorker` is itself already a producer on this exact channel (curiosity-fetch outcomes, `bus_worker.py:602-611`/`1044-1068`). Without a guard, Redis pub/sub delivers to all subscribers including the publisher, double-counting/looping. Guard: skip processing when `env.source.name == self.cfg.service_name` (i.e. only process outcomes from *other* services — today that means only `orion-execution-dispatch-runtime`'s Layer-9 outcomes reach the new extractor, not `ConceptWorker`'s own curiosity-fetch ones, which already have their own dedicated in-process path).
- Add `"action.outcome.emit.v1"` to the concept-induction-skip set (`bus_worker.py:1097`, currently `{"substrate.self_state.v1", "feedback.frame.v1"}`) so this structured signal doesn't spuriously trigger concept extraction/clustering.

### Operator inform

`services/orion-hub/scripts/substrate_execution_dispatch_routes.py`'s `/latest` route currently returns a bare `ExecutionDispatchFrameV1.model_dump()` with no dispatch/dry-run breakdown and no visibility into the runtime's in-process `theater_tripwire_active` (that flag lives only on `orion-execution-dispatch-runtime`'s own `/latest`, per P1 — the Hub route reads Postgres directly and has no access to it). Minimal fix, matching what's actually derivable from data already in scope: add a small summary block to the Hub route's response — `{"dispatched_count": len(dispatched_candidates), "prepared_for_dispatch_count": <count from candidates>, "dry_run_count": <count from candidates>}` — computed from the frame already loaded, no new fields, no cross-service call. Exposing the live `theater_tripwire_active` flag itself would require either a cross-service HTTP call from Hub to the runtime's own `/latest` (a real architectural choice, not a one-line addition) or persisting the flag into the dispatch-frame row — **explicitly deferred**, named as a non-goal below; the count breakdown is the thin, honest slice.

### Non-goals

- No full `drive_origin` bridge between the proposal/goal pipeline and the Layer-9 dispatch pipeline.
- No cross-service Hub→runtime call (or persistence change) to surface `theater_tripwire_active` on the Hub route — only the count breakdown ships.
- No change to the legacy (`leaky_math_enabled=False`) DriveEngine path beyond leaving it inert-safe.
- No touching `extract_tensions_from_feedback`'s existing failure-only behavior.

### Acceptance checks

1. Existing `DriveEngine.update()` tests (find and run) still pass unchanged.
2. New test: a tension with `drive_impacts={"coherence": -0.5}` and non-zero prior pressure reduces pressure, floored at 0, never negative.
3. New test: existing (non-negative) tensions produce byte-identical pressures to before the clamp change (regression guard against a sign-handling mistake).
4. New test: `extract_tensions_from_action_outcome` on `success=True` mints exactly one tension with a negative weight for the mapped drive; on `success=False`/`None` mints nothing.
5. New test: `bus_worker.py`'s self-publish filter — an `action.outcome.emit.v1` envelope with `source.name == "orion-spark-concept-induction"` is skipped; one with `source.name == "orion-execution-dispatch-runtime"` is processed.
6. Hub route test: `/latest` response includes the three counts, computed correctly from a fixture frame.

---

## P4 — Capability vocabulary: 1 verb → 3

### Ground truth (confirms most of the original brainstorm, corrects the "client class" assumption)

`orion/autonomy/capability_policy.py::evaluate_capability(capability_id, ctx)` is a per-capability-id evaluator, not a fan-out/selector — the caller (`policy_act.py`) hardcodes which `capability_id` to attempt at each call site and asks the evaluator to approve/deny it. `orion/autonomy/fanout_policy.py` is unrelated (chat-stance SPARQL fan-out breadth, not capability selection) — do not build on it.

No `RecallClient` class exists; recall queries are issued via inline `bus.rpc_request(...)` using `build_recall_query_v1` (`orion/cognition/recall_query.py`) against channel `orion:exec:request:RecallService` (`orion/bus/channels.yaml`, schema `RecallQueryV1`, reply `orion:exec:result:RecallService:*`). `producer_services` on that channel does not yet include any autonomy-side service.

`services/orion-self-experiments`: `SELF_EXPERIMENTS_DISPATCH_ENABLED` defaults **false** — dispatch is currently a no-op (`status="queued", reason="dispatch_disabled"`) regardless of what this patch adds. No `endogenous_safe` field exists on `EXPERIMENT_REGISTRY` entries (a plain `dict[str,str]`, not a pydantic model) or `SelfExperimentSpecV1`. No `memory_contradiction_probe`-equivalent `SelfExperimentType` exists; `belief_origin_check` is the closest but is about provenance, not contradiction detection. No `autonomy`/`endogenous` `SelfExperimentSource` literal exists.

**Scope call, given the above**: `self_experiment.create` as originally scoped (new experiment type + new source literal + `endogenous_safe` registry field + a bounded depth-2 probe) is real, non-trivial new surface area on a *different service* whose own dispatch mechanism is disabled by default regardless — building it now would ship inert code with no way to verify it end-to-end (dispatch is off). **Deferred as a named non-goal.** `recall.query.readonly` alone is the real, buildable, verifiable-in-principle half of P4: it rides an existing bus RPC pattern, targets an already-running service (`orion-recall`), and its policy gate (`auto_execute` under existing `capability_policy.py` machinery) is provably testable without needing a second service's disabled dispatch path.

### Proposed changes

**`config/autonomy/capability_policy.v1.yaml`**: one new rule —
```yaml
- capability_id: recall.query.readonly
  side_effect_class: readonly
  auto_execute: true
  requires_goal_status: proposed
  required_drive_origins: [predictive]
  required_signal_kinds: [world_coverage_gap]
  budget_per_cycle: 2
```
(mirrors `web.fetch.readonly` exactly — same signal kind, same drive origin, same budget; deliberately reuses the existing `world_coverage_gap` vocabulary rather than inventing a new signal kind).

**`orion/autonomy/policy_act.py`**: new `maybe_execute_readonly_recall_after_goal(...)` mirroring `maybe_execute_readonly_fetch_after_goal` (`policy_act.py:37-99`) exactly in shape: gate on signal kinds present → build `CapabilityEvaluationContext` → `evaluate_capability("recall.query.readonly", ctx)` → on allow, issue the recall RPC (inline `bus.rpc_request` + `build_recall_query_v1`, matching `orion/cognition/recall_prefetch.py:169-183`'s exact pattern — no new client class, riding the existing seam) → record an `ActionOutcomeRefV1`/emit via the same `append_action_outcome`/bus-emit convention already established, so a recall-first check is itself visible the same way a Layer-9 dispatch or curiosity-fetch is. Behavioral intent from the original brainstorm preserved: "check what I already know first" — call this before the existing readonly-fetch path in the same tick, and if recall finds something, DO NOT budget-consume the fetch capability (fetch stays available for when recall comes up empty).

**`orion/bus/channels.yaml`**: add whichever service actually issues this RPC (the autonomy/`orion-spark-concept-induction` `ConceptWorker`, same process P3's satisfaction wiring touches, since `policy_act.py`'s call sites are invoked from there) to `orion:exec:request:RecallService`'s `producer_services`, if not already covered by a wildcard (verify in-patch; the P1-era finding for the cortex-exec-request channel found a `:background` sibling channel already multi-producer — check whether an equivalent already exists here before adding a new entry).

### Non-goals

- `self_experiment.create` — deferred, named above, because `SELF_EXPERIMENTS_DISPATCH_ENABLED` is off by default and building the full new-type/new-source/registry-field surface now would ship unverifiable dead code. Recommended as its own future patch once dispatch is enabled for real.
- No new `RecallClient` class — reuses the existing inline RPC pattern.
- No change to `required_signal_kinds`/`required_drive_origins` vocabulary — reuses `world_coverage_gap`/`predictive` exactly as the existing fetch capability does.

### Acceptance checks

1. `evaluate_capability("recall.query.readonly", ctx)` returns `allowed` under the same conditions `web.fetch.readonly` would (mirrored policy shape) — new test.
2. A live tick where `world_coverage_gap` fires: recall is tried first; if recall's result is non-empty, the fetch budget is not consumed that cycle (new test, mocked recall RPC).
3. If recall's RPC fails or times out, degrades to falling through to the existing fetch path — never raises, never blocks the tick.

---

## P5 — Attention-bound proposals

### Ground truth (corrects two premises from the original brainstorm)

`ProposalTemplateV1` (`orion/proposals/policy.py`) has `model_config = ConfigDict(extra="forbid")` and only literal `target_kind`/`target_id` fields — `target_binding` is a wholly new field, not an extension of an existing mechanism. `orion/proposals/builder.py::_build_candidate` already has `self_state: SelfStateV1` in scope at exactly the point `target_id`/`target_kind` get set (`_build_candidate(*, template_key, template, self_state, attention, policy)`, called from a loop already iterating `policy.proposal_templates.items()`) — so the resolution point is real and available, just needs a new branch.

`ProposalCandidateV1.provenance` **does not exist** — the original brainstorm's plan to add `binding_resolved_from` to it was based on a false premise. What exists instead: `evidence_refs: list[str]` (free-form string refs already populated with `self_state:{id}`/`attention:{id}`/`field:{id}`), and flat fields `source`/`thought_id` (already used for a different provenance purpose — reverie-injected candidates). The correct home for binding provenance is a new flat field alongside `source`/`thought_id`, matching this schema's actual existing style, not a new nested object.

`SelfStateV1.dominant_attention_targets: list[str]` and `dominant_attention_target_details: list[AttentionTargetSummaryV1]` (fields: `target_id`, `target_kind: Literal["node","capability","channel","edge","field","system"]`, `pressure_score`, `dominant_channel`, `reason`) both confirmed present exactly as the original design expected.

### Proposed changes

**`orion/proposals/policy.py`**: `ProposalTemplateV1` gains `target_binding: str | None = None` (still `extra="forbid"` for anything else — this is the one new, explicitly-typed field, not a loosening of the schema). Only one recognized binding path string for v1: `"self_state.dominant_attention_targets[0]"` — validated/matched literally, not a general expression parser (no invented DSL).

**`config/proposals/proposal_policy.v1.yaml`**: one new template, `inspect_attended_target`:
```yaml
- kind: inspect
  target_kind: capability   # placeholder default kind; overridden per-candidate when binding resolves to a different kind
  target_id: capability:orchestration   # fallback literal, used only if the binding fails to resolve
  target_binding: "self_state.dominant_attention_targets[0]"
  proposed_effect: increase_observability
  required_policy_gate: read_only
  base_priority: 0.34   # shipped live per explicit user go-ahead ("turn on whatever you need"), not dark-shipped at 0.0 -- read-only inspect proposal under the existing policy gate, same risk class as every other already-live inspect template
  base_risk: 0.05
  reversibility: 1.0
  dimensions:
    field_intensity: 0.30
```

**`orion/proposals/builder.py::_build_candidate`**: new branch — if `template.target_binding == "self_state.dominant_attention_targets[0]"` and `self_state.dominant_attention_target_details` is non-empty, resolve `target_id`/`target_kind` from `self_state.dominant_attention_target_details[0]` (`.target_id`, `.target_kind` — using the richer `_details` list, not the bare-string `dominant_attention_targets`, since it carries a real typed `target_kind` rather than requiring a second lookup); **whitelist**: only accept `target_kind` values already present in `AttentionTargetSummaryV1`'s own literal (`node/capability/channel/edge/field/system`) intersected with `ProposalCandidateV1`'s existing accepted `target_kind`s (verify overlap in-patch — `cast_target_kind` already exists per P1-era grounding; if a resolved kind isn't in its accepted set, fail closed to the literal fallback, never raise). If the binding can't resolve (empty details list, unrecognized kind), fall back to the template's literal `target_id`/`target_kind` — never block candidate construction.
- Set the new flat field (e.g. `binding_resolved_from: str | None`) on `ProposalCandidateV1` when a binding actually resolved (not set when falling back to literal) — traceability for the eval below, matching this schema's existing flat-field style.

**`orion/schemas/proposal_frame.py`**: `ProposalCandidateV1` gains `binding_resolved_from: str | None = None`.

**Kill-criterion eval** (per the original brainstorm's own falsifiable design, preserved): a small script/eval, mirroring `orion/autonomy/evals/run_origination_eval.py`'s shape, checking over a real window whether `distinct(target_id)` for this template's candidates is `>= 3` — not run automatically in this patch (needs real traffic accumulation over days), but the script itself ships so it can be run once enough live data exists. Documented in the template's own YAML comment as the kill criterion, matching the design's explicit self-falsifying intent.

### Non-goals

- No general binding-expression DSL — one literal, recognized path string only.
- No new `target_kind` values beyond what `AttentionTargetSummaryV1`/`cast_target_kind` already accept.
- Not running the kill-criterion eval's real 7-day verdict in this patch — no live traffic window available in this sandbox; the eval ships, its verdict doesn't.

### Acceptance checks

1. New test: template with the binding resolves `target_id`/`target_kind` from a fixture `SelfStateV1` with populated `dominant_attention_target_details`.
2. New test: empty `dominant_attention_target_details` falls back to the template's literal `target_id`/`target_kind`, `binding_resolved_from` stays `None`.
3. New test: a resolved `target_kind` outside the accepted set falls back to literal rather than raising.
4. `binding_resolved_from` is set only on the binding-resolved path, verified by test.
