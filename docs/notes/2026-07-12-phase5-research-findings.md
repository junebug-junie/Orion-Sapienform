# Phase 5 research findings — ladder reach + phi encoder input — 2026-07-12

Context: Phase 5 ("Ladder + phi closure") of
`docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md` requires two
read-only research answers before any implementation. Both are answered here from live
code, live `.env` values, and the live active phi-encoder manifest — not from README claims.

## Q1 — Does L7-L11 reach real behavior, or terminate in Hub debug tiles?

**Verdict: L7-L11 currently terminates in Hub debug tiles. No consumer beyond that was
found — and the one candidate exception (`orion-dream/compaction_applier.py`) is not even
wired into the live pipeline, so it isn't a real exception, it's a second dead end.**

### `compaction_applier.py` is not called by anything live

`services/orion-dream/app/compaction_applier.py:53-58` defines a `CompactionMemoryStore`
Protocol whose docstring says outright: "A real (canonical) implementation is provided ONLY
in a dedicated, signed-off change — never imported here — so this module is inert by
construction." That signed-off change has not happened:

- `grep -rn "CompactionMemoryStore"` across the repo finds only the Protocol definition
  and its use as a type hint in `compaction_applier.py` itself — zero real implementations
  anywhere.
- `grep -rn "apply_compaction_delta"` finds callers **only** in
  `services/orion-dream/tests/test_compaction_applier.py`. No `worker.py`, cron, or bus
  handler in `orion-dream/app/` ever calls it.
- `PolicyDecisionFrameV1` is imported in `compaction_applier.py:38` for a type hint only —
  nothing in `services/orion-dream/app/` fetches a live `PolicyDecisionFrameV1` from
  Postgres to pass into it. `grep -rn "PolicyDecisionFrame\|ExecutionDispatchFrame"
  services/orion-dream/app/*.py` returns matches only inside `compaction_applier.py`.
- The hot gate is off live: `services/orion-dream/.env:68` —
  `ORION_DREAM_COMPACTION_APPLY_ENABLED=false`.

So the arsonist review's "known consumer" framing was optimistic: `compaction_applier.py`
is not merely gated off, it is **not invoked by any live code path at all** — it is a
standalone, fully-tested, entirely dormant module. The adjacent Phase F module
(`services/orion-dream/app/rem_compaction.py:1-8`) says the same thing about its own layer
in its own docstring: "REM sleep reads the awake reverie's Phase-E compaction *requests*
and narrates what sleep *would* do to memory... **and nothing applies it**." Confirmed live:
`services/orion-hub/scripts/substrate_observability_routes.py:141-142` labels the pending
compaction-request queue section "Queue only — applied by nothing."

### The one non-debug reader of L11 output doesn't causally use it

`orion-consolidation-runtime` (L11) writes to four tables
(`services/orion-consolidation-runtime/app/store.py`): `substrate_consolidation_frames`,
`substrate_expectations`, `substrate_tensor_slices`, `substrate_schema_candidates`.

Grepping for readers of each table across the whole repo (excluding `.worktrees/`), the only
non-Hub-debug hit is `services/orion-thought/app/grounding.py:55-65`
(`default_motif_loader`), which selects `frame_id` from `substrate_consolidation_frames` and
feeds it to `services/orion-thought/app/reverie.py`'s spontaneous-thought pipeline. This is
live: `ORION_REVERIE_ENABLED=true` and `ORION_REVERIE_GROUND_CONSOLIDATION=true` in
`services/orion-thought/.env`.

But tracing what that reference actually *does* (`reverie.py:358-411`):

1. The LLM call that produces the thought's actual content happens first
   (`client.execute_plan(...)` at line 372, prompt built by `build_reverie_plan_request` /
   `build_reverie_context`, which only take `broadcast` (attention) and `concern_cards` —
   never consolidation data).
2. Only *after* the thought is already generated does `collect_grounding(...)` at
   `reverie.py:404-411` attach `motif_refs` (bare consolidation `frame_id` strings, not
   motif content) to the already-finished `thought` object via `model_copy(update=...)`.

So L11's output never enters the LLM prompt or shapes thought content — it is a bare ID tag
appended post-hoc for provenance. Tracing where that tag goes next: the thought is published
on `reverie.thought.v1` (`grep`'d — subscribed by nothing outside `orion-thought` itself) and
persisted via `persist_reverie_thought` (`services/orion-thought/app/store.py:46-80`) into
`substrate_reverie_thought`, whose docstring says it "Backs the hub `_reverie_section`
panel." That panel
(`services/orion-hub/scripts/substrate_observability_routes.py:112-138`, part of the
`substrate_*` Hub-route family) doesn't even surface `motif_refs` in its response — it only
returns `thought_id`, `salience`, `interpretation`, `attended_node_ids`,
`selected_open_loop_id`. So the one candidate non-debug consumer bottoms out in a debug/
observability route anyway, and even that route drops the field before display.

### Full ladder confirmation

- `EXECUTION_DISPATCH_MODE=dry_run` confirmed live in
  `services/orion-execution-dispatch-runtime/.env:16`, read at
  `services/orion-execution-dispatch-runtime/app/worker.py:76`
  (`override_dispatch_mode=self._settings.execution_dispatch_mode`).
- Every `substrate_execution_dispatch_frames`, `substrate_feedback_frames`,
  `substrate_consolidation_frames` (etc.) reader outside the ladder services themselves is
  either another ladder service's own store, or a `services/orion-hub/scripts/substrate_*.py`
  route explicitly self-labeled `"""Read-only debug API for substrate ... frames."""`
  (confirmed for `substrate_consolidation_routes.py`, `substrate_execution_dispatch_routes.py`,
  `substrate_feedback_routes.py`, `substrate_lattice_routes.py`).

### Verdict table

| Layer | Writes to | Real consumer beyond Hub debug tiles? |
|---|---|---|
| L9 execution-dispatch | `substrate_execution_dispatch_frames` | No — `EXECUTION_DISPATCH_MODE=dry_run` live, only reader outside the ladder is `substrate_execution_dispatch_routes.py` (self-labeled debug API) |
| L10 feedback | `substrate_feedback_frames` | No — only reader outside the ladder is `substrate_feedback_routes.py` (self-labeled debug API) |
| L11 consolidation | `substrate_consolidation_frames`, `substrate_expectations`, `substrate_tensor_slices`, `substrate_schema_candidates` | Only `orion-thought/grounding.py` reads outside the ladder/Hub — but only appends an inert ID tag to an already-generated thought; that thought's persisted record backs a Hub observability panel that doesn't even render the tag |
| `orion-dream/compaction_applier.py` | (would write canonical memory) | **Not applicable — not wired.** No `CompactionMemoryStore` implementation exists anywhere; `apply_compaction_delta` is called only from tests; hot gate `ORION_DREAM_COMPACTION_APPLY_ENABLED=false` live |

**Bottom line for the plan's Phase 5 checkbox:** L7-L11's real-world reach is now a
documented fact, not an open question — it does not currently reach real behavior. The
whole ladder (proposal → policy → execution-dispatch → feedback → consolidation) plus its
one non-Hub reader (reverie grounding) is rehearsal: frames are computed, persisted, and
displayed, but nothing downstream changes memory, chat content, or triggers any
side effect a human or Orion would notice. This is not ambiguous — every path was traced to
either an unwired module, a dry-run flag, or a route whose own docstring says "debug."

---

## Q2 — Does the phi encoder train on raw channels, dimension scores, or `InnerStateFeaturesV1`?

**Verdict: the encoder reads `InnerStateFeaturesV1` (option d), and every SelfStateV1-sourced
value inside that vector is a `.score` value (option a) — `.confidence` is never read
anywhere in the path. No phi-corpus impact statement is needed for Phase 1's confidence-formula
change; the `policy_pressure` removal (already shipped) also required none, confirmed
below.**

### Traced call path

`handle_self_state` (`services/orion-spark-introspector/app/worker.py:2506`) does two
separate things with the incoming `SelfStateV1` (`ss`):

1. `phi_now = _phi_from_self_state(ss)` (line 2515) — a **separate**, older heuristic that
   produces four EKG display stats (`coherence`, `energy`, `novelty`, `valence`) directly
   from `ss.dimensions[key].score` via the local `_s()`/`_t()` helpers
   (`worker.py:222-295`). This function is **not** the trained encoder's input path — it
   feeds `_get_phi_stats()` / the legacy WS EKG headline only, and is a red herring for the
   "trained encoder" question if read in isolation.
2. The actual trained-encoder path, gated on `settings.inner_features_enabled`
   (`worker.py:2521-2568`):
   - `inner, ... = build_inner_state_features(ss, _INNER_SCALER, ...)` — builds an
     `InnerStateFeaturesV1` from `ss` (`services/orion-spark-introspector/app/inner_state.py:354-495`).
   - `x = enc.feature_vector_from_inner(inner)` — builds the literal numpy input vector
     (`services/orion-spark-introspector/app/phi_encoder.py:99-116`).
   - `out = enc.forward(x)` — the actual MLP forward pass
     (`phi_encoder.py:118-134`, `W1/b1/W2/b2/W3/b3/w_phi/b_phi` loaded from
     `ORION_PHI_ENCODER_WEIGHTS`).

`feature_vector_from_inner` (`phi_encoder.py:99-116`) reads `inner.features` — a list of
named `InnerFeatureV1{name, raw_value, scaled_value, source}` rows — filtered/ordered
strictly by `manifest.input_features`, and uses `scaled_value` for the tensor. It never
touches `SelfStateV1` directly; `InnerStateFeaturesV1` is the only thing it sees.

### What builds `InnerStateFeaturesV1`, and whether `.confidence` ever appears

`build_inner_state_features` (`inner_state.py:354-495`) constructs every feature from:

- `_dim_score(ss, key)` (`inner_state.py:149-151`) — **`float(dim.score)`, never
  `dim.confidence`** — for each `key` in `FELT_DIMENSIONS`
  (`inner_state.py:76-87`: `coherence, field_intensity, agency_readiness,
  execution_pressure, reasoning_pressure, resource_pressure, reliability_pressure,
  continuity_pressure, social_pressure, introspection_pressure`), each tagged with
  `source=f"self_state.dimensions.{key}"` (line 389/398).
- `ss.overall_intensity` (top-level score-like float, line 403) tagged
  `source="self_state.overall_intensity"`.
- Four cognitive features (`recall_gate_fired`, `reasoning_present`, `execution_load`/
  `exec_step_fail_rate`, `reasoning_load`/`execution_friction`) derived from
  `execution_trajectory` / `reasoning_activity` projections fetched separately over HTTP
  (`worker.py:2523-2524`) — **not** `SelfStateV1` fields at all.
- `dominant_field_channels` values for `INFRA_CHANNELS` (`bus_health`,
  `delivery_confidence`, `transport_integrity`, `contract_pressure`,
  `catalog_drift_pressure`) go into the `infra` list only, explicitly commented
  `scaled_value=0.0,  # infra never scaled/read by φ` (`inner_state.py:456`) — recorded for
  provenance, excluded from the trainable set by `encoder_trainable_feature_names`
  (`inner_state.py:130-146`).

`grep -n "confidence" inner_state.py phi_encoder.py` returns exactly one hit — the string
`"delivery_confidence"` (an infra channel name, not `SelfStateDimensionV1.confidence`,
and explicitly excluded from training per above). `SelfStateDimensionV1.confidence` is
never read anywhere in `services/orion-spark-introspector/app/`.

`DROPPED_DIMENSIONS = frozenset({"policy_pressure", "uncertainty"})` (`inner_state.py:89`)
confirms the already-shipped PR #888 decontamination; `ENCODER_EXCLUDED_FELT` and
`SEEDV4_THEATER_FELT` (`inner_state.py:91-101`) further narrow what seed-v4 actually trains
on beyond what's merely collected.

### Confirmed against the live active encoder manifest

`/mnt/telemetry/models/phi/encoders/active/manifest.json` (`encoder_id:
"phi-encoder:v20260710-seedv4-full"`, `features_version: "seed-v4"`, `hidden_dim: 16`,
`latent_dim: 8` — matches live `PHI_ENCODER_HIDDEN_DIM=16` / `PHI_ENCODER_LATENT_DIM=8` in
`services/orion-spark-introspector/.env:160-161`) lists `input_features`:

```
agency_readiness, execution_pressure, reasoning_pressure, overall_intensity,
recall_gate_fired, reasoning_present, execution_load, reasoning_load
```

Three of these (`agency_readiness`, `execution_pressure`, `reasoning_pressure`) are
`SelfStateV1.dimensions[*].score` values; `overall_intensity` is a top-level SelfStateV1
score-like field; the remaining four are cognitive features sourced from execution-trajectory
telemetry, not `SelfStateV1` at all. No `confidence` field, of any dimension, appears in the
live encoder's trained feature set.

### Recommendation

**No phi-corpus impact statement is needed for Phase 1's per-dimension `confidence` formula
change** — the encoder's input vector never reads `SelfStateV1.dimensions[*].confidence` at
any point in the traced path (`handle_self_state` → `build_inner_state_features` →
`feature_vector_from_inner` → `forward`). Changing the confidence formula changes a field the
encoder structurally cannot see.

The `policy_pressure` removal (Phase 0, already shipped) is confirmed retroactively safe by
the same evidence: `policy_pressure` is already in `DROPPED_DIMENSIONS` and was never in
`FELT_DIMENSIONS`, so it was never part of any encoder's trainable set even before removal.

The one thing Phase 1/2 changes *do* touch that the encoder reads directly is `.score` for
`agency_readiness`, `execution_pressure`, and `reasoning_pressure`, plus whatever feeds
`overall_intensity` — if a later phase changes the **scoring formula** (not just confidence)
for those three dimensions, or `overall_intensity`'s aggregation, *that* would need a
retrain/version-pin decision. Per the current plan text, Phase 1's scoring-math change is
scoped to the `channel_dimension_map` double-counting fix (11 channels feeding capability
dimensions, not `agency_readiness`/`execution_pressure`/`reasoning_pressure` directly) — but
this should be re-checked against the corrected Phase 1 changes before Phase 1 ships, since
that's a `.score`-level change and the encoder's structural blindness only covers
`.confidence`, not `.score`.

## Addendum (orchestrator re-check, post Phase 1 implementation, 2026-07-12)

The re-check this section asked for was done against Phase 1's actual landed diff to
`config/self_state/self_state_policy.v1.yaml`. **The caveat above is confirmed, not
theoretical**: `execution_load: execution_pressure` and `reasoning_load: reasoning_pressure`
are 2 of the 11 `channel_dimension_map` entries the double-counting fix removes. Before the
fix, `execution_pressure`'s `.score` came from `max(raw execution_load, diffused
execution_pressure)`; after, it comes from the diffused capability channel alone. Same
mechanism for `reasoning_pressure` (`reasoning_load` removed). Both are direct phi-encoder
training inputs per the live active manifest's `input_features`. `agency_readiness` (a
direct function of `execution_pressure`/`reliability_pressure`/`coherence`/`uncertainty`/
`resource_pressure`) and `overall_intensity` (the weighted average over all dimensions,
`execution_pressure`/`reliability_pressure`/`resource_pressure` included) both change value
transitively for the same reason — `reliability_pressure` loses `execution_friction` and
`failure_pressure`'s direct entries the same way.

**Verdict: this is a real `.score`-level distribution shift on all four of the phi
encoder's live-manifest `SelfStateV1`-sourced input features, not a no-op.** The fix is
correct — it repairs a genuine bug where topology edge weights were functionally inert —
but the seedv4 encoder was trained on the pre-fix (double-counted, raw-dominated)
distribution of these four features. This is exactly the case design invariant 5 in
`docs/superpowers/specs/2026-07-12-self-state-mesh-substrate-redesign-design.md` exists
for: a dimension-formula change needs an explicit retrain / version-pin / accept-and-document
decision before going live, not a silent deploy. **Not decided here — this needs Juniper's
call**, not an orchestrator default, given the phi corpus was already flagged fragile
(relaxed 6/8 requirement) before this change. Options, roughly in ascending cost:

1. **Accept and document**: ship the fix, log/monitor phi score distribution post-deploy,
   accept some drift given the corpus was already loosely calibrated. Cheapest, but risks
   quietly degrading an already-fragile signal with no detection mechanism in place today.
2. **Version-pin**: gate the encoder to flag/reject inputs whose feature distribution has
   drifted past some threshold from its training distribution, surfacing degradation instead
   of silently absorbing it. Requires building a drift check that doesn't exist yet.
3. **Retrain**: recompute the phi training corpus against post-fix `SelfStateV1` output
   before deploying the self-state fix live, so the encoder's weights match the corrected
   distribution from day one. Most correct, most expensive (full corpus regeneration).

**Recommendation for sequencing, not for which option**: the fix itself (already committed
to code, fully tested) does not need to be reverted or held — it is correct. What needs to
wait is *restarting `orion-self-state-runtime` in production* until one of the three options
above is chosen. Code-complete and deploy-ready are different gates here.
