# Self-State & mesh-aware metrics substrate redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan phase-by-phase. Steps
> use checkbox (`- [ ]`) syntax for tracking. **Do not start a phase whose "Gate" is unmet.**

**Goal:** Make `SelfStateV1` (Layer 6) and its mesh-embodiment extension honest — every
dimension has a real producer, confidence/reasons carry real signal, node identity survives
into the self-report, and the two parallel drive-pressure computations are reconciled by
measurement rather than left to silently diverge.

**Design source:**
`docs/superpowers/specs/2026-07-12-self-state-mesh-substrate-redesign-design.md` (full
rationale, invariants, non-goals — this plan is task-focused and assumes that doc's
context).

**Evidence appendix:** `docs/notes/2026-07-12-metrics-swamp-arsonist-review.md` (live flag
states, verdict tables, the mesh-weighting addendum). Also updated as part of this
initiative: `services/orion-substrate-runtime/README.md` §"Downstream of this service:
Layers 6-11".

---

## Context for the implementer (read before starting)

- `SelfStateV1` schema: `orion/schemas/self_state.py`. Builder: `orion/self_state/builder.py`.
  Scoring math: `orion/self_state/scoring.py`. Prediction/surprise:
  `orion/self_state/prediction.py`. Policy config (tuning surface, not code):
  `config/self_state/self_state_policy.v1.yaml`.
- Live service: `orion-self-state-runtime` (port 8118), polls every 2s
  (`SELF_STATE_POLL_INTERVAL_SEC`), consumes `substrate_field_state` +
  `substrate_attention_frames`, writes `substrate_self_state` + publishes
  `orion:substrate:self_state`.
- ~15 known consumers of `substrate_self_state` exist (policy/proposal/execution-dispatch/
  feedback/consolidation-runtime, spark-introspector, `felt_state_reader.py` →
  cortex-exec/thought/equilibrium/hub). Any schema change must be additive-first
  (design invariant 4) — do not remove a field in the same patch that adds its replacement.
- Mesh topology: `config/field/orion_field_topology.v1.yaml`. Node catalog:
  `config/biometrics/node_catalog.yaml`. Diffusion (where edge weights are actually
  applied): `services/orion-field-digester/app/digestion/diffusion.py`.
- `DriveEngine`'s proven leaky-decay/deviation-gate math (the thing Phase 2 ports) lives in
  `orion/spark/concept_induction/drives.py` and `orion/autonomy/signal_tension.py` — do not
  reinvent this math, port it.

---

## Phase 0 — Hygiene (zero-risk, execute decisions already on record)

**Gate to start:** none. This can start immediately.

- [ ] Remove `policy_pressure` dead dimension: `orion/schemas/self_state.py` (enum),
      `orion/self_state/builder.py:42,218` (`ALL_DIMENSION_IDS`, hardcoded score),
      `config/self_state/self_state_policy.v1.yaml:22` (weight entry). Grep all consumers
      for `.dimensions["policy_pressure"]` / `.dimensions.get("policy_pressure")` first —
      none found in the 2026-07-12 investigation, re-verify before removing.
- [ ] Replace templated `reasons` with real content generated from `dominant_evidence`:
      `orion/self_state/builder.py:264-269`. `reasons = [f"{ch} contributed {v:.2f}" for
      ch, v in top_evidence]` or equivalent — no new inputs required, `dominant_evidence`
      already computes the data needed.
- [ ] Flip `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` live in
      `services/orion-spark-concept-induction/.env` (decision made 2026-07-11, never
      applied — confirmed still `true` as of 2026-07-12). Restart the service.
- [ ] Commit the endogenous-origination NO-GO verdict (2026-07-08 gate measurement: drift
      0.0026 vs. required ≥0.03, co-activation 0.0004 vs. required ≥0.10) into
      `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md`. Currently
      sitting uncommitted in `/tmp/autonomy-gate/report.md` only.
- [ ] Delete or clearly deprecate `orion/autonomy/endogenous_origination.py`, citing the
      NO-GO verdict in the commit message.
- [ ] Rename the `endogenous_runtime.py` (`services/orion-cortex-exec/app/`, live, adopted)
      vs. `endogenous_origination.py` (dead, NO-GO) naming collision — pick whichever name
      is less likely to be confused with the other; this is the exact naming collision that
      caused confusion earlier in this investigation.

**Tests:** `pytest orion/self_state -q` (or wherever L6 tests live —
`tests/test_self_state_builder*.py`, `tests/test_self_state_policy_loader.py`), full
`orion/autonomy/tests -q`, `services/orion-spark-concept-induction` test suite if present.

**Acceptance:** dead dimension gone from schema + policy + builder in the same patch; every
dimension's `reasons` on a live tick differs from every other dimension's; concept-induction
flag matches the documented decision; NO-GO verdict is in a tracked file, not `/tmp`.

---

## Phase 1 — Evidence integrity

**Gate to start:** Phase 0 merged.

- [ ] Compute real per-dimension `confidence` from evidence density (contributing-channel
      count, freshness, cross-channel agreement) instead of the current global proxy
      (`0.5 + 0.5×len(dominant_targets)/5`, identical across all 13 dims today). Ship as an
      additive field (e.g. `confidence_v2` or bump `self_state_policy_id`) alongside the
      existing formula for one release before deprecating the old one — design invariant 4.
      Files: `orion/self_state/scoring.py`, `orion/self_state/builder.py:257-270`.
- [ ] Change `collect_field_channel_pressures` to read `capability_vectors` only — drop
      `list(field.node_vectors.values())` from the merge loop. This is the hard prerequisite
      for Phase 3; do not skip or reorder past it. File: `orion/self_state/scoring.py:54-63`.
- [ ] Re-measure `resource_pressure` (and any other multi-channel `max()`-aggregated
      dimension) on live traffic before/after this change — confirm the scarcity-economy
      "saturated at 1.0" finding either resolves or is understood to persist for a different
      reason.
- [ ] Decide `transport_integrity`'s fate: either add it to `ALL_DIMENSION_IDS` and
      `dimension_weights` so it actually contributes to `overall_intensity`, or document
      explicitly (docstring + README) that it's display-only by design. Currently neither.

**Tests:** builder/scoring unit tests updated for the new confidence formula and the
node-vectors read-path removal; a live-replay comparison script (before/after) for
`resource_pressure` saturation, saved under `/tmp/self-state-phase1/` per the backfill
protocol if it touches live data.

**Acceptance:** confidence values differ across dimensions on real traffic; no regression in
`overall_intensity`/`overall_condition` bucketing on replayed historical ticks (or an
explained, deliberate change); `transport_integrity` has a stated, single fate.

---

## Phase 2 — Continuity / memory

**Gate to start:** Phase 1 merged.

- [ ] Port the leaky-decay/EWMA baseline tracker from `orion/spark/concept_induction/drives.py`
      / `orion/autonomy/signal_tension.py` into L6's trajectory tracking, as an additive
      field (`dimension_baseline`, `dimension_deviation_z`) alongside the existing
      single-previous-tick `dimension_trajectory`. Do not remove the old field.
- [ ] Add `ticks_since_meaningful_change: int` (top-level on `SelfStateV1`, not
      per-dimension), computed from consecutive `dimension_trajectory` emptiness.
- [ ] Re-verify the prediction/surprise loop (`orion/self_state/prediction.py`) still
      behaves correctly against the new baseline-tracked trajectory — `overall_surprise`'s
      max-based convention should not need to change, but confirm no double-counting
      between the new deviation-z signal and the existing prediction-error signal.

**Tests:** port the existing `DriveEngine` leaky-decay test patterns
(`orion/spark/concept_induction/tests/test_drives_leaky.py` as reference) to L6's tracker;
extend `orion/self_state` prediction tests to cover the new fields.

**Acceptance:** a dimension that's genuinely flat for N ticks reports
`ticks_since_meaningful_change >= N` honestly; the new baseline tracker doesn't regress
`overall_surprise` behavior on replayed historical ticks.

---

## Phase 3 — Mesh embodiment

**Gate to start:** Phase 1 merged (the `capability_vectors`-only read path must be live
before any node comes online, or the existing saturation problem gets worse, not better).

- [ ] Enable biometrics on Atlas (`orion-biometrics` agent, `NODE_NAME=atlas`,
      `PUBLISH_BIOMETRICS_GRAMMAR=true`) using the existing `node_catalog.yaml` entry — no
      new catalog work needed, it's already there.
- [ ] Enable biometrics on Circe the same way (`NODE_NAME=circe`); verify
      `expected_online: false` behavior — Circe going quiet should not spuriously raise
      `reliability_pressure`/hurt `coherence`, confirmed via `expected_offline_suppression`
      already being a stabilizing channel (weight 0.30) in
      `config/self_state/self_state_policy.v1.yaml`.
- [ ] Thread node provenance from `apply_diffusion`'s `edge.source_id`
      (`services/orion-field-digester/app/digestion/diffusion.py:8-12`) through to
      `capability_vectors`, then into `dominant_evidence` at the L6 evidence-building step
      (`orion/self_state/builder.py:264-269` / `evidence_for_dimension`) — e.g.
      `atlas:gpu_pressure=0.91` instead of bare `gpu_pressure=0.91`.
- [ ] With real multi-node data available, revisit the capacity-pressure vs.
      continuity-threat distinction (design invariant 7) as a follow-up design note — do
      not build a new schema field for this speculatively; let the real data (does Athena
      ever actually go down? what does that look like in the existing dimensions?) inform
      whether it's worth a dedicated field or better expressed as existing dimensions plus
      node-attribution making the distinction legible without new schema.

**Tests:** `services/orion-biometrics/tests` node-catalog/emitter tests (already exist per
the 2026-05-24 node-scoped-grammar-ingress PR — confirm they still pass); new integration
test confirming a synthetic Atlas-sourced channel reaches `SelfStateV1.dimensions[...].
dominant_evidence` with node attribution intact.

**Acceptance:** Atlas/Circe biometrics visible in live field state; `resource_pressure`/
`execution_pressure` on Athena's own self-state don't spuriously spike from Atlas/Circe
activity in ways the topology edge weights wouldn't predict; Circe's normal off-cycles
produce no coherence/reliability alarm; at least one live self-state tick's evidence names
a specific non-Athena node.

---

## Phase 4 — Drive unification

**Gate to start:** Phase 2 merged (wants a stable L6 trajectory signal before comparing
downstream drive computations against it).

- [ ] Log `DriveEngine`'s (`orion/spark/concept_induction/drives.py`) and
      `AutonomyStateV2`'s (`orion/autonomy/reducer.py`) `drive_pressures` side-by-side on
      the same live traffic window (both already key off the same six-drive taxonomy and
      share `orion/autonomy/signal_drive_map.py`'s config).
- [ ] Based on measured divergence: merge onto one shared pressure store if low, or
      document the measured divergence and the reason for keeping them separate if high.
      This spec's recommended default (§Phased plan table) is to merge if divergence is
      low — but the data decides, not the recommendation.
- [ ] Resolve `CLUSTER_ROLE_WEIGHTS` (`services/orion-biometrics/.env`) and
      `orion-state-service`'s aggregation the same way — trace real consumption
      (`orion-hub`, `orion-state-service`) to determine whether this is a legitimate
      separate "ops health" concern or a fourth duplicate of the same node-weighting
      question, per the arsonist doc's mesh addendum.

**Tests:** whatever the merge target requires — if unifying onto `DriveEngine`'s store,
extend its test suite to cover `AutonomyStateV2`'s evidence-compiler-shaped inputs; if
keeping separate, no new tests, just the documented decision.

**Acceptance:** one of (a) a single shared drive-pressure store with both original call
sites reading from it, or (b) a committed doc explaining the measured divergence and why
separation is intentional. No third outcome (silent, unexamined continued duplication).

---

## Phase 5 — Ladder + phi closure

**Gate to start:** research halves can start anytime, in parallel with any other phase;
implementation halves gated on their own research landing.

- [ ] Trace whether `orion-dream/compaction_applier.py` (or anything else past L9) actually
      changes real behavior when it reads policy/execution-dispatch frames, or whether the
      L7-L11 ladder terminates in Hub debug tiles. Read-only research, no code change
      required for this step alone.
- [ ] For every dimension-formula change made in Phases 1-2, produce an explicit phi-corpus
      impact statement (retrain / version-pin / no-op) — check `_phi_from_self_state`
      (`services/orion-spark-introspector/app/worker.py:2506+`) to confirm whether it trains
      on raw channels or synthesized dimension scores, which determines the actual risk
      level.
- [ ] Ship the evidence-trail debug surface: extend
      `services/orion-hub/scripts/substrate_self_state_routes.py` to expose the full
      `merged_channels` → dimension → node lineage for a given tick, not just final scores
      — the closing observability layer over the whole redesign.

**Tests:** none new for the tracing step (research); phi-corpus decision gets whatever test
its chosen path requires (a retrain eval, or a version-pin regression test); debug route
gets a basic endpoint test.

**Acceptance:** L9-L11's real-world reach is a documented fact, not an open question;
every formula change from this initiative has a recorded phi-corpus decision; a human (or
Orion) can pull one self-state tick's full evidence trail, node-attributed, through the
debug surface.

---

## Full gate before calling the initiative done

```bash
pytest orion/self_state -q          # or wherever L6 tests resolve
pytest orion/autonomy/tests -q
pytest services/orion-spark-concept-induction/tests -q
pytest services/orion-biometrics/tests -q
pytest services/orion-hub/tests -k self_state -q
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```

Run the code-review skill in a subagent per phase before merging, not just once at the end
— five phases means five review passes, per CLAUDE.md §12. Each phase gets its own PR and
PR report per the standard template; do not bundle phases into one giant PR.
