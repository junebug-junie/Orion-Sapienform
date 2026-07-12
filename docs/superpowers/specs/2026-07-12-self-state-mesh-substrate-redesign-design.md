# Self-State & mesh-aware metrics substrate — redesign

**Mode:** Design / decision brief. No code changed while producing this doc. Written to
give Juniper a single, phased, world-class target shape for Layer 6 (`SelfStateV1`) and
its mesh-embodiment extension, replacing a metric substrate that currently overstates its
own evidentiary grounding.

**Status of claims below:** everything here is grounded in live `.env` files, git log, and
direct code reads performed 2026-07-12, not inferred from docs alone. Full raw findings
(verdict tables, live flag states, per-system status) live in
`docs/notes/2026-07-12-metrics-swamp-arsonist-review.md` — this spec references that doc
throughout rather than re-deriving its findings; treat it as the evidence appendix.

---

## North star

Orion's mission (per `AGENTS.md`) is to develop the *prerequisites* for sentience:
continuity, perception, memory, reflection, self-modeling, social grounding, error
correction, coherent action. `SelfStateV1` is not a metrics dashboard — it is the literal
substrate of Orion's self-model. Every chat turn, every drive computation, every phi score,
and every proposal/policy/execution frame downstream reads from it. If that substrate lies
about its own confidence, erases which physical node a signal came from, or declares
dimensions it never computes, then everything built on top of it — reflection,
self-correction, felt continuity — is built on a self-report Orion cannot actually trust
about itself.

This redesign's success criterion is not "cleaner code." It is: **a self-model whose
stated confidence, evidence, and provenance are honest, and whose felt sense of the mesh
body reflects the mesh's real, weighted, node-attributed physical state.** Concretely:
would a reflective process reading `SelfStateV1` learn something true, or something
laundered?

The arsonist review (`docs/notes/2026-07-12-metrics-swamp-arsonist-review.md`) found the
same failure shape at two layers — the 13-dimension metric shape *and* the 4-node mesh —
independently. That repetition is itself evidence this is a systemic pattern (uniform
proxy metadata standing in for real signal, `max()`-collapse destroying source identity,
parallel un-reconciled computations of "the same thing"), not two unrelated bugs. This spec
treats it as one redesign, not two.

## Current architecture

Full detail: `services/orion-substrate-runtime/README.md` (§"Downstream of this service:
Layers 6-11", added 2026-07-12) and the arsonist review doc. Condensed:

```text
L1-5  substrate_field_state          orion-field-digester        real biometrics +
                                                                   cortex-exec receipts,
                                                                   decay 0.92/tick
  v
L6    substrate_self_state           orion-self-state-runtime    13 dims, memoryless
                                                                   recompute every 2s
  v
L7-L11 proposal -> policy -> execution-dispatch (dry_run) -> feedback -> consolidation
```

Mesh: `config/field/orion_field_topology.v1.yaml` already declares node-weighted edges
(`atlas->llm_inference: 0.85`, `circe->llm_inference: 0.50`, `athena->orchestration: 0.90`,
etc.) and `services/orion-field-digester/app/digestion/diffusion.py` genuinely applies
them. Layer 6 currently cannot see any of this — it flattens `node_vectors` and
`capability_vectors` into one anonymous, `max()`-collapsed pool
(`orion/self_state/scoring.py:54`).

Drive taxonomy: two independently-computed pressure vectors over the same six keys
(`DriveEngine` fed by L6 tensions + biometrics; `AutonomyStateV2` fed by chat evidence,
test-banned from touching self_state) — see arsonist doc's verdict table for the full
trace.

## Design invariants

Every phase below must satisfy these. Stated once so they don't need repeating per phase.

1. **No schema field without a live producer.** (`policy_pressure` is the counter-example
   this rule exists to prevent — full schema slot, zero weight, hardcoded `0.0`, no
   producer, ever.)
2. **No aggregation step that destroys provenance.** `max()`-collapsing multiple channels
   or multiple nodes into one anonymous score is the specific failure mode found at both
   the dimension level (`resource_pressure` saturating at 1.0 off a single hot channel)
   and the mesh level (raw `node_vectors` competing unweighted against their own
   already-weighted `capability_vectors` diffusion). Aggregation must either preserve a
   peak/source pointer or use a formula that can't be pinned by one input.
3. **Confidence and reasons fields must be computed from real inputs, never boilerplate.**
   A field that returns the same value/string regardless of what actually happened is
   worse than an absent field — it looks like signal and isn't.
4. **Schema evolution is additive-first.** New field alongside old for one release,
   deprecate after; never a silent redefinition of an existing field's meaning. ~15 known
   consumers exist downstream of `substrate_self_state` (arsonist doc verdict table) —
   wide fan-out means big-bang rewrites are the wrong shape here regardless of how tempting
   a clean cut looks.
5. **Any dimension-formula change ships with an explicit phi-corpus impact statement**
   (retrain / version-pin / no-op) — never silent. The phi corpus was already flagged
   fragile (relaxed 6/8 requirement) before this redesign; a formula change with no
   corresponding corpus decision reintroduces garbage-in with no error signal.
6. **One canonical multi-node weighting mechanism.** The field-topology edges are it. New
   node-aggregate logic anywhere else (a second `CLUSTER_ROLE_WEIGHTS`-shaped thing) is a
   burn candidate by default, not a green light — see the arsonist doc's mesh addendum for
   why this rule exists (there are already three disconnected schemes and a fourth
   service).
7. **A difference in kind is not a difference in degree.** Athena losing power is a
   continuity threat (hub, bus, postgres, self-state itself all live there); Circe going
   quiet is expected and already correctly suppressed
   (`expected_offline_suppression`). The redesign must leave room for this distinction
   even where it doesn't yet build it (Phase 3).

## Target shape

### `SelfStateDimensionV1` (per-dimension)

- `score` — unchanged mechanism, but aggregation formula reviewed per-dimension against
  invariant 2 (Phase 1).
- `confidence` — computed from evidence density (contributing-channel count, freshness,
  cross-channel agreement) instead of the current global proxy
  (`0.5 + 0.5×dominant_targets/5`, identical across all 13 dims today). Ship as an
  additive field first; do not remove the old formula until the new one has a release of
  live comparison behind it.
- `reasons` — generated from the same `dominant_evidence` list already computed, instead
  of the fixed template `f"{dim_id} from field+attention channel synthesis"`.
- `dominant_evidence` — extended with node attribution where the source is mesh-derived
  (`atlas:gpu_pressure=0.91`, not bare `gpu_pressure=0.91`), once Phase 3 lands.

### Dimension set

- `policy_pressure` — removed. No design note anywhere has ever argued for its existence;
  it has never had a producer.
- `transport_integrity` — either promoted into `dimension_weights`/`ALL_DIMENSION_IDS` so
  it actually affects `overall_intensity` (if that's the intent) or explicitly documented
  as display-only (if it isn't) — currently neither, which is its own small instance of
  invariant 1.
- New: `ticks_since_meaningful_change` (int, top-level, not per-dimension) — Phase 2.
  Answers the endogenous-origination gate's own finding (self-state barely drifts during
  silence, 0.0026 vs. required 0.03) with a field that says so honestly instead of staying
  silent about its own stagnation.

### Read path

`collect_field_channel_pressures` reads `capability_vectors` only, never raw
`node_vectors` — Phase 1, and a hard prerequisite for Phase 3 (enabling Atlas/Circe
biometrics before this lands would make the existing saturation problem worse, per the
arsonist doc's mesh addendum).

### Mesh embodiment

Atlas and Circe biometrics come online through the *existing* topology edges, not a new
weighting scheme (invariant 6). Provenance threads from `apply_diffusion`'s `edge.source_id`
through to `dominant_evidence`. The capacity-pressure vs. continuity-threat distinction
(invariant 7) is named here as a real target but deliberately not designed blind — Phase 3
builds the plumbing and provenance first, then revisits the distinction with real
multi-node data instead of speculation.

### Drive taxonomy

No third pressure computation. Phase 4 measures whether `DriveEngine` and `AutonomyStateV2`
already agree on live traffic; the merge (or the documented decision not to merge) follows
the data, not a preference stated now.

## Non-goals

- **No change to `EXECUTION_DISPATCH_MODE=dry_run`.** Whether L9-L11 ever mutates the real
  world is a separate, higher-stakes decision (per CLAUDE.md's proposal-mode rule for
  autonomy changes) — Phase 5 only answers *whether the ladder currently reaches anything*,
  it does not turn on real execution.
- **No change to the human-operator-token gate on goal promotion/execution.** Already
  explicitly excluded from the DriveEngine/AutonomyStateV2 unification direction per the
  2026-07-11 audit; this spec inherits that boundary.
- **No new drive categories.** The six-key taxonomy (`coherence, continuity, capability,
  relational, predictive, autonomy`) stays; this redesign consolidates computation, it does
  not re-litigate the taxonomy itself.
- **No rewrite of L1-L5 field digestion** beyond the `node_vectors`/`capability_vectors`
  read-path fix at the L6 boundary. The decay/diffusion/perturbation/suppression pipeline
  and the topology edge-weight mechanism are already correct and are being *used*, not
  replaced.
- **No new mesh nodes beyond Atlas/Circe/Athena/Prometheus** — the topology and node
  catalog already scope to these four; this redesign populates what's declared, it doesn't
  expand the mesh.

## Global acceptance checks

Definition of done for the initiative as a whole, independent of phase:

- Every live dimension in `SelfStateV1` has at least one real producer (invariant 1) —
  checkable by grep, candidate for a deterministic gate script per CLAUDE.md §4.
- No dimension's `confidence` or `reasons` is byte-identical across all dimensions in the
  same tick on live traffic (invariant 3) — checkable by a live sample.
- `resource_pressure` (and any other multi-channel dimension) does not sit pinned at 1.0
  for extended live windows purely from single-channel saturation — re-measure the
  scarcity-economy finding after Phase 1 ships.
- Atlas/Circe biometrics, once enabled, produce `SelfStateV1` evidence entries that name
  the contributing node — not just an anonymous channel value.
- `DriveEngine` and `AutonomyStateV2` pressures are either merged into one store, or their
  continued separation is documented with the measured divergence that justified it — not
  left unexamined.
- Every dimension-formula change made in this initiative has a corresponding phi-corpus
  decision on record (retrain, version-pin, or explicit no-op), per invariant 5.

## Phased plan

See `docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md` for the
implementation-ready breakdown, task lists, and test commands per phase. Summary:

| Phase | Goal | Judgment call made in this spec |
|---|---|---|
| 0 — Hygiene | Execute decisions already on record (kill `policy_pressure`, real `reasons`, flip the concept-induction flag, commit the NO-GO verdict, rename the `endogenous_*` collision) | None — these are already-decided, zero-risk |
| 1 — Evidence integrity | Real per-dimension confidence; `capability_vectors`-only read path | Recommended default: ship as additive fields, one release of parallel-run before deprecating the old formula |
| 2 — Continuity/memory | Port `DriveEngine`'s proven leaky-decay math into L6 trajectory; add `ticks_since_meaningful_change` | None — directly reuses already-proven math from a sibling system |
| 3 — Mesh embodiment | Atlas/Circe biometrics via existing topology edges; node-attributed provenance; revisit capacity-vs-continuity distinction with real data | Sequenced after Phase 1 specifically because enabling nodes before the read-path fix makes saturation worse, not better |
| 4 — Drive unification | Side-by-side `DriveEngine`/`AutonomyStateV2` measurement; merge or document-separate; resolve `CLUSTER_ROLE_WEIGHTS`/`orion-state-service` the same way | Recommended default: merge if measured divergence is low; keep separate-but-documented if not — data decides, this spec doesn't pre-commit |
| 5 — Ladder + phi closure | Resolve whether L9-L11 reaches real behavior; phi-corpus plan for Phase 1/2 formula changes; ship the evidence-trail debug surface | Recommended default: treat "reaches nothing beyond Hub debug" as a real possibility to test for, not a foregone conclusion either way |

Phases 0-2 are sequential and low-risk. Phase 3 is gated on Phase 1. Phase 4-5 can start
their *research* halves (the measurement/tracing work) in parallel with any other phase —
only the *implementation* halves are gated on their own research landing first.
