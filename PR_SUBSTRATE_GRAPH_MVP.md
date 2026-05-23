# Substrate Graph MVP: shared grammar + 7-day experiment harness

## Summary

This PR lands the **Substrate Graph MVP** — a tiny shared-grammar layer that lets multiple Orion organs perturb one common reality model instead of each owning its own bespoke schema. It is *additive*; no existing organ code paths are modified.

Core stance: **organs do not own reality. Organs emit transformations over a shared substrate.**

What this is:
- A shared substrate grammar
- A field-stabilization scaffolding layer
- A common reality model for multiple Orion organs

What this is **not**: a memory system, an ontology platform, an agent framework, an AGI stack, a GraphDB project.

## What's inside

### Phase 1–2, 4 — Schema kernel (`orion/schema_kernel/`)
- `ConceptAtomV1` with a closed set of **12 atom kinds** that are *invariants*, not domain nouns: `signal`, `constraint`, `attention`, `state`, `change`, `relation`, `context`, `agency`, `evidence`, `gradient`, `persistence`, `boundary`.
- `ConceptRelationV1`: source, predicate, target, `weight ≥ 0`, `polarity ∈ [-1, 1]`.
- `CompositeV1`: small kernel-level bundle.
- Canonical gradient vector: `salience`, `contradiction`, `novelty`, `coherence`.
- `SchemaKernelRegistry` + `default_registry()` seed.
- `validate_atom` / `validate_relation` / `validate_composite`.

### Phase 3, 5, 6, 8 — Substrate molecules (`orion/substrate/`)
- `SubstrateMoleculeV1` — every organ emits the *same* shape: `{molecule_id, molecule_kind, atoms, relations, gradients, provenance, payload, created_at, last_touched_at}`.
- Operators that mutate gradients only:
  - `reinforce_molecule` — salience + coherence up.
  - `decay_molecule` — slow loss when untouched.
  - `amplify_contradiction` — contradiction + salience up.
  - `stabilize_coherence` — coherence up, contradiction down.
- `find_resonant_molecules(gradients, threshold)` — O(n) traversal, descending by summed gradient pressure.
- `MoleculeJsonlStore` — append-only JSONL + in-memory mirror; round-trips through Pydantic.

### Phase 7 — Organ integration
- `orion/mind/substrate_emit.py`: `emit_observation`, `emit_claim`.
- `orion/autonomy/substrate_emit.py`: `emit_pressure`, `emit_contradiction`.
- Both produce identical `SubstrateMoleculeV1` shape — proves the no-bespoke-schemas claim.
- Existing chat / autonomy code is untouched.

### Phase 9 — 7-day experiment harness (`orion/substrate/experiment/`)
- `SubstrateExperimentHarness` records `emit`, `traversal`, gradient deltas (reinforce/decay/contradiction/stabilize).
- `compute_daily_rollup` → `DailyMetricsV1`:
  - molecule_count, organ_coverage
  - gradient_distribution (min/mean/max for each canonical key)
  - resonance_hits, cross_organ_reuse
  - reinforcement_count, decay_count
  - contradiction_clusters (grouped by shared atom signature)
  - orphan_molecule_rate
  - substrate_health_score = `coherence + cross_organ_rate + reinforcement_rate − contradiction − orphan_rate`, clamped to `[-1, 1]`.
- `write_daily_rollup` → `runs/YYYY-MM-DD.json`.
- `generate_week_report` → one Markdown verdict answering: *Did the shared substrate become more useful than bespoke organ state?*

## What this PR deliberately avoids

- No giant ontology.
- No tensors, no embeddings.
- No GraphDB integration.
- No autonomous agents, no symbolic reasoner.
- No dashboards, no ML, no causal claims.
- No invented atom types beyond the 12-kind closed set.
- No changes to any existing organ code paths.

## Architecture rules respected

| Rule | Status |
|------|--------|
| Atoms are invariants, not domain nouns | ✅ 12 kinds, all dimensions of interaction |
| Same molecule shape across organs | ✅ mind + autonomy both emit `SubstrateMoleculeV1` |
| Operators mutate gradients only | ✅ four operators, no structural rewrites |
| Substrate owns shared reality | ✅ `MoleculeJsonlStore` is the source of truth |
| Inspectable | ✅ JSONL on disk, JSON daily rollups, Markdown report |
| Evolvable | ✅ kernel sits beside (not on top of) the existing `orion.substrate` graph layer |

## Files likely to touch in follow-up work

- A bridge between `MoleculeJsonlStore` and the existing `SubstrateGraphMaterializer`.
- A scheduler that fires `compute_daily_rollup` once per UTC day.
- Real wiring of `mind_emit.emit_*` into `orion/chat.py` and of `autonomy_emit.emit_*` into the existing pressure loop. (Done intentionally as a separate change to keep this PR additive.)

## Non-goals (out of scope here)

- Replacing or refactoring `orion/substrate/materializer.py`, `dynamics.py`, or `relational/`.
- Persisting molecules into Postgres or the graph store.
- Embedding-based traversal.
- Cross-organ reasoning beyond gradient resonance.

## Test plan

- [x] `venv/bin/pytest tests/test_substrate_kernel_atoms.py tests/test_substrate_kernel_molecules.py tests/test_substrate_kernel_operators.py tests/test_substrate_organ_emit.py tests/test_substrate_experiment_harness.py` — 26 passed in 0.78s
- [x] Existing `orion.substrate` package still imports without error.
- [ ] Run the harness against a real 7-day window and inspect `weekly.md`.
- [ ] Wire `mind_emit` into a chat code path (follow-up PR).
- [ ] Wire `autonomy_emit` into the pressure loop (follow-up PR).

## Acceptance checks (per the spec)

1. ✅ Chat-shaped molecules can be emitted via `mind_emit.emit_observation` / `emit_claim`.
2. ✅ Pressure-shaped molecules can be emitted via `autonomy_emit.emit_pressure` / `emit_contradiction`.
3. ✅ Gradients evolve over time via the four operators.
4. ✅ `find_resonant_molecules` retrieves by gradient pressure.
5. ✅ Both organs share the same `SubstrateMoleculeV1` grammar — verified by `test_substrate_organ_emit::test_both_organs_emit_substrate_molecules`.
6. ✅ No bespoke organ-specific state systems introduced.
