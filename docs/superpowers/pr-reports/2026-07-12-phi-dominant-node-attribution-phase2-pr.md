## Summary

- Phase 2 of the inner-state unification plan: `PhiIntrinsicRewardV1.dominant_node`/`dominant_node_reason`, naming the real hardware node most salient this tick — sourced from Phase 1's `SelfStateV1.dominant_attention_target_details`.
- Phase 1 isn't deployed yet, so the plan's Phase 2 start-gate ("does the dominant target actually vary tick-to-tick") was checked against the pre-existing bare-string `dominant_attention_targets` field (already live in production) instead — confirmed real variance, but also surfaced something the plan hadn't anticipated: a `target_kind == "system"` entry wins the #1 salience slot most ticks, so filtering only the two synthetic pseudo-nodes (as planned) wasn't enough.
- Code review flagged a third candidate gap (a parameterized `harness_closure:<uuid>` node). Investigated and disproved — traced the actual write path and confirmed it goes to a completely disconnected subsystem, so it can never reach this filter. Documented as a general (not this-specific) known limitation instead of "fixed."

## Outcome moved

`PhiIntrinsicRewardV1` can now name which physical node is under stress, not just that something is — the first genuinely embodied (spatially-attributed) signal in the phi/self-state pipeline.

## Current architecture

`handle_self_state()`'s encoder block (where golden phi was wired in earlier today) already had `ss` (the parsed `SelfStateV1`) in scope. `ss.dominant_attention_target_details` (Phase 1, merged) is a salience-ordered list of `AttentionTargetSummaryV1` spanning node/capability/system target kinds — attention's own sort (`orion/attention/field_attention/builder.py:34,50`) doesn't segregate by kind.

## Architecture touched

`orion/schemas/telemetry/phi_encoder.py`, `services/orion-spark-introspector/app/worker.py`. No new service, no new dependency, no new config mount.

## Files changed

- `orion/schemas/telemetry/phi_encoder.py`: additive `dominant_node: Optional[str]`, `dominant_node_reason: Optional[str]` on `PhiIntrinsicRewardV1`.
- `services/orion-spark-introspector/app/worker.py`: new `_dominant_hardware_node()` (filters `target_kind == "node"` AND excludes `_SYNTHETIC_PSEUDO_NODES`), wired into the existing encoder-tick reward construction. Extended code comment documenting the blacklist's known limitation and the investigated-and-ruled-out `harness_closure` candidate.
- `services/orion-spark-introspector/tests/test_phi_reward_emit.py`: 3 unit tests on `_dominant_hardware_node()`, 1 end-to-end test through `handle_self_state()`.
- `orion/self_state/inner_state_registry.py`, `orion/self_state/README.md`, `services/orion-spark-introspector/README.md`, `docs/superpowers/plans/2026-07-12-inner-state-unification-plan.md`: registry notes + documentation + plan checklist updated.

## Schema / bus / API changes

- Added: `PhiIntrinsicRewardV1.dominant_node`, `.dominant_node_reason` (additive, both `Optional[str] = None`).
- Removed: none.
- Compatibility notes: only one other constructor call site exists (`services/orion-sql-writer/tests/test_phi_reward_sql_shape.py`), unaffected — both new fields optional, and the reward is stored as a JSON blob there, not per-field columns.

## Env/config changes

None.

## Tests run

```text
PYTHONPATH=. :services/orion-spark-introspector pytest services/orion-spark-introspector/tests -q
138 passed, 1 skipped
1 pre-existing FAILED (test_inner_features_settings_defaults -- confirmed via git stash
against clean main, unrelated to this change: asserts a settings default that
doesn't match another branch's not-yet-merged work)

PYTHONPATH=. pytest tests/test_inner_state_registry_gate.py -q
8 passed

python scripts/check_inner_state_registry.py
inner_state_registry gate OK (9 entries checked)
```

## Evals run

None applicable.

## Docker/build/smoke checks

Not run — not deployed.

## Review findings fixed

- Finding: attention's `dominant_targets` is salience-sorted across node/capability/system kinds together (confirmed by reading `orion/attention/field_attention/builder.py`), and live data showed a system-kind entry (`field:recent_perturbations`) winning the #1 slot most ticks — the original plan's "filter the two pseudo-nodes" was insufficient.
  - Fix: `_dominant_hardware_node()` filters `target_kind == "node"` in addition to the pseudo-node exclusion.
  - Evidence: new test `test_dominant_hardware_node_skips_system_and_pseudo_nodes` asserts the correct node wins over a system entry, a capability entry, and a pseudo-node entry all present in the same fixture.
- Finding (review, investigated and disproved rather than fixed): a candidate third gap — `orion-substrate-runtime`'s `harness_closure:<uuid>` prediction-error nodes might evade the hardcoded pseudo-node blacklist.
  - Investigation: traced `_write_prediction_error_node` to confirm it writes exclusively to the cognitive-substrate graph store (`ConceptNodeV1`/`store.upsert_node`), then confirmed `orion-field-digester` (the actual producer of `FieldStateV1.node_vectors`, what this filter operates on) has zero references to that graph store anywhere in its codebase. These nodes structurally cannot reach `FieldAttentionTargetV1`/`dominant_attention_target_details` — the review's claimed connection didn't hold. Documented as a ruled-out candidate, not silently dropped.
  - Follow-up named, not built: switching `_SYNTHETIC_PSEUDO_NODES` from a hardcoded blacklist to a whitelist against `config/biometrics/node_catalog.yaml` (the authoritative real-node list) would close the general "could a future pseudo-node go unnoticed" risk, but needs a new config mount + settings field on a service that has neither today — out of scope for this phase, documented in the code comment.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml \
  up -d --build spark-introspector
```

Not run — deliberately left for Juniper. Depends on Phase 1 also being deployed first (this reads `SelfStateV1.dominant_attention_target_details`, which only populates once `orion-self-state-runtime` is rebuilt with Phase 1's change).

## Risks / concerns

- Severity: low
- Concern: `_SYNTHETIC_PSEUDO_NODES` is a maintained blacklist (2 known strings), not a whitelist against the authoritative node catalog — a genuinely new pseudo-node type added to the field topology in the future would silently evade this filter until someone adds it here too.
- Mitigation: documented explicitly in the code comment with the exact fix path (`NodeCatalog` whitelist) and why it wasn't done now (new infrastructure needed). Not a live risk today — no third pseudo-node type currently exists in `config/field/orion_field_topology.v1.yaml`.
- Severity: low
- Concern: like Phase 1, live-traffic verification (does `dominant_node` actually name real, varying nodes correctly in production) is blocked on deployment of both this phase and Phase 1.
- Mitigation: explicitly left as an open acceptance item in the plan doc, not assumed complete.

## PR link

Branch pushed: `feat/phi-dominant-node-attribution`
