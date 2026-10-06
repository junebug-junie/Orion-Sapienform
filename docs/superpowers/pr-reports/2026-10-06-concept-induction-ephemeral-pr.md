## Summary

Follows #2508 (built on it; #2508 merged to main while this was in progress, so this targets main). Unified-turn latency L6 step 2 (spec `docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md`, corrections #1 and #4). Step 1 (decay fix, #2504) is merged and was verified live before this.

- Every stance build used to write Orion's concept nodes back into the graph. That write made the store think something changed, so it reloaded the whole graph from Falkor a second time in the same build (about 4.5 s). The concept producer now keeps its copy for the turn only and writes nothing. One reload per build instead of two.
- The concept producer read from its own private copy of the graph that was loaded once when the service started and never refreshed. It now reads the stance layer's own store, which the layer has just refreshed, so new concepts show up and no second Falkor connection is opened.
- Since the concept producer's copies and the stored nodes are now both in the result, each concept is shown once; the stored node's values win.
- The relationship and juniper anchors were rebuilt every turn: a producer that legitimately had nothing to add (autonomy) was never marked as freshly pulled, so the layer judged freshness by the seed concepts' day-old timestamp. A completed pull with nothing to add now counts as fresh for its 300 s window. Failures still retry next turn.

## Outcome moved

- Rehydrates per stance build: 2 -> 1 (pinned by `test_stance_build_does_one_rehydrate_and_no_writes`; the same build with the old write-through tier measures 2 in `test_write_through_tier_is_what_cost_the_second_rehydrate`).
- Cold fan-out for relationship/juniper: every build -> once per 300 s. Live evidence of the bug, read-only: all 7 shared-spine builds in `orion-athena-cortex-exec-background` over the last 6 h logged `cold_anchors=['relationship', 'juniper']`.
- The concept re-save that undid activation decay (correction #1) is gone.

## Current architecture

`CognitiveUnificationLayer.beliefs_for_stance` snapshots the durable store, fans out cold producers, materializes write-through producers into the durable store and others into a per-call ephemeral store, then snapshots again and concatenates durable + ephemeral nodes per anchor. `concept_induction` was `CONCEPT_INDUCED` (write-through) and read a module-level Falkor store hydrated once at boot.

## Architecture touched

- `orion/substrate/relational` (registry tier, layer, concept adapter). Live registry `orion/cognition/projection_builder.py`; fallback registry `services/orion-cortex-exec/app/chat_stance.py`. No service boundary, bus, or schema change.

## Files changed

- `orion/substrate/relational/registry.py`: `CONCEPT_INDUCED_EPHEMERAL` tier (same name/rank, `write_through=False`); `ProducerUnavailableError`.
- `orion/substrate/relational/adapters/autonomy_ctx.py`: raises on transient failure instead of returning None.
- `orion/substrate/relational/tests/test_concept_induction_ctx_adapter.py`: failure cases expect the raise.
- `orion/substrate/relational/__init__.py`: export it.
- `orion/cognition/projection_builder.py`, `services/orion-cortex-exec/app/chat_stance.py`: concept_induction uses the new tier; registry builders take `concept_store` and bind the adapter to the layer's store.
- `orion/substrate/relational/adapters/concept_induction_ctx.py`: `store=` keyword; unbound fallback calls `snapshot()` before reading.
- `orion/substrate/relational/layer.py`: dedupe by node_id (durable wins); a None pull is recorded as fresh.
- `orion/substrate/relational/tests/test_concept_induction_ephemeral.py`: new.
- `services/orion-cortex-exec/tests/test_turn_latency_l1_l3.py`: both registries ephemeral + bound.

## Schema / bus / API changes

- Added: `CONCEPT_INDUCED_EPHEMERAL`, `ProducerUnavailableError` (Python only).
- Removed / Renamed: none.
- Behavior changed: concept_induction no longer writes to Falkor; lineage string unchanged (`concept_induction:concept_induced`).
- Compatibility notes: `build_projection_unification_registry()` / `_build_unification_registry()` keep their no-arg form (adapter falls back to its own store, now refreshed).

## Env/config changes

- None. No `.env_example` touched; sync not needed.

## Tests run

```text
pytest orion/substrate/relational/tests/test_concept_induction_ephemeral.py   14 passed
pytest orion/substrate/relational orion/cognition/tests/test_projection_builder.py   123 passed
pytest services/orion-cortex-exec/tests/test_turn_latency_l1_l3.py   24 passed
pytest orion/substrate/relational orion/cognition/tests orion/substrate/tests   1181 passed, 3 failed (pre-existing on base: test_felt_state_self_definition_lane.py), 3 collection errors (pre-existing: missing orion_cognition module)
cortex-exec stance/unification/latency/concept/belief files   185 passed, 15 failed (same 15 fail on base)
cortex-orch mind projection files   27 passed, 1 failed (same on base)
Mutation check: removing the dedupe or the None-is-fresh line each turns a new test red.
```

## Evals run

```text
No eval harness for the unification layer. The rehydrate-count test above is the measurement the spec asked for.
```

## Docker/build/smoke checks

```text
Not run (no deploy authorized). Read-only log scan of orion-athena-cortex-exec-background only.
```

## Review findings fixed

Review ran in a subagent against the explicit diff `origin/fix/turn-latency-l1-l3..HEAD`. No blockers.

- Finding (should): counting a None return as fresh would hide transient failures, because the autonomy and concept adapters returned None when their source was down.
  - Fix: new `ProducerUnavailableError` (`registry.py`). The concept adapter raises it on store-init, snapshot, or query failure and on a degraded read. Autonomy raises it on repository-init or fetch failure (logged at warning). The layer then marks the producer degraded and leaves it untracked. "Gate off" or "nothing to return" still returns None.
  - Evidence: `test_transient_concept_failure_keeps_the_anchor_cold`; adapter tests updated to expect the raise (their fake stores now have `snapshot()`, so they fail at the query and not on a missing method).
- Finding (should): the one-rehydrate test used concept_induction alone, and spark is still write-through.
  - Fix: `test_full_registry_rehydrates_per_cold_build` runs the whole live registry, with and without spark state in ctx. Both measure 1 rehydrate and 0 writes. Spark's node is a `state_snapshot`, which Falkor refuses to store, so spark degrades before any write lands. Live: no spark materialize failures in 24 h, so live ctx carries no spark state.
  - Evidence: the test asserts `(1, 0, False)` without spark and `(1, 0, True)` with it.
- Finding (nit): the dedupe applied to every ephemeral producer.
  - Fix: it now applies only to node ids re-read by a non-write-through cold producer.
  - Evidence: `test_dedupe_is_scoped_to_rereads`.
- Finding (nit): `TIER_BY_NAME["concept_induced"]` resolves to the write-through tier.
  - Fix: a guard test pins that a name lookup is not how a producer's tier is chosen.
  - Evidence: `test_tier_name_lookup_still_resolves_to_write_through_concept_tier`.
- Finding (nit): the adapter's `concept_type` default no longer reaches the stance.
  - Fix: comment at the dedupe. The stance's anchor fallback gives the same bucket for orion, relationship and juniper; only `claude` would differ, and it is not an anchor.
- Finding (nit, not fixed): in orch, the unbound fallback now rehydrates Falkor about every 30 s, where before it loaded once at boot and never refreshed. This is the intended trade (correct concepts over a frozen copy), but nothing measures it. The fallback test uses an in-memory store, so it proves `snapshot()` is called, not the Falkor cost.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-orch up -d --build
```

(After #2508 and this PR merge to main; deploy from the primary checkout on main.)

## Risks / concerns

- Severity: low. Concern: spark is still `CONCEPT_INDUCED` write-through. Today its only node kind cannot be stored in Falkor, so it never writes; if that changes, a cold turn with spark state goes back to 2 rehydrates (the full-registry test will catch it).
- Severity: low. Concern: orch's concept fallback now refreshes from Falkor about every 30 s (review nit, not measured).
- Severity: low. Concern: the autonomy adapter now raises on init/fetch failure where it used to return None; the only caller is the unification layer, which catches it and marks autonomy degraded.
- Severity: medium. Concern: live proof is UNVERIFIED -- not deployed. Proof after deploy: `cold_anchors=[]` on most shared-spine builds, and Falkor reads of `sub-concept-seed-juniper` keep following the 30-day half-life curve with no reset to 1.0 after a stance build.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2510

🤖 Generated with [Claude Code](https://claude.com/claude-code)
