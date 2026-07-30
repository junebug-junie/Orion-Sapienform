# Materializer node_id-collision fallback gap fix

## Summary

- Found during code review of PR #1501 (bus_synaptic materializer clobber fix): `SubstrateGraphMaterializer.apply_record()`'s existing-node lookup only fell back to a direct `get_node_by_id(node.node_id)` check when `identity_key` was `None` outright.
- When `identity_key` resolved to something (the common case) but the identity-index lookup found nothing, `apply_record()` took the "created" branch: a full wholesale overwrite via `node.model_copy()`, no `merge_node()` call, no `skip_metadata_keys` protection at all — worse than a metadata-only clobber, since it replaces the entire node.
- Fix: widen the fallback to fire whenever the identity lookup comes up empty, regardless of whether `identity_key` was `None` or just unresolved.
- Verified correct and beneficial standing alone (converts a full-node replace into a same-call metadata-preserving merge via `reconcile.merge_node()`'s existing-wins precedence), and verified to combine cleanly with PR #1501 via `git merge-tree` (no conflicts, complementary line ranges).

## Outcome moved

Closes a narrow but real gap: any node_id collision where the existing node's registered identity_key doesn't match what `SubstrateIdentityResolver.canonical_node_key()` independently computes for a colliding incoming node — exactly the shape of every reducer-owned node (`_write_prediction_error_node` registers `f"substrate_prediction_error|{node_id}"`, never reproducible by the resolver) — no longer risks a full wholesale node replacement.

## Current architecture

`apply_record()` resolves each incoming node's canonical identity via `SubstrateIdentityResolver.canonical_node_key()`, looks that up in the store's identity index, and only fell back to a raw `node_id` check when the resolver itself returned `None`. The durable Falkor write (`MERGE (n:SubstrateNode:{label} {node_id: $node_id})`) is keyed purely on raw `node_id` regardless of any of this Python-level identity bookkeeping — so a collision at that layer was always inevitable once two records shared a `node_id`; this patch only changes which merge *strategy* gets applied to a collision the store was always going to make anyway.

## Architecture touched

`orion/substrate/materializer.py` only (shared library, used by `orion-cortex-exec-background`'s concept-induction pipeline and other materializer callers). No contract/schema/bus changes.

## Files changed

- `orion/substrate/materializer.py`: widened the existing-node fallback; the merge-decision `reason` field now correctly reports `"node_id_match"` (not `"identity_match"`) when the raw-id fallback is what found the match.
- `orion/substrate/tests/test_materializer_node_id_collision_fallback.py` (new): two tests — one confirms a node_id collision with an unregistered/mismatched identity_key takes the merge branch (not created) and preserves reducer-owned metadata; the other confirms genuinely new node_ids still take the created branch normally.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests/test_materializer_node_id_collision_fallback.py -q
2 passed

$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests --ignore=orion/substrate/relational/tests -q
491 passed (489 baseline + 2 new), zero regressions.
```

Red-before-green confirmed via `git show HEAD:<path>` (not `git stash` — a real cross-worktree stash leak happened during this exact investigation; see the "Incident" note below and `feedback_git_stash_shared_across_worktrees` memory, updated this session).

## Evals run

Not applicable.

## Docker/build/smoke checks

Not run for this specific patch — it's a narrow, low-frequency edge-case fallback (a node_id collision with a mismatched identity_key), not a demonstrated live symptom the way the sibling bus_synaptic fixes were. Covered by unit tests; live verification wasn't pursued given the low blast radius and that `orion-cortex-exec-background` was already rebuilt/redeployed once this session for the sibling materializer-clobber fix (PR #1501), which this patch is designed to combine cleanly with.

## Review findings fixed

- Finding: three places (commit message, module docstring, `_reducer_owned_node()` helper docstring) claimed `_write_prediction_error_node` calls `store.upsert_node(identity_key=None, ...)`. Confirmed false via reading `services/orion-substrate-runtime/app/worker.py:1044-1047` — it registers a real, non-`None` identity_key (`f"substrate_prediction_error|{node_id}"`), present since 2026-07-01 (`git log -p -S`), not a recent drift.
  - Fix: corrected all three docstrings/comments and the test's own seed call to register that real identity_key convention instead of `None`. Re-verified the test still passes — it never depended on the false claim; `SubstrateIdentityResolver.canonical_node_key()` never reproduces the reducer's identity_key format regardless of whether the registered value was `None` or that real string, so the raw-node_id fallback under test is exercised correctly either way, now for the accurate reason. Follow-up commit `f45d6a4c3`.
  - Evidence: `git log -p -S "identity_key=f\"substrate_prediction_error"` on `worker.py`; `pytest orion/substrate/tests/test_materializer_node_id_collision_fallback.py -q` — 2/2 passing after the correction.
- Finding: does this PR stand alone safely without #1501's `skip_metadata_keys`, and do the two combine cleanly?
  - Verified (not a defect, confirmation only): `reconcile.merge_node()`'s `existing.model_copy(update={...})` construction and `{**incoming.metadata, **existing.metadata}` precedence mean even standing alone, this PR converts a full-node clobber into a same-call metadata-preserving merge. `git merge-tree` against PR #1501's branch produced no conflict — the two PRs touch adjacent, non-overlapping lines within the same merge branch (this PR: the `existing_node` lookup + `reason` field; #1501: the `upsert_node()` call's `skip_metadata_keys` kwarg).
  - Evidence: `git merge-tree <common-ancestor> <this-branch> <1501-branch>`; direct reading of `reconcile.py:244-328`, `falkor_store.py:511-590`.
- No correctness regressions found in the widened fallback itself, the `reason` field's accuracy, or any downstream consumer (`relational/layer.py`, `frontier_landing.py`, `concept_atlas_routes.py`'s `_CountingSubstrateStore` — none branch on `reason` or rely on `created`/`merged` counts in a way this changes).

## Restart required

No restart required for this branch alone (not yet deployed independently — see Docker/build/smoke checks above). Once merged, will take effect on the next `orion-cortex-exec-background` rebuild:

```bash
bash scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: Low
- Concern: This patch alone (without #1501) still routes newly-caught collision cases into an *unprotected* merge branch `upsert_node()` call (no `skip_metadata_keys` yet in this branch's own history) — a strict improvement over the prior full-overwrite behavior, but not the full protection until #1501 also lands.
- Mitigation: Verified the two PRs merge cleanly in either order; recommend merging both together or #1501 first.
- Severity: Low
- Concern: Not live-verified against a running service the way the sibling bus_synaptic fixes were.
- Mitigation: Narrow, low-frequency edge case; covered by targeted unit tests; low blast radius.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/materializer-identity-fallback-gap
