# Concept-induction blind-upsert null-write fix

## Summary

- Third fix in a same-day incident chain (PR #1498: bus_synaptic write-skip-on-zero; PR #1501: materializer merge-branch clobber). Found during PR #1501's own code review.
- `orion/spark/concept_induction/falkor_materialization.py`'s `materialize_concept_profile_to_falkor()` calls `store.upsert_node()` directly, no existing-node read, no `skip_metadata_keys` protection at all.
- Worse than the materializer bug: since this is a blind upsert, `falkor_codec.py`'s `encode_node_properties()` always emits a `prediction_error` param (default `None` when the freshly-mapped concept's own metadata doesn't carry one, which it never does), and `set_assignments()` never filters `None` out of the Cypher `SET` clause — an unprotected write here **nulls** reducer-owned fields, not just freezes them.
- Live default backend (`CONCEPT_PROFILE_GRAPH_BACKEND=falkor`), not opt-in.
- Fix: `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`, same reused protection as PR #1501.
- Live-verified: rebuilt and restarted `orion-spark-concept-induction`, confirmed clean startup and normal event processing.

## Outcome moved

Closes a real data-corruption path: any node whose `node_id` happens to be reused by an induced concept (the same collision shape already confirmed live for `node:substrate.bus_synaptic` via the materializer bug) could previously have its `prediction_error`/`contributing_turn_ids` silently nulled by this path, on the live default backend.

## Current architecture

`orion-spark-concept-induction`'s `ConceptWorker` (`bus_worker.py`) runs concept extraction/clustering on chat/cognition-trace intake and, when `CONCEPT_PROFILE_GRAPH_BACKEND=falkor`, calls `materialize_concept_profile_to_falkor()` to durably upsert the resulting concept nodes/edges into FalkorDB — one `store.upsert_node()` call per concept, unconditionally, with no read of any existing node at that identity first.

## Architecture touched

`orion/spark/concept_induction` only (shared library code; deployed as `services/orion-spark-concept-induction`). No contract/schema/bus changes.

## Files changed

- `orion/spark/concept_induction/falkor_materialization.py`: `SubstrateWriteStore` Protocol now declares `skip_metadata_keys`; the `store.upsert_node()` call in `materialize_concept_profile_to_falkor()` passes `skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS`.
- `orion/spark/concept_induction/tests/test_falkor_materialization.py`: `_ExplodingStore.upsert_node()` test double updated to accept (and ignore) the new kwarg, matching a real regression PR #1501 found and fixed for a different test double. New test `test_materialize_does_not_null_a_reducer_owned_field_it_collides_with` seeds a node with a reducer-owned value present, re-materializes the same profile, and asserts both the stored value survives and the Cypher `SET` clause from post-seed calls never references `n.prediction_error`.

## Schema / bus / API changes

None.

## Env/config changes

None. No `.env_example` changes; nothing to sync.

## Tests run

```
$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/spark/concept_induction/tests/test_falkor_materialization.py -q
10 passed, 1 pre-existing failure (test_materialize_populates_write_through_cache)

$ /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/spark/concept_induction/tests -q
76 passed, 1 pre-existing failure — confirmed identical on main (75 passed + 1 failure), same failure, byte-for-byte, before this change existed. Zero regressions.
```

Red-before-green confirmed via `git stash` (this file only — see PR report note below on why later fixes in this session switched away from `git stash` for this purpose).

The one failure (`test_materialize_populates_write_through_cache`) is pre-existing and unrelated: `FalkorSubstrateStore(hydrate=False)` leaves `_last_snapshot_generation` at its `-1` sentinel, so the first `snapshot()` call always triggers `_hydrate_from_durable()` regardless of any writes; `RecordingFalkorClient()`'s default (no configured `hydrate_node_rows`) returns empty rows for the hydrate query, wiping the write-through cache the test expects to still hold. Independently reproduced against unmodified `main` by both this session and its code-review pass — confirmed structural, not caused or masked by this patch (which never touches `snapshot()`/hydrate logic). The new regression test in this PR deliberately asserts via `store.get_node_by_id()` (direct cache read) instead of `store.snapshot()`, sidestepping this unrelated bug entirely rather than depending on it being fixed.

## Evals run

No dedicated eval harness for this path; not applicable.

## Docker/build/smoke checks

```
$ bash scripts/safe_docker_build.sh orion-spark-concept-induction up -d --build
... rebuilt and restarted ...
```

Live-verified: clean startup log (`concept_induction_worker_started`, stream consumer group ready, subscribed to all intake channels), and normal event processing observed immediately after restart (`concept_induction_worker_event_received` / `concept_induction_trigger_received` / `concept_induction_trigger_decision` for real incoming events) — no crash, no exception.

## Review findings fixed

Code review pass (subagent) found **no material issues** — fix confirmed correct, complete, and well-tested. Two things worth recording from that pass:

- Confirmed no other caller of `materialize_concept_profile_to_falkor()` or test double implementing the `SubstrateWriteStore` Protocol shape needed the same fix — searched every `upsert_node` implementation and every caller reachable from this function; the only test double requiring an update (`_ExplodingStore`) was already caught and fixed in this same patch.
- The review's own investigation independently surfaced the same cross-worktree `git status` anomaly this session separately found and fixed while working on a sibling PR (`fix/materializer-identity-fallback-gap`) in a different worktree — a `git stash`-across-worktrees leak, not caused by or affecting this branch's own commit (verified clean both by this session and by the review pass). Noted here for the record; the sibling PR's own report has the full incident writeup. As a direct consequence, later fixes in this session stopped using `git stash` for red-before-green checks in favor of `git show HEAD:<path> > <path>` / restore-from-backup, which doesn't touch the shared `refs/stash`.

## Restart required

```
bash scripts/safe_docker_build.sh orion-spark-concept-induction up -d --build
```

Already run during this session's live verification; the running container reflects this fix. No further restart needed unless this branch is rebuilt from a fresh checkout.

## Risks / concerns

- Severity: Low
- Concern: None material identified by review.
- Mitigation: N/A.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/bus-synaptic-concept-induction-null-fix
