## Summary

- The person-graph pass now renames a linked cluster's graph node when its name differs from the name the interlocutor typed (`vision_individual.label`).
- The rename uses the ask listener's own rename step. That step moved from the listener into `orion/substrate/adapters/vision_individual.relabelled_individual_entity`, so both writers rename the same way. The old name is kept as an alias.
- A rename happens only when the database row's `labeled_at` is at least as new as the node. A rename the listener just wrote from the bus is never undone.

## Outcome moved

Before this change, a rename the listener never heard left the graph node on the old name forever. That covers a missed bus event, and also a direct database correction. The live case: on 10-27 the mixed cluster was given back the typed name "Teddy" directly in the database. Its graph node still says "Juniper" until this PR is deployed.

## Current architecture

Only the ask listener renamed cluster nodes, and only from `orion:ask:answered`. `person_graph_loop` (PRs #2858 and #2897) created missing nodes but never renamed existing ones.

## Architecture touched

orion-substrate-runtime `app/person_graph_loop.py` and `app/ask_answered_listener.py`; the shared adapter `orion/substrate/adapters/vision_individual.py`. No schema, channel or env changes.

## Files changed

- `orion/substrate/adapters/vision_individual.py`: `relabelled_individual_entity` and `MAX_ALIASES`, moved verbatim from the listener.
- `services/orion-substrate-runtime/app/ask_answered_listener.py`: uses the helper.
- `services/orion-substrate-runtime/app/person_graph_loop.py`: the links query also reads `labeled_at`. A stale node is renamed. The log line gains `clusters_relabelled=`.
- `services/orion-substrate-runtime/tests/test_person_graph_loop.py`: 2 tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
test_person_graph_loop.py + test_ask_answered_listener.py + test_walkway_seen_loop.py: 49 passed, 2 skipped
Mutation check: with the rename disabled, the stale-name test fails.
```

## Evals run

```text
Live check: the boy's cluster has labeled_at 2026-10-27 03:04Z. That is newer than its graph node (10-21), so the first pass after deploy renames it.
```

## Docker/build/smoke checks

```text
No new dependency.
```

## Review findings fixed

- Finding (low-medium): the pass could undo a rename for up to one pass, or for as long as the database writer is down. The listener writes from the bus before orion-sql-writer applies the answer to the database.
  - Fix: rename only when `labeled_at` is not older than the node's `observed_at`.
  - Evidence: `test_a_rename_the_writer_has_not_applied_yet_is_not_undone`.
- Noted (behaviour, kept): a "yes" on an "Is this <name>?" card for an already-named cluster is refused by the writer. The listener's rename of that node is now corrected back to the typed database name on the next pass. The database is the truth.

## Restart required

Merge and pull. The post-pull rebuild redeploys orion-substrate-runtime.

Expected log line: `person_graph cluster_relabelled individual=f5242d40... old='Juniper' new='Teddy'`.

## Risks / concerns

- Severity: low. Concern: a hand-edited label that does not bump `labeled_at` does not reach the graph. Mitigation: the typed-name path always bumps it.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
