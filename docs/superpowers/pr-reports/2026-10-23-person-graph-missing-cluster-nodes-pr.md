## Summary

- The person-graph pass now builds the graph node of a linked camera cluster when that node is missing. It uses the cluster's `vision_individual` row and the same adapter the ask listener uses.
- It only does this for clusters the interlocutor named herself. Copied-name clusters (`label_copied`) never get a node, and existing nodes are never touched.
- One failing cluster write is logged and skipped. It does not stop the claims for the rest of the pass.

## Outcome moved

7 named clusters had no graph node, so their person links waited forever (live `waiting=7`). They were named on 10-15, 10-19 and 10-21 through ordinary "who is this?" answers. The ask listener writes a cluster node only when it hears `orion:ask:answered` on the bus, and the bus does not replay a missed event. After this change, those 7 get their nodes and their claims in the first pass.

## Current architecture

A cluster node is written by `app/ask_answered_listener.py` from the bus, and only from there. `app/person_graph_loop.py` (PR #2858) waited for that node before claiming.

## Architecture touched

orion-substrate-runtime `app/person_graph_loop.py` only. No new keys, schemas or channels.

## Files changed

- `services/orion-substrate-runtime/app/person_graph_loop.py`: the links query joins `vision_individual` (label, kind, ask, stream, label_copied) and `orion_ask.answered_at`. New `_ensure_cluster_nodes` step. Log line gains `clusters_written=`.
- `services/orion-substrate-runtime/tests/test_person_graph_loop.py`: 2 tests.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
test_person_graph_loop.py + test_walkway_seen_loop.py + test_ask_answered_listener.py: 47 passed, 2 skipped
Mutation check: with the label_copied guard removed, the copied-cluster test fails.
```

## Evals run

```text
New links query run read-only on live Postgres: 35 links, 35 with label, 35 with answered_at.
```

## Docker/build/smoke checks

```text
No new dependency. The image is unchanged apart from Python source.
```

## Review findings fixed

- Finding (medium, latent): the repair also built nodes for copied-name clusters. That falsely stamps them as named by the interlocutor, and it breaks `walkway_seen_loop`'s rule that a copied cluster has no node of its own.
  - Fix: skip rows where `label_copied` is true.
  - Evidence: the copied-cluster test. Live there are 0 copied rows today.
- Finding (low): one bad cluster write aborted the whole pass.
  - Fix: a per-cluster try/except that logs `person_graph cluster_node_failed` and continues.
- Not fixed (low, pre-existing): if the listener also misses a later rename, the cluster node keeps the old name. Listener-written nodes have the same gap, so this is not a regression.

## Restart required

Merge and pull. The post-pull rebuild redeploys orion-substrate-runtime.

Expected first pass: `clusters_written=7 claims=7 waiting=0`.

## Risks / concerns

- Severity: low. Concern: the stale-name-after-missed-rename gap above.
- Mitigation: none in this PR.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2897

🤖 Generated with [Claude Code](https://claude.com/claude-code)
