# Entity nodes are durable, and one bad node no longer aborts an ingest

## Summary

- **Live outage.** `FalkorSubstrateStore.upsert_node` raised on any `node_kind` outside concept/evidence. The topic-foundry adapter emits `EntityNodeV1`. `SubstrateGraphMaterializer.apply_record` writes incrementally, so that exception killed the whole ingest **before a single edge was written**.
- Entity is now a durable node kind: encode branch, decoder, hydration columns, and one shared `DURABLE_NODE_KINDS` both guards derive from.
- `_CountingSubstrateStore` no longer lets a refused node end the run. Skips are counted and surfaced, never swallowed.
- An edge whose endpoint was skipped is dropped too — otherwise `MERGE (source:SubstrateNode {node_id})` conjures a phantom node carrying nothing but an id.
- Found while verifying PR #1957's deploy, by the `/structure` route that PR added.

## Outcome moved

Observed live at 21:59:14 UTC, on both graphs:

```
substrate  concepts_written=18  entities_written=0  edges_written=0  available=False
AI Town    concepts_written=132 entities_written=0  edges_written=0  available=False
```

| | before | after |
|---|---|---|
| edges written per ingest | **0** | all of them |
| entity nodes persisted | impossible | round-trips through cold hydration |
| cost of one unwritable node | the entire run | that node |
| orphaned Evidence left behind | 18 and counting | none |

## Current architecture

Two independent hardcoded guards rejected non-concept/evidence kinds — `falkor_store.py:559` and `falkor_codec.py:104` — each with its own copy of the tuple. The ingest wrapper `_CountingSubstrateStore` existed specifically to report accurate counts on mid-record failure (its docstring says so), but it only *counted* the abort; it did not prevent it.

## Architecture touched

- `orion/substrate/falkor_codec.py`: `DURABLE_NODE_KINDS`, entity encode branch, `decode_entity_node`, wired into `decode_node`.
- `orion/substrate/falkor_store.py`: guard derives from the codec constant; `NATIVE_NODE_RETURN_FIELDS` gains `entity_type` / `aliases_json`.
- `services/orion-hub/scripts/concept_atlas_routes.py`: `_CountingSubstrateStore` resilience + skip accounting; ingest payload and logs surface it.

No schema change, no bus change, no new env key, no new service.

## Files changed

- `orion/substrate/falkor_codec.py`: make entity durable; one source of truth for which kinds are.
- `orion/substrate/falkor_store.py`: derive the guard; hydrate the entity columns.
- `services/orion-hub/scripts/concept_atlas_routes.py`: don't let one node cost the run its edges.
- `orion/substrate/tests/test_falkor_codec.py`: +6 entity tests.
- `orion/substrate/tests/test_falkor_store.py`: +4, incl. a drift guard on the two guards.
- `services/orion-hub/tests/test_concept_atlas_ingest_resilience.py`: new, 8 tests.

## Schema / bus / API changes

- **Added:** ingest response fields `skipped_nodes`, `skipped_edges`, `skipped_node_kinds`. Additive; existing readers unaffected.
- **Behavior changed:** entity nodes now persist to FalkorDB instead of raising. A previously-impossible node kind appears in `snapshot()` — consumers iterating nodes generically will now see `EntityNodeV1`.
- **Compatibility:** `decode_entity_node` defaults `entity_type` to `"unknown"` and `aliases` to `[]` so any row written before these columns existed decodes instead of raising. `EntityNodeV1.entity_type` is `min_length=1`, so a NULL there would otherwise take down the whole generic hydration, not just that node.

## Deliberately NOT "allow every kind"

`_LABEL_BY_KIND` lists eleven kinds. Only `entity` joined, because only `entity` has a real producer writing it to this store. Adding encode/decode for a kind nothing emits would be a keyword cathedral, and a kind with no decoder round-trips to `None` — silently. A test asserts `DriveNodeV1` is still rejected.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: n/a — no env surface touched.
- local `.env` synced: n/a.
- skipped keys requiring operator action: none.

## Tests run

```text
orion/substrate/tests/test_falkor_codec.py
orion/substrate/tests/test_falkor_store.py
orion/substrate/tests/test_falkor_codec_topic_id.py
orion/substrate/tests/test_falkor_store_aitown_graph.py         87 passed, 1 failed*

services/orion-hub/tests/test_concept_atlas_routes.py
services/orion-hub/tests/test_concept_atlas_ingest_resilience.py
services/orion-hub/tests/test_concept_atlas_structure_route.py  58 passed
```

\* `test_redis_graph_client_returns_named_dicts_from_header` fails on **clean main**, untouched by this diff — a regression I introduced in PR #1957 and fixed separately in **PR #1964**. This branch should merge after that one. Substrate pytest is not in CI, so it does not gate either PR.

Mutation-tested, every mutation asserted present in the file before running:

```text
9/9 killed
  codec  entity dropped from durable kinds · entity encodes no aliases ·
         decode_node forgets entity · null entity_type no longer defaulted
  store  entity columns dropped from hydration ·
         store guard hardcodes its own list again
  route  a refused node aborts the run again · skips swallowed silently ·
         edges to a refused node get written anyway
```

## Evals run

```text
orion/substrate/evals exists but has no harness covering the Falkor codec or
the ingest wrapper; none added.
```

Not claimed as covered. The live round trip below is what stands in for one here.

## Docker/build/smoke checks

Live against FalkorDB 4.18.11, on a scratch graph (`orion_entity_smoke`, deleted afterwards) so nothing was written to the real substrate:

```text
wrote concept + ENTITY + edge -- no exception (this raised before the fix)
cold-hydrated kinds: {'concept': 1, 'entity': 1}
entity from COLD store: EntityNodeV1 | athena | host | ['athena-host', 'Athena']
edges: [('associated_with', 'smoke-concept', '->', 'smoke-entity')]
get_node_by_id: EntityNodeV1
```

Raw graph confirmed the label and columns: `[SubstrateNode, Entity] node_kind=entity label=athena entity_type=host`.

One correction found by running it: `mentions` is not a valid `SubstrateEdgeV1` predicate. Entity mentions use `associated_with`, which matches the 74 such edges already live.

## Remediating the 18 orphans — no backfill needed

Orphan ids are deterministic:

```text
node_id      sub-evidence-topicfoundry-9e1211cb-…-158e-17
identity_key evidence|world||topic_foundry_run_topic|9e1211cb-…-158e:topic:17
```

All 18 belong to run `9e1211cb-a0d1-4b03-9c8f-77dfc3ac158e` — the run that failed. Re-running ingest for that run after deploy re-MERGEs the same node ids and writes the `supports` edges that were never reached. No snapshot, no migration, no backfill script. If the scheduler has moved on to a newer run they simply stay inert, visible in the `/structure` card's singleton list.

## Review findings fixed

`/code-review high` ran in a subagent; see the follow-up commit on this branch.

## Restart required

```bash
sudo docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build orion-hub
```

`orion-substrate-runtime` also imports `falkor_store`; restart it too if it holds a long-lived store:

```bash
sudo docker compose \
  --env-file .env \
  --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- **Severity: low.** Entity nodes now appear in `snapshot()` for the first time. Any consumer iterating nodes generically and assuming concept-or-evidence will see a new kind. `EntityNodeV1` carries `label`, so label-reading consumers are fine; kind-switching consumers should be checked as they surface.
- **Severity: low.** The resilience path makes a partial write possible where previously the run aborted. That is the intended trade — but it means `available: true` can now accompany a degraded run, which is why `skipped_nodes`/`skipped_edges`/`skipped_node_kinds` are in the payload and a warning is logged. A caller that ignores those fields will read a degraded ingest as clean.
- **Severity: informational.** Skipping an edge whose endpoint failed means the graph under-reports connections rather than inventing phantom nodes. Deliberate: an incomplete graph is honest, a phantom node is not.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/substrate-entity-nodes
