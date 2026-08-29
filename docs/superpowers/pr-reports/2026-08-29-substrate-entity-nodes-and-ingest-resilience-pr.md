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
orion/substrate/tests/test_falkor_store_aitown_graph.py
orion/graph/tests                                              194 passed

services/orion-hub/tests/test_concept_atlas_routes.py
services/orion-hub/tests/test_concept_atlas_ingest_resilience.py
services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py
services/orion-hub/tests/test_concept_atlas_structure_route.py 110 passed
```

**304 passed, 0 failed.** An earlier run showed one failure — `test_redis_graph_client_returns_named_dicts_from_header`, a regression I introduced in PR #1957 and fixed separately in **PR #1964**, which has since merged and is included here.

Mutation-tested, every mutation asserted present in the file before running:

```text
original      9/9 killed
  codec  entity dropped from durable kinds · entity encodes no aliases ·
         decode_node forgets entity · null entity_type no longer defaulted
  store  entity columns dropped from hydration ·
         store guard hardcodes its own list again
  route  a refused node aborts the run again · skips swallowed silently ·
         edges to a refused node get written anyway

review fixes 13/13 killed
  store  third guard reverts to a hardcoded tuple ·
         topic_id dropped from hydration again
  route  circuit breaker removed · failure counter never resets ·
         wrote_anything ignores edges · empty node_id poisons the skip set ·
         origin reverts to metadata-only · synthetic_label reverts to
         metadata-only · origin claims every node · break-even guard removed ·
         guard flips to zero-writes only · guard fires on any skip at all
```

One review-fix mutation initially **survived** — removing the wrote-nothing guard. I had tested the `wrote_anything` property but never the route path that consumes it, so nothing exercised the guard. Fixed by adding three route-level ingest tests; that same gap is what then surfaced the break-even correction, because the pre-existing partial-write test proved my first threshold was too permissive.

**A harness note, again.** The first review-fix mutation sweep hit the 2-minute tool timeout and was killed by SIGTERM, so its `try/finally` never ran and one mutation was left in the working tree — the same leak recorded earlier this session, this time from a signal rather than an exception. `finally` does not cover SIGTERM. The rerun installs `atexit` plus `SIGTERM`/`SIGINT`/`SIGHUP` handlers and runs in the background; it reported `restored: True` and the tree was verified clean by grepping for the mutation markers.

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

Re-run after the review fixes, on a second scratch graph (also deleted):

```text
kinds: ['concept', 'entity']
entity from cold store: host ['Athena']
topic_id survives hydration: 17 (was always None before)
provenance.producer survives: topic_foundry_adapter
metadata['source'] survives: None  <- confirms why origin needed producer
edges: [('associated_with', 'smoke-c1', 'smoke-e1')]
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

`/code-review high` in a subagent. **Four findings, all real, all fixed.**

- **Finding (major):** the blanket per-node `except` turned a *dead store* into `available: true` with every count at zero — strictly worse than the abort it replaced, and indistinguishable in the scheduler's tick log from "no new data". It also removed fail-fast (N × connect-timeout) and still spent sequential LLM calls afterward.
  - **Fix:** two independent guards. A consecutive-failure breaker (`MAX_CONSECUTIVE_NODE_FAILURES`) bounds how long an unreachable store is retried and lets the exception propagate to the pre-existing `_unavailable("substrate_store_write_failed")`. Separately, a **break-even** check — more nodes skipped than written — decides how the run is reported. Break-even is not a tuned ratio; it is the point past which the run did not produce a usable graph, and it is exactly what separates "one unwritable kind" (many land, a few do not → available, skips surfaced) from "broken store" (few or none land → unavailable).
  - **Evidence:** the pre-existing `test_ingest_partial_store_write_reports_precise_successful_counts` — which asserts a store failing everything after the first write reports `available: false` with real partial counts — **passes unchanged**, so the prior contract is preserved rather than replaced. Three mutations killed, including both wrong thresholds (`written == 0` only, and "any skip at all").

- **Finding (medium):** a **third** hardcoded `("concept", "evidence")` guard survived centralization, in `_migrate_legacy_payload_nodes`. A legacy `payload_json` row holding an `EntityNodeV1` would never migrate and never enter the cache — invisible forever, while logging a skip on every hydrate. My own `test_the_two_durable_kind_guards_cannot_drift` asserted *two* call sites and passed while the third was out of sync, so it did not catch the drift it is named for.
  - **Fix:** third guard now derives from `DURABLE_NODE_KINDS`; the test asserts the *literal tuple is absent anywhere in the module*, which finds any copy including one added later.
  - **Evidence:** `grep -c 'not in ("concept", "evidence")'` → 0. Mutation reverting it fails 3 tests.

- **Finding (medium):** `topic_id` has had a complete encode/decode pair since the topic-foundry work and was simply never listed in `NATIVE_NODE_RETURN_FIELDS`, so the decoder could only ever see NULL — **every hydrated node lost its cluster id, and the atlas colours nodes by that field.** Separately, `origin` and `synthetic_label` gated on `metadata["source"]`, which is not in the codec's closed allowlist and does not survive a rehydrate. Under the live `SUBSTRATE_STORE_BACKEND=falkor`, `origin` was permanently `"concept"` and `synthetic_label` permanently `False` — meaning a genuinely unlabeled `topic_<id>` cluster rendered as if it were a real concept name, the exact dishonest label that field exists to prevent. Pre-existing for concepts; this patch is what puts entities in front of it.
  - **Fix:** `topic_id` added to the return fields. `origin`/`synthetic_label` now read `provenance.producer`, a native column that does survive, rather than growing the metadata allowlist.
  - **Evidence, live:** `topic_id survives hydration: 17 (was always None before)`, `provenance.producer survives: topic_foundry_adapter`, and `metadata['source'] survives: None` — the last one confirming the finding directly. Four mutations killed, including one that would make `origin` claim every node.

- **Finding (low):** a node with no `node_id` put `""` into `_failed_node_ids`, where it would match any later edge whose endpoint ref was missing, dropping it and blaming `endpoint_not_written` for an unrelated cause. My own test explicitly blessed this.
  - **Fix:** only a real id joins the set; the test now asserts an unrelated edge is *not* dropped.
  - **Evidence:** confirmed the review's reasoning that this cannot misfire on today's schema-typed path — `NodeRefV1.node_id` is `min_length=3`, which I hit for real while writing the live smoke. Fixed anyway: the wrapper is duck-typed and a sentinel comparison is the wrong shape regardless.

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
