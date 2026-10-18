# fix(substrate-runtime): walkway day node is evidence, not event

## Summary

- The walkway "seen on this day" pass never wrote anything live. After PR #2755 was deployed, every pass failed with `ValueError: FalkorSubstrateStore durable writes support concept, evidence, entity, assertion nodes only; got node_kind='event'`.
- The day node is now an `EvidenceNodeV1` (id `evidence:walkway_day:<stream>:<date>`, `evidence_type=walkway_day`, `content_ref=vision_rollup:<stream>:day:<date>`). The edges are now entity `-observed_in->` evidence.
- The tests now run on the production `FalkorSubstrateStore` (with a recording client), not on the in-memory store. The in-memory store accepts any node kind, so the old tests passed for the wrong reason.
- Added a real-FalkorDB test. It writes the day, queries the `observed_in` edges, rehydrates a fresh store and checks that a restart adds nothing. It runs in the `substrate-neighborhood` lane (FalkorDB v4.18.11 + 6.0.1) and must not skip.

## Outcome moved

The five named walkway people (Maeve, Adeline, Esme, Teddy, Amanda Feld) get one `observed_in` link per complete day they were seen. Before this fix they had zero links. Live read-only baseline on `orion_substrate` (2026-10-18): 5 `entity:vision_individual:*` nodes, 0 `observed_in` edges, 0 `walkway_day` nodes.

## Current architecture

`app/walkway_seen_loop.py` runs every 1800 s. It reads complete `vision_rollup` day rows from the last 3 days and calls `map_walkway_day_seen_to_substrate`, which built an `EventNodeV1` day. The pass then applies that through `SubstrateGraphMaterializer` into the service's Falkor store. The store accepts only `DURABLE_NODE_KINDS` (`orion/substrate/falkor_codec.py:157`). The node upsert raised before any edge was written, so no partial writes are live.

## Why evidence, not concept

- A day's rollup is a sensed observation record that points back at its source row. That is what `evidence` is for: `evidence_type` plus `content_ref`.
- This repo already uses the same shape. `services/orion-memory-consolidation/app/referent_projector.py` writes `EvidenceNodeV1(evidence_type="episode_memory", content_ref="episode_memory:<id>")` with `observed_in` edges from the nodes into it.
- A concept is an abstract label. It has activation decay seeded at the schema boundary, and that is the wrong behaviour for one dated record.
- The evidence identity key (`reconcile.py:140-143`) is `evidence|scope|subject_ref|evidence_type|content_ref`. Both refs carry the stream and the date, so days never merge. A re-run lands on the same node.

## Architecture touched

- Adapter `orion/substrate/adapters/vision_individual.py` (the day node kind and id, plus `content_ref`).
- Tests and CI only. There is no change to the bus, schema, env or SQL.

## Files changed

- `orion/substrate/adapters/vision_individual.py`: the day is now `EvidenceNodeV1`, the edge target kind is `evidence`, and new helpers `walkway_day_content_ref` and `WALKWAY_DAY_EVIDENCE_TYPE`.
- `services/orion-substrate-runtime/tests/test_walkway_seen_loop.py`: moved to the Falkor store. Added a durable-kind test, a harness regression test (the exact ValueError) and a real-FalkorDB test.
- `.github/workflows/substrate-neighborhood.yml`: new path triggers. New must-not-skip step that selects tests by node id and requires `3 passed`.
- `services/orion-substrate-runtime/app/walkway_seen_loop.py`, `app/settings.py`, `docs/superpowers/plans/2026-10-18-walkway-rollups-design.md`: comment and doc wording (event -> evidence).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: day node id `event:walkway_day:...` -> `evidence:walkway_day:...`. The old id was never written live, and nothing in the repo references it.
- Behavior changed: the day node kind is now evidence.
- Compatibility notes: the day's line is kept in node metadata (`summary`). The durable codec does not persist evidence metadata, so a node loaded back from Falkor carries only `content_ref`. The line itself lives in the `vision_rollup` row that `content_ref` names. Edge metadata (`label`, `first_seen_local`) is also cache-only. That behaviour existed before this patch for all edges.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
python -m pytest -q services/orion-substrate-runtime/tests/test_walkway_seen_loop.py
  5 passed, 2 skipped (env-gated Falkor / Postgres)
ORION_TEST_FALKOR_URI=redis://127.0.0.1:16399 (throwaway falkordb/falkordb:v4.18.11 container, stopped after)
  python -m pytest -q <the 3 CI node ids>  -> 3 passed
Old adapter (git show HEAD:...) + new tests -> 4 failed with the exact live ValueError
python -m pytest -q test_ask_answered_listener.py test_walkway_seen_loop.py orion/substrate/tests/test_falkor_store.py
  78 passed, 2 skipped
```

## Evals run

```text
No walkway_seen eval exists in orion-substrate-runtime/evals. The real-FalkorDB test in the
substrate-neighborhood lane is the end-to-end proof for this seam: the record is written,
the edges are queried from the graph, the store is rehydrated and the pass is idempotent.
Follow-up: a fixture eval scoring days linked vs. named sightings, if this grows.
```

## Docker/build/smoke checks

```text
No build or deploy from this branch. Live FalkorDB was read only (GRAPH.RO_QUERY baseline above).
```

## Review findings fixed

- Finding: the CI `-k` filter fails open if the real-Falkor test is renamed (the other tests still match, and nothing reads SKIPPED).
  - Fix: the step selects tests by node id and requires `3 passed`.
  - Evidence: the 3 node ids pass against a throwaway FalkorDB.
- Finding: the regression test passes with or without the fix, so it reads as coverage it does not give.
  - Fix: the docstring now says it is a harness guard, and it names the tests that guard the fix.
  - Evidence: on the old adapter, the main pass test and the durable-kind test fail with the exact ValueError.
- Finding: `GRAPH.DELETE` in `finally` would hide the real failure if the graph was never created.
  - Fix: catch `redis.ResponseError`.
  - Evidence: the code change in the test.
- Finding (nit): `ensure_indexes=False` differs from production.
  - Fix: the test uses the default (indexes on).
  - Evidence: the real-Falkor test passes.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Run this from the clean primary `main` checkout after merge. The loop runs every 1800 s with a 3-day lookback, so it fills in the last 3 complete days by itself.

Post-deploy check (read-only):

```bash
docker exec orion-athena-falkordb redis-cli GRAPH.RO_QUERY orion_substrate "MATCH (p:SubstrateNode)-[e]->(d:SubstrateNode) WHERE p.node_id STARTS WITH 'entity:vision_individual:' AND e.predicate = 'observed_in' RETURN p.label, d.node_id, d.node_kind ORDER BY d.node_id"
```

Expect `evidence:walkway_day:walkway:<date>` rows with `node_kind=evidence` for every named person seen on a complete day. Also check the log line `walkway_seen linked day=... edges_created=N` and that `walkway_seen pass failed` is gone.

## Risks / concerns

- Severity: low. Concern: the day line does not survive rehydration from Falkor; only `content_ref` does. Mitigation: no reader uses the line from the graph today, and the source row holds it.
- Severity: low. Concern: live linking is UNVERIFIED until deploy. Mitigation: use the post-deploy query above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2756

🤖 Generated with [Claude Code](https://claude.com/claude-code)
