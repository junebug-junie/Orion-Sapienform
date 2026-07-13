# PR report: fix graphiti_core /v1/search RELATES_TO schema gap

Follow-up to PR #993 (`feat/graphiti-core-backend-activation`). That PR activated the `graphiti_core` backend live but left `/v1/search` in a documented known-FAIL state: it ran without crashing but returned zero results for real data. This PR closes that gap.

## Summary

- Root-caused `/v1/search` returning empty results to two stacked bugs: (1) entities were written keyed on a custom `.id` property while every graphiti-core query (including `Graphiti.search()`) matches on `.uuid`; (2) edges were written as custom `:HAS_EPISODE`/`:RELATED` shapes while `Graphiti.search()` only reads `RELATES_TO`-shaped edges with `fact`/`fact_embedding` properties.
- Rewrote `ingest_episode()` to use graphiti-core's own `EntityNode`/`EntityEdge` classes (`uuid`-keyed identity, `attributes` dict flattened onto node/edge properties by `.save()`) instead of hand-rolled Cypher. Every crystallization gets a self-referential `RELATES_TO` edge (fact = deterministic template over its own subject/summary) so link-less crystallizations — most of the real data — stay searchable; linked crystallizations additionally get a real cross-entity edge per link.
- Fixed `_extract_crystallization_ids()`, which read a guessed `.source_node`/`.target_node` object shape that never matched real `EntityEdge` results (`.source_node_uuid`/`.target_node_uuid` strings).
- Found and fixed a third bug only visible under live verification: `graphiti-core==0.19.0`'s single-entity `EntityNode`/`EntityEdge.save()` never casts embeddings to FalkorDB's native `Vectorf32` type (only an unused bulk-save path does), so every embedded write 500'd on search. Added a narrow `vecf32(...)` cast after each save.
- Added idempotent fulltext/range index bootstrap (`ensure_graphiti_indices`, gated by new `GRAPHITI_AUTO_BUILD_INDICES`, called once from `main.py`'s `lifespan`) — FalkorDB's `CREATE FULLTEXT INDEX` has no `IF NOT EXISTS` guard and errors on a second call.
- `scripts/smoke_graphiti_search_e2e.sh` now PASSes live (was committed known-FAIL); also fixed a self-inflicted collision where the script's fixed-template subject/summary tripped crystallization duplicate-detection on repeated runs.

**Design decision (the "design decision issue" this PR was asked to resolve):** no LLM re-extraction, per the original non-goal in `docs/superpowers/specs/2026-07-06-graphiti-rail-activation-design.md`. All `fact`/`name`/`summary` text written to the graph is a deterministic string template over already-governed `subject`/`summary` fields (e.g. `f"{subject}: {summary[:280]}"`, `f"{subject} {relation} {target_subject}"`) — never a model call. This uses graphiti-core's own explicit-payload write API (`EntityNode`/`EntityEdge` + `.save()`), which is exactly what the original Phase C spec called for ("use graphiti-core driver write APIs with explicit node/edge payloads") and had not actually been implemented that way until now.

## Outcome moved

`/v1/search` went from "runs without crashing but proves nothing" to actually finding real ingested crystallizations. Live evidence: `RELATES_TO` edge count on this host's FalkorDB went from 0 to 14 (16 after this session's additional smoke runs), and `scripts/smoke_graphiti_search_e2e.sh` passes repeatably across independent runs.

## Current architecture (before this patch)

See PR #993's report. In short: `graphiti_core` backend ran live but wrote a schema (`HAS_EPISODE`/`RELATED` edges, `.id`-keyed entities) that graphiti-core's own search implementation could never read, because it queries `.uuid`-keyed `RELATES_TO` edges exclusively.

## Architecture touched

- `services/orion-graphiti-adapter/app/backends/graphiti_core.py` — the fix
- `services/orion-graphiti-adapter/app/main.py` — index bootstrap wiring at startup, `embed_url` passthrough to `ingest_episode`
- `services/orion-graphiti-adapter/app/settings.py`, `.env_example`, `docker-compose.yml` — new `GRAPHITI_AUTO_BUILD_INDICES` key, full env-parity surface
- `scripts/smoke_graphiti_search_e2e.sh` — now passes; also fixed independent of the main bug (duplicate-detection collision on repeated runs)

## Files changed

- `services/orion-graphiti-adapter/app/backends/graphiti_core.py`: `ensure_graphiti_indices()` (new), `_extract_crystallization_ids()` fixed for real `EntityEdge` shape, `_cast_embedding_to_vecf32()` (new, works around a graphiti-core FalkorDB gap), `_lookup_subject()` (new, Postgres subject lookup for link fact text), `_ensure_target_entity_stub()` (new, non-clobbering — checks existence before creating, since `EntityNode.save()` is a full property replace), `ingest_episode()` rewritten around `EntityNode`/`EntityEdge`
- `services/orion-graphiti-adapter/app/main.py`: calls `ensure_graphiti_indices()` once in `lifespan` when `graphiti_core`+FalkorDB are both active; passes `CRYSTALLIZER_EMBED_HOST_URL` into `ingest_episode`
- `services/orion-graphiti-adapter/app/settings.py`, `.env_example`, `docker-compose.yml`: `GRAPHITI_AUTO_BUILD_INDICES` (default `true`)
- `services/orion-graphiti-adapter/tests/test_graphiti_core_backend.py`: 6 new regression tests — `EntityNode`/`EntityEdge` field-mapping/attribute-flattening, self-referential edge always written, link edges + non-clobbering target-stub creation, real `EntityEdge`-shaped extraction
- `scripts/smoke_graphiti_search_e2e.sh`: known-FAIL header removed; two independently-random tokens per run instead of a fixed template (avoids duplicate-detection collision after repeated runs); explicit approve-status check instead of silently swallowing a failed approve
- `services/orion-graphiti-adapter/README.md`, `docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md`: updated with root cause, field mapping, and live verification evidence

## Schema / bus / API changes

- Added: `GRAPHITI_AUTO_BUILD_INDICES` env key (adapter-local, no schema/bus/API surface)
- Removed: none
- Renamed: none
- Behavior changed: `/v1/search` now returns real matches for real data (previously always empty). FalkorDB graph shape changed from custom `HAS_EPISODE`/`RELATED` edges to graphiti-core's native `RELATES_TO` edges — this is a projection rebuild, not a migration; canonical Postgres crystallization data is untouched (`canonical_mutated` remains `false` throughout). A fresh `/v1/rebuild` re-projects existing crystallizations into the new shape; the old shape's nodes/edges are simply superseded (same `uuid`/entity-id scheme, `MERGE` semantics mean re-ingest overwrites in place — no orphaned old-shape nodes were left behind on this host since backfill was re-run during verification).
- Compatibility notes: `orion_postgres` backend and neighborhood/BFS (Phase B) are unaffected — verified via `smoke_graphiti_links_e2e.sh` still passing.

## Env/config changes

- Added keys: `GRAPHITI_AUTO_BUILD_INDICES=true` (adapter-local)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-graphiti-adapter/.env_example`)
- Local `.env` synced: **not automatically** — `python scripts/sync_local_env_from_example.py` was run and is worth flagging as a real operational hazard found during this session (see Risks below); the correct live values were restored by hand afterward and verified.
- Skipped keys requiring operator action: none

## Tests run

```text
$ source venv/bin/activate && python -m pytest services/orion-graphiti-adapter/tests -q
21 passed, 2 warnings in 0.77s
```
Independently re-run by the orchestrator after the implementing agent's session, not taken on report alone.

## Evals run

No eval harness exists for this service (same gap noted in PR #993). The live e2e smokes below remain the deterministic gate.

## Docker/build/smoke checks

```text
$ curl -sS http://localhost:8640/health
{"service":"orion-graphiti-adapter","postgres":true,"falkordb_enabled":true,"backend":"graphiti_core"}

$ bash scripts/smoke_graphiti_links_e2e.sh
PASS smoke_graphiti_links_e2e seed=12b378e3-0b3a-456e-a190-691c73d7bfa6 linked=174f97fe-933e-467a-b588-3434ed8358dc

$ bash scripts/smoke_graphiti_search_e2e.sh
PASS smoke_graphiti_search_e2e seed=f6618e38-1b23-4321-8110-7e156b4e1a27 embed_used=true

$ redis-cli GRAPH.QUERY graphiti_temporal "MATCH ()-[r:RELATES_TO]->() RETURN count(r)"
count(r): 14 (0 before this fix; confirmed live before/after)

# Privacy check (real data, independent of the implementing agent's own report):
$ psql -c "select crystallization_id from memory_crystallizations where status='active' and governance->>'sensitivity'='intimate' limit 1" -> 0fc34990-541e-48f0-adca-aa8686bb1521
$ redis-cli GRAPH.QUERY graphiti_temporal "MATCH (n) WHERE n.crystallization_id = '0fc34990-541e-48f0-adca-aa8686bb1521' RETURN n.id"
(empty — confirmed absent)

# consolidation_gate.py / low_info_social.py / orion-memory-consolidation diff:
$ git diff origin/main -- orion/memory/consolidation_gate.py orion/memory/low_info_social.py services/orion-memory-consolidation/ | wc -l
0
```

All output above independently re-run by the orchestrator; not taken from the implementing agent's report alone.

## Review findings fixed

The implementing agent ran its own medium-effort code review pass (4 finder agents) before returning, and fixed:
- Link edges were silently dropping the `confidence` field present in the old raw-Cypher path — restored via `EntityEdge.attributes` (flattens the same way as entity attributes).
- Two independent, serial embed-host calls (node-name embedding, self-fact embedding) parallelized with `asyncio.gather` — halves embed-host round-trip latency on the ingest hot path.
- Two near-duplicate `vecf32` cast call sites merged into one parameterized helper.

Left as documented, not fixed (thin-seam judgment calls, not defects):
- `created_at` resets on re-ingest (the old schema had no such field at all, so this isn't a regression).
- Multi-step FalkorDB writes (entity save → vecf32 cast → edge save → vecf32 cast) aren't transactional; a partial failure raises a real 500 rather than corrupting silently, and re-sync is idempotent via `uuid` `MERGE`, so a retry self-heals.
- Narrow race window between `EntityNode`/`EntityEdge.save()` and the follow-up `vecf32` cast where a concurrent search could hit a List-typed embedding — avoiding it entirely means bypassing `.save()` for the embedded-write path, out of scope for this fix.

The orchestrator additionally independently re-read the full diff of `graphiti_core.py`, `main.py`, `settings.py`, and the README/spec updates (see Docker/build/smoke checks above for what was independently re-verified rather than trusted from the report).

## Restart required

Already executed live during this session:

```bash
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

If merged and redeployed elsewhere: same command, plus `GRAPHITI_AUTO_BUILD_INDICES=true` (or leave default) in that environment's `services/orion-graphiti-adapter/.env`.

## Risks / concerns

- **Severity: High (operational, not code) — flagging for Juniper explicitly.** Running `python scripts/sync_local_env_from_example.py` from repo root during this session **reverted the live activation state** set by PR #993: it reset `GRAPHITI_BACKEND` back to `orion_postgres`, `FALKORDB_ENABLED` back to `false`, cleared `CRYSTALLIZER_EMBED_HOST_URL`, and reset Hub's `GRAPHITI_ADAPTER_URL` back to the unreachable container-DNS form. The script appears to treat `.env_example`'s checked-in defaults as authoritative and overwrites local `.env` values that intentionally differ (e.g. this node's live-activated runtime state vs. the fresh-deployment default documented in `.env_example`), rather than only adding missing/new keys. Caught and manually corrected in this session (values restored, verified live afterward), but **anyone who runs that sync script on this host in the future will silently deactivate the graphiti_core backend again** unless this is fixed or the intentional deviation is otherwise protected. Not fixed in this PR (separate decision: should the sync script preserve local overrides for keys that already differ from the example, or should the intentional deployment-specific overrides live somewhere the sync script won't touch?) — recommend a follow-up.
- Severity: Medium — same concurrent-multi-agent-host caveat as PR #993: this host runs multiple Claude Code sessions that can rebuild/restart shared containers mid-task. Final state independently re-verified after the implementing agent's session ended.
- Severity: Low — `search()` still rebuilds the FalkorDB driver, `Graphiti` instance, and stub embedder/llm/cross-encoder clients fresh per request (noted as a follow-up in PR #993, not addressed here — out of this fix's scope).

## PR link

Not opened via `gh` (same reason as PR #993 — no token, SSH-only remote). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/graphiti-core-search-relates-to-schema?expand=1`
