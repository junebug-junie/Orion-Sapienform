# PR report: graphiti-core backend activation

## Summary

- Live-activated the already-merged `graphiti_core` backend on `orion-graphiti-adapter` (`GRAPHITI_BACKEND=graphiti_core`, `FALKORDB_ENABLED=true`) against a real FalkorDB container — this had never been run against live infrastructure before, only mocked unit tests.
- Found and fixed 3 crash bugs in `backends/graphiti_core.py` / `crystallization_ids.py` that made `/v1/episodes`, `/v1/rebuild`, and `/v1/search` 500 against real data and a real `graphiti-core==0.19.0` driver — invisible to the existing fully-mocked test suite.
- Backfilled all 59 `status=active` crystallizations into FalkorDB (58 landed; the 1 excluded is a real `sensitivity=intimate` row, correctly filtered — verified directly, not just via the synthetic unit test).
- Added `scripts/smoke_graphiti_search_e2e.sh`, a live (non-mocked) search smoke. It intentionally documents a **known FAIL**: `/v1/search` runs cleanly and calls the real embed host, but returns zero results for real ingested data — `graphiti-core==0.19.0`'s own search only reads its `RELATES_TO`-shaped edges, which this adapter's raw-Cypher `ingest_episode()` never writes. Root cause fully documented in the script header and README; not fixed here (out of scope — needs a follow-up decision on write path).
- Confirmed backend-agnostic neighborhood/BFS traversal (Phase B, `smoke_graphiti_links_e2e.sh`) is unaffected either direction.
- Confirmed rollback (`GRAPHITI_BACKEND=orion_postgres`) restores `/v1/search` → 501 with neighborhood unaffected, then re-forwarded to leave the live system in the activated state.

## Outcome moved

The `graphiti_core` backend went from "code exists, zero runtime evidence" to "running live against real FalkorDB with real Orion belief data, with the one broken capability (`/v1/search` result matching) explicitly failing loud instead of silently returning empty and looking fine." Per AGENTS.md's "runtime truth beats config truth" — this is the first real evidence this backend does or doesn't work, in either direction.

## Current architecture (before this patch)

`orion-graphiti-adapter` ran with `GRAPHITI_BACKEND=orion_postgres`, `FALKORDB_ENABLED=false`. The `graphiti_core` backend module existed and had 100% mocked test coverage (driver, `graphiti_core.Graphiti` class, and module import were all patched out — never touched a real FalkorDB instance or a real `graphiti-core` package call).

## Architecture touched

- `services/orion-graphiti-adapter` runtime env (live `.env`, not committed — gitignored, see Env/config below)
- `services/orion-graphiti-adapter/app/backends/graphiti_core.py` — bug fixes, no new capability added
- `services/orion-graphiti-adapter/app/crystallization_ids.py` — validation bug fix
- `scripts/` — one new live smoke, one existing smoke's propose payload fixed (pre-existing validator drift, unrelated to graphiti, but blocking this task's "when done" checklist)

## Files changed

- `docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md`: the spec this PR implements
- `scripts/smoke_graphiti_search_e2e.sh` (new): live, non-mocked `/v1/search` verification; documents the known-failing root cause in its header
- `scripts/smoke_graphiti_links_e2e.sh`: propose payload was missing `planning_effects`/`retrieval_affordances`, so `validate_proposal()` auto-quarantined every crystallization the script created (pre-existing drift in `orion/memory/crystallization/validator.py`, unrelated to this task; fixed because it silently broke this Phase B regression smoke)
- `services/orion-graphiti-adapter/README.md`: documents live runtime state, the search-smoke's known-fail, and clarifies neighborhood is backend-agnostic
- `services/orion-graphiti-adapter/app/backends/graphiti_core.py`:
  - `validate_crystallization_id()` call sites no longer reject bare-UUID `crystallization_id` values (real data shape; the `crys_` prefix is only used for derived ids)
  - `driver.execute_query()` calls fixed to pass a single params dict as keyword args, matching `graphiti-core==0.19.0`'s `FalkorDriver.execute_query(self, cypher_query_, **kwargs)` signature (previously passed a second positional dict, raising `TypeError` against the real driver)
  - Added `_no_op_llm_client()`, `_no_op_cross_encoder()`, `_orion_embedder_client()` — `Graphiti()` eagerly builds OpenAI-backed clients unless given explicit ones; there is no `OPENAI_API_KEY` configured and none of this adapter's usage needs an LLM (writes are raw Cypher, not `add_episode()` entity extraction). The embedder stub bridges to Orion's own `CRYSTALLIZER_EMBED_HOST_URL` instead of OpenAI.
- `services/orion-graphiti-adapter/app/crystallization_ids.py`: `validate_crystallization_id()` now accepts bare UUIDs (real shape) in addition to `crys_`-prefixed ids (derived-id shape); still rejects anything that isn't `[a-zA-Z0-9_-]+` (injection guard — moot in practice since all Cypher calls are parameterized, but kept as defense in depth)
- `services/orion-graphiti-adapter/tests/test_crystallization_ids.py`: regression test for the bare-UUID fix
- `services/orion-graphiti-adapter/tests/test_graphiti_core_backend.py`: regression test asserting `execute_query` call args match the real driver's calling convention; updated the search test's fakes for the new stub-client constructor signature

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `/v1/episodes`, `/v1/rebuild`, and `/v1/search` on `orion-graphiti-adapter` now work against real data when `GRAPHITI_BACKEND=graphiti_core` (previously crashed). `/v1/search` still returns empty `crystallization_ids` for real data — a pre-existing, now-documented gap, not a regression from this patch (there was no working `/v1/search` before this patch to regress from).
- Compatibility notes: `orion_postgres` backend and Phase B neighborhood/link behavior are byte-for-byte unchanged; rollback verified.

## Env/config changes

- Added keys: none (all keys already existed in `.env_example`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no (key shapes unchanged)
- Live runtime `.env` changes (gitignored, not committed, host-level only):
  - `services/orion-graphiti-adapter/.env`: `GRAPHITI_BACKEND=orion_postgres` → `graphiti_core`; `FALKORDB_ENABLED=false` → `true`; `CRYSTALLIZER_EMBED_HOST_URL=` → `http://orion-athena-vector-host:8320/embedding`
  - `services/orion-hub/.env`: `GRAPHITI_ADAPTER_URL` changed from `http://orion-athena-graphiti-adapter:8000` to `http://127.0.0.1:8640` — pre-existing bug unrelated to this task's plan, found because Hub runs `network_mode: host` and cannot resolve the adapter's container DNS name; the fix matches the pattern already used for `CRYSTALLIZER_EMBED_HOST_URL`/`RECALL_PG_DSN` in the same file. Without this fix, Hub-mediated smokes (`smoke_graphiti_links_e2e.sh`, `smoke_graphiti_search_e2e.sh`, the active-packet check) could not reach the adapter at all.
- skipped keys requiring operator action: none

## Tests run

```text
$ source venv/bin/activate && python -m pytest services/orion-graphiti-adapter/tests -q
15 passed, 2 warnings in 0.72s
```
(Independently re-run by orchestrator after the implementing agent's session, not just taken on report.)

## Evals run

No eval harness exists for `orion-graphiti-adapter`. The two live e2e smokes below are the closest equivalent and are the deterministic gate for this capability going forward.

## Docker/build/smoke checks

```text
$ curl -sS http://localhost:8640/health
{"service":"orion-graphiti-adapter","postgres":true,"falkordb_enabled":true,"backend":"graphiti_core"}

$ bash scripts/smoke_graphiti_links_e2e.sh
PASS smoke_graphiti_links_e2e seed=53d31c3a-e308-456c-bad9-33e6c11b1480 linked=b2918b5e-e8e1-40f2-820b-fccae45a7e52

$ bash scripts/smoke_graphiti_search_e2e.sh
FAIL smoke_graphiti_search_e2e: crystallization_ids missing seed=4c45a09e-e796-47cf-801d-0a6c310755e3
response={"crystallization_ids":[],"trace":{"backend":"graphiti_core","rails":["vector","graph"],
"query":"Graphiti search smoke subject zz20260713051855","embed_used":true,"result_count":0,
"raw_type":"list"}}
(expected FAIL — documented root cause in script header and README; exit 1 is correct, not flaky)

# Backfill verification (orchestrator, independent of agent's report):
$ psql -c "select count(*) from memory_crystallizations where status='active'"        -> 59
$ psql -c "select count(*) from memory_crystallizations where status='active' and
           governance->>'sensitivity'='intimate'"                                      -> 1
$ redis-cli GRAPH.QUERY graphiti_temporal "MATCH (n:Entity) RETURN count(n)"           -> 58
# 59 active - 1 correctly-excluded intimate = 58 in FalkorDB. Exact match.

# Privacy check (orchestrator, independent — corrects an inaccuracy in the implementing
# agent's own report, which claimed "0 intimate active crystallizations" when there is
# actually 1; the exclusion itself is verified correct on that real row):
$ redis-cli GRAPH.QUERY graphiti_temporal \
    "MATCH (n) WHERE n.crystallization_id = '0fc34990-541e-48f0-adca-aa8686bb1521' RETURN n.id"
(empty result — confirmed absent from FalkorDB)
```

All smoke/test output above was re-run independently by the orchestrator after the implementing subagent's session ended, not taken on the subagent's report alone.

## Review findings fixed

Code review skill was not run as a separate subagent pass for this PR — the orchestrator directly read every changed file (`graphiti_core.py`, `crystallization_ids.py`, both smoke scripts, README, both test files) and independently re-ran tests and both live smokes rather than trusting the implementing agent's self-report. One material inaccuracy was caught this way: the implementing agent's report claimed "0 intimate active crystallizations" in real data; there is actually 1, and the report is corrected above. No other discrepancies found between the agent's claims and independently observed behavior.

## Restart required

Already executed live as part of this activation (not pending):

```bash
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

If this PR is merged and redeployed on a different host/environment, the same command plus the three `.env` key changes listed above under Env/config changes must be applied there — they are runtime-only and do not travel with the git merge.

## Risks / concerns

- Severity: Medium — `/v1/search` (the headline new capability) does not find real data today. `scripts/smoke_graphiti_search_e2e.sh` is committed in a known-FAIL state on purpose (a failing gate is the correct signal per AGENTS.md, not a passing-but-lying test). Closing this requires a follow-up decision: either wire a real LLM client into `graphiti-core`'s own `add_episode()`/`add_triplet()` write path, or hand-write `RELATES_TO`-shaped edges with a synthesized fact + embedding to match what `graphiti-core`'s search actually reads. Both are larger patches with real design tradeoffs, out of scope here.
- Severity: Medium — this host had multiple concurrent Claude Code sessions active during implementation; one was observed rebuilding `orion-graphiti-adapter` from `main` mid-task, twice reverting the implementing agent's live fixes until redeployed. Final state is verified correct at time of this report (re-verified independently by the orchestrator after the fact), but the live container's correctness is not guaranteed to persist without this branch being merged and redeployed as the new baseline.
- Severity: Low — `search()` rebuilds the FalkorDB driver and stub `Graphiti`/embedder/llm/cross-encoder clients fresh on every `/v1/search` request. Not a correctness bug, flagged as a legitimate follow-up (connection reuse) not addressed here to keep this patch to the activation's actual scope.

## PR link

Not opened via `gh` (no `GH_TOKEN`/`.netrc` credential exists in this environment and the git remote is SSH-only, so there is no way to call the GitHub REST API to create a PR object without `gh`). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/graphiti-core-backend-activation?expand=1`
