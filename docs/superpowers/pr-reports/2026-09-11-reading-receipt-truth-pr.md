# Reading receipt truth and live queue recovery

## Arsonist summary

PR #2199's deployed reading tool reached Hub but the live database had not received its final additive migration. `enqueue_seeds()` selected `duplicate_of`; Postgres raised `UndefinedColumnError`, and the listener collapsed that useful local evidence into `reading_queue_unavailable`. The assistant then treated the failed tool call as durable acceptance and invented a description of an unread paper.

This patch keeps Postgres as the acceptance authority, validates the full `ok=true` receipt at Hub and MCP boundaries, and derives deterministic response grounding from correlated FCC `tool_use`/`tool_result` records. An unconfirmed recommendation now replaces model prose with acceptance-unknown truth; it cannot retain a queued/future-action claim or an unsupported source summary. A confirmed receipt preserves the response and appends the canonical request ID and status.

Worktree: `/mnt/scripts/Orion-Sapienform-reading-receipt-truth`; branch: `fix/reading-receipt-truth`; base: `52ca280cb` (current `main` after PR #2200).

## Root cause and deployed-path evidence

Observed bad-turn correlation: `f9f9d41d-11c7-4873-bef1-983c395e5c83`. Governor logs showed two selections of `mcp__orion-reading__recommend_reading`; their RPC correlations were `f94dd703-c7a7-4790-a00c-65d89749bb78` and `4d487ad6-ada8-4344-b723-05924f84b44a`.

Hub's full local exception for both calls was:

```text
ReadingListener.handle -> enqueue_reading -> enqueue_seeds -> ACTIVE_URL_SQL
asyncpg.exceptions.UndefinedColumnError: column "duplicate_of" does not exist
```

The structured tool error and correlated logs prove MCP discovery/execution, internal Pub/Sub RPC, Hub listener dispatch, `memory_pg_pool`, pool acquisition, the configured asyncpg connection, and the enqueue query were reached. `memory_pg_pool_ready dsn_configured=true` was also present. This rules out missing tool discovery, bus routing, listener registration, absent pool, and DSN initialization as the failing boundary. The demonstrated cause was live schema/code skew.

The stored bad turn in `chat_history_log` and `harness_turn_trace.run_artifact` contained the false queued/future-processing claims and the fabricated LLM-agent-benchmark description. The actual URL is the condensed-matter paper “Information dynamics of our brains in dynamically driven disordered superconducting loop networks.”

## Exact migration state observed

Before repair, live `world_pulse_read_seed` had 21 columns: the original 14 plus the seven Stage 2 columns `handoff_json`, `handoff_at`, `stage2_status`, `stage2_claimed_at`, `stage2_completed_at`, `stage2_error`, and `stage2_trace_id`. It had the base status check, a kind check limited to `finding`/`digest_item`, the Stage 2 status check, its primary key, and three World Pulse claim/run indexes.

Repository-order comparison showed:

1. `manual_migration_world_pulse_read_seed_queue_v1.sql` — present live.
2. `manual_migration_world_pulse_read_stage2_v1.sql` — present live.
3. `manual_migration_general_reading_v1.sql` — absent live.

The missing third migration owns `request_id`, `request_json`, `root_request_id`, `duplicate_of`, `stage2_result_json`, `landing_at`, the `reading` kind, and three `idx_reading_*` indexes. After the additive migration, the live table had exactly 27 columns, the kind constraint accepted `reading`, and all seven indexes were present. The deterministic schema/constraint/index query printed `MIGRATION_VERIFY_PASS`.

The first migration attempt waited behind the scheduled `pg_dump`. Its backend was inspected in `pg_stat_activity`; only this session's waiting `psql` backends were cancelled, and the backup was left untouched. No migration statement had committed. After `pg_dump` completed, the repository migration was applied atomically with a five-second lock timeout and 30-second statement timeout.

## Architectural boundary and files changed

- `orion/schemas/reading.py`: typed durable receipt, status receipt, and per-recommendation grounding outcome.
- `orion/world_pulse_read/tools.py`: validates explicit success, durable/matching IDs, and returns the complete wrapper to the MCP transcript; retains deterministic same-turn IDs.
- `orion/harness/reading_receipts.py`: correlates FCC tool records, aggregates retries, recognizes source fetch evidence, and enforces receipt-grounded output.
- `orion/harness/runner.py`, `finalize.py`, and `orion/schemas/harness_finalize.py`: carry and enforce receipt truth before draft materialization and again after voice finalization.
- `services/orion-harness-governor/app/bus_listener.py`: preserves receipt outcomes through all reply/finalization branches.
- `services/orion-hub/scripts/reading_listener.py`: validates row-backed results and logs sanitized `no_pool`, `connection_failure`, `schema_incompatible`, `enqueue_failure`, or `status_failure` categories.
- Focused Hub, MCP, harness, finalizer tests and `services/orion-hub/evals/test_reading_receipt_truth_eval.py`.
- Hub/governor README operational contracts and this report.

The server-owned Stage 1 materializer remains unchanged. Source readers still receive only `WebFetch`/`WebSearch` under strict MCP configuration. No Cypher, RDF, Graphiti, graph query, or graph-write capability was restored.

## Truth contract

1. Acceptance requires a full `ReadingToolResultV1` with `ok=true`, no contradictory error, a row-backed status, a nonempty seed ID, and the deterministic matching request ID.
2. Tool errors, timeouts, malformed/mismatched receipts, and missing results produce `acceptance=unknown`.
3. Unknown acceptance discards all model prose. The deterministic replacement can therefore not preserve semantic variants of queued/saved/logged/future-action claims.
4. Failed retries aggregate under the deterministic request ID without an arbitrary retry ceiling. A later valid same-turn receipt can resolve earlier uncertainty.
5. The replacement says the source was not read unless a successful, nonempty same-source WebFetch, Firecrawl scrape, or current-turn Context Mode fetch result was correlated in the transcript. A Context Mode `Cached:` instruction is not current-turn read evidence.
6. Accepted outcomes append the canonical durable request ID and current status. `reading_status` independently validates its returned request ID.

## Tests and evals

Focused final receipt regression:

```text
27 passed
```

Combined receipt/finalize/MCP/eval checks:

```text
36 passed, 1 skipped
```

Governor-specific checks with its service import path:

```text
28 passed
```

Final full affected reading/Postgres/migration gate:

```text
483 passed, 1 skipped, 103 warnings in 34.44s
```

That gate covered Hub ingress and real disposable-Postgres integration, all World Pulse reading tests, curiosity callers, turn orchestration, MCP config/protocol, FCC motor, receipt/finalize tests, root reading tests, and the new eval. Warnings were pre-existing Pydantic/FastAPI/pytest-asyncio deprecations.

Hub reading evals, run separately from the Hub directory to avoid the repository's two unrelated `scripts` packages colliding in one Python process:

```text
11 passed in 1.15s
```

An initial attempt to combine those evals into the repository-root pytest process stopped during collection with `ModuleNotFoundError: scripts.world_pulse_read_pipeline`; the separated invocation above is green. No test body failed in that assembly attempt.

An expanded harness-related run produced `369 passed, 1 skipped, 3 failed`. All three failures reproduced unchanged on current `main`: two missing `mind_coloring` template fixtures and the known `fcc_timeout` grounding assertion. No unrelated fix is mixed into this branch.

Static checks passed: metric lineage, zero definition drift, stdlib shadowing, service hostnames, compose relative mounts, host Claude config mounts, journal registry, async route blocking, Python compilation, and `git diff --check`. No new metric or telemetry signal was introduced, so the metric quality gate does not apply beyond confirming definition drift remained zero.

## Migration, builds, and deployment actually executed

```bash
docker exec -i orion-athena-sql-db sh -lc \
  'PGOPTIONS="-c lock_timeout=5s -c statement_timeout=30s" psql \
   -v ON_ERROR_STOP=1 -U postgres -d conjourney --single-transaction -f -' \
  < services/orion-sql-db/manual_migration_general_reading_v1.sql

scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-harness-governor build
scripts/safe_docker_build.sh orion-hub up -d --no-deps hub-app
scripts/safe_docker_build.sh orion-harness-governor up -d --no-deps harness-governor
```

Only Hub and the harness governor were rebuilt/restarted. Hub and governor health checks passed, deployed image IDs matched the built images, and in-container source checks confirmed the new code. Operator configuration used the existing Tailscale Redis URL. No env key or `.env_example` changed; env synchronization was not applicable.

## Sanitized live smoke

The first real Unified Chat smoke used session `reading-receipt-live-smoke-20260911` and correlation `79da9740-9cde-40b0-8112-69f4eaa03fc5`. The model fetched the URL through Context Mode and corrected the paper description, but did not invoke `recommend_reading`. Its stored `reading_receipts=[]`, and it made no queue/persistence claim. This verifies the no-tool boundary but is not presented as queue acceptance.

The same deployed stdio MCP configuration then exercised the real MCP protocol, internal bus RPC, Hub, and Postgres. Durable receipt:

```text
request_id=f3283057-e329-59e6-8ce9-72287aa58510
seed_id=reading:f3283057-e329-59e6-8ce9-72287aa58510
status=queued
requested_by=juniper
invocation_context=unified_chat
parent_run_id=reading-live-smoke-20260911
parent_trace_id=reading-live-smoke-20260911
url=https://arxiv.org/abs/2310.19279
```

The returned wrapper was `ok=true,error=null`; a deployed `reading_status` call returned the same ID and queued state. The indexed Postgres row had priority 0, kind `reading`, matching root request, null duplicate, and pending Stage 1/2 state.

Stage 1 was forced once through the deployed worker class because the live schedule was outside its configured window. It used restricted source tools and completed under trace `76188ff9-a530-4371-888b-863a566f1f1a`:

- row `status=done`, `handoff_len=5295`, four priors, four concepts, four open threads;
- Wallet A moved 5/6 → 6/6 for local date 2026-09-10;
- stable journal entry `30e0105c-dad4-5182-ac7b-3e9da7dd87d2` exists with source ref `world_pulse_read:76188ff9-a530-4371-888b-863a566f1f1a`, the exact URL, and request ID;
- Concept Atlas returned all four deterministic nodes (`fluxon`, `Josephson junction oscillator`, `disordered superconducting loop network`, and `topological abstraction of network information flow`) as proposed concepts with confidence/salience 0.5.

Stage 2 was then forced once through the deployed worker class. Wallet B moved 0/6 → 1/6 and the exact target was claimed under correlation `49aa2cad-8328-4291-aabc-1f414ee1acca`. The live process exposed exactly `WebFetch,WebSearch` and strict MCP config. It fetched the source twice and produced a 2,300-character source-grounded draft after 39 recorded steps, but voice finalization rewrote the required JSON into ordinary prose. The Stage 2 parser rejected it at the correct boundary rather than fabricating a structured result:

```text
stage2_status=failed
stage2_trace_id=null
stage2_result_json=null
landing_at=null
failure=Could not parse JSON object from LLM text
```

Per the requested smoke rule, the lifecycle stopped there. No row reset, retry, journal repair, landing confirmation, or `completed` claim was attempted. A final deployed `reading_status` MCP call returned a valid `ok=true` status wrapper for the same request and truthfully reported `status=failed`, `stage1_status=done`, `stage2_status=failed`, the Stage 1 trace/summary, and null Stage 2 trace/landing. The Stage 1 journal and Concept Atlas artifacts remain durable.

No generated `graphify-out` artifact was changed or added. This follows the post-#2200 source-only Graphify workflow.

## Review findings fixed

- An arbitrary 20-attempt grounding cap could discard later receipts: removed; retries aggregate without a fixed ceiling.
- An enqueue `ValueError` could bypass sanitized database handling: command validation and enqueue phases are now separate.
- Quoted/space-separated libpq passwords could survive redaction: URI and keyword DSN patterns are covered by regressions.
- Empty fetch results and malformed non-list FCC content could be misclassified/crash: both now fail closed.
- A Context Mode cache-hit instruction could count as a fresh read: exact `Cached:` results are rejected as current-turn fetch evidence.

Independent subagent review found no remaining material issue after those fixes.

## Rollback

Revert this branch's code commit and rebuild/redeploy Hub and the harness governor with the same safe wrapper. The additive database migration should normally remain: old code ignores its nullable columns and widened kind constraint, while dropping them risks destroying request lineage/results created after deployment. If database rollback is nevertheless required, first stop reading ingress/workers, preserve affected rows, obtain explicit destructive-operation approval, and use a separately reviewed down migration. No automatic destructive rollback is supplied.

## Remaining risks

- Unified Chat model selection remains nondeterministic: the live chat chose direct Context Mode fetch instead of the explicitly requested recommendation. The actual deployed MCP rail was therefore exercised separately and is clearly labeled.
- Redis Pub/Sub tool RPC and lifecycle notifications remain ephemeral. Postgres receipts/state are authoritative; an RPC timeout correctly remains acceptance unknown even if a commit may have raced it.
- Source-read evidence proves a successful fetch result in the same FCC turn, not semantic correctness of the resulting summary. Stage outputs remain attributed candidates.
- The live Stage 2 finalizer can violate a structured-output imperative by converting valid JSON-shaped work into prose. The parser fails closed, but Wallet B is still spent and the request cannot land without a separately scoped repair/retry. This patch does not broaden into Stage 2 structured-output architecture.
- Repository-wide harness tests have the three unchanged baseline failures listed above.

## Non-goals

No new queue, service, wallet, scheduler, intent router, graph writer, reading taxonomy, or source-summary feature. The patch repairs the existing acceptance seam and makes its truth observable.

## PR link / completion status

[PR #2201](https://github.com/junebug-junie/Orion-Sapienform/pull/2201). **DONE_WITH_CONCERNS** — the queue migration, durable receipt contract, deterministic false-success prevention, deployment, Stage 1, journal, Concept Atlas, and status lookup are verified. The live request correctly remains failed rather than completed because Stage 2 voice finalization destroyed its required JSON shape, as documented above.
