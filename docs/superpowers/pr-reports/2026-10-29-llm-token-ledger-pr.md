## Summary

Orion now keeps a record of every model call: how many tokens went in, how many came out, and which turn or act asked for it. Before this, the gateway wrote the token count to its log and then lost it.

- One event per model call: `llm.call.usage.v1` on bus channel `orion:llm:call:usage`. orion-sql-writer stores each one as a row in the new table `llm_call_usage`.
- The LLM gateway records every call it sends to a model. This covers bus calls and both HTTP passthroughs (`/v1/chat/completions`, `/v1/messages`), streamed or not.
- The harness now sends the turn id on every model call (header `X-Orion-Correlation-Id`). The gateway now reads that header. Before this, every harness call logged `corr=-`.
- The harness warm relay records one kind of call itself: a chat turn with no GPU lease. That call goes through the FCC proxy, and the FCC proxy drops our headers, so the gateway cannot see the turn id.
- An eval checks the table after deploy. It checks that served calls have token counts, that the counts fit inside the measured time, and how many rows have a turn id.

## Outcome moved

"Cost per 1,000 tokens" and "joules per token" can now be computed. Before, they could not: no table held tokens per call, and nothing joined tokens to a turn. The sibling physiology report joins `llm_call_usage` to per-act energy (`grammar_acts.cost_json`, `wh_by_host`) on `correlation_id`.

## Table contract (for the physiology report agent)

Table `llm_call_usage`. One row per model call. Insert-only. Primary key `call_id`.

| column | type | meaning |
| --- | --- | --- |
| `call_id` | text PK | uuid4 hex |
| `recorded_at` | timestamptz | when the reply (or the stream's end) came back |
| `recorder` | text | `gateway` or `harness_relay` |
| `path` | text | `bus_rpc`, `http_openai`, `http_anthropic`, `relay_anthropic` |
| `correlation_id` | text | the turn or act id; NULL when the caller sent none |
| `route` | text | gateway route key (the lane), e.g. `agent`, `chat`, `metacog` |
| `model` | text | model the reply named, else the requested model |
| `served_by` | text | worker label, e.g. `circe-worker-chat` |
| `pool_role` | text | GPU-pool role that was granted |
| `holder` | text | who asked: bus source name, `http:anthropic`, `http:openai`, `harness:fcc_warm_relay` |
| `verb` | text | bus path only: `options.verb` |
| `node` | text | recorder's own node |
| `outcome` | text | `served`, `upstream_timeout`, `upstream_http_5xx`, `client_gone`, ... |
| `stream` | boolean | streamed reply |
| `prompt_tokens` | integer | ALL prompt tokens, KV-cache reuse included |
| `cached_prompt_tokens` | integer | of those, reused from the cache (not recomputed) |
| `completion_tokens` | integer | generated tokens |
| `token_source` | text | `usage` (reply's usage block), `timings` (llama.cpp timings), `none` |
| `duration_ms` | integer | gateway: grant -> reply, model time only. relay: request -> reply end, includes the pool line |
| `wait_ms` | integer | gateway: time in the GPU pool's line |
| `decode_tps` | double | llama.cpp's own `timings.predicted_per_second`; NULL when not reported |
| `created_at` | timestamptz | row insert time |

Indexes: `(recorded_at)`, `(correlation_id, recorded_at)`, `(route, recorded_at)`.

Join rules. Follow these or you count a call twice:

- Totals over time (tokens per hour, joules per token): `WHERE recorder = 'gateway'`.
- Tokens for one turn or act: `WHERE correlation_id = <id>`, all recorders. A relay row's call also has a gateway row, but that gateway row has `correlation_id` NULL.

## Current architecture

- The gateway logged `completion_tokens=… finish_reason=…` per bus call (`llm_backend.py`, log `llm_gateway_provider_result`) and stored nothing.
- The gateway's grammar windows (`app/grammar_emit.py`) sum `prompt_tokens`/`completion_tokens` per node per minute, for bus calls only. Their contract forbids correlation ids. They are aggregates, so they cannot answer "how many tokens did this turn use".
- HTTP passthrough calls (all harness traffic) had no token record at all.
- The harness sent `X-Orion-Correlation-Id` only from the warm relay. The gateway read only `x-request-id`. Live logs: all 115 `anthropic_passthrough` calls in 48 h show `corr=-`.
- Narrow token columns exist (`orion_metacognitive_trace.token_count`, `episode_distill_run`, `compression_jobs.llm_tokens_used`) and `dev_economics_ledger_log` (Claude dev usage, not Orion's models). None is per call and joinable.

## Architecture touched

- Contract: `orion/schemas/llm_call_usage.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- Shared helper: `orion/llm/call_usage.py` (token parser, bounded queue, publisher).
- orion-llm-gateway: producer, header read, publisher task.
- orion-harness-governor and `orion/harness`: header on spawns, relay recording, publisher task.
- orion-sql-writer: model, route, channel, migration.

## Files changed

- `orion/schemas/llm_call_usage.py`: new event schema and its join rules.
- `orion/llm/call_usage.py`: reads tokens from a reply dict or from a stream's first and last 4 KB (both OpenAI and Anthropic shapes from llama.cpp); bounded thread-safe queue; bus publisher.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: register `LlmCallUsageV1` and `orion:llm:call:usage`.
- `services/orion-llm-gateway/app/call_usage_emit.py`: builds and queues the gateway's event.
- `services/orion-llm-gateway/app/main.py`: records each bus call; starts the publisher.
- `services/orion-llm-gateway/app/passthrough_proxy.py`: records each passthrough call; keeps a stream's first 4 KB too (an Anthropic stream's prompt counts are in its first event).
- `services/orion-llm-gateway/app/anthropic_passthrough.py`: reads `X-Orion-Correlation-Id` first (also used by the OpenAI passthrough).
- `services/orion-llm-gateway/app/settings.py`, `.env_example`, `README.md`: `LLM_GATEWAY_CALL_USAGE_ENABLED`.
- `orion/harness/fcc_motor.py`: per-turn spawns send `X-Orion-Correlation-Id` via `ANTHROPIC_CUSTOM_HEADERS`; an inherited one is dropped; a value with a newline is refused.
- `orion/harness/fcc_warm_relay.py`: records no-lease turns (decodes gzip/deflate for its own copy).
- `services/orion-harness-governor/app/main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `HARNESS_LLM_CALL_USAGE_ENABLED`, publisher task, drain on shutdown.
- `services/orion-sql-writer/app/models/llm_call_usage.py`, `models/__init__.py`, `settings.py`, `worker.py`, `.env_example`, `README.md`: table, route, forced subscription.
- `services/orion-sql-db/manual_migration_llm_call_usage_v1.sql`: the table and indexes (additive, auto-apply allowlist).
- `config/metrics/metric_definitions.lock.json`: re-locked for the new channel.
- Tests: `orion/llm/tests/test_call_usage.py` (+ real captured replies in `fixtures/call_usage/`), `services/orion-llm-gateway/tests/test_call_usage_emit.py`, `services/orion-llm-gateway/tests/test_concurrent_handlers.py`, `orion/harness/tests/test_call_usage_harness.py`, `services/orion-sql-writer/tests/test_llm_call_usage_sql_shape.py`.
- Eval: `services/orion-llm-gateway/evals/run_call_usage_eval.py` (+ its test).
- CI: `.github/workflows/orion-fcc-warm-pool-tests.yml` runs the new harness and helper tests; `orion-durable-runs-tests.yml` gateway job triggers on the helper and schema.

## Schema / bus / API changes

- Added: schema `LlmCallUsageV1`, kind `llm.call.usage.v1`, channel `orion:llm:call:usage` (producers orion-llm-gateway, orion-harness-governor; consumer orion-sql-writer), table `llm_call_usage`.
- Added: the gateway reads request header `X-Orion-Correlation-Id`.
- Removed: none.
- Renamed: none.
- Behavior changed: gateway passthrough logs now show the harness turn id instead of `corr=-`.
- Compatibility notes: additive. sql-writer must be deployed (and the migration applied) before rows land; until then the events are published and dropped (pub/sub).

## Env/config changes

- Added keys: `LLM_GATEWAY_CALL_USAGE_ENABLED=true` (gateway), `HARNESS_LLM_CALL_USAGE_ENABLED=true` (governor). sql-writer: `orion:llm:call:usage` in `SQL_WRITER_SUBSCRIBE_CHANNELS`, `llm.call.usage.v1` in `SQL_WRITER_ROUTE_MAP_JSON`.
- Removed / renamed keys: none.
- `.env_example` updated: yes, all three services.
- local `.env` synced: gateway and governor keys added with `python scripts/sync_local_env_from_example.py --all-keys orion-llm-gateway orion-harness-governor orion-sql-writer`.
- Skipped keys requiring operator action: none left. sql-writer's two JSON lists differ from the example on this host, so the sync script left them alone; the new channel and route were added to the local `.env` by hand, keeping the host's other entries.

## Metric gate (CLAUDE.md, 6 steps)

1. **Where the numbers come from.** llama.cpp's own reply. OpenAI shape: `usage.prompt_tokens` (includes cache reuse), `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens`. Anthropic shape: `usage.input_tokens` (EXCLUDES cache reads) + `cache_read_input_tokens` + `cache_creation_input_tokens`, `usage.output_tokens`. When there is no `usage` (an OpenAI stream), llama.cpp `timings.prompt_n + cache_n` and `predicted_n`. Parser: `orion/llm/call_usage.py::token_counts`. Gateway bus path: `llm_backend.py` `raw_data` -> `result["raw"]` -> `call_usage_emit.record_bus_call`. Passthroughs: `passthrough_proxy._CallReport.finish`. Relay: `fcc_warm_relay.record_relay_call`. Time: `grammar_emit.CallClock.model_ms` (grant -> reply).
2. **Independence.** Not a new signal. It is the same llama.cpp counter that the grammar windows already sum per node per minute, now kept per call with a turn id. `decode_tps` is the same reading as the windows' `decode_tps_p50`. Nothing in this patch feeds it into a model or a cognition loop. If a later consumer uses both this table and the window totals, they are redundant.
3. **Why it measures what it claims.** A token count is the model's own tokenizer count, reported by the server that ran it. Cost per token is energy divided by tokens: an accounting identity, not a theory. No inference is made here.
4. **Live data.** 24 h of gateway logs (2026-10-29): 7,032 bus calls with a completion count; none missing, none zero; min 2, max 7,809; per-route medians 19 (metacog_background) to 756 (agent). Grammar windows, last 2 h: about 1,050 prompt tokens per call. Live probes through the real gateway today: OpenAI non-stream 14 prompt / 0 cached / 8 out; OpenAI stream 14 / 10 / 8 (from timings, no usage chunk); Anthropic stream 14 / 10 / 8; Anthropic stream "Name three colors" 16 / 0 / 18, and the reply streamed 17 text pieces (close to 18: a piece is not always exactly one token). Rest state: a counter ledger has no calm value. No calls means no rows, not a row with zero. The eval treats "no rows" as a failure, not as calm, because an idle-looking table and a broken writer look the same.
5. **Existing mechanism.** Checked: grammar windows (aggregates, no turn id by contract), `orion_metacognitive_trace.token_count` (metacog only), `episode_distill_run`, `compression_jobs.llm_tokens_used` (one job type each), `dev_economics_ledger_log` (Claude dev usage). None is per call and joinable. Extending the grammar windows was rejected: their contract says "no correlation ids".
6. **Reversibility.** Cheap. Two flags turn both producers off. The table is additive and has no readers yet except the planned physiology report.

## Tests run

```text
pytest orion/llm/tests/test_call_usage.py                                  10 passed
pytest services/orion-llm-gateway/tests services/orion-llm-gateway/evals   474 passed (468 tests + 6 evals)
pytest orion/harness/tests/{test_fcc_warm_pool,test_fcc_warm_pool_chat_binding,test_fcc_motor_cancel,
       test_gpu_lease_transport,test_fcc_chat_auto_memory,test_call_usage_harness}.py   61 passed
pytest services/orion-harness-governor/tests                               96 passed
PYTHONPATH=.:services/orion-sql-writer pytest services/orion-sql-writer/tests   1138 passed, 175 skipped
static gates (every pytest file in orion-static-gates.yml)                 645 passed, 2 skipped, 1 failed
  failed: tests/test_check_postgres_connection_headroom.py::test_a_crash_in_the_dedupe_path_still_cards_the_alarm
  -- not touched by this branch; it reads the live database (280/300 connections in use) on this host.
check_migration_required_by --base origin/main, check_env_key_single_source, check_metric_lineage --gate,
check_definition_drift --gate (after re-lock), check_env_template_parity, check_inner_state_registry,
check_body_economy_contract, check_async_routes_not_blocking, check_service_hostname_refs: pass
```

## Evals run

```text
pytest services/orion-llm-gateway/evals/test_call_usage_eval.py   4 passed
run_call_usage_eval.py against live llm_call_usage: NOT RUN -- table does not exist until deploy (UNVERIFIED).
```

## Docker/build/smoke checks

```text
Live calls through the running gateway (main code), parsed by this branch's token_counts: see metric gate step 4.
A built LlmCallUsageV1 envelope passed the bus's own payload validation (OrionBusAsync._validate_payload).
End-to-end (patched gateway -> bus -> sql-writer -> row): UNVERIFIED. Needs deploy.
No Docker build was run.
```

## Review findings fixed

Code review ran in a subagent on the branch diff. Findings and fixes:

- Finding (HIGH): sql-writer overwrote every row's `correlation_id` with the envelope's id. The envelope id is a fresh random UUID when the producer has none, or when the turn id is not a UUID. So "no turn id" rows got a random id, and non-UUID turn ids were replaced, which breaks the per-turn join.
  - Fix: the `LlmCallUsageSQL` write passes no envelope extras, the same as `HarnessTurnTraceSQL` (`services/orion-sql-writer/app/worker.py`).
  - Evidence: new test `test_envelope_path_keeps_the_payload_correlation_id_not_the_envelope_one` goes through `handle_envelope`. It fails without the fix (checked) and passes with it.
- Finding (MEDIUM): a per-turn FCC spawn with no GPU lease goes to the FCC proxy. The proxy drops the header, and the relay is not in that path, so no row has the turn id.
  - Fix: documented as a known gap in the schema docstring, the migration header, and Risks below. These calls are rare (about 8 in 48 h) and are still counted in totals.
  - Evidence: `orion/schemas/llm_call_usage.py` "Known attribution gaps".
- Finding (MEDIUM): the relay lost its row if `resp.aclose()` raised or was cancelled. That is the client-disconnect case.
  - Fix: the row is recorded in a `finally` around `aclose`.
  - Evidence: `orion/harness/fcc_warm_relay.py`; harness tests pass.
- Finding (LOW): the relay recorded even when the ledger flag or the bus was off. Its queue was also created lazily, without a lock.
  - Fix: the queue is now created at import. Recording stays off until the governor turns it on (`set_usage_recording`), and the governor turns it off again at shutdown.
  - Evidence: new test `test_relay_records_nothing_until_the_governor_turns_recording_on`.
- Finding (LOW): an `X-Orion-Correlation-Id` longer than 128 characters would fail the pool lease request (`turn_correlation_id` max 128).
  - Fix: the header is clipped to 128.
  - Evidence: new test `test_an_overlong_correlation_header_is_clipped_not_refused`.
- Finding (LOW): the new channel's comment block split another channel's comment from its entry.
  - Fix: moved in `orion/bus/channels.yaml`.
- Finding (LOW): a partial publish failure logged the whole batch as dropped.
  - Fix: `PublishError` carries the sent count, and the log shows sent and dropped.
- Finding (found by me while running the full suite): `test_concurrent_handlers` hung. `main()` now starts the ledger publisher, which runs forever, like the bus listener.
  - Fix: the test stubs the publisher. It also asserts that `main()` starts the publisher.

## Restart required

After merge, from the clean primary `main` checkout synced with `origin/main`:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The post-pull rebuild applies `manual_migration_llm_call_usage_v1.sql` itself (additive, has the REQUIRED-BY header). sql-writer's `create_all` also makes the table at boot. Deploy sql-writer first so the first events are not dropped.

Check after deploy:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select recorder, path, count(*), count(correlation_id), sum(prompt_tokens), sum(completion_tokens) from llm_call_usage where recorded_at > now() - interval '1 hour' group by 1,2"
```

## Risks / concerns

- Severity: low. Concern: `http_openai` callers (AI Town NPCs, the FCC proxy) mostly send no turn id, so those rows have `correlation_id` NULL. Mitigation: they still count in totals; a caller can send `X-Orion-Correlation-Id` to join.
- Severity: low. Concern: a bus call that never got a GPU grant and failed upstream with no tokens is not recorded (non-pool routes only). Mitigation: no tokens were spent; the grammar windows still count it.
- Severity: low. Concern: Anthropic-shape replies carry no `timings`, so `decode_tps` is NULL for most harness calls. Mitigation: `completion_tokens / duration_ms` gives a wall-clock rate; it is not stored as `decode_tps` on purpose.
- Severity: low. Concern: a per-turn FCC spawn with no GPU lease leaves no row with its turn id (the FCC proxy drops headers; the relay only covers the warm pool). Mitigation: rare (about 8 lease-less harness calls in 48 h); still in the totals; documented in the schema.
- Severity: low. Concern: the hub's own FCC bridge (`services/orion-hub/scripts/fcc_claude_bridge.py`) does not pass a turn id. Mitigation: out of scope; its calls are still counted.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2974

🤖 Generated with [Claude Code](https://claude.com/claude-code)
