## Summary

Harness model calls now reach the token ledger with their turn id. Before this fix, every harness row in `llm_call_usage` had no turn id, so tokens could not be joined to a turn or act.

- The FCC proxy now forwards the `X-Orion-Correlation-Id` header to llm-gateway. Before, FCC built a new upstream request and dropped the header.
- The warm relay's own token recorder (`recorder='harness_relay'`) is removed. It existed only because of this gap. With the header forwarded, it would count the same call twice for one turn.
- The new regression test runs the real patched FCC app against a stub upstream and checks the header that arrives there. It fails on the image now deployed and passes with the fix.

## Root cause (with evidence)

The harness sends the header correctly. FCC drops it.

1. The gateway log shows every harness call as `anthropic_passthrough corr=- model=harness route=harness`. The last hour had 36 such calls.
2. The live spawned `claude` process has the right environment (read from `/proc/<pid>/environ` in `orion-athena-harness-governor`):
   `ANTHROPIC_CUSTOM_HEADERS=X-Orion-Correlation-Id: 9da970f0-...` and `ANTHROPIC_BASE_URL=http://host.docker.internal:8082`. So the header is set, and the call goes to the FCC proxy on port 8082, not to the gateway.
3. The governor log shows `fcc_warm_path_decision ... decision=spawn:not_chat_reply`. This is a per-turn spawn with no GPU lease. It is the common harness path, not the rare one. FCC received 41 `POST /v1/messages` in the same hour that the gateway logged its 36 harness calls.
4. FCC's llama.cpp provider (`providers/transports/anthropic_messages/transport.py`) builds each upstream request with `_request_headers()`. That function returns only `Content-Type`.
5. Reproduced. The same probe was run against the live image (`orion-fcc-fcc`) and the fixed image. The stub upstream received:
   - live image: `Content-Type`, `Content-Length` and no `X-Orion-Correlation-Id`.
   - fixed image: `X-Orion-Correlation-Id: turn-abc`.
6. There are no `harness_relay` rows. That recorder covered only warm-pool turns with no lease. Every harness call seen in the window was a spawned turn.

PR #2974 was wrong about this. It called the no-lease spawn path "rare (about 8 calls in 48 h)". That count came from FCC's request log, and it was wrong.

## Outcome moved

After deploy, harness `http_anthropic` rows in `llm_call_usage` carry the turn id. The physiology report's per-act join (`grammar_acts` <-> `llm_call_usage` on `correlation_id`) can then match harness acts.

## Current architecture

Claude Code (per-turn spawn, no lease) -> FCC proxy :8082 -> llm-gateway `/v1/messages` -> llama.cpp. Before this fix, the turn id stopped at FCC.

## Architecture touched

- orion-fcc's adapter (`orion_fcc_messages_transport.py`) and its install patch.
- The harness warm relay and governor: the ledger recorder is removed.
- `orion:llm:call:usage` now has one producer, orion-llm-gateway.

## Files changed

- `services/orion-fcc/orion_fcc_messages_transport.py`:
  - `bind_correlation(request)` saves the header in a request-scoped context variable. It drops a value containing CR/LF and clips to 128 characters, the gateway's limit.
  - `forward_correlation_header()` wraps `AnthropicMessagesTransport._request_headers` so every upstream messages call carries the header. It is idempotent, and it fails loudly if a version bump moves the hook.
- `services/orion-fcc/install_transport_patch.py`: the patched route calls `bind_correlation(request)`, and importing the app installs the forward.
- `services/orion-fcc/tests/test_correlation_forward.py`: the regression test. It runs the real FCC app and the real llama.cpp provider in a fresh process. It checks streamed and non-streamed turns, a request with no header, the 128-character clip, and 8 concurrent turns that must never swap ids. It also checks that the forward is installed when the app is imported.
- `orion/harness/fcc_warm_relay.py`, `services/orion-harness-governor/app/main.py`, `app/settings.py`, `docker-compose.yml`, `.env_example`, `README.md`: the relay recorder, its publisher and `HARNESS_LLM_CALL_USAGE_ENABLED` are removed. These files are restored to their state before #2974.
- `orion/harness/tests/test_call_usage_harness.py`: the relay tests are removed. The spawn-header tests stay.
- `orion/schemas/llm_call_usage.py`, `orion/llm/call_usage.py`, the sql-writer model docstring and README: `harness_relay` is marked retired, and the join rules are updated. The value stays in the schema so old rows still validate.
- `orion/bus/channels.yaml`, `orion/llm/tests/test_call_usage.py`: the only producer is now orion-llm-gateway.
- `services/orion-durable-runs/app/physiology_report.py`: the per-act token join counts gateway rows only (review finding).
- `config/metrics/metric_definitions.lock.json`: re-locked for the producer change.

## Schema / bus / API changes

- Added: none.
- Removed: orion-harness-governor as a producer on `orion:llm:call:usage`.
- Renamed: none.
- Behavior changed: FCC forwards `X-Orion-Correlation-Id` upstream. `recorder='harness_relay'` is retired and still accepted.
- Compatibility notes: no table change.

## Env/config changes

- Removed keys: `HARNESS_LLM_CALL_USAGE_ENABLED` (governor). It is also removed from the local `.env`.
- Added / renamed keys: none.
- `.env_example` updated: yes.
- Skipped keys requiring operator action: none.

## Tests run

```text
docker build -f services/orion-fcc/Dockerfile -t orion-fcc-corrtest:local .   (fixed image)
  python -m unittest discover -s services/orion-fcc/tests     28 tests OK
same correlation tests against the live image orion-fcc-fcc               FAILED (failures=4, errors=1)  <- the regression test catches the bug
pytest orion/llm/tests/test_call_usage.py orion/harness/tests/{test_call_usage_harness,test_fcc_warm_pool,
       test_fcc_warm_pool_chat_binding,test_gpu_lease_transport}.py          48 passed
pytest services/orion-harness-governor/tests                                  96 passed
pytest services/orion-llm-gateway/tests services/orion-llm-gateway/evals      474 passed
check_definition_drift --gate (after re-lock), check_metric_lineage --gate,
check_env_template_parity, check_env_key_single_source                         pass
```

## Evals run

```text
None new. After deploy, run services/orion-llm-gateway/evals/run_call_usage_eval.py.
Its with_corr figure for gateway/http_anthropic should move from 0% toward 100%.
```

## Docker/build/smoke checks

```text
Built orion-fcc test image locally (not deployed). End-to-end probe in that image: stub upstream received X-Orion-Correlation-Id.
Live after deploy: UNVERIFIED.
```

## Review findings fixed

A code review ran in a subagent. It found no blocking issues. It checked that the turn-id value cannot leak between requests, that it reaches stream mode, non-stream mode, retries and recovery, and that the governor restore removed only the ledger lines.

- Finding (low): if FCC is deployed before the governor, the old governor's relay keeps writing `harness_relay` rows for lease-less warm turns. The matching gateway row now has the same turn id. The physiology report's per-act join had no recorder filter, so it would count those calls twice.
  - Fix: the per-act token join in `services/orion-durable-runs/app/physiology_report.py` now counts `recorder = 'gateway'` only. The restart section below deploys the governor before FCC.
  - Evidence: `pytest services/orion-durable-runs/tests/test_physiology_report.py` passes.
- Finding (low): a comment in `physiology_report.py` still described the retired relay recorder.
  - Fix: the comment is updated.
  - The applied migration's comments are left unchanged on purpose, so the migration file is not changed after it was applied.

## Restart required

After merge, from the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-fcc up -d --build
```

Check after deploy:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select path, count(*), count(correlation_id) from llm_call_usage where recorder='gateway' and recorded_at > now() - interval '30 minutes' group by 1"
```

## Risks / concerns

- Severity: low.
  - Concern: the forward depends on the FCC provider's `_request_headers` hook.
  - Mitigation: the install fails loudly if the hook moves, and a test confirms that importing the app installs the forward.
- Severity: low.
  - Concern: the physiology report's act join counts every recorder.
  - Mitigation: no relay rows exist, and none will be written now, so it cannot double count.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2981

🤖 Generated with [Claude Code](https://claude.com/claude-code)
