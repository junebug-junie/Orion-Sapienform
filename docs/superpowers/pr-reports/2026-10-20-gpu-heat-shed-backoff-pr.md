# fix(gpu-pool): back off from heat-shed refusals instead of retrying every call

## Summary

- When the cabinet is hot, the GPU pool refuses background/system work at once with `shed:cabinet_hot`. That part is correct and is not changed.
- The callers did not wait. vision-council asked again for every camera window (~every 6 s). topic-foundry asked once per topic or boundary in a batch, hundreds of times a minute. Live 2026-10-20: 14,257 refused leases in 14 h.
- New shared rule `orion/gpu_pool/shed_backoff.py`. After a heat-shed refusal, later calls wait 30 s. Each further refusal doubles the wait, up to 180 s, with +-20 % jitter. One call per window probes. The first real answer ends the wait at once.
- The rule runs in two places. The pool client (inside orion-llm-gateway, per lease priority) stops the refusals from reaching the pool. vision-council and topic-foundry also run it, so their skipped calls do not reach the gateway.
- The gateway now tells callers how long to wait (`retry_after_sec`, plus a `Retry-After` header on HTTP).
- Each backoff episode logs one line when it starts and one line when it ends, with the count of skipped calls. Nothing is logged per call.

## Outcome moved

Replay eval (`python -m orion.gpu_pool.evals.shed_backoff_replay --from-db 14`). The input is the real `admitted` times from `gpu_pool_events` for the last 14 h, replayed as if the shed lasted the whole window:

| caller | pool attempts/h before | after |
|---|---|---|
| vision-council | 678 | 18 |
| orion-topic-foundry | 232 | 0.4 |
| cortex-exec | 68 | 10 |
| orion-memory-consolidation | 9.7 | 0.3 |
| total | 995 | 29 |

Recovery: once the heat clears, vision-council is served again within 204 s in the worst case (synthetic replay over 98 clear moments). The bound is 180 s cap + 20 % jitter + one 6 s window.

## Current architecture

- `orion/gpu_pool/scheduler.py` rule U4/D3 refuses a one-shot request of a shed priority at once with `Unavailable("shed:<name>")`. The pool sends no retry-after hint. Its shed signals are refreshed every hardware-watch tick (30 s), so `valid_until` says nothing about when the heat clears.
- `orion/gpu_pool/client.py gpu_lease` raised `LeaseUnavailable(reason)`. orion-llm-gateway turned that into an empty chat result with `raw.error=gpu_pool_unavailable` and `raw.details.reason=shed:cabinet_hot`.
- vision-council (`app/main.py _call_llm_raw`) and topic-foundry (`app/services/llm_client.py request_json`) treated that result as an empty reply. They moved on to the next work item and called again at once.

## Architecture touched

- Shared library: `orion/gpu_pool/shed_backoff.py` (new), `orion/gpu_pool/client.py`.
- orion-llm-gateway: bus refusal result and HTTP passthrough 503.
- orion-vision-council, orion-topic-foundry: a caller-side gate in front of the gateway RPC.

## Files changed

- `orion/gpu_pool/shed_backoff.py`: `ShedBackoff` state machine and the `gateway_shed_refusal` reply parser.
- `orion/gpu_pool/client.py`: a per-priority gate in `gpu_lease`, used only for one-shot requests (never holds, attach children or retryable leases). `LeaseUnavailable` gains `retry_after_sec` and `local`. The acquire exchange moved unchanged into `_acquire`.
- `services/orion-llm-gateway/app/main.py`, `app/passthrough_proxy.py`: add `retry_after_sec` / `Retry-After`. Locally answered refusals log at debug.
- `services/orion-vision-council/app/main.py`: skip the LLM call while the route is shed.
- `services/orion-topic-foundry/app/services/llm_client.py`: skip the call while shed (returns `None`, as a refusal did). `_bus_chat` is replaced by `_bus_chat_result` so the client can read `raw`.
- Tests: `orion/gpu_pool/tests/test_shed_backoff.py`, `test_shed_backoff_replay.py`, `conftest.py` (resets the process-wide backoff), `services/orion-llm-gateway/tests/test_shed_backoff_hint.py`, `services/orion-vision-council/tests/test_shed_backoff_wiring.py`, `services/orion-topic-foundry/tests/test_llm_client_shed_backoff.py`.
- Eval: `orion/gpu_pool/evals/shed_backoff_replay.py`.
- READMEs of the three services.

## Schema / bus / API changes

- Added: optional `raw.details.retry_after_sec` on gateway `gpu_pool_unavailable` bus results for shed refusals. Optional `error.retry_after_sec` and a `Retry-After` header on HTTP passthrough 503s for shed refusals.
- Removed: none.
- Renamed: none.
- Behavior changed: during a heat-shed episode, one-shot leases of the shed priority are refused locally, with the same reason string, without a pool RPC, except for one probe per window.
- Compatibility notes: `raw` and `details` are free-form dicts, so the new key is additive. No registered schema changes. The pool is untouched.

## Env/config changes

- Added keys: none. Backoff constants live in `shed_backoff.py`, with the reasoning next to them.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest orion/gpu_pool/tests services/orion-vision-council/tests -q  -> 596 passed
cd services/orion-llm-gateway && pytest tests -q                    -> 459 passed (4 new)
topic-foundry new file                                              -> 5 passed
cd services/orion-topic-foundry && pytest tests -q (local venv lacks joblib/sklearn: 4 files not collectable)
                                                                    -> 113 passed, 1 failed (test_chat_corpus_builder_stages, fails on main too)
cd services/orion-gpu-pool && pytest tests -q                       -> 194 passed, 15 skipped
cd services/orion-dream && pytest tests -q                          -> 270 passed, 7 skipped
Mutation check (after review fixes): 22 mutants (no growth, no cap, no clear on success, all reasons shed, neutral
removed, multiple probes, any call grows, any call clears, jitter bound, client gate removed, client gates retryable,
queued-probe timer removed, abandon on cancel removed, vision-council gate/resolve removed, topic-foundry gate/hint/None
guard removed, gateway hint/local removed, gateway warning always on, passthrough header removed) -> 22 killed, 0 survived
```

## Evals run

```text
python -m orion.gpu_pool.evals.shed_backoff_replay --from-db 14
  TOTAL before 994.6/h, after 28.6/h
python -m orion.gpu_pool.evals.shed_backoff_replay   (synthetic, live cadences)
  TOTAL before 1027.8/h, after 31.7/h
  worst delay after heat clears: vision-council 204 s, cortex-exec 144 s (its own arrival gap)
Gate: orion/gpu_pool/tests/test_shed_backoff_replay.py
```

## Docker/build/smoke checks

```text
No new dependencies (stdlib only). All three Dockerfiles COPY the whole orion/ package.
Docker build not run. Live path UNVERIFIED until deploy: after restart, expect one gpu_shed_backoff_start line per
service and a drop in gpu_pool_events 'unavailable' rows with reason 'shed:cabinet_hot'.
```

## Review findings fixed

- Finding (should-fix): if the heat cleared, the probe could sit in the pool queue behind real work, and every other call in the gateway was refused locally under a false `shed:` reason for up to 120 s.
  - Fix: if the pool queues a probe instead of refusing it, the episode ends after 3 s (`QUEUED_PROBE_CLEARS_SEC`). The pool refuses a shed lease on its first 1 s tick, so 3 s is enough. `PROBE_STALE_SEC` drops from 120 s to 60 s and is now only a backstop.
  - Evidence: `test_client_probe_the_pool_queues_ends_the_episode`, and the mutant "client no queued timer" is killed.
- Finding (should-fix): any non-shed reason ended the episode, including `deadline`, `cancelled` and pool-unreachable. One short-deadline caller could make the episode flap.
  - Fix: those reasons are now "neutral". They release the probe but neither end the episode nor make the wait longer. Only a grant, or a refusal the pool made after judging the lease, ends the episode.
  - Evidence: `test_neutral_outcome_releases_the_probe_without_ending_or_growing`, `test_client_probe_deadline_is_neutral`, and topic-foundry `test_gateway_deadline_releases_the_probe_without_ending_the_episode`.
- Finding (should-fix): topic-foundry raised AttributeError when the worker thread returned None. That path is not about shedding, and before the patch it returned None.
  - Fix: a `None` result releases the probe and returns None.
  - Evidence: `test_worker_thread_returning_none_is_a_failed_call_not_a_crash`.
- Finding (nit): a refusal that was already in flight could make the wait longer or let a second probe through. A late grant from before the episode could end it.
  - Fix: `admit()` gives the probe a token. Only the outstanding probe's outcome makes the wait longer or ends the episode.
  - Evidence: `test_only_the_probe_moves_the_episode`, and the mutants "any call grows" and "any call clears" are killed.
- Finding (nit): downstream consumers could not tell a pool refusal from a refusal the gateway answered locally.
  - Fix: `details.local=true` (bus) and `error.local=true` (HTTP).
  - Evidence: `test_bus_reply_carries_retry_after_on_a_shed_refusal`.
- Finding (nit, tests): the gateway tests had no backoff reset, and no test covered a cancelled probe.
  - Fix: an autouse reset in `services/orion-llm-gateway/tests/conftest.py`, plus `test_client_cancelled_probe_is_released`.
  - Evidence: the tests pass, and the mutant "client no abandon on cancel" is killed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-topic-foundry up -d --build
```

Run these from the clean primary `main` checkout after merge. orion-world-model and orion-thought also import `gpu_lease`. They pick up the gate on their next rebuild. Neither showed shed refusals in the last 3 days, so no restart is needed for them.

Live check after deploy:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select date_trunc('hour',created_at), holder, count(*) from gpu_pool_events where event='unavailable' and reason like 'shed:%' and created_at > now()-interval '3 hours' group by 1,2 order by 1,3 desc;"
```

## Risks / concerns

- Severity: low. Concern: after the heat clears, system-priority work on the gateway (cortex-exec, memory, dream) can wait up to ~3.6 min before its next call reaches the pool. Mitigation: the cap was chosen against the 30 s reflex tick. The shed lifts only after the cabinet cools past rearm, which takes minutes anyway.
- Severity: low. Concern: a topic-foundry batch that runs while the heat clears can skip the rest of its burst, up to ~3.6 min of calls. Those calls return `None`, as refused calls did before. Mitigation: the episode-end log line counts them. Re-running such a batch is a separate follow-up.
- Severity: low. Concern: the backoff state lives in each process. Several gateway replicas would each probe on their own schedule. Mitigation: today there is one gateway, and the worst case is N× ~20 probes/h.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
