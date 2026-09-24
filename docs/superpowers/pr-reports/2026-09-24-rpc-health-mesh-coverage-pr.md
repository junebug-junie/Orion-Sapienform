## Summary

Seven more services now report how long their calls to other services take, and how often those calls time out. Before this, only cortex-exec, cortex-orch and hub reported it. This is the "Mesh transport coverage" section of `docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md`.

**Stacked on #2312. Merge #2312 first, then retarget this PR to `main`.**

- **Shared core (`orion/core/bus`):**
  - `SharedRpcHealthSink` is a lock-guarded collection point. A bus that lives for one tick or one call, often inside `asyncio.run` on a worker thread, hands its stats to the sink before it is thrown away. The publish loop then moves them into the published window. Before this, those stats were thrown away with the bus.
  - `RpcHealthAggregator.absorb` merges one aggregator into another. `OrionBusAsync.take_rpc_health_aggregator` / `absorb_rpc_health` expose that on the bus.
  - `rpc_health_publish_loop(sinks=)` drains sinks into each window.
  - `RpcHealthPublisher` wraps start/stop. With `connect_bus=True` it retries the bus connection with backoff.
  - `http_health.normalize_id_path` collapses numeric ids, UUIDs and long hex strings in URL paths to `:id`, so hop keys stay bounded.
- **Services instrumented:** durable-runs, execution-dispatch-runtime, mind, thought, harness-governor, actions, embodiment. The governor also reports FCC motor time as `fcc:<served_model>`.
- **orion-fcc skipped:** the container runs only the third-party `fcc-server`. It has no `orion` package and no redis, so there is nothing to attach a bus to.
- **cortex-orch:** `svc.bus` is drained as a hop-only bus. This is a defensive fix for the verb-hop nit from #2312.
- **orion/harness/runner.py:** now carries `exit_code` from `fcc_nonzero_exit` error events, and records `probed_served_model`.

## Outcome moved

Before, per-hop latency baselines existed only for cortex-exec, cortex-orch and hub. The hops that most often hide slowness had no measurement at all:

- the thought→mind HTTP call
- durable-runs→gateway/cabinet/elastic controller
- mind→LLM gateway
- dispatch→exec
- governor→exec `:background`
- the FCC motor

Execution-dispatch and mind also threw away every tick's or call's RPC outcomes along with their short-lived bus. All of these now reach `RpcHealthSnapshotV1.channel_latency`, which feeds the log-only EWMA baseline from #2310.

## Current architecture

#2312 added per-hop `channel_latency`, the `record_hop_*` API, the HTTP timing transport, and publishers in cortex-exec, cortex-orch and hub. Every other service either:

- recorded `rpc_request` outcomes on a bus that nothing ever drained (durable-runs, actions, embodiment, governor), or
- opened a new bus per tick or per call and discarded it (dispatch, mind, most of thought).

## Architecture touched

- `orion/core/bus/{rpc_health,rpc_health_publish,async_service,http_health}.py`
- `orion/harness/runner.py`
- `services/{orion-durable-runs,orion-execution-dispatch-runtime,orion-mind,orion-thought,orion-harness-governor,orion-actions,orion-embodiment,orion-cortex-orch}`

## Files changed

- `orion/core/bus/rpc_health.py`: `absorb`, `SharedRpcHealthSink`, hop-key docs.
- `orion/core/bus/rpc_health_publish.py`: `sinks=`, pre-loop sink drain, `RpcHealthPublisher` (connect retry).
- `orion/core/bus/async_service.py`: take/absorb the aggregator.
- `orion/core/bus/http_health.py`: `normalize_id_path`, proxy note.
- `orion/core/bus/tests/test_rpc_health_sink.py`: the sink across threads and event loops, absorb caps, publisher retry and stop, pre-loop drain, normalizer.
- `orion/harness/runner.py`: `exit_code` from error events; `probed_served_model`.
- Per service: settings, `.env_example`, compose, README, lifespan/main wiring, `tests/test_rpc_health_publish.py` (durable-runs: `tests/test_rpc_health_http_hops.py`, `app/http_hops.py`).
- thought: `app/rpc_health.py`.
- governor: `record_fcc_hop` in `app/bus_listener.py`.

## Schema / bus / API changes

- Added: no schema or channel changes. Publishers use the existing `orion:rpc_health:snapshot` channel and `RpcHealthSnapshotV1` from #2312.
- New hop keys:
  - `http:<host>[:port]<normalized path>` from durable-runs and thought
  - `fcc:<served_model>` from the governor
  - request channels from every new publisher
- New organ ids (signal-gateway pass-through, unregistered, exogenous): `rpc_health_durable_runs`, `rpc_health_execution_dispatch_runtime`, `rpc_health_mind`, `rpc_health_thought`, `rpc_health_harness_governor`, `rpc_health_actions`, `rpc_health_embodiment`.
- Behaviour changed: `HarnessMotorResult.exit_code`, and so `HarnessRunV1.exit_code`, is now the real exit code on `fcc_nonzero_exit`. It was `None`.

## Env/config changes

- Added keys, in all seven services: `RPC_HEALTH_PUBLISH_ENABLED=true`, `RPC_HEALTH_PUBLISH_INTERVAL_SEC=30` (gt 0), `RPC_HEALTH_CHANNEL_LATENCY_ENABLED=true`. Code defaults match.
- `.env_example` updated: yes, including both deploy prerequisites.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys <7 services>`: yes, 21 keys added.
- Skipped keys requiring operator action: none. Pre-existing diverged keys were left alone: `DURABLE_RUNS_ELASTIC_RESTORATION`, `DURABLE_RUNS_ELASTIC_THERMAL_ENABLED`, `MIND_LLM_SYNTHESIS_ENABLED`, `ORION_VISUAL_ELASTIC_STATUS_ENABLED`.

## Tests run

```text
orion/core/bus/tests + orion/signals + orion/harness/tests   503 passed
orion-durable-runs      93 passed, 42 skipped (Postgres DSN tests; CI runs them)
orion-execution-dispatch-runtime (repo root)   63 passed
orion-mind              102 passed
orion-thought           442 passed, 12 skipped, 1 failed (test_mind_enrichment_defaults_off -- also fails on base)
orion-harness-governor  53 passed
orion-actions           172 passed
orion-embodiment        79 passed, 14 failed (same 14 fail on base: fixtures lack _abandon_participating_since_ms)
orion-cortex-orch       192 passed, 34 failed (same 34 fail on base feat/rpc-health-per-hop)
Static gates (from .github/workflows/orion-static-gates.yml): metric lineage PASS, definition drift PASS
(no re-lock needed), env template parity PASS, inner-state, system-health producers, async routes,
chat-route poachers, compose mounts, hostname refs, sentience instruments --static-only, journal
dispatch, control-surface parity: all PASS; env/schema-registry pytest 32 passed; git diff --check clean.
```

## Evals run

```text
None. This is transport telemetry, and no eval harness covers it. The durable-runs evals
(admission/gateway/elastic fairness) need ORION_ADMISSION_TEST_DSN and run in CI.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh <svc> build -> rc=0 for orion-durable-runs,
orion-execution-dispatch-runtime, orion-mind, orion-thought, orion-harness-governor,
orion-actions, orion-embodiment. Nothing brought up or deployed. The live path is UNVERIFIED.
```

## Review findings fixed

- Finding (BLOCKER): on the current equilibrium build, the legacy transport-metacog trigger reads every service's pooled timeouts and p95 (>=5s). These new publishers make LLM-bound calls by design, so the trigger would fire every cooldown.
  - Fix: PR #2310 already limits that trigger (timeouts and p95) to cortex-exec/cortex-orch. That build is now a hard deploy prerequisite, written into every new `.env_example` and the restart order below. No code change here.
  - Evidence: `origin/feat/transport-ewma-baseline-gate` `services/orion-equilibrium-service/app/service.py` has `LEGACY_TIMEOUT_SERVICES` guarding `build_transport_metacog_trigger_from_snapshot`.
- Finding: a Hub cancel (SIGKILL) was recorded as a successful FCC run, and the test passed only because it built the motor directly.
  - Fix: the runner now carries `exit_code` from error events.
  - Evidence: a new test drives the real `HarnessRunner` with the cancel event sequence; no hop is recorded.
- Finding: FCC timeouts before the first assistant event landed in `fcc:unknown`.
  - Fix: fall back to `probed_served_model`.
  - Evidence: test `..._keys_by_probed_model`.
- Finding: FCC docs said "subprocess wall time", and output-limit kills were counted as successes.
  - Fix: docs now say "motor leg wall time", and output-limit kills are skipped.
  - Evidence: parametrized skip test.
- Finding: if the publish bus failed to connect at boot, publishing stayed off for the life of the process.
  - Fix: `RpcHealthPublisher(connect_bus=True)` retries with backoff, used by dispatch, mind and thought.
  - Evidence: `test_publisher_connect_bus_retries_until_connected`, `test_publisher_stop_during_connect_retry_returns_promptly`.
- Finding: the first window could cover the whole time since process start, and `stop()` didn't await the cancelled task.
  - Fix: pre-loop sink drain; `stop()` awaits the task.
  - Evidence: the sink publish test asserts a pre-start hop is absent.
- Finding: an interval of 0 would busy-loop.
  - Fix: `gt=0.0` in all seven settings.
- Finding: thought only stopped the publisher on the success path.
  - Fix: stop moved into `finally`.
- Finding: the cortex-orch comment was misleading.
  - Fix: comment corrected. `dream_hunter.bus` was checked: it only publishes, so it has nothing to drain.
- Finding: passing `transport=` turns off httpx env proxies.
  - Fix: documented in `http_health`. The thought and durable-runs compose files and env have no proxy.
- Not fixed (nit): thought chain workers fold once per run, so a run longer than 30s lands in a later window. Nothing is lost. `normalize_id_path` does not collapse ULIDs or base62 ids; current paths are static and the 200-key cap is the backstop.

## Restart required

Deploy consumers first, in this order. The schema is `extra="forbid"`, and equilibrium's legacy trigger must already be limited to exec/orch.

```bash
# 1. #2312 consumers + #2310 equilibrium (after both merge)
scripts/safe_docker_build.sh orion-signal-gateway up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build   # must include PR #2310
# 2. #2312 producers (cortex-exec lanes, cortex-orch, hub) per #2312's report
# 3. this PR's producers
for s in orion-durable-runs orion-execution-dispatch-runtime orion-mind orion-thought \
         orion-harness-governor orion-actions orion-embodiment; do
  scripts/safe_docker_build.sh "$s" up -d --build
done
```

## Risks / concerns

- **High:** deploying any of these seven services before #2310's equilibrium is live would trigger transport metacog every cooldown.
  - Mitigation: the prerequisite is in every `.env_example` and in the restart order. Setting `RPC_HEALTH_PUBLISH_ENABLED=false` per service is the kill switch.
- **Medium:** channel latency defaults to true, unlike #2312's false. Deploying before signal-gateway and equilibrium are rebuilt makes every snapshot fail validation, and you get warning spam.
  - Mitigation: consumer-first order above; set `RPC_HEALTH_CHANNEL_LATENCY_ENABLED=false` to back out.
- **Low:** the `/v1/gpu-slots/activate` hop (1200s timeout) will have rare, long samples, so its baseline will be noisy.
- **Low:** the metric quality gate's live-data check (step 4) has not been done for the new hops. That needs a post-deploy pull of real windows.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2314 (base `feat/rpc-health-per-hop`; merge #2312 first, then retarget to `main`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
