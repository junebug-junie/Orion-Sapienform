## Summary

- Per-hop latency (`RPC_HEALTH_CHANNEL_LATENCY_ENABLED`) now defaults to **on** for cortex-exec and cortex-orch, and Hub's RPC-health publishing is on. The template, settings defaults, compose defaults and READMEs now match what has run live since 2026-09-24.
- **Materiality floor for the transport baseline gate.** A spike or saturation only counts when latency is also at least `EQUILIBRIUM_TRANSPORT_BASELINE_MIN_EXCESS_MS` (250) above normal in absolute terms. Immaterial deviations are learned into the baseline instead of freezing it.
- **Review fix: the governor timeout grammar atom has its own flag,** `HUB_GOVERNOR_TIMEOUT_GRAMMAR_ENABLED`, default false. Turning on Hub publishing used to arm it too.
- **Review fix: an open saturation closes on a proportionally lower materiality floor** (125 ms), so the band doesn't collapse for fast hops.

## Outcome moved

- **No more millisecond-jitter spikes.** Live on 2026-09-24, a GPU-slot poll that normally takes about 10 ms read about 14 ms and opened a "spike". That can no longer happen.
- **Cortex hops are now baselined.** Before, cortex snapshots carried no per-hop latency, so they couldn't be. Live check after the restart: all four exec lanes and orch now carry `channel_latency`, and metacog's own call shows up as `orion:cortex:exec:request:background#log_orion_metacognition`.

## Current architecture

- **Where the flags were before:** the per-hop latency flags shipped `false`, holding for the consumer-first rollout. Hub publishing shipped `false` because of the fixed 5 s p95 gate, which PR #2310 has since removed.
- **How the gate judged spikes:** only on z-score and ratio, which ignore scale. So a 4 ms change on a 10 ms hop counted as a spike.

## Architecture touched

- **Services:** orion-cortex-exec, orion-cortex-orch and orion-hub (config defaults, plus the Hub governor client's timeout-atom gate).
- **Library:** `orion/metacog/transport_baseline.py`.
- **Equilibrium config:** one new key.

## Files changed

- `services/orion-cortex-{exec,orch}/{.env_example,app/settings.py,docker-compose.yml,README.md}`: latency flag on.
- `services/orion-hub/{.env_example,app/settings.py,docker-compose.yml,README.md}`: publishing on, the new timeout-atom flag, and corrected comments.
- `services/orion-hub/scripts/harness_governor_client.py`: the timeout atom is gated on its own flag.
- `orion/metacog/transport_baseline.py`: `min_excess_ms`, the materiality checks, and saturation close hysteresis.
- `services/orion-equilibrium-service/{app/settings.py,app/transport_baseline_gate.py,.env_example,docker-compose.yml,README.md}`: the new key is exposed.
- Tests: `orion/metacog/tests/test_transport_baseline.py` (3 new) and `services/orion-hub/tests/test_harness_governor_client_rpc_health.py` (updated).

## Schema / bus / API changes

- None. The gate's config fingerprint changes, so the baselines cold-start once when this deploys (logged). That is expected during the log-only week.

## Env/config changes

- **Added keys:**
  - `EQUILIBRIUM_TRANSPORT_BASELINE_MIN_EXCESS_MS=250` (equilibrium)
  - `HUB_GOVERNOR_TIMEOUT_GRAMMAR_ENABLED=false` (hub)
- **Defaults changed to true:**
  - `RPC_HEALTH_CHANNEL_LATENCY_ENABLED` (exec, orch, hub)
  - `RPC_HEALTH_PUBLISH_ENABLED` (hub)
- **Local `.env` synced** with `sync_local_env_from_example.py --all-keys` run from this worktree. The flag values and the removal of the dead cortex-exec keys (`CORTEX_METACOG_RETURN_LOGPROBS`, `..._LOGPROB_PROBE_MODE`, `..._UNCERTAINTY_PROBE_ENABLED`) were edited by hand, because the sync script only adds keys.
- **Skipped keys:** none.

## Tests run

```text
pytest orion/metacog/tests services/orion-equilibrium-service/tests services/orion-equilibrium-service/evals  -> 414 passed
pytest services/orion-hub/tests/test_harness_governor_client_rpc_health.py                                   -> 7 passed
scripts/check_env_template_parity.py                                                                         -> PASS
Mutation checks: min_excess_ms=0 -> immaterial test fails; close-floor proportional term removed -> hysteresis test fails;
                 atom gated on RPC_HEALTH_PUBLISH_ENABLED again -> flag-off test fails.
```

## Evals run

```text
services/orion-equilibrium-service/evals (mesh eval) -> pass (included above)
```

## Docker/build/smoke checks

```text
Live, env-only restart of cortex-exec (4 lanes), cortex-orch and hub via scripts/safe_docker_build.sh <svc> up -d:
100 s snapshot sample -> cortex-exec {background,chat,legacy,spark} and cortex-orch all carry channel_latency; hub publishing.
```

## Review findings fixed

- **Finding (material):** turning on Hub publishing also armed the governor `rpc_transport_timeout` atom. That atom fires transport metacog directly, with no baseline gate, and also fires on a failed liveness check.
  - **Fix:** a separate `HUB_GOVERNOR_TIMEOUT_GRAMMAR_ENABLED` flag, default false.
  - **Evidence:** the test asserts that publish-on with the flag off emits no atom, and the mutation check fails without the fix. Live: 0 governor-timeout transport triggers in the ~40 min since Hub was restarted.
- **Finding (medium):** the materiality term had no hysteresis, so the saturation band collapsed for hops with a floor of 500 ms or less.
  - **Fix:** an open episode closes on `min_excess_ms * (close_ratio-1)/(ratio-1)`.
  - **Evidence:** new test, mutation-checked.
- **Finding (nit):** stale Hub settings comments.
  - **Fix:** rewritten.
- **Accepted (low):** immaterial outliers now fold raw, which inflates `fast_var` a little on fast hops. The reviewer confirmed a real 400 ms excursion on a 100 ms hop still reaches about z 4.6. It is left as-is and will be revisited in the log-only week.

## Restart required

```bash
# after merge + pull, from a worktree
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build   # materiality floor (cold-starts baselines, logged)
scripts/safe_docker_build.sh orion-hub up -d --build                   # governor atom flag
```

## Risks / concerns

- **Severity: low.**
  - **Concern:** until Hub is rebuilt, the running Hub image still arms the governor timeout atom, because publishing is on.
  - **Mitigation:** none has fired so far. Rebuild Hub after merge.
- **Severity: low.**
  - **Concern:** 250 ms is provisional.
  - **Mitigation:** it is exposed as an env key, and the log-only week sets it.
