# feat(mesh-guardian): host-wide stability checks

## Summary

- Adds an observe-only check that runs every 60 s in `orion-mesh-guardian` and raises Hub Pending Attention cards for four failures. The 2026-10-02 incident hit all four, and the existing service checks saw none of them for 8 days:
  - **Crash loops:** a container's Docker `RestartCount` rises 3 or more times within 2 h. Applies to every container on the host, not only the roster.
  - **Slow-consumer kills:** the bus Redis `client_output_buffer_limit_disconnections` counter rises, meaning Redis dropped a pub/sub client and its queued messages.
  - **Snapshots on bus Redis and FalkorDB:** a background save running 15 min or longer (critical), a failed last save, or no save for 6 h with unsaved changes while saving is configured.
  - **Graph inflation:** `orion_bus_synapse` has more than 2× as many Channel nodes as the channel catalog. The catalog is read from the live repo mount at `/repo`, not from the copy baked into the image.
- Cards that fail to deliver are now logged. `NotifyClient` returns `ok=False` silently, and the guardian never checked it.
- Two latent problems fixed while verifying live:
  - The image has no `docker` CLI (Debian's `docker.io` package no longer ships it). Restart counts now come from the Docker Engine API over the mounted socket. The existing `docker compose` auto-remediation is still affected; see Risks.
  - stdlib logging was never configured, so every `logger.info` in the service was dropped.

## Outcome moved

The 10-02 incident replay (`evals/test_stability_incident_replay.py`, real numbers) gives the first card per check:

- stuck snapshot and graph inflation: cycle 0
- slow-consumer kill: 9 min, at the first disconnect after baseline
- crash loop: 56 min, at the third restart

The healthy baseline produces no cards. Before this patch, the incident produced 0 cards over 8 days.

## Metric quality gate

1. **Provenance:**
   - Docker Engine `RestartCount` (`GET /containers/{id}/json`)
   - Redis `INFO stats` / `INFO persistence` / `CONFIG GET save`
   - FalkorDB `GRAPH.RO_QUERY count(Channel)`
   - catalog size from `orion.bus.census.load_channel_catalog_names`
2. **Independence:** four different sources. The only coupling is that a crash-looping slow consumer moves both the restart and the disconnect signals, which on 10-02 rose 1:1. They are kept separate on purpose: a disconnect can happen without a restart, if the client reconnects in-process, and a restart can happen without a disconnect.
3. **Theory:** all four are definitional, not inferred.
   - A rising restart count means the process exited.
   - The disconnect counter is incremented by Redis when it closes a client over its buffer limit.
   - `rdb_current_bgsave_time_sec` is the age of the running save child.
   - Channel nodes are created by MERGE on each distinct channel name.
4. **Live data (2026-10-02):**
   - Incident values: 581 restarts and 581 disconnects; a save running 136,812 s; 118,112 nodes against a 332-entry catalog.
   - Rest values after the fix: highest RestartCount 5 and flat; disconnects flat at 581; `bgsave_sec` -1 when idle; 243 nodes against 332.
   - Every signal returns to a genuine rest state. Counters are compared as deltas, so 581 is not an absorbing state.
5. **Existing mechanism:**
   - `scripts/bus_core_health_watchdog.py` does crash-loop detection for `bus-core` only. It runs from host cron and alerts by marker file.
   - The guardian's roster probes only check point-in-time liveness.
   - Nothing covered the other three signals.
6. **Reversibility:** `MESH_GUARDIAN_STABILITY_ENABLED=false` turns it off. No schema, bus, or storage changes.

## Files changed

- `services/orion-mesh-guardian/app/stability.py`: new. Pure check logic, trackers, and an alert gate that re-sends at most once per 6 h per alert.
- `services/orion-mesh-guardian/app/service.py`: stability loop, collectors (Docker API socket, Redis, FalkorDB), and per-source isolation.
- `services/orion-mesh-guardian/app/attention.py`: logs undelivered cards.
- `services/orion-mesh-guardian/app/main.py`: enables INFO logging.
- `services/orion-mesh-guardian/app/settings.py`, `.env_example`, `docker-compose.yml`: four new keys.
- `services/orion-mesh-guardian/tests/test_stability.py`, `evals/`: tests and the incident replay.

## Schema / bus / API changes

None. Cards use the existing notify `/attention/request` with `heartbeat_name="stability"` and `context.event` set to the alert kind.

## Env/config changes

- Added keys: `MESH_GUARDIAN_STABILITY_ENABLED=true`, `MESH_GUARDIAN_STABILITY_INTERVAL_SEC=60`, `FALKORDB_URI`, `FALKORDB_BUS_GRAPH`.
- `.env_example` updated: yes.
- Local `.env` synced: yes, with `python3 scripts/sync_local_env_from_example.py orion-mesh-guardian --all-keys`. The default run skips these keys because they fall outside `SYNC_PREFIXES`. The sync also added the previously missing `HEARTBEAT_INTERVAL_SEC`.
- `check_env_template_parity.py orion-mesh-guardian`: PASS.

## Tests run

```text
pytest services/orion-mesh-guardian/tests services/orion-mesh-guardian/evals -q -> 70 passed
```

## Evals run

```text
evals/test_stability_incident_replay.py -> 4 passed
first alert per check: crash_loop 56 min, slow_consumer_kill 9 min, snapshot_stuck 0, graph_inflation 0; healthy baseline silent
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build -> restarts=0, /ready ok:true
live first cycle 04:14:33Z:
  stability cycle {'containers': {'containers': 95, 'max_restarts': 5},
                   'bus_redis': {'disconnections': 581, 'bgsave_sec': -1},
                   'falkordb': {'bgsave_sec': -1, 'channel_nodes': 243, 'catalog': 332}} alerts=0
probe loop errors since deploy: 0
```

## Review findings fixed

- **Finding:** a FalkorDB graph-query failure discarded FalkorDB's stuck-save alert, because both were in one try.
  - **Fix:** separate try.
  - **Evidence:** `test_one_broken_source_does_not_blind_the_others` now expects both `snapshot_stuck` alerts.
- **Finding:** crash history was keyed by container name, so a recreated container blended two histories.
  - **Fix:** keyed by `name@id`; the card still shows the name.
  - **Evidence:** `test_recreated_container_is_a_new_history`.
- **Finding:** the catalog file was read on the event loop.
  - **Fix:** `asyncio.to_thread`.
- **Finding:** the `redis is None` branch was dead.
  - **Fix:** removed. The error surfaces as a per-check error.
- **Found live, not in review:**
  - The docker CLI is missing from the image. Fixed by reading the Docker API over the socket; covered by `test_docker_restart_counts_reads_the_engine_api`.
  - INFO logs were dropped. Fixed with `basicConfig`.

## Restart required

Already deployed from the worktree. After merge, redeploy from main:

```bash
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build
```

## Risks / concerns

- **Medium: the existing auto-remediation shells out to `docker compose`, which this image does not have.** It is off (`MESH_GUARDIAN_AUTO_REMEDIATE=false`), so nothing breaks today, but turning it on would fail. Follow-up: install `docker-cli` plus the compose plugin in the Dockerfile, or drop the feature.
- **Low: live delivery of a stability card is UNVERIFIED.** No condition has fired since deploy. Cards go through the same `publish_transition` and notify path that delivered guardian cards on 2026-09-29, and failures are now logged as `attention card NOT delivered`.
- **Low: the 6 h re-send gate is in memory.** A guardian restart re-sends cards for conditions that are still present.
- **Low: thresholds are fixed constants** chosen from one incident and one healthy baseline. A legitimately long FalkorDB save, over 15 min on a much larger graph, would raise a false critical.
