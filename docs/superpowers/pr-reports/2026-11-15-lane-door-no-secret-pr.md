# Lane door: no shared secret, no manual deploy steps

## Summary

- The lane door no longer uses a shared secret. The gateway sends only the lease id it already has
  (`X-Orion-Lease-Id`). The door marks each call `leased`, `unleased` or `invalid`. Enforce mode refuses a
  model call that has no lease id. The key, its settings, the gateway's key-drift check and the key hash
  on `/door/stats` are deleted, with no fallback.
- The door is for catching ACCIDENTAL bypass by Orion's own evals and scripts. It is not a fence against
  a deliberate caller. The spec now says this plainly: the Hub holds the docker socket on athena, so a
  secret raised no real bar. A made-up lease id is still caught offline: the report joins lease ids to
  `gpu_pool_events`.
- Deploying the doors needs no runbook steps. circe's post-pull rebuild now starts the four circe doors.
  On the first deploy only, it also moves the running chat, agent and fast engines from their public lane
  ports to loopback. hecate's `hecate-deep.sh up` starts the agent-deep door, then its engine.
- A door takes its lane port only once its engine answers on loopback. So a door never steals a port an
  engine still publishes, and the deploy order does not matter.
- Doors ship ON in shadow mode. On a redeploy, a door lets calls in flight finish (up to 100 s) before it
  exits.

## Outcome moved

Juniper does not have to generate, copy or rotate a secret, and does not have to run any per-lane
commands. A normal pull deploys the doors. The one sudo line left (the kev 8009 firewall reinstall) is
unrelated to the doors.

## Current architecture

PRs #3417/#3422 shipped the door with a per-deployment secret (`LLM_GATEWAY_LANE_DOOR_KEY` /
`LANE_DOOR_KEY`). The doors were excluded from every automatic rebuild. Starting a door had to happen by
hand, together with moving its engine off the lane port (`sed` on `*_ENGINE_PUBLISH`, then
`up` engine, then `up` door, per lane). Live state: the gateway key was empty, no door ran, and the engines
published their own ports.

## Architecture touched

- `orion-llm-gateway`: sends only `X-Orion-Lease-Id`. The background `/door/stats` key check and the
  `lane_doors` block in `/health` are removed.
- `orion-lane-door`: lease states instead of key states. Host network. Waits for its engine on loopback,
  then binds. Drains on SIGTERM. Health check that reads healthy while waiting. Deploy hooks
  (`pre_rebuild.sh`, `post_rebuild.sh`, `move_engines_behind_doors.py`).
- `orion-llamacpp-host` atlas-workers compose: the `*_ENGINE_PUBLISH` defaults are now `127.0.0.1:180xx`.
- `orion-vllm-host`: the `docker-compose.deep.yml` default is now `127.0.0.1:18021`. `hecate-deep.sh`
  starts and stops the door.
- `mesh-utilities/common/include_services_circe.txt`: adds `orion-lane-door`. It stays excluded on athena.

## Files changed

- `services/orion-llm-gateway/app/lane_door.py`: the header only. The key, the check and the status are deleted.
- `services/orion-llm-gateway/app/{main,passthrough_proxy,llm_backend,settings}.py`, `.env_example`,
  `tests/conftest.py`, `tests/test_lane_door_headers.py`: the key is removed from every path.
- `services/orion-lane-door/app/door.py`: `lease_state`, `take_port`, `drain`, `serve` with signal handling.
- `services/orion-lane-door/app/settings.py`: no key. Adds `bind_poll_sec` and `drain_sec`. The listen host defaults to all interfaces.
- `services/orion-lane-door/app/health.py` (new): the container health check.
- `services/orion-lane-door/Dockerfile`: the health check (host network, no fixed container port).
- `services/orion-lane-door/docker-compose.yml`: host network. Loopback upstreams. The circe doors have no
  profile. `stop_grace_period: 120s`.
- `services/orion-lane-door/.env_example`: no key, no `PROJECT` (root `.env` names containers). Loopback
  upstreams. Drain and poll keys.
- `services/orion-lane-door/scripts/{pre_rebuild.sh,post_rebuild.sh,move_engines_behind_doors.py}` (new).
- `services/orion-lane-door/tests/test_door.py`, `tests/test_move_engines.py` (new),
  `evals/measure_latency.py`, `README.md`, `requirements.txt`.
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`, `.env_example`, `README.md`: loopback defaults.
- `services/orion-vllm-host/docker-compose.deep.yml`, `.env_example`, `scripts/hecate-deep.sh`.
- `mesh-utilities/common/include_services_circe.txt`, `exclude_services.txt`.
- `scripts/lane_door_report.py`, `tests/test_lane_door_report.py`: lease states. READY criteria updated.
- `tests/test_lane_door_exempt_parity.py`: the lease header parity. No key remains on either side.
- `scripts/gpu_power_bench.py`, `tests/test_gpu_power_bench.py`: door detection no longer reads `key_hash`.
- `orion/evals/leased_llm.py`: comment.
- `.github/workflows/lane-door-tests.yml`: also runs on changes to the engine compose files.
- `docs/runbooks/2026-11-14-lane-door.md`: rewritten (what a deploy does, check, roll back, Phase 3).
- `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md`: Phase 2/3 revised (no secret, why).

## Schema / bus / API changes

- Added: none.
- Removed: HTTP header `X-Orion-Lane-Key`. Gateway `/health` field `lane_doors`. Door `/door/stats` field
  `key_hash`.
- Renamed: door log row field `key` (`ok|missing|bad`) -> `lease` (`leased|unleased|invalid`). Stats
  `unkeyed_by_*` -> `unleased_by_*`. Container log `lane_door_unkeyed` -> `lane_door_unleased`.
- Behavior changed: enforce refuses calls with no valid lease id (was: no valid key).
- Compatibility notes: no door has run in production yet, so there are no old-format logs to read. The
  report counts an old-format row as unreadable. No bus or schema registry change.

## Env/config changes

- Added keys: `LANE_DOOR_BIND_POLL_SEC=2`, `LANE_DOOR_DRAIN_SEC=100` (orion-lane-door).
- Removed keys: `LLM_GATEWAY_LANE_DOOR_KEY`, `LLM_GATEWAY_LANE_DOOR_CHECK_ENABLED`,
  `LLM_GATEWAY_LANE_DOOR_CHECK_SEC` (orion-llm-gateway). `LANE_DOOR_KEY`, `PROJECT` (orion-lane-door).
- Changed meaning: `ATLAS_{CHAT,FAST,AGENT,AGENT_BURST}_ENGINE_PUBLISH` and `VLLM_ENGINE_PUBLISH` (deep.yml):
  empty now means loopback (`127.0.0.1:180xx`), where it used to mean the lane port. `LANE_DOOR_*_UPSTREAM`
  defaults are now `127.0.0.1:180xx`.
- `.env_example` updated: yes (gateway, lane-door, llamacpp-host, vllm-host).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: ran. athena has no lane-door
  `.env` (doors never run there). The three dead gateway keys were removed by hand from athena's
  `services/orion-llm-gateway/.env` (all empty or default). On circe, `pre_rebuild.sh` creates the door
  `.env` from the template.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-lane-door: pytest tests -q                        -> 45 passed
services/orion-llm-gateway: pytest tests -q                      -> 558 passed
tests/test_lane_door_report.py tests/test_lane_door_exempt_parity.py tests/test_gpu_power_bench.py
  tests/scripts/test_host_allowlist_buildable.py tests/test_compose_logging_journald.py -> 70 passed
scripts/check_env_template_parity.py                             -> ok
orion-static-gates.yml run steps, locally                        -> 40/40 passed
scripts/rebuild_affected_services.py --base origin/main --host circe  -> orion-lane-door build (only)
scripts/rebuild_affected_services.py --base origin/main --host athena -> orion-llm-gateway build (only)
```

## Evals run

```text
services/orion-lane-door/evals/measure_latency.py --n 200
  POST chat (leased, logged): added p50 0.31 ms; GET /props: 0.76 ms; stream first byte: 0.85 ms
```

## Docker/build/smoke checks

```text
docker build services/orion-lane-door/Dockerfile                 -> ok
docker run --network host (spare ports 47911/47912 on athena):
  engine absent  -> door does not bind, app.health rc=0 (waiting)
  engine started -> door binds within one poll, /door/stats answers, proxied GET -> engine's 404, app.health rc=0
  docker stop    -> exits in 1 s (graceful)
docker compose config (lane-door): 4 circe services, network_mode host, loopback upstreams
circe (read-only): compose `config --hash` for atlas-chat/agent/fast equals the running containers' hashes,
  so the first deploy changes only the published port.
circe (read-only): `config --format json` with ATLAS_CHAT_ENGINE_PUBLISH=127.0.0.1:18011 -> host_ip 127.0.0.1,
  the shape move_engines_behind_doors.py reads.
```

## Review findings fixed

Code review ran in a subagent. It found nothing material (key fully removed; real pool lease ids are
`uuid4().hex`, which the door reads as `leased`; the dual-stack bind retry, the drain, the deep.yml
`!override`, the compose project label and the port gate coverage all check out). Minor findings, all fixed:

- Finding: the docs said agent-gpu2's door "waits while the seat is unloaded". In fact it holds 8016 after
  its first bind and answers 502. Also, the seat is standby warm, so its first move can take days.
  - Fix: compose comment, README and runbook corrected.
  - Evidence: `services/orion-lane-door/docker-compose.yml`, runbook step 4.
- Finding: the engine-move hook was silent when no atlas-workers engines were found (doors would wait forever, deploy green).
  - Fix: prints a WARNING line.
  - Evidence: `test_no_engines_under_the_project_is_a_loud_warning_not_silence`.
- Finding: no end-to-end `serve()` test, and the dual-stack default was never exercised.
  - Fix: added `test_serve_on_sigterm_stops_listening_and_lets_the_call_in_flight_finish` and `test_take_port_on_every_interface_answers_ipv4_and_ipv6`.
  - Evidence: 45 passed.
- Finding: the door handling in hecate-deep.sh was untested and had no CI trigger. A foreground `up` would start the door only after Ctrl-C.
  - Fix: the door now starts BEFORE the engine (safe, because it waits for the engine on loopback). Added
    `tests/test_hecate_deep_script.py` (stubbed commands, asserts call order and `.env` creation). Added the script to the CI paths.
  - Evidence: 3 tests pass.
- Finding: stale "lane door: 127.0.0.1:18021" comments on hecate compose files the door does not front.
  - Fix: reworded to "public; stop the door first".
- Finding: the llamacpp-host `.env_example` had lost the chat/fast/agent `*_ENGINE_PUBLISH` lines its comment refers to.
  - Fix: restored them as empty lines (empty = loopback default).
- Finding: rows from the old keyed door read as "unreadable".
  - Fix: counted apart as pre-revision lines and printed as such.
  - Evidence: `test_lease_ids_are_collected_and_rows_from_the_old_keyed_door_are_counted_apart`.
- Not changed (accepted trade-off, documented): a door redeploy closes all four listeners at once and
  drains for up to 100 s, so new calls are refused at connect for that window. `LANE_DOOR_DRAIN_SEC` can
  shorten it.

## Restart required

No commands. A merge, then the normal pull on each host, does it:

- athena: the post-pull rebuild rebuilds `orion-llm-gateway` (a few seconds).
- circe: the post-pull rebuild starts the four doors. The first deploy recreates `atlas-chat`,
  `atlas-agent` and `atlas-fast` once, together. Chat and agent are Strata and take several minutes to
  reload. Fast takes about a minute. Those lanes are down while they reload: Hub chat cannot answer until
  the chat engine is back. agent-gpu2 is not restarted. Its door takes 8016 at the pool's next swap.
  Later deploys never restart an engine.
- hecate: the next `services/orion-vllm-host/scripts/hecate-deep.sh up -d` restarts the deep engine on
  loopback and starts its door. That is the normal start of agent-deep, so nothing extra is needed.

The only sudo line, unrelated to the doors (kev 8009 firewall; the installed gate on circe is older than
the repo copy):

```bash
ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/circe_llm_port_gate.sh install
```

## Risks / concerns

- Severity: medium. Concern: the first circe pull restarts chat, agent and fast together, so chat is down
  for several minutes. Mitigation: pull circe at a quiet time. To keep one lane out of the move, add
  `ATLAS_CHAT_ENGINE_PUBLISH=8011` to circe's llamacpp-host `.env` first. The hook leaves a pinned engine
  alone.
- Severity: low. Concern: until the pool next swaps agent-gpu2, its old engine keeps its public 8016. If
  that engine crashed and docker restarted it, the door would NOT take the port, because the door waits
  for the loopback upstream. So the lane cannot be lost to the door. Mitigation: built in (`take_port`).
- Severity: low. Concern: the door itself cannot tell a made-up lease id from a real one. Mitigation: the
  report's join to `gpu_pool_events` blocks READY for that lane. This is by design (no pool RPC on the
  chat path).
- Severity: low. Concern: hecate's other agent-deep definitions (`docker-compose.hecate.yml`, the Strata
  challenger) still publish 8021 themselves. Mitigation: `hecate-deep.sh stop` stops the door too, and
  the runbook says so.
- UNVERIFIED: the live circe deploy (doors bind, engines move, pool seats return, Hub chat answers),
  live shadow logs, and hecate. None of these can be proven before merge.
- Issues opened/closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
