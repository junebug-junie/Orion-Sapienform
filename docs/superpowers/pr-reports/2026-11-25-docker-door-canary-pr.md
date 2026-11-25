# Docker door canary (with review fixes)

## Summary

- Adds `orion-docker-door`. This small proxy sits in front of the Docker socket. Orion's turns can look (`ps`, `logs`, `inspect`) and `docker exec` into ordinary organs. Changes to containers are refused. So is exec into any container that leads back to the host. Every request is recorded to a host JSONL file.
- Ships as a **canary**. A turn uses the door only inside a scheduled window (`ORION_DOCKER_DOOR_CANARY_WINDOWS`). Outside the window, it keeps the real socket.
- Fixes all seven findings from the first code review (design PR #3790, "Code review of the held build"). It also fixes the high findings and the cheap low findings from a second, independent review.
- Replaces the tests' fake daemon with a keep-alive one that also accepts a bare `\n`, as dockerd does. The old fake closed after every request, which hid both smuggling holes.

## Outcome moved

Before this change, two working ways past the filter existed: bare-LF smuggling, and `Upgrade` switching off the close. Each one got a hidden `kill` to the daemon. Both now fail, and the record shows the refusal. Running the original door against the new fake proves both holes again; the new door stops both.

The second review found a way to read host files with a "look": copying the host's shutdown SSH key out of `power-guard`. That is now refused, and I checked the refusal live.

## Current architecture

Orion's turns run in the harness governor as root with the real Docker socket. Nothing filtered or recorded what they asked Docker to do.

## Architecture touched

- New service `services/orion-docker-door`: stdlib asyncio, host port none, `app-net`, socket mounted, record at `/mnt/storage-warm/orion-record`.
- Governor: `orion/harness/docker_door_canary.py` chooses the door or the socket for each turn. `fcc_motor.py` (cold spawn and resume) and `fcc_warm_pool.py` (warm acquire) pass the choice into the turn's env.
- The GitHub MCP server is pinned to the real socket in `orion/fcc/fcc_claude_mcp.template.json`.

## Files changed

- `services/orion-docker-door/app/door.py`: the door. Strict parsing, one request per connection, the host-reach rule, the record.
- `services/orion-docker-door/app/report.py`: what broke in a window.
- `services/orion-docker-door/tests/test_door.py`: 141 tests, including a keep-alive fake daemon and a test for every bypass.
- `services/orion-docker-door/{Dockerfile,docker-compose.yml,.env_example,README.md}`: the service.
- `orion/harness/docker_door_canary.py`: the window, an async `/_ping` check (cached 30 s), and the per-turn choice.
- `orion/harness/fcc_motor.py`, `orion/harness/fcc_warm_pool.py`: apply the choice. The check runs off the event loop.
- `orion/harness/tests/test_docker_door_canary.py`: the canary tests.
- `orion/fcc/fcc_claude_mcp.template.json`: GitHub MCP pinned to the socket.
- `services/orion-harness-governor/.env_example`: the window and the door URL.

## Schema / bus / API changes

- Added: none on the bus. New HTTP proxy on `app-net` (`orion-athena-docker-door:2375`).
- Removed / Renamed: none.
- Behavior changed: only inside a canary window.
- Compatibility notes: outside a window, turns are unchanged. No check runs and `DOCKER_HOST` is not touched.

## Env/config changes

- Added keys: `ORION_DOCKER_DOOR_CANARY_WINDOWS` (empty) and `ORION_DOCKER_DOOR_URL` (governor). `DOCKER_DOOR_ALLOWED_CALLERS`, `DOCKER_DOOR_SOCKET`, `DOCKER_DOOR_RECORD_DIR` and `DOCKER_DOOR_PORT` (door).
- `.env_example` updated: yes (in the base commit `bad427158`; the review-fix commits change no env keys).
- Local `.env`: the governor's live `.env` has the window **empty on purpose**. The door's live `.env` exists in the primary checkout.
- Skipped keys: none.

## Tests run

```text
pytest services/orion-docker-door/tests orion/harness/tests/test_docker_door_canary.py  -> 157 passed
pytest orion/harness/tests orion/fcc/tests                                             -> 1144 passed
Old door (bad427158) against the new keep-alive fake: both smuggles reach the fake ("kill" seen); new door: refused, fake sees nothing.
Long-poll leak test fails with the fix disabled, passes with it.
```

## Evals run

```text
No eval harness for this service. The canary window is the eval: report.py lists every refusal in the window.
```

## Docker/build/smoke checks

A temporary door (`orion-docker-door-smoke`) was built from this branch, called through the live governor, then removed. Nothing was deployed. Refusals were tested only against containers whose refusal could not cause a change.

```text
works:   docker ps; exec -i into thought (stdin "print(40+2)" -> 42); dig path exec -i sql-db psql -> 5;
         exec -d thought; docker cp out of thought; logs -f abandoned (door releases the daemon side)
refused: restart (non-existent target) -> "changing containers is off"
         exec hub -> network_host; exec --privileged thought -> "a privileged exec is off"
         exec power-guard, cp power-guard:/etc/powerguard/ssh_key -> mount:/root/.ssh/powerguard_shutdown
         exec hardware-watch -> docker_root:/mnt/docker; exec dream / recall -> writable:/mnt/scripts/...
         another caller (thought -> door) -> 403 before it sent a byte
```

## Review findings fixed

First review (design PR #3790):

1. Bare-LF smuggling.
   - Fix: strict head parsing. A bare LF, a stray CR, a folded header, a version other than HTTP/1.0 or 1.1, or a method that is not a plain token gets a 400.
   - Evidence: `test_bare_lf_smuggle_is_refused`, `test_parse_head_is_strict`.
2. `Upgrade` switched off the close.
   - Fix: the door always forwards `Connection: close`. It sends the daemon exactly the head plus the body. Later bytes go through only after a non-detached exec start, when they are already that exec's stdin.
   - Evidence: `test_upgrade_or_body_cannot_carry_a_second_request` (resize, detached start, a short body), `test_exec_start_streams_stdin_and_never_parses_it`.
3. Exec privileges and unreadable bodies.
   - Fix: an exec that asks for `Privileged` is refused at create and at start (`ProcessConfig.privileged`). So is an exec create the door cannot read: chunked, over 1 MB, a missing or bad `Content-Length`, or `Expect`.
   - Evidence: `test_exec_create_refusals`.
4. `host_reach` was too narrow.
   - Fix: added `CapAdd`, `Devices`, host network, IPC and user namespaces, runtime sockets matched by source, and `/proc`, `/sys`, `/dev`, `/etc`, `/root`.
   - Evidence: `test_host_reach_refuses`.
5. Hangs.
   - Fix: the caller check runs before the head is read. Head, body and daemon reads have timeouts. Connections close in `finally`. A failed record write refuses with 503.
   - Evidence: `test_unknown_caller_is_refused_before_it_sends_anything`, `test_slow_head_is_dropped`, `test_short_body_is_dropped`, `test_slow_body_is_dropped`, `test_failed_record_write_refuses`, `test_daemon_down_answers_and_closes`.
6. The door check blocked the governor's event loop.
   - Fix: the check is async on the cold-spawn, resume and warm paths.
   - Evidence: `test_slow_door_does_not_block_the_event_loop`, `test_build_subprocess_env_takes_the_door_from_the_caller`.
7. Encoded paths dodged the attach rule (`%61ttach`).
   - Fix: the path is decoded once. Double encoding and dot or empty segments are refused.
   - Evidence: `test_normalize_decodes_once_before_classify`, `test_normalize_refuses_tricky_paths`, `test_encoded_attach_is_refused`.

Second review (independent subagent; it found no protocol bypass):

- HIGH: a "look" could copy files out of any container, including the host's SSH key.
  - Fix: `archive` and `export` are checked like an exec. `/images/get` is refused.
  - Evidence: `test_copy_out_of_host_reaching_container_is_refused`, plus the live smoke.
- HIGH: exec into containers with a writable bind of Orion's code. The host imports `orion/` after every pull.
  - Fix: refuse a writable bind under `/mnt/scripts`, `/home`, `/usr`, `/var`, and the other system roots.
  - Evidence: `test_host_reach_refuses` (`writable:` cases), plus the live smoke (`dream`, `recall`).
- HIGH: more host-reach gaps.
  - Fix:
    - Docker's real data root (`/mnt/docker`, read from `/info`, failing closed).
    - `container:<x>` namespaces (x is inspected too).
    - UTS and cgroup host modes, `unconfined` security options, device cgroup rules.
    - Any bind under `/run` except audio and sensors.
    - Local volumes that are really binds (`o=bind`).
  - Evidence: `test_host_reach_refuses`, `test_exec_create_refusals`, `test_missing_docker_root_refuses_exec`, `test_exec_into_ordinary_namespace_and_volume_passes`.
- MEDIUM: an abandoned long-poll kept the daemon connection open.
  - Fix: the door watches for the client to leave, as dockerd does.
  - Evidence: `test_abandoned_long_poll_releases_the_daemon` (fails with the fix disabled).
- LOW: the exec body was not read the way Go reads it.
  - Fix: keys are matched without case, duplicates that differ only in case are refused, and `RecursionError` is caught.
  - Evidence: `test_exec_spec_reads_keys_like_go`.
- LOW: record writes blocked the event loop.
  - Fix: `asyncio.to_thread`.
- LOW: the canary check proved only a TCP connect, and its threads could pile up.
  - Fix: a native async `/_ping` that requires 200, cached for 30 s.
  - Evidence: `test_door_answers_needs_a_real_ping`, `test_probe_result_is_cached`.

## Restart required

After merge, from the clean primary `main` checkout (the post-pull rebuild already plans `orion-docker-door build`):

```bash
scripts/safe_docker_build.sh orion-docker-door up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Then Juniper picks a full-day window. Set `ORION_DOCKER_DOOR_CANARY_WINDOWS` in `services/orion-harness-governor/.env` and recreate the governor. After the window:

```bash
python3 services/orion-docker-door/app/report.py --since <start> --until <end>
docker logs orion-athena-harness-governor 2>&1 | grep DOCKER_DOOR_CANARY
```

## Risks / concerns

- Severity: by design. The canary is **not enforcement**. The governor still mounts the socket, so a turn that points `docker` at the socket directly goes around the door. The canary measures what breaks; only the later swap (patch 1b) makes the door a wall.
- Severity: medium. The wider rule refuses exec into these organs: retro-gateway, gpu-cluster-power, vision-retina (all four), zwave-js-ui, power-guard, hardware-watch, dream, recall, and the vision frame router. Some may be places Orion really uses. The canary report will show it.
- Severity: low. A "look" still shows every container's env, secrets included (#3981, needs Juniper).
- Severity: low. The caller check is the source IP on `app-net` (#3980).
- Issues opened: #3980, #3981. Closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3982

🤖 Generated with [Claude Code](https://claude.com/claude-code)
