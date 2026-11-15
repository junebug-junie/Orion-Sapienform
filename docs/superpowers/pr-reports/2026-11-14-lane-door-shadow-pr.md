# Leased evals, part 2 of 3: the shadow lane door

Design: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md`, Phase 2.
Stacked on part 1 (#3404). Part 3 (refusal + firewall) is stacked on this one.

## Summary

- A small proxy, the "lane door", can now stand in front of each model server (chat, agent, agent-gpu2,
  fast on circe; agent-deep on hecate). It listens on the lane's usual port and passes each call to the
  engine, which moves to circe's loopback.
- The gateway adds two headers to every call it sends to a model server: a shared secret and the GPU
  pool lease id. The door checks the secret, removes both headers, and writes one line per model call:
  who, which path, key ok/missing/bad, lease id, status, time. No prompt text, never the key.
- Shadow mode (the default): the door refuses nothing. It exists to answer "who still calls around the
  gateway?" with data.
- `scripts/lane_door_report.py` turns the door logs into a daily table per lane and a verdict:
  READY TO ENFORCE only after 7 complete days with zero unkeyed calls, every lease found in
  `gpu_pool_events`, and the door's count within 1 % of the gateway's.
- The gateway checks in the background that each door holds the same key (`lane_door_key_mismatch`).

## Outcome moved

How often something calls a model server without a lease was UNVERIFIED (the spec: server logs have no
client address). Once deployed, every such call is logged with its source. The latency cost is measured:
about 0.3-0.9 ms per call.

## Current architecture

Each engine published its lane port on circe. The gateway sent nothing that named the lease. The only
fences were the athena-only firewall (cannot tell the gateway from other athena containers) and the CI
address gate (cannot see env, loopback or tunnel URLs).

## Architecture touched

- New service `services/orion-lane-door` (stdlib only; one compose service per lane).
- `services/orion-llm-gateway`: door headers on both dispatch paths; background key check; `/health`
  shows `lane_doors`.
- `services/orion-llamacpp-host` and `services/orion-vllm-host` compose: `*_ENGINE_PUBLISH` decides
  where an engine publishes. Empty = today's layout.
- `scripts/gpu_power_bench.py`, `scripts/check_circe_worker_refs.py`, new `scripts/lane_door_report.py`.

## Files changed

- `services/orion-lane-door/` (new): `app/door.py` (proxy), `app/settings.py`, `app/main.py`,
  `Dockerfile`, `docker-compose.yml`, `.env_example`, `requirements.txt` (nothing to install), `README.md`,
  `tests/test_door.py`, `evals/measure_latency.py`.
- `services/orion-llm-gateway/app/lane_door.py` (new), `passthrough_proxy.py`, `llm_backend.py`,
  `upstream_cancel.py` (`lease_id` on the cancel handle), `main.py`, `settings.py`, `.env_example`,
  `tests/conftest.py` (no background checks in unit tests), `tests/test_lane_door_headers.py` (new).
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`, `docker-compose.hecate.yml`,
  `.env_example`; `services/orion-vllm-host/docker-compose.yml`, `.env_example`.
- `scripts/lane_door_report.py` (new), `tests/test_lane_door_report.py` (new).
- `tests/test_lane_door_exempt_parity.py` (new): the door's no-key reads equal the eval helper's worker
  reads; door and gateway hash the key the same way.
- `scripts/gpu_power_bench.py` + `tests/test_gpu_power_bench.py`: bench the engine's loopback port;
  refuse a door port.
- `scripts/check_circe_worker_refs.py`: the door is a zone; its compose names count as worker names.
- `docs/runbooks/2026-11-14-lane-door.md` (new): deploy per lane, check, daily evidence, roll back.
- `.github/workflows/lane-door-tests.yml` (new).

## Schema / bus / API changes

- Added: request headers `X-Orion-Lane-Key`, `X-Orion-Lease-Id` (gateway to model server; the door strips
  both). Door endpoint `GET /door/stats`. Gateway `/health` field `lane_doors`.
- No bus channel, schema or registry change.

## Env/config changes

- Added keys:
  - gateway: `LLM_GATEWAY_LANE_DOOR_KEY` (secret, empty), `LLM_GATEWAY_LANE_DOOR_CHECK_ENABLED=true`,
    `LLM_GATEWAY_LANE_DOOR_CHECK_SEC=600`.
  - llamacpp-host: `ATLAS_CHAT_ENGINE_PUBLISH`, `ATLAS_FAST_ENGINE_PUBLISH`, `ATLAS_AGENT_ENGINE_PUBLISH`,
    `ATLAS_AGENT_BURST_ENGINE_PUBLISH`, `HECATE_AGENT_DEEP_ENGINE_PUBLISH` (all empty).
  - vllm-host: `VLLM_ENGINE_PUBLISH` (empty).
  - lane-door (new): `LANE_DOOR_KEY`, `LANE_DOOR_MODE=shadow`, log dir/retention, ports, upstreams.
- `.env_example` updated: yes.
- Local `.env` synced: yes, `scripts/sync_local_env_from_example.py --all-keys orion-llm-gateway
  orion-llamacpp-host orion-vllm-host` (missing keys added only; nothing overwritten). The door's `.env`
  lives on circe/hecate; the runbook creates it.
- Skipped / operator action:
  - The shared secret is set by the runbook's step 1 one-liner (no sudo).
  - `*_ENGINE_PUBLISH` ship empty on purpose. The door mode flag ships ON (`shadow`). Moving an engine to
    loopback is a deploy step done together with starting its door: an engine moved with no door in front
    is a lane nobody can reach, so this is not shipped as a default a `.env` sync could apply alone.

## Tests run

```text
services/orion-lane-door/tests                          16 passed
services/orion-llm-gateway/tests (full)                564 passed
tests/test_lane_door_exempt_parity.py, test_lane_door_report.py, test_gpu_power_bench.py,
  test_check_circe_worker_refs.py                       141 passed
tests/test_circe_llm_port_gate.py, test_compose_logging_journald.py, test_ci_service_image_sources.py,
  scripts/test_host_allowlist_buildable.py, test_no_personal_terms.py, test_no_private_eval_artifacts.py
                                                        24 passed
check_circe_worker_refs PASS; check_compose_no_relative_mounts PASS; check_env_key_single_source OK;
check_service_hostname_refs OK; check_scripts_dir_no_stdlib_shadow clean; git diff --check clean
```

## Evals run

```text
services/orion-lane-door/evals/measure_latency.py (new connection per call, like the gateway):
  local stand-in, POST chat keyed+logged: added p50 0.09-0.31 ms, p99 0.57-0.94 ms
  local stand-in, stream first byte:      added p50 0.76-0.86 ms, p99 0.83-1.35 ms
  live circe fast engine, GET /props (door on athena): added p50 0.80 ms, p99 1.08 ms
  live circe chat engine (Strata), GET /props:          added p50 0.45 ms, p99 0.69 ms
lane_door_report.py against live Postgres (read-only): a real lease id from gpu_pool_events matched,
a made-up one was flagged; gateway count for fast on 2026-11-14 read 3,752.
```

The spec sets no number, only "no per-call pool check in the door". A chat turn takes seconds; under
1 ms is noise against that.

## Docker/build/smoke checks

```text
docker build -f services/orion-lane-door/Dockerfile .           ok
docker run (shadow, no engine): /door/stats 200 with key_hash; /health 502 (seat reads down)
docker compose config, door compose --profile circe: 4 doors, ports 8011/8015/8016/8013, mode shadow
docker compose config, atlas-workers with ATLAS_CHAT_ENGINE_PUBLISH=127.0.0.1:18011: chat publishes
  127.0.0.1:18011; others unchanged on their lane ports
Nothing deployed.
```

## Review findings fixed

A code-review subagent reviewed the door commits. No blocker on the gateway-to-engine chat path.

- Finding: an engine's `100 Continue` ended the exchange; the caller never got the real reply (curl sends
  `Expect` on large bodies).
  - Fix: the door answers `Expect: 100-continue` itself, strips `Expect`, and skips any interim 1xx.
  - Evidence: `test_expect_100_continue_gets_the_final_reply_not_just_the_interim_one`.
- Finding: the runbook's agent-gpu2 step claimed the door retries a taken port; it stays `Created`.
  - Fix: the step refuses while 8016 is taken (`ss -ltn` guard) and shows `docker ps` after.
- Finding: the live hecate definition (`docker-compose.deep.yml`) hard-coded 8021, and the runbook had no
  real hecate command.
  - Fix: deep.yml publishes `${VLLM_ENGINE_PUBLISH:-8021}`; the runbook uses `hecate-deep.sh up`.
- Finding: athena's post-pull rebuild would try to build/start the door.
  - Fix: `orion-lane-door` in `mesh-utilities/common/exclude_services.txt`; deliberately not in circe's
    include list (hand deploy only, stated in the runbook).
- Finding: a reset (not a clean close) from the caller did not reach the engine.
  - Fix: any error on the caller's side sets `client_gone`.
- Finding: the readiness verdict printed READY without the lease join, and skipped the gap check when the
  gateway count was 0.
  - Fix: both are NOT READY; the day window is cast to UTC explicitly; lanes with no log are listed.
  - Evidence: `test_seven_clean_days_without_the_lease_join_are_not_ready`, `test_joined_gaps_and_unmatched_leases_block`.
- Finding: a truncated request body counted as an engine error.
  - Fix: classified as the caller's (`client_request_incomplete`, 400).
  - Evidence: `test_a_truncated_request_body_is_the_callers_fault_not_the_engines`.
- Finding: the parity test collided with the gateway's `app` package in one pytest session.
  - Fix: evict `app*` before and after the import. The combined run passes (706).
- Nits fixed: 30 s head timeout (`test_a_silent_caller_is_dropped_after_the_head_timeout`); both sides
  keyless reads `unkeyed`, not `mismatch`; per-lane `LANE_DOOR_MODE_*` keys in `.env_example`; the key
  goes to circe over ssh stdin, never on a command line, and the step stops if the gateway's `.env` lacks
  the key line; README notes on bridge-address sources and the 502-vs-connect telemetry change.

## Restart required

```text
athena, after merge (post-pull rebuild does this): scripts/safe_docker_build.sh orion-llm-gateway up -d --build
circe/hecate: follow docs/runbooks/2026-11-14-lane-door.md steps 1-3, one lane at a time, chat last.
Recreating an engine reloads its model; chat is unreachable for several minutes during its step.
```

## Risks / concerns

- Severity: high if done wrong. Concern: a lane whose engine moved to loopback with no door up is
  unreachable (Orion mute on chat). Mitigation: the runbook moves the engine and starts the door in one
  command per lane; `*_ENGINE_PUBLISH` ship empty; rollback is one command.
- Severity: medium. Concern: a calm shadow log on a lane proves only that lane; agent-gpu2 is often
  unloaded, so it may need longer than 7 days to show enough keyed traffic. Mitigation: the verdict needs
  keyed calls every day.
- Severity: low. Concern: the native logprobs path and the Strata JSON retry make two engine calls for one
  gateway call, so "door keyed = gateway count ±1 %" can read a little high. Mitigation: the report shows
  the gap; Juniper judges it.
- Severity: low. Concern: loopback publishing still lets processes on circe's host reach an engine. The
  door fences other machines, which is where evals run.
- Issues opened/closed: none.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
