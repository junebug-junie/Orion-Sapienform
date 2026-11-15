# Leased evals, part 3 of 3: the lane door can refuse, and kev goes behind the firewall

Design: `docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md`, Phase 3.
Stacked on part 2 (#3417). Part 1 (#3404) is merged.

## Summary

- The lane door gains `enforce` mode: a model call without the gateway's key gets HTTP 401
  `lane_door_unleased` (the spec's exact body) and never reaches the engine. Health checks, the pool's
  probes and the other no-model reads still pass.
- It ships OFF. Every door stays in `shadow`. Juniper flips one lane at a time with one line, after the
  shadow report says `READY TO ENFORCE` for that lane. Rollback is one line and takes seconds.
- A door told to enforce with no key refuses to start, rather than refusing every call.
- The circe firewall's port list now includes kev's port 8009, so the earlier one-off iptables line
  becomes permanent once the gate is re-installed.
- The runbook says exactly what evidence to read before a flip, the flip order, and how to check it.

## Outcome moved

The refusal half of the design is built and tested, waiting on evidence. kev stops being reachable from
any machine but athena after one re-install.

## Current architecture

Part 2's door counts and logs only. The firewall unit covered 8011-8017, 8090, 8099; kev's 8009 was open
to the whole tailnet and LAN unless the one-off line from #3242 was run (it does not survive a reboot).

## Architecture touched

`services/orion-lane-door` (mode), `scripts/lane_door_report.py`, `scripts/ops/circe_llm_port_gate.sh`,
runbooks.

## Files changed

- `services/orion-lane-door/app/settings.py`: `enforce` mode; refuses to start with no key.
- `services/orion-lane-door/app/door.py`: 401 before any engine connection; `refused` counter and log flag.
- `services/orion-lane-door/.env_example`, `README.md`: the mode, and that it ships `shadow`.
- `services/orion-lane-door/tests/test_door.py`: enforce tests.
- `scripts/lane_door_report.py`, `tests/test_lane_door_report.py`: refused calls shown apart.
- `scripts/ops/circe_llm_port_gate.sh`: `PORTS="8009,8011:8017,8090,8099"`.
- `tests/test_circe_llm_port_gate.py`: pins kev.
- `docs/runbooks/2026-11-14-lane-door.md`: Phase 3 section. `docs/runbooks/2026-10-01-circe-llm-port-firewall.md`: 8009.

## Schema / bus / API changes

- Added: door response `401 {"error":{"type":"lane_door_unleased",...}}` in enforce mode only.
- No bus, schema or registry change.

## Env/config changes

- Added keys: none new (`LANE_DOOR_MODE` and the per-lane `LANE_DOOR_MODE_<LANE>` came with part 2; this
  PR adds the value `enforce`). Default stays `shadow`.
- `.env_example` updated: comment only. Local `.env` sync: not needed (the door's `.env` lives on circe).
- Operator action: the flip itself, per lane, by Juniper (runbook).

## Tests run

```text
services/orion-lane-door/tests                          26 passed
tests/test_lane_door_report.py, test_lane_door_exempt_parity.py, test_circe_llm_port_gate.py,
  test_check_circe_worker_refs.py, services/orion-llm-gateway/tests, test_leased_llm.py (one session)   688 passed
check_circe_worker_refs PASS; git diff --check clean
```

## Evals run

```text
services/orion-lane-door/evals/measure_latency.py (local stand-in, n=300): added p50 0.25-0.38 ms on
plain calls, 1.2 ms on a stream's first byte (shadow; enforce adds one comparison).
No live enforce run: no door is deployed yet, and the flip is Juniper's call after 7 days of evidence.
```

## Docker/build/smoke checks

```text
No new image change beyond part 2. Firewall: DRY_RUN=1 apply prints --dports 8009,8011:8017,8090,8099
for both iptables and ip6tables (test_circe_llm_port_gate.py).
```

## Review findings fixed

A code-review subagent reviewed this PR's commits.

- Finding (blocker for the agent-deep flip): the pool's vLLM probe reads `/orion/server-info`, which was
  not exempt. In shadow agent-deep could never read READY; in enforce the pool would mark it down.
  - Fix: exempt it (door and eval helper lists), in part 2 (#3417) where the list lives, plus a test that
    reads every path the pool's `_probe` and the launch `ready` paths request and pins them to the list.
  - Evidence: `test_every_path_the_pool_and_the_lane_controller_read_is_exempt`.
- Finding: the flip command had no `--build`; a door image older than this PR rejects `enforce` and the
  lane goes down.
  - Fix: flip and rollback use `up -d --build`, then print the mode the door really runs.
- Finding: the flip line did not fit agent-deep (hecate, `--profile hecate`).
  - Fix: its own command.
- Finding: refused wrong-key calls were excused by the report, hiding gateway key drift after a flip.
  - Fix: only keyless refused calls are set aside; refused wrong-key calls block the verdict.
  - Evidence: `test_a_refused_wrong_key_still_blocks_it_is_gateway_key_drift`.
- Finding: no key-rotation procedure for an enforcing lane.
  - Fix: runbook line (shadow, rotate both, check hash, flip back).
- Finding: refusing without reading the body could reset a caller mid-upload.
  - Fix: after the 401 the door half-closes and drains up to 1 MiB / 2 s.
  - Evidence: `test_a_refused_large_upload_still_reads_the_401` (3 MB body).
- Nits fixed: the `refused` flag comes from the refusal branch, not re-derived; the journald line is at
  most once a minute per source and path (the JSONL keeps every call); the flip `sed` fails loudly if the
  key line is missing; rollback wording aligned to `shadow`; refused callers in their own `refused_from` map.

## Restart required

Firewall (Juniper, on circe, sudo; re-installs the unit so 8009 is covered at every boot):

```bash
ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/circe_llm_port_gate.sh install
```

Door flips: none now. Per lane, later, by the runbook's "Phase 3" one-liner.

## Risks / concerns

- Severity: high if flipped early. Concern: an enforced lane refuses any caller the shadow phase did not
  see. Mitigation: ships shadow; the report's READY verdict needs 7 complete days, the lease join, and
  the gateway's own count within 1 %; chat flips last; rollback in seconds.
- Severity: medium. Concern: key drift after a flip would refuse Orion's own calls on that lane.
  Mitigation: the gateway logs `lane_door_key_mismatch` every 10 minutes per server; the runbook checks it
  before a flip; a door in enforce with no key does not start.
- Severity: low. Concern: kev at 8009 is reachable from athena only after the re-install; any non-athena
  caller of kev breaks. Checked: the only caller in the repo is orion-substrate-runtime on athena.
- Issues opened/closed: none.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
