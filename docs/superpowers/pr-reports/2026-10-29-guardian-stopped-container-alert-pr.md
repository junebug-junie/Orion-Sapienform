## Summary

On 2026-10-29 a post-pull rebuild died halfway through rebuilding the harness worker. The old worker was stopped, and its replacement was created but never started. No agent turn could run for about 9.5 hours, and nothing told anyone. This patch closes both gaps.

- The mesh guardian now raises a Hub card when a container that should be running is stopped. It does this when the restart policy is `always` or `unless-stopped`, the state is `exited`, `created` or `dead`, and the container has stayed that way for more than 10 minutes. It also flags a half-finished compose recreate: a `<12 hex>_<name>` container in `created`.
- The rebuild script now writes an `ABORT` line when it is interrupted. The line names the step in flight and the services it did not rebuild, and it gives the one command that finishes them.
- New `--services "a b" [--recreate "a"]` option. It rebuilds exactly the named stacks, with the same rules for each stack. It is the finish command that `ABORT` prints.
- New `scripts/check_rebuild_log.py`. It finds runs that died with no `END` line (SIGKILL, which no trap can catch) and aborted runs that left work undone. It runs at the start of every rebuild, which reports each run once and then marks it, and at Claude Code session start.

## Outcome moved

Failure mode: "a rebuild dies mid-recreate, a service stays down, and nobody knows." Before this patch, nobody saw it for 9.5 hours. After this patch:
- A Hub card appears about 1 minute after the old container exits. The replacement container has usually been in `created` for longer than the grace period by then.
- The run log names the unfinished services, plus the finish command.
- The next rebuild run and the next agent session repeat that report.

## Current architecture

- The guardian's stability cycle (`app/stability.py`, every 60 s) watched restart-count growth (crash loops), Redis slow-consumer kills, stuck snapshots and graph inflation. A container that was stopped cleanly never restarts, so the guardian could not see it. The roster (`config/mesh_remediation_roster.yaml`) does not list the harness worker.
- `rebuild_services_from_git_diff.sh` logged `START`/`END` around each compose step and `DONE` at the end. It had no trap. A killed run just stopped, and the log ended at a `START`.

## Architecture touched

- `orion-mesh-guardian`: new stability check `stopped`, with its own docker read. It lists all containers and inspects only the containers that are not running. Cards use the existing `_publish_alerts` -> AlertGate -> notify `attention_request` path, so they show up on the Hub **Pending Attention** surface, the same place as the crash-loop cards (`heartbeat: stability`, `event: stopped_container`).
- Local docker only. The circe census (`ContainerRestartsV1`) carries restart counts, not state, and its schema forbids extra fields, so adding state there needs its own migration that updates consumers first. That is a follow-up.
- Rebuild script: traps, `--services`/`--recreate`, and a dead-run check at startup.
- `.claude/settings.json`: one more SessionStart hook (`check_rebuild_log.py --session-start`, 0.1 s).

## Files changed

- `services/orion-mesh-guardian/app/stability.py`: `ContainerState`, `stopped_containers`, `StoppedContainerTracker` (one card per stop episode, grouped by compose service, marked only after delivery).
- `services/orion-mesh-guardian/app/service.py`: `_check_stopped_containers`, `_docker_container_states`, `_parse_docker_time`. `_publish_alerts` now returns the delivered alerts.
- `services/orion-mesh-guardian/app/attention.py`: `publish_transition` returns whether notify accepted the card.
- `services/orion-mesh-guardian/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: two new keys, plus docs.
- `services/orion-mesh-guardian/tests/test_stopped_containers.py`: incident replay, rules, collector, end-to-end cycle, retry of an undelivered card.
- `scripts/rebuild_services_from_git_diff.sh`: trap, `--services`/`--recreate`, dead-run check.
- `scripts/check_rebuild_log.py`: finds died or aborted runs.
- `tests/scripts/test_rebuild_services_from_git_diff.py`, `tests/scripts/test_check_rebuild_log.py`: SIGTERM, SIGKILL, abort during the backup wait, finish-command round trip.
- `.claude/settings.json`: SessionStart hook.

## Schema / bus / API changes

- Added: none on the bus. There is a new alert kind `stopped_container` in the existing attention-card context.
- Removed: none
- Renamed: none
- Behavior changed: `AttentionPublisher.publish_transition` returns `bool` (it returned `None` before). Its callers ignored the return value.
- Compatibility notes: none

## Env/config changes

- Added keys: `MESH_GUARDIAN_STOPPED_CONTAINER_WATCH_ENABLED=true`, `MESH_GUARDIAN_STOPPED_CONTAINER_GRACE_SEC=600` (orion-mesh-guardian)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- Local `.env` synced: the two keys were appended to `/mnt/scripts/Orion-Sapienform/services/orion-mesh-guardian/.env`. The worktree has no `.env`, so the sync script had nothing to write to there.
- Skipped keys requiring operator action: none

## Tests run

```text
services/orion-mesh-guardian: pytest tests evals -q                           150 passed
tests/scripts/test_rebuild_services_from_git_diff.py
  tests/scripts/test_check_rebuild_log.py tests/test_deploy_migration_gate.py  109 passed
tests/scripts/test_session_agent_board_hooks.py tests/scripts/test_cursor_bridge.py  32 passed
bash -n scripts/rebuild_services_from_git_diff.sh                               ok
python3 scripts/check_env_template_parity.py                                    PASS
```

These regression tests would have caught the incident:
- `TestIncidentReplay`: the real timeline gives one card, then silence for 9.5 h.
- `test_sigterm_mid_step_logs_abort_...`
- `test_sigkill_leaves_no_end_and_the_next_run_reports_it_once`
- `test_the_incident_run_is_reported_...`: uses the real log lines.

## Evals run

```text
Live-data check (metric gate step 4), athena, 2026-10-29, real collector + tracker:
  133 containers, 2 not running.
  Card: orion-athena-kev -- unless-stopped, exited 2026-10-15 (332 h). kev moved to circe
        (config/gpu_pool.yaml); this is a leftover container.
  Not flagged: orion-analytics-lightdash-object-storage-init-1 (restart=no, one-shot init job).
  Second cycle: no card.
check_rebuild_log.py against the real ~/.orion/post_pull_rebuild.log (whole history):
  exactly one report -- pid 3250904, in flight `orion-harness-governor up -d --build`,
  not rebuilt: harness-governor, hub, thought, cortex-orch.
```

The live data shows no pile of old dead containers that the detector would nag about. Only one container is flagged.

## Docker/build/smoke checks

```text
Not built or deployed (this session does not deploy). The guardian already mounts the docker socket
(docker-compose.yml:23), and the live check above used the same Engine API calls.
```

## Review findings fixed

- Finding: an ABORT line was treated as "closed", so an interrupt by SIGHUP (the terminal closed) was never reported anywhere that someone reads.
  - Fix: `check_rebuild_log.py` reports an ABORT that left services unfinished until `--mark`.
  - Evidence: `test_an_abort_that_left_services_is_reported_until_marked`.
- Finding: an episode was marked as carded before delivery, so a card that notify refused was lost for the whole outage.
  - Fix: an episode is marked only after notify accepts the card. Until then the gate retries once per 6 h.
  - Evidence: `test_a_card_notify_refused_is_retried_after_the_gate_window`.
- Finding: an abort during the backup wait said `unfinished=none` while a consumer stayed refused, and `--services` does not apply migrations.
  - Fix: the abort now lists those consumers and writes `ABORT-MIGRATIONS` with the apply command.
  - Evidence: `test_abort_during_the_backup_wait_names_the_refused_consumer_and_the_migration`.
- Nits fixed:
  - The report text now says how to silence it.
  - `--recreate` must be a subset of `--services`.
  - The `dead` state is flagged.
  - No start command is printed for a stack outside `services/`.
- Nits not fixed:
  - A SIGKILL during the retry pass leaves the retry services out of the finish command. This is rare, and the deploy gate still holds those services.
  - AlertGate's key map grows by one entry per stop episode. The growth is small.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build
```

Run this from the clean primary main checkout after merge. The rebuild script and the hook change need no restart.

## Risks / concerns

- Severity: low
  - Concern: the first guardian deploy raises a card for `orion-athena-kev`. Each later guardian restart raises it again, because episodes are kept in memory.
  - Mitigation: if kev on athena is retired, run `docker rm orion-athena-kev`.
- Severity: low
  - Concern: the SessionStart hook reports today's harness incident for up to 72 h, until a rebuild runs or someone runs `python3 scripts/check_rebuild_log.py --mark`.
  - Mitigation: the report text names that command.
- Severity: low
  - Concern: circe containers are not covered by the stopped check.
  - Mitigation: follow-up. Add state to the circe census, with a consumer-first migration.
- Severity: info
  - Concern: SIGKILL cannot be trapped.
  - Mitigation: `check_rebuild_log.py` covers that case at the next run and at the next session.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2968
