# PR 4 of the ladder retirement: hardware-watch's disk rule asks for the image prune

## Summary

- Orion prunes dangling docker images (leftover, untagged image layers from rebuilds) through a skill in
  orion-cortex-exec. Until now only the proposal -> policy -> dispatch ladder asked for it: the
  `prune_dangling_images` template, 156 runs in 14 days, 145 of them "nothing to reclaim".
- orion-hardware-watch now has a disk rule beside its heat rules. Every 30 s tick it reads how full the
  docker volume is. At or above 70 % (the skill's own floor) it asks cortex-exec to run
  `skills.runtime.image_prune.v1`, then waits 12 h before it asks again (30 min if the ask got no answer).
  The skill still runs its own gate: it prunes only with >= 100 dangling images.
- The old trigger is removed in the same PR: the three prune templates (`prune_build_cache`,
  `prune_dangling_images`, `prune_stopped_containers`), their routes, the `maintain` fallback route, the
  reserved maintenance slot, `allow_mutating_dispatch`, `approved_maintenance` and the envelope's derived
  execute mode. The image prune has one trigger.
- Build cache and stopped containers are no longer pruned automatically. Their skills stay, hand-run
  (Juniper, decision 4). In the 14-day window dispatch sent neither of them (0 runs).
- A replay over 14 real days runs the shipped rule on the real disk record. Result: 28 asks (2 a day) -> 12
  prunes, 1,924 images removed, 60 dangling images left on the host on average. The real record: 156 runs
  -> 11 prunes, 1,993 images removed, 97 left on average.

Spec: `docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md` ("PR 4. Docker pruning").

## Outcome moved

- **The image prune no longer depends on the ladder.** PR 6 can retire proposal, policy and dispatch
  without silencing it.
- **About 80 % fewer asks, and a cleaner host.** 28 asks against 156 runs. 12 prunes against 11, and an
  average of 60 dangling images left on the host against 97.
- **Each tick is visible.** `GET /health` -> `image_prune` shows the disk reading, the verdict, the ask in
  flight, the last answer (decision, dangling count, images removed, bytes reclaimed, or the error) and a
  count of every verdict and outcome.
- **Dispatch can no longer mutate the host.** No route admits the maintenance scope, and no envelope asks a
  skill to execute.

## Current architecture (before this PR)

- orion-proposal-runtime emitted the three `maintain` templates when the field's `resource_pressure` or
  `deviation_pressure` was high. Live, the disk sat at 79-81 % the whole window, so "pressure" was mostly
  other things (CPU, memory, stream backlog).
- orion-policy-runtime approved them `approved_maintenance`. orion-execution-dispatch-runtime held one
  reserved slot for `maintenance_bounded`, checked `mode.allow_mutating_dispatch`, derived
  `skill_args.mode=execute` from its own dry-run state, and sent a bus RPC to cortex-exec on
  `orion:cortex:exec:request:background` (720 s budget).
- The skill (`services/orion-cortex-exec/app/verb_adapters.py`, `ImagePruneVerb`) measures the disk with
  `shutil.disk_usage("/hostfs/docker")` and counts dangling images with `docker images --filter
  dangling=true`. It prunes (`docker image prune -f`, never `-a`) only when disk >= 70 % **and** >= 100
  dangling. The audit's "75 % / 40 GB" is the build-cache skill's gate, not this one.
- orion-hardware-watch had no disk rule. `scripts/disk_threshold_watchdog.py` alerts at 90 % only.

## Architecture touched

How one disk-rule tick works (`services/orion-hardware-watch/app/image_prune.py`, pure verdict
`orion/hardware_watch/rules.py::image_prune_verdict`):

1. Read the docker volume: `shutil.disk_usage` on the read-only bind of `ORION_HOST_DOCKER_DATA_ROOT` at
   `/hostfs/docker`. A path that is not a mount point reads as unreadable (never as the container's own
   overlay disk).
2. Read the rule's own state, then take the clock.
3. Decide, in order: one ask at a time (`in_flight`); no ask in the first 10 min after a start
   (`starting`); no ask on an unreadable disk (`disk_unreadable`); no ask below 70 % (`below_threshold`);
   after an answered ask wait 12 h (`cooldown`), after an unanswered one 30 min (`retry_wait`).
4. Ask: a background task sends one RPC, with `require_listener=True` so a cortex-exec mid-restart fails at
   once instead of after 720 s. Its answer (or error) is recorded, and the hold starts from that moment.

The rule runs only when both `HARDWARE_WATCH_ENABLED` and `HARDWARE_WATCH_IMAGE_PRUNE_ENABLED` are on. A
failure in it is logged and never stops the heat rules (separate try in the tick loop).

### Why this invocation path

The rule sends the same bus RPC to cortex-exec that dispatch sent: the same channel, the same
`PlanExecutionRequest` built by `orion.cognition.plan_loader.build_plan_for_verb`, and the same
`context.skill_args.mode=execute`.
- cortex-exec already owns the docker socket and the skill, and the skill's own gate is the safety. A
  second copy in hardware-watch would need the docker socket inside a watcher.
- The channel and its schema already exist. Only the producer list grows by one, so there is no new
  contract.
- I rejected two other paths. Hub's skill runner is Juniper's hand lane (HTTP, a catalogue of prompts).
  cortex-orch's chat request shape is for LLM verbs.
- No body-economy bid is placed. The family `execution_dispatch.maintain` had 0 rows before this PR, and
  the spec asks for none. A bid can come with attention patch 3.

### Clock and freshness (the #3134 lesson)

- The disk is read first, then the rule's state, then the clock (`ImagePruneRule.tick`).
- An ask that finished after the tick's clock is age 0, so it holds the full cooldown. It is never a
  negative age that reads as "long ago" (`max(0, ...)` in `image_prune_verdict`).
- `test_the_clock_is_taken_after_the_disk_read_with_a_real_read_lag` uses the live order: the clock
  first, then a disk read that takes 2.4 s. `test_an_ask_finished_after_the_ticks_clock_is_age_zero_not_negative`
  checks the negative-age case.
- The post-pull rebuild restarts services alphabetically, and cortex-exec may be mid-restart. The first
  tick is harmless: there is a 10 min start delay, and a failed RPC is recorded `rpc_error` and retried
  after 30 min (`test_a_failed_first_ask_after_a_restart_is_harmless_and_retried`).

## Metric quality gate: the docker volume's fill level

1. **Provenance.** `read_docker_volume` -> `shutil.disk_usage("/hostfs/docker")`, `used / total`. This
   is the same formula as the skill's `_builder_prune_disk_usage` (`verb_adapters.py:2849`), on the same
   bind (`/mnt/docker`). In the built image it read 81.303 %. The skill's last live run read 81.306 %.
2. **Independence.** It is deliberately **not** independent: the rule must agree with the skill's own
   disk gate. It feeds no model or cognition loop. It is the same filesystem statistic as orion-biometrics'
   `disk_capacity` pressure (max over 5 mounts). It is not a new signal, and nothing combines the two.
3. **Theory anchor.** Dangling images occupy space on exactly this filesystem. The threshold is the
   skill's own reasoned floor (70 %: no reuse value is traded away by reclaiming a dangling image).
4. **Live data.**
   - Over the 14-day window (10-22 .. 11-05) the volume read 79.09-81.53 %. It never went below the
     floor, so in that window the cooldown, not the disk, set the ask rate. Said plainly: the rule acts as
     a 12 h timer, and the skill's dangling-image count decides whether anything is pruned.
   - The reading can go back below the floor: on 10-11/10-12 it fell to 68.6-69.8 % after a cleanup, and
     then climbed back. So it is not pinned.
   - Cross-check: athena's biometrics `disk_capacity` matched the skill's own `disk_before` within 0.047
     points on all 173 dispatch runs since 10-08 (mean difference 0.003).
5. **Existing mechanism.**
   - Biometrics `disk_capacity`: rejected as the live source. It is the max over five mounts, not this
     one, and it would add a Postgres read lag. It is used only for the replay history.
   - `scripts/disk_threshold_watchdog.py`: alerts at 90 %. It does not prune.
6. **Reversibility.** Cheap to undo. There is no schema, no table and no persisted state. The kill switch
   is `HARDWARE_WATCH_IMAGE_PRUNE_ENABLED=false`, and reverting the PR puts the dispatch templates back.

## Replay: 14 real days

Command: `cd services/orion-hardware-watch && PYTHONPATH=../..:. python evals/run_image_prune_replay_eval.py`.
- Window: 2026-10-22 00:00 to 2026-11-05 00:00 UTC (the spec's audit window).
- Fixture: `evals/fixtures/image_prune_replay.json.gz`, exported from live Postgres with `--export`.

How the replay works:
- It runs the shipped `ImagePruneRule` every 30 s on a simulated clock.
- The disk comes from athena's biometrics `disk_capacity`. The fixture keeps the max reading per 5 min.
- The skill's answer comes from the real dangling-image counts in dispatch's real runs (156 in the window,
  plus the runs from the 30 days before it as the seed). New dangling images between two real runs are
  spread evenly over that gap. The replay keeps its own count: it prunes at >= 100 dangling images and
  >= 70 % disk, and resets the count to 0 after a prune.
- The "backlog" is the time-weighted mean of dangling images on the host, every tick, on both sides.

| | Asks | Prunes | Declined | Images removed | Mean backlog |
|---|---|---|---|---|---|
| Real (dispatch) | 156 | 11 | 145 | 1,993 | 97.0 |
| Disk rule, 12 h cooldown (shipped) | 28 (2.0/day) | 12 | 16 | 1,924 | 59.9 |

- Every one of the 11 real prunes has a replay prune within 12 h of it.
- The disk was readable on all but 10 ticks (a gap in the biometrics record).

Cooldown sweep (`--sweep`):

| Cooldown | Asks | Prunes | Declined | Images removed | Mean backlog |
|---|---|---|---|---|---|
| 1 h | 336 | 16 | 320 | 1,956 | 44.1 |
| 3 h | 112 | 15 | 97 | 1,970 | 53.0 |
| 6 h | 56 | 13 | 43 | 1,970 | 58.8 |
| **12 h** | **28** | **12** | **16** | **1,924** | **59.9** |
| 24 h | 14 | 10 | 4 | 1,830 | 79.1 |
| 48 h | 7 | 6 | 1 | 1,701 | 124.3 |

Why 12 h:
- Going from 6 h to 12 h halves the asks and barely moves the backlog (58.8 -> 59.9).
- 24 h is the first step where the backlog rises clearly (79.1).
- 48 h is worse than dispatch (124.3 against 97.0), and the backlog gate below refuses it.
- Each prune reclaims only 2-8 MB (one reclaimed 861 MB), so pruning sooner buys almost nothing.

Gates, all PASS:
- asks only on a readable disk at or above 70 %;
- never two asks inside the cooldown (minimum gap 43,200 s);
- at most one ask per cooldown;
- the replay removes >= 90 % of the images the real record removed;
- the host carries no more dangling images on average than under dispatch (59.9 <= 97.0).

Fault injection, both PASS:
- the disk held at 69 % for a whole day: 0 asks that day;
- a 2 h gap in the disk record: 0 asks in it. This checks the replay's own stale-reading rule. Live, the
  rule reads the disk itself, and the unit tests cover an unreadable or stuck read.

What the replay does not model:
- **Restarts.** Each restart resets the in-memory cooldown, so the first ask comes 10 min after the start.
  hardware-watch was rebuilt on 10 of the last 15 days, so live asks can be about 2-4 a day, not 2.
- **RPC failures.**
- **Hand prunes.**
- **The seed.** The count at the window start comes from the last real run before it (10-18) plus the
  arrivals spread to 10-22. Dispatch sent no run between 10-18 and 10-22 11:23, so that gap's 428 new images
  are spread over 4 days. The real backlog side uses the same model, so the two are compared on the same
  terms.

## Files changed

Disk rule:
- `orion/hardware_watch/rules.py`: `IMAGE_PRUNE_MIN_DISK_PCT`, `DiskReading`, `ImagePruneRuleConfig`,
  `ImagePruneState`, `image_prune_verdict` (pure).
- `services/orion-hardware-watch/app/image_prune.py`: new. The disk read, the rule object (background ask,
  counters, `/health` snapshot), the skill-answer parser and the bus ask.
- `services/orion-hardware-watch/app/main.py`: builds the rule, ticks it beside the watcher, `/health` ->
  `image_prune`, cancels an in-flight ask on shutdown.
- `services/orion-hardware-watch/app/settings.py`, `.env_example`, `docker-compose.yml`: eight new keys
  and the read-only `/hostfs/docker` bind.
- `services/orion-hardware-watch/tests/test_image_prune.py` (29 tests): new.
- `services/orion-cortex-exec/tests/test_image_prune_hardware_watch_request.py`: new. The disk rule's own
  payload, fed through cortex-exec's `build_exec_ctx` -> `_plan_request_from_step_ctx` -> `_skill_args`
  -> `_resolve_docker_prune_run_mode`, comes out `execute`.
- `services/orion-hardware-watch/evals/run_image_prune_replay_eval.py` and
  `evals/fixtures/image_prune_replay.json.gz`: new.
- `services/orion-hardware-watch/README.md`: the disk rule section, the kill switch and the eval commands.
- `.github/workflows/orion-hardware-watch-tests.yml`: runs the replay and the parity test, plus a new
  `cortex-exec-contract` job (cortex-exec's requirements) for the execute-mode test. Its paths include
  the verb, the plan loader, the cortex schemas, and cortex-exec's adapter, executor and exec_ctx.

Old trigger removed:
- `config/proposals/proposal_policy.v1.yaml`: the three templates are removed, with a tombstone.
  `orion/proposals/templates.py`: their copy is removed.
- `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the `maintain` route, the two template
  routes, `allow_mutating_dispatch`, `approved_maintenance` and the reserved slot are removed.
- `orion/execution_dispatch/policy.py`: `allow_mutating_dispatch` and `MAINTENANCE_SCOPE` are removed.
  `builder.py`: the scope no longer admits `maintenance_bounded`, and its warning and reason string are
  removed. `envelopes.py`: no derived `mode`, and the constraints are always read-only.
- `config/policy/substrate_policy.v1.yaml`: the `maintain` rule is marked inert until PR 6 (the same
  treatment PR 5 gave `self_regulate`).

Contract and comments:
- `orion/bus/channels.yaml`: orion-hardware-watch is added as a producer of
  `orion:cortex:exec:request:background` and a consumer of `orion:exec:result:*`.
- `config/metrics/metric_definitions.lock.json`: re-locked for those two routing changes.
- `config/physiology/families.yaml`, both prune verb yamls, cortex-exec `.env_example`/`settings.py`,
  dispatch `README.md`/`.env_example`, `orion/inner_state_registry.py`, `orion/metrics/liveness.py`:
  the comments now say who asks for each prune.

Tests:
- `tests/test_image_prune_disk_rule_parity.py`: new. The rule's floor equals the skill's (read with ast);
  no prune template, route, slot, flag or decision is left; the hand-run skills stay; hardware-watch is a
  declared producer.
- `tests/test_maintenance_dispatch_gating.py`: rewritten. A stale maintenance route is blocked by scope
  even with the old flag on; no envelope derives an execute mode; a stray `maintain` candidate still
  evaluates and dispatch refuses it. The general dry-run, env-template, fault-isolation and
  confidence-gate tests are kept.
- `tests/test_tension_driven_dispatch_gating.py`: deleted. It only tested the two removed routes; the
  camera route keeps its pin in `test_cortex_route_resolution.py`.
- `tests/test_dispatch_starvation.py`: the reserved lane is tested on a test-only summarize reservation
  in the 5-slot arena it was written for. This fixes 12 failures that were already on main (the arena cap
  became 50 on 08-30).
- `tests/test_cortex_route_resolution.py`, `test_execution_dispatch_policy_loader.py`,
  `test_express_outward_action.py`, `test_image_prune_skill.py` (the timeout ladder now ends at
  hardware-watch's 720 s), `test_proposal_policy_loader.py`, `test_proposal_scoring.py`,
  `test_self_study_analysis_wiring.py`.

## Schema / bus / API changes

- **Added:**
  - `/health` key `image_prune` on orion-hardware-watch;
  - orion-hardware-watch on `orion:cortex:exec:request:background` (producer) and `orion:exec:result:*`
    (consumer).
- **Removed:**
  - the proposal templates `prune_build_cache`, `prune_dangling_images` and `prune_stopped_containers`;
  - their dispatch routes, and the `maintain` kind route;
  - `mode.allow_mutating_dispatch`, `DispatchModeConfigV1.allow_mutating_dispatch` and `MAINTENANCE_SCOPE`;
  - `approved_maintenance` from dispatch's allowed decisions;
  - `limits.reserved_slots_by_scope.maintenance_bounded`;
  - the frame warning `mutating_dispatch_disabled_by_policy` and the reason `approved_maintenance_dispatch_v1`.
- **Behavior changed:**
  - A route naming `maintenance_bounded` is blocked `route_scope_not_read_only`.
  - Dispatch envelopes never carry a derived `skill_args.mode`.
  - The image prune is asked for by hardware-watch, not dispatch.
- **Compatibility:**
  - `PolicyDecisionV1`'s `maintenance_bounded` literals stay, so stored frames still parse until PR 6.
  - The `maintain` proposal kind literal and policy rule stay too, inert.
  - A policy file that still names `allow_mutating_dispatch` loads (the mode model ignores the key).
  - No table changes.

## Env/config changes

- **Added keys (orion-hardware-watch).** Flags ship ON.
  - `HARDWARE_WATCH_IMAGE_PRUNE_ENABLED=true`
  - `HARDWARE_WATCH_IMAGE_PRUNE_MOUNT_PATH=/hostfs/docker`
  - `HARDWARE_WATCH_IMAGE_PRUNE_COOLDOWN_SEC=43200`
  - `HARDWARE_WATCH_IMAGE_PRUNE_RETRY_SEC=1800`
  - `HARDWARE_WATCH_IMAGE_PRUNE_START_DELAY_SEC=600`
  - `HARDWARE_WATCH_IMAGE_PRUNE_RPC_TIMEOUT_SEC=720`
  - `CORTEX_EXEC_CHANNEL=orion:cortex:exec:request:background`
  - `CORTEX_EXEC_RESULT_PREFIX=orion:exec:result`
  - `ORION_HOST_DOCKER_DATA_ROOT=/mnt/docker`
- **Removed keys:** none. No env key named the removed routes. The cortex-exec keys
  (`BUILDER_PRUNE_*`, `IMAGE_PRUNE_TIMEOUT_SEC`, `SKILLS_ALLOW_MUTATING_RUNTIME_HOUSEKEEPING`) are still
  read by the hand-run skills.
- **`.env_example` updated:** yes, hardware-watch. Comments only in cortex-exec and dispatch.
- **Local `.env` synced:** the main checkout's `services/orion-hardware-watch/.env` got the 9 keys by hand.
  The sync script cannot reach the main checkout from a worktree.
- **Skipped keys:** none.

## Tests run

```text
services/orion-hardware-watch/tests (whole folder)                       110 passed (29 new)
  same, in a throwaway venv with only requirements.txt + pytest (the CI job's install)   106 passed (before the review fixes)
services/orion-cortex-exec tests/test_image_prune_hardware_watch_request.py
  + test_executor_docker_prune_skill_args.py + test_skill_verbs.py (both orders)          all pass
tests/test_maintenance_dispatch_gating.py, test_image_prune_disk_rule_parity.py,
  test_dispatch_starvation.py, test_execution_dispatch_policy_loader.py,
  test_execution_dispatch_result_extraction.py, test_express_outward_action.py,
  test_image_prune_skill.py, test_measure_arena_degeneracy.py, test_proposal_policy_loader.py,
  test_proposal_scoring.py, test_rung3_act_candidates.py, test_self_monitor_act_gate.py,
  test_self_study_analysis_wiring.py, test_cortex_route_resolution.py, test_artifact_rating.py
                                                                          all pass (see note)
orion-static-gates commands run locally (pip steps skipped)              all OK after the metric re-lock
scripts/check_definition_drift.py --gate                                 PASS (re-locked: 2 routing changes)
```

Note: three failures do not come from this branch.
- `test_self_study_analysis_wiring.py::test_the_verb_is_registered_under_that_exact_name` passes on its
  own. It fails only in a combined run, because an `app` package from another service is already
  imported.
- `tests/test_execution_dispatch_runtime_worker.py` needs a local Postgres on :5432. It fails the same way
  on main.
- The whole `services/orion-cortex-exec/tests` folder in one run has 14 collection errors ("Verb already
  registered") with or without the new file. That is a pre-existing double-import between test files.

## Evals run

```text
python evals/run_image_prune_replay_eval.py           28 asks -> 12 prunes, 1,924 images, backlog 59.9 vs 97.0; 7/7 gates PASS
python evals/run_image_prune_replay_eval.py --sweep   1/3/6/12/24/48 h table above
python orion/body_atlas/evals/run_wire_eval.py (body-atlas-tests workflow, run locally)   PASS
python evals/run_rules_replay_eval.py                 PASS (unchanged heat/AC replay)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-hardware-watch/Dockerfile .                OK
in the image (live hardware-watch .env, /mnt/docker bound read-only, --network none):
  settings: image_prune_enabled=True, cooldown 43200, mount /hostfs/docker,
            channel orion:cortex:exec:request:background
  read_docker_volume -> DiskReading(used_pct=81.303)   (the skill's last live run: 81.306)
  build_plan_for_verb("skills.runtime.image_prune.v1") -> one step, timeout_ms 660000
  app.main imports
```

UNVERIFIED on the live rail: no ask has reached cortex-exec from hardware-watch yet (not deployed). No
prune was run by hand for this PR.

## Review findings fixed

A code-review subagent read the whole diff. It traced the disk rule's payload through cortex-exec's real
functions to the skill's run mode (`execute`) and the reply decode against `CortexExecResultPayload`. It
also checked the clock order and the shutdown order, and grepped the repo for leftovers. It found no
blockers.

- Finding (should-fix): the settings default for `HARDWARE_WATCH_IMAGE_PRUNE_ENABLED` was false, while
  `.env_example` and compose say true.
  - Fix: true in code, and the README states the exception.
  - Evidence: `test_the_settings_default_ships_the_rule_on_like_the_env_template` pins the flag and the
    three timing defaults to `.env_example` and the rule config.
- Finding (should-fix): no test pinned the cortex-exec side. If cortex-exec stopped carrying
  `ctx["skill_args"]`, the skill would quietly answer `would_act` twice a day.
  - Fix: `services/orion-cortex-exec/tests/test_image_prune_hardware_watch_request.py`, plus a CI job that
    installs cortex-exec's requirements.
  - Evidence: 4 passed with the docker-prune skill-args tests, in either order with `test_skill_verbs.py`.
- Finding (should-fix): `would_act` counted as an answer and held the 12 h cooldown.
  - Fix: it is out of `IMAGE_PRUNE_ANSWERED`, so it now waits the 30 min retry and logs
    `image_prune_preview_despite_execute` as a warning.
  - Evidence: `test_a_preview_answer_is_retried_and_warned_not_held`, and `would_act` added to the retry
    parametrisation.
- Finding (nit): the replay's seed window held no run, so the seed path was dead.
  - Fix: `SEED_DAYS` 3 -> 30 (seeds from the 10-18 run) and the fixture re-exported. The numbers above
    are from the new fixture.
- Finding (nit): the "keeps up" gate could not tell 12 h from 24 h.
  - Fix: a time-weighted mean-backlog metric on both sides, a gate that it is no worse than dispatch, and a
    48 h row in the sweep. The 12 h choice is argued from the sweep (see "Why 12 h").
- Finding (nit): the 2 h silence check mostly tests the eval itself.
  - Fix: renamed to say so ("replay check ... live reads the disk itself").
- Finding (nit): a stuck disk read would stall the heat rules in the same loop.
  - Fix: a 5 s `asyncio.wait_for` maps to `DiskReading(None, "read_timeout")`.
  - Evidence: `test_a_stuck_disk_read_times_out_as_unreadable`.
- Finding (nit): the cooldown is lost on a restart, and a shutdown leaves cortex-exec's prune running.
  - Fix: both are stated in the README (and the restart point in "Risks / concerns"). No code change: one
    extra cheap ask per restart.
- Finding (nit): `skill_answer` reads only `result.final_text`.
  - Fix: the docstring says so and why: it is what cortex-exec sets for one local skill step, and what
    dispatch's stored results show.

## Restart required

Deploy from the clean primary `main` checkout after merge. The post-pull rebuild restarts services in
alphabetical order. Among the ones that matter here, that order is: orion-cortex-exec, then
orion-execution-dispatch-runtime, then orion-hardware-watch, then orion-policy-runtime, then
orion-proposal-runtime.

1. **orion-cortex-exec** rebuilds. Only comments changed; the skill is unchanged.
2. **orion-execution-dispatch-runtime** rebuilds. Its prune routes, its maintenance slot and its flag are
   gone. A `maintain` proposal the old proposal-runtime still sends is blocked (no route, and
   `approved_maintenance` is not allowed). From here dispatch sends no prune.
3. **orion-hardware-watch** rebuilds. The disk rule starts. Its first ask comes 10 min later. If
   cortex-exec is not listening yet, the ask fails at once and is retried after 30 min.
4. **orion-policy-runtime** and **orion-proposal-runtime** rebuild. The templates stop.

This order never has two triggers, because dispatch loses the routes before hardware-watch gains the rule.
If hardware-watch were rebuilt first by hand, there could be one overlap. It is harmless: the second ask
finds fewer than 100 dangling images and declines. There is no gap risk either, because a few minutes with
no trigger misses at most a few MB.

By hand, in the same order:

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-hardware-watch up -d --build
scripts/safe_docker_build.sh orion-policy-runtime up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

Stale env keys: none. No sed is needed. Each `.env` key that names a prune is still read by a hand-run
skill.

Live checks after the deploy:
- `curl -fsS http://localhost:8131/health | jq .image_prune` shows `enabled: true` and `last_tick.disk_used_pct`
  near 81.
- After 10 min, `last_ask.outcome` is set (`pruned` or `declined_nothing_to_reclaim`).
- Within a day: 0 new `substrate_dispatch_results` rows with `dispatch_kind = 'maintain'`.

## Risks / concerns

- **Severity: low.** Concern: in the 14-day window the disk never fell below the floor, so the rule is a
  12 h timer in practice, and the skill's dangling count decides. Mitigation: that is the skill's own gate,
  the same one dispatch relied on. Below 70 % the rule stops asking (fault injection).
- **Severity: low.** Concern: each restart resets the in-memory cooldown, which gives one extra ask 10 min
  later. hardware-watch was rebuilt on 10 of the last 15 days. Each extra ask is one `docker images` call
  and at most one cortex-exec background slot for about 1 s. Mitigation: none needed. Persisting the last
  ask would need a table or a Redis key for a few MB of benefit.
- **Severity: low.** Concern: the build cache is no longer pruned automatically. Dispatch sent it 0 times
  in the window anyway, and its skill declines below 75 % / 40 GB. It is not on the Hub Skill Runner
  dropdown; it never was. Mitigation: Juniper's decision 4. The skill can still be run by hand as a cortex
  verb.
- **Severity: low.** Concern: the body-economy family name `execution_dispatch.maintain` is now historical.
  Nothing bids as it (0 rows). Mitigation: a comment in `families.yaml`. A rename is left for when a bid is
  added, so the want ledger has one name from its first row.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3153

🤖 Generated with [Claude Code](https://claude.com/claude-code)
