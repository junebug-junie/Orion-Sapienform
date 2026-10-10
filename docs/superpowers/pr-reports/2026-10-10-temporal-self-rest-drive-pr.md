## Summary

Orion's tiredness (the dream service's sleep pressure) is now something other parts of Orion can read, and two of them ease off when Orion is tired.

- The dream service turns every 600 s pressure check, and the moment right after every sleep, into a `DriveReadingV1`: `resting`, `building`, `due` (tired, waiting to sleep), `refractory` (inside the 6 h minimum), or `no_reading` (a source read failed). It writes the latest one to Redis `orion:drive:rest:latest` with an 1800 s expiry.
- Hub **outreach** doubles its 45-minute cooldown (to 90 min) while the drive reads `due`.
- Hub **curiosity** doubles each line's cooldown (investigation, self-inquiry, self-sense eval) while the drive reads `due`.
- Nothing else moves: daily caps, quiet hours, recent-chat, turn-in-flight, waking windows, forced runs and Door-A (a finished curiosity run with something to say) are untouched. A missing, expired, unparseable or `no_reading` value, or a Redis error, is **unknown**, and unknown is exactly the old behaviour.
- One flag per reader, all shipped ON. A replay eval measures how often each reader would have eased off per day.

## Outcome moved

Before: only the dream service listened to tiredness. After: tiredness changes when Orion reaches out and when it starts curiosity runs, and every refusal it causes is logged as `reason=rest_drive_cooldown` (outreach decisions table and curiosity logs), so the effect is countable.

How much, measured (details under Evals):

- Orion reads as tired about **50 min/day** on average over 13 full replayed days (median 12 min, range 0-232, 2 days with none). The one real post-#2557 day: **20 min**, the stretch between pressure crossing 3 (00:41 UTC) and the sleep (01:01 UTC).
- **Outreach** would have been held back **8 times in 14 days** (out of 55 background sends; 1-3 on 4 days, 0 on the other 10). Each hold is a delay to the 90-minute mark or until Orion sleeps, not a cancelled message.
- **Curiosity** would have been held back **0 times** (98 investigation turns; only 1 started inside a tired stretch at all). At today's settings the curiosity half is wired and tested but has near-zero live effect. That is reported, not tuned away: raising the multiplier to manufacture an effect would be picking a number to look busy.

## Current architecture

- `services/orion-dream/app/cycle.py` `run_cycle_once` reads pressure every 600 s (`read_pressure` -> `compute_pressure`, novelty formula since #2557), saves each check to `dream_pressure_observation` (#2563), and sleeps when pressure >= 3.0, chat idle >= 45 min, and >= 6 h since the last attempt, or when 48 h pass with material (overdue backstop).
- The Hub tiredness gauge (#2559, `static/js/biometrics-view.js`) reads `GET /dreams/cycle/pressure` through `/api/dream/pressure`, which runs the same `read_pressure`.
- Curiosity (`curiosity_investigation.py` `scheduling_block_reason`) and outreach (`endogenous_outreach.py` `outreach_block_reason`) each pace on their own private cooldown, cap and window. Neither read tiredness.

## Architecture touched

- **Producer**: orion-dream, via a new optional `CycleDeps.publish_drive_reading`. Built by the pure `orion/regulation/rest_drive.py::read_rest_drive` from the exact values the sleep gate uses (threshold, `too_soon`, `overdue` + candidates), so there is one copy of "due". A publish failure is logged (`rest_drive_publish_failed`) and never changes or fails a sleep.
- **Contract**: `orion/schemas/drive_reading.py` (`DriveReadingV1`, `extra="forbid"`), registered in `_REGISTRY`, `orion/inner_state_registry.py` (required: name contains "drive"; kept distinct from the retired `DriveStateV1`), and the metric lock.
- **Readers**: `services/orion-hub/scripts/rest_drive_reader.py`, one instance per consumer. One Redis GET per tick; staleness re-judged at the caller's `now`, so a reader that stops being refreshed decays to unknown rather than staying tired.
- **Debug surfaces**: `GET /dreams/cycle/pressure` now includes `rest_drive`; the outreach status endpoint includes `rest_drive` and `rest_drive_cooldown_sec`; each reader logs `rest_drive_view ...` once per change.

## Files changed

- `orion/schemas/drive_reading.py`: the reading contract and Redis key.
- `orion/regulation/rest_drive.py`, `orion/regulation/__init__.py`: pure producer (`read_rest_drive`, `no_rest_reading`) and reader rule (`rest_drive_view`, `eased_cooldown_sec`).
- `orion/schemas/registry.py`, `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: registration (lock re-generated: 4 added definitions, nothing else changed).
- `services/orion-dream/app/cycle.py`: publish at every check (built and sent inside one guard with a 5 s timeout), `no_reading` on a failed read, a fresh `dp-postsleep-*` reading right after each sleep (after the story trigger); `check_id` now shared with the saved observation.
- `services/orion-dream/app/main.py`: Redis publisher; `rest_drive` on the pressure endpoint.
- `services/orion-dream/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `DREAM_REST_DRIVE_PUBLISH_ENABLED`, `DREAM_REST_DRIVE_REDIS_TTL_SEC`.
- `services/orion-hub/scripts/rest_drive_reader.py`: the Hub reader.
- `services/orion-hub/scripts/curiosity_investigation.py`: `SchedulingGateInputs.rest_drive_cooldown_sec`, reason `rest_drive_cooldown`, refresh at the top of `tick`, all three lines; forced runs override it like the plain cooldown.
- `services/orion-hub/scripts/endogenous_outreach.py`: `OutreachGateInputs.rest_drive_cooldown_sec`, reason `rest_drive_cooldown` (skipped by Door-A), refresh at the top of `_outreach_once`, status fields.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`, `README.md`: wiring and the four Hub keys plus the shared max age.
- `scripts/analysis/measure_rest_drive_easing.py`: the eval.
- `docs/superpowers/evidence/2026-10-10-temporal-self-rest-drive/rest-drive-easing.json`: eval output.
- Tests: `tests/test_rest_drive.py`, `services/orion-dream/tests/test_rest_drive_producer.py`, `services/orion-hub/tests/test_rest_drive_readers.py`, `scripts/analysis/tests/test_measure_rest_drive_easing.py`.
- `.github/workflows/rest-drive.yml`: runs all four in CI.

## Schema / bus / API changes

- Added: `DriveReadingV1` (`drive.reading.v1`); Redis key `orion:drive:rest:latest`; `rest_drive` on `GET /dreams/cycle/pressure`; `rest_drive`, `rest_drive_cooldown_sec` on the outreach status payload; block reason `rest_drive_cooldown` (curiosity logs, `endogenous_outreach_decisions.reason`).
- Removed / renamed: none.
- Behavior changed: outreach and curiosity cooldowns double only while the drive reads `due`.
- Compatibility notes: **no bus channel.** The spec names `orion:drive:reading`; it was built first and dropped, because nothing subscribes to it yet (the spec's regulation step in durable-runs is unbuilt) and the metric-lineage gate correctly refused the orphan channel (`bus_channel` orphans 17 -> 18). History already exists per check in `dream_pressure_observation`, joined by `source_ref = check_id`. Readers use Redis, the same pattern as `orion:situation:latest`.

## Env/config changes

- Added keys: orion-dream `DREAM_REST_DRIVE_PUBLISH_ENABLED=true`, `DREAM_REST_DRIVE_REDIS_TTL_SEC=1800`; orion-hub `ORION_REST_DRIVE_MAX_AGE_SEC=1800`, `HUB_ENDOGENOUS_OUTREACH_REST_DRIVE_ENABLED=true`, `HUB_ENDOGENOUS_OUTREACH_REST_DRIVE_COOLDOWN_MULTIPLIER=2.0`, `HUB_CURIOSITY_REST_DRIVE_ENABLED=true`, `HUB_CURIOSITY_REST_DRIVE_COOLDOWN_MULTIPLIER=2.0`.
- Removed / renamed keys: none.
- `.env_example` updated: yes (orion-dream, orion-hub); settings defaults match (flags ON, multipliers 2.0, `ge=1.0`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, `orion-dream orion-hub`, then `--all-keys` for the three keys outside the default sync prefixes. All seven present in the primary checkout's `.env` files.
- skipped keys requiring operator action: none. (Pre-existing, unrelated divergence left alone: `HUB_ROOM_CLAUDE_ENABLED` local `true` vs example `false`.)

## Metric gate (AGENTS.md 0A) for the tiredness value

1. **Provenance.** `level` = `SleepPressureV1.pressure` from `compute_pressure` (`services/orion-dream/app/replay.py`), read in `read_pressure` (`app/cycle.py`). `state` from `read_rest_drive` (`orion/regulation/rest_drive.py`), fed by the same `too_soon` / `overdue` / threshold the sleep gate uses in `run_cycle_once`.
2. **Independence.** Not a new signal: it is the existing dream pressure, now readable. It does not overlap the readers' own inputs: outreach's recent-chat gate reads Juniper's last message, and pressure counts new metacog problems, compaction themes, resonance themes and crystallizations, not chat recency. One causal chain exists and is kept on purpose: curiosity runs can create crystallizations, which add pressure, which slows curiosity. That is negative feedback (work raises tiredness, tiredness slows optional work). Crystallizations were 17 of 12,482 source rows in the 14-day window.
3. **Theory anchor.** Process S of Borbély's two-process model of sleep regulation (1982): homeostatic sleep pressure builds with time awake, discharges in sleep, and while high reduces discretionary activity. Spec R2 admits rest as the only drive that passes the gate.
4. **Live data, including rest.** 125 saved checks, 2026-10-09 06:53 to 2026-10-10 03:32 UTC (first pull). Pressure reads exactly 0.0 at the first check after a sleep (01:32 UTC 10-10) and on 35 checks of 10-09, and that zero is structural, not decay: the window restarts at the sleep's start, so zero means nothing new arrived since. It built to 2.3, crossed 3 at 00:41, the dream slept at 01:01, and reset. States seen live: refractory 50, resting 35, building 37, due 3. Not saturated: the post-#2557 maximum is 13.3 against a ceiling of 200 (50 per source x 4 sources x weight <= 1).
5. **Existing mechanism.** Reused, not duplicated: the dream's own pressure and gates; the gauge's source (`/dreams/cycle/pressure`, which now also returns the reading); the reader shape of `energy_stakes_gate.py` (fresh-or-unknown, unreadable never holds).
6. **Reversibility.** Cheap. One flag per reader plus the producer flag; a Redis key with an expiry; no table, no migration, no bus channel. Removing it is deleting the reader wiring and one registry entry.

## Tests run

```text
python -m pytest tests/test_rest_drive.py                                   15 passed
(services/orion-dream) python -m pytest tests/ evals/                       163 passed, 1 skipped
(services/orion-hub)   python -m pytest tests/test_rest_drive_readers.py    30 passed
(services/orion-hub)   readers + curiosity + outreach + self-inquiry + self-sense + admission suites   476 passed
python -m pytest tests/test_rest_drive.py scripts/analysis/tests/test_measure_rest_drive_easing.py scripts/analysis/tests/test_regulation_measurements.py tests/test_agent_trace_schema_registry.py   62 passed
Clean CI-equivalent venv (uv, services/orion-hub/tests/requirements-reading.txt + sql-writer reqs): all four rest-drive steps green
scripts/check_metric_lineage.py --gate         PASS
scripts/check_metric_lineage.py --prompt-semantics   PASS
scripts/check_definition_drift.py --gate       PASS (after --update; 4 added)
scripts/check_inner_state_registry.py          OK (20 entries)
scripts/check_env_template_parity.py           PASS
tests/test_agent_trace_schema_registry.py tests/test_metric_prompt_semantics.py   44 passed
static gates (async routes, control surface, system health, chat poachers, hostnames, stdlib shadow, sentience instruments, journal dispatch)   all pass
```

Full Hub suite (`services/orion-hub/tests`, e2e excluded): 40 failures, and the same 40 test ids fail identically on a clean `origin/main` checkout on this host (UI asset strings, missing local Postgres, route selector). None touches a file in this PR.

What the tests pin, per reader: tired -> the cooldown stretches and the reason is `rest_drive_cooldown`; resting, building, refractory -> unchanged; absent, unparseable, stale, `no_reading`, Redis error, flag off -> same decision as with the drive switched off (each compared directly). Producer: due then refractory across a real sleep, building/resting, refractory beats a high level, overdue backstop, `no_reading` on a failed and on a partial read, a publish failure never changes the sleep, the Redis key and TTL. Lifecycle (pure): the live 10-09 to 10-10 shape, rest -> building -> due -> sleep -> refractory -> rest, plus an unrefreshed `due` decaying to unknown.

## Evals run

```text
python scripts/analysis/measure_rest_drive_easing.py --print-sql --start 2026-09-26T04:00:00Z --end 2026-10-10T04:00:00Z
  (run as one READ ONLY transaction against conjourney; 12,945 rows)
python scripts/analysis/measure_rest_drive_easing.py rows.jsonl --start ... --end ...
  -> docs/superpowers/evidence/2026-10-10-temporal-self-rest-drive/rest-drive-easing.json
```

| Lane | What it is | Tired minutes | Outreach held | Curiosity held |
|---|---|---|---|---|
| saved | 127 real checks, 10-09 06:53 to 10-10 ~04:00 | 0 on 10-09, 20 on 10-10 | 0 of 4 | 0 of 7 |
| replay | 2,016 checks over 14 days, novelty pressure recomputed with the dream's own loaders, threshold-3 schedule simulated with the live gate order (all-chat idle, refractory from the last attempt of any status) | 50/day mean over 13 full days, median 12, range 0-232 | 8 of 55 (09-26: 2, 09-27: 1, 10-02: 3, 10-05: 2); 1 Door-A send excluded | 0 of 98 (1 turn started in a tired stretch, outside the stretched window) |

Replay states across all 2,016 checks: building 1,032, refractory 840, due 88, resting 56. The replay simulated 23 sleeps in 14 days.

How "held" is counted: a send/turn that started while the reader would have judged Orion tired, and whose gap since the previous one was between the normal cooldown and twice it. A tired reading holds until the next check, the 1800 s staleness bound, or the next sleep end (the dream publishes a fresh reading right after each sleep), whichever is first.

Limits, stated: the replay before 10-09 06:33 is counterfactual (the novelty formula was not live then; it validated exact against all 114 saved checks it overlaps, per #2576). "Held" is first-order: a held send would have shifted the ones after it, which is not simulated. Curiosity turns come from `curiosity_offer_decisions.turn_started_at` (the investigation line, same start the live cooldown stamps) with the 1800 s base; that table does not record whether a run was forced, and forced runs override the drive.

## Docker/build/smoke checks

```text
Not run. Juniper asked for no deploy. No live smoke of the Redis key: UNVERIFIED until orion-dream and orion-hub are rebuilt.
scripts/check_service_env_compose_parity.py orion-dream / orion-hub: both declare env_file, all keys reach the container.
```

## Review findings fixed

Review: `orion-repo-agent` subagent over `git diff origin/main...HEAD`. Three major, seven minor, three nits. All fixed except two notes recorded under Risks.

- Finding (major): the Hub reader called `getattr(bus, "redis")` outside its try. On the real bus that is a property which raises `RuntimeError` when the bus is not connected, so an unknown state would have aborted the whole curiosity/outreach tick instead of behaving as before.
  - Fix: the whole read is inside one try, with a 2 s timeout so a hung Redis cannot stall a tick.
  - Evidence: `test_redis_failure_is_unknown_never_an_exception[_NotConnected]`, `test_a_hung_redis_does_not_stall_the_tick`, `test_outreach_tick_survives_a_disconnected_bus`.
- Finding (major, test passing for the wrong reason): the Redis-failure test used a shape the real bus never has.
  - Fix: added the property-raises bus, a plain object with no `redis`, and a hanging `get`.
  - Evidence: same tests as above.
- Finding (major, eval could mislead): outreach events included Door-A sends, which the drive never gates.
  - Fix: Door-A rows (`source=curiosity_outreach`) reset the clock but are never counted.
  - Evidence: `test_door_a_sends_reset_the_clock_but_are_never_counted`; evidence re-run, outreach held 9 -> 8, 1 Door-A send excluded.
- Finding (minor, eval): curiosity used `completed_at`, but the live cooldown is stamped at turn start.
  - Fix: switched to `curiosity_offer_decisions.turn_started_at`.
  - Evidence: export test, re-run numbers.
- Finding (minor, eval): the replay seeded the refractory clock from good cycles only; the live clock uses the last attempt of any status.
  - Fix: seeded from the latest `ended_at` across all cycles.
  - Evidence: re-run.
- Finding (minor, eval): a `due` reading right before a sleep was counted for its full 600 s slot, but live readers see the post-sleep reading within seconds.
  - Fix: the hold is cut at the next sleep end in both lanes.
  - Evidence: `test_a_sleep_end_cuts_the_tired_hold_like_the_post_sleep_publish`. Tired minutes fell from 57 to 50 a day.
- Finding (minor): the post-sleep reading had no observation row, but the docs claimed every `source_ref` joins `dream_pressure_observation`.
  - Fix: its id is `dp-postsleep-*`, and the docs say it has no row.
  - Evidence: `test_post_sleep_reading_is_marked_and_comes_after_the_story`.
- Finding (minor): the publish was awaited inline with no timeout, and building the reading sat outside any guard.
  - Fix: one guarded helper builds and publishes, with a 5 s timeout. A bad reading or a dead Redis is logged and the loop carries on.
  - Evidence: `test_a_hung_publisher_cannot_hold_up_the_sleep_decision`, `test_a_reading_that_fails_to_build_never_kills_the_loop`.
- Finding (minor): the post-sleep read ran before the story trigger, and it dropped idle-read errors.
  - Fix: it now runs after the story trigger and merges `deps.read_errors`.
  - Evidence: the ordering test above. Cost: one extra pressure read per sleep, 1-4 a day. That cost is documented, not removed.
- Finding (minor): `/dreams/cycle/pressure` did not fold in source errors, so it could show `due` where the loop would publish `no_reading`.
  - Fix: it passes and merges the read errors.
  - Evidence: `test_pressure_endpoint_folds_in_source_errors`.
- Finding (nit): Hub tests stamped readings at module import, so they would go flaky in a session longer than 30 minutes.
  - Fix: readings are now built per test, and a future-stamped case was added.
  - Evidence: 30 passed.
- Finding (nit): the CI paths filter missed the registry, the Hub settings and the Hub main wiring.
  - Fix: added them.
  - Evidence: `.github/workflows/rest-drive.yml`.
- Finding (nit, design note): `due` ignores idleness. While Juniper is chatting, or during a long LLM outage (failed sleeps then 6 h refractory windows), it can read `due` for hours.
  - Fix: not changed. That only lengthens cooldowns, which is within the contract. It is recorded under Risks.

## Restart required

Deploy order: orion-dream first (so the key exists), then orion-hub. Either order is safe: a Hub with no key reads unknown and behaves as today. From the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-dream up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

Then verify within 10 minutes: the key `orion:drive:rest:latest` exists on the bus Redis (`redis-cli -u "$ORION_BUS_URL" GET orion:drive:rest:latest`), the dream logs `rest_drive_reading state=...`, and the dream service's `GET /dreams/cycle/pressure` shows `rest_drive`. Hub logs `rest_drive_view reader=outreach verdict=...` on its first outreach tick.

## Risks / concerns

- Severity: low. Concern: curiosity easing has ~zero live effect at today's settings (0 of 98 runs held in replay). Mitigation: reported as measured; the gain is one env value per reader and the spec's missing question 12 recommends re-fitting after two weeks of data.
- Severity: low. Concern: `due` is usually brief (the dream sleeps soon after crossing whenever Juniper has been quiet), but it ignores idleness: while she is chatting, or during a long LLM-gateway outage (failed sleeps alternating with 6 h refractory windows), it can read `due` for hours and keep outreach and curiosity at double cooldown that whole time. Mitigation: that only lengthens cooldowns, never loosens a gate; each reader has its own off switch.
- Severity: low. Concern: the dream now connects its bus every 600 s check (it used to connect only during a sleep) to write the Redis key. Mitigation: one connection, reused; a publish is bounded at 5 s and never changes the sleep decision.
- Severity: low. Concern: spec drift. Rev 4 says the rest drive has "trace consumers only" in v1 and that curiosity/outreach read arousal. Juniper asked for tiredness to drive them directly; arousal is unbuilt. The spec's bus channel `orion:drive:reading` is not built (no subscriber; see above). The spec's repair 1 (dream idle counts only Juniper's turns) is not in this PR; the replay uses the live all-chat idle gate.
- Severity: low. Concern: a stale Hub reading during a dream outage. Mitigation: 1800 s max age plus the key's own TTL; tested to decay to unknown.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
