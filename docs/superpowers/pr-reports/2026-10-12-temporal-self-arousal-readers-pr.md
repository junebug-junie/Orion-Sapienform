# Temporal Self rev 4, order 6: the first arousal readers (R3 readers 1, 2, 3 and 7)

## Summary

- Orion's arousal reading (engaged / idle / strained / unknown, live since #2594) now changes four real decisions. Before this PR nothing read it.
- **Dream (reader 1):** a sleep the dream was about to take is held while Juniper is talking (`engaged`) or the body is under load (`strained`). Arousal only vetoes: it never lets the dream sleep when its own Juniper-filtered idle query says no. A level that has lasted 6 hours stops holding sleep (so a jammed GPU queue cannot cost a day of sleep).
- **Curiosity investigation line (reader 2):** `strained` holds a run (`arousal_strained`); `engaged` doubles the cooldown (`arousal_engaged_cooldown`).
- **Curiosity self-inquiry (reader 3):** runs only when arousal reads `idle`.
- **Background outreach (reader 7):** refused while `strained`.
- Every reader treats a missing, stale (> 360 s), malformed or `unknown` state, a Redis error, or its own flag off as "arousal does not exist" and makes exactly today's decision. Each has its own flag, shipped ON.

## Outcome moved

Behaviour: the slow "what is Orion's situation" reading now sets the gain of four dials, only in the conservative direction. The replay says what that would have changed on saved history (below): on a fault-jammed day it would have held every optional run; on a normal day it changes 0-2 decisions.

## Current architecture

- `orion-durable-runs` regulate node writes `RegulationStateV1` to Redis `orion:regulation:latest` (TTL 360 s) every 120 s and on every Juniper turn; each level change is an `arousal_transition` row in `temporal_self_event`.
- The registry entry said "no reader yet". The rest drive (#2584) was the only regulation input any dial read (Hub curiosity + outreach cooldown stretch while tired).

## Architecture touched

- `orion/regulation/arousal_view.py` (new, pure): one freshness rule for every reader and the shared refusal-reason spellings.
- `services/orion-dream`: `CycleDeps.read_arousal`; the veto sits after every existing gate and is read only when the dream would otherwise sleep.
- `services/orion-hub`: `scripts/arousal_reader.py` (new, same shape as `rest_drive_reader.py`: one Redis GET per tick, 2 s timeout, re-judges staleness at the caller's `now`); new fields on `SchedulingGateInputs` and `OutreachGateInputs`, checked last in each gate function.
- **Rest drive stays a separate input.** Tired (sleep pressure due) and strained (heat / GPU backlog) are different facts; both can hold, and each names itself in the refusal reason. Folding rest into arousal would make arousal read a drive that the spec says reads arousal (a loop). The live rest readers are untouched.
- **Spec order and this PR:** the R3 table order is dream, curiosity pacing, self-inquiry, visual reverie baseline, reverie chain, mutation pressure, outreach. Readers 1-3 and 7 are here. Reader 5 (reverie chain) lives in orion-thought, which a parallel session owns; readers 4 (visual reverie baseline) and 6 (mutation pressure) are the next PRs. One PR across two services because both use the same new pure module and the dream reader is one function; each reader is still its own commit and its own flag.

## Files changed

- `orion/regulation/arousal_view.py`: pure reader view + `engaged_cooldown_sec`.
- `services/orion-dream/app/cycle.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: reader 1.
- `services/orion-hub/scripts/arousal_reader.py`: Hub reader.
- `services/orion-hub/scripts/curiosity_investigation.py`: readers 2 and 3 (gate fields, force-overridable reasons, per-line flags).
- `services/orion-hub/scripts/endogenous_outreach.py`: reader 7, status shows `arousal`.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`, `README.md`: wiring and flags.
- `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`: four declared consumers (routing change, re-locked).
- `scripts/sync_local_env_from_example.py`: `DREAM_IDLE_FROM_AROUSAL_ENABLED`, `DREAM_AROUSAL_MAX_HOLD_HOURS` exact keys (no `DREAM_` prefix is synced).
- `scripts/analysis/measure_arousal_reader_replay.py` (new) + test; `measure_arousal_reducer_replay.py` exposes `replay_readings` and lists every step's reasons for a strained span.
- Tests: `tests/test_regulation_arousal_view.py`, `services/orion-dream/tests/test_dream_reads_arousal.py`, `services/orion-hub/tests/test_regulation_arousal_readers.py`, `services/orion-hub/tests/test_regulation_must_not_readers.py`.

## Schema / bus / API changes

- Added: none on the bus. New optional dataclass fields on Hub's internal gate inputs. Outreach status JSON gains `arousal`.
- Removed / Renamed: none.
- Behavior changed: the four dials above, conservative direction only.
- Compatibility notes: `RegulationStateV1` is read, not changed. No migration.

## Env/config changes

- Added keys: orion-dream `DREAM_IDLE_FROM_AROUSAL_ENABLED=true`, `ORION_REGULATION_READER_MAX_AGE_SEC=360`, `DREAM_AROUSAL_MAX_HOLD_HOURS=6`; orion-hub `ORION_REGULATION_READER_MAX_AGE_SEC=360`, `HUB_CURIOSITY_AROUSAL_ENABLED=true`, `HUB_CURIOSITY_AROUSAL_ENGAGED_COOLDOWN_MULTIPLIER=2.0`, `HUB_CURIOSITY_SELF_INQUIRY_AROUSAL_ENABLED=true`, `HUB_ENDOGENOUS_OUTREACH_AROUSAL_ENABLED=true`.
- Removed / renamed: none.
- `.env_example` updated: yes (both services), plus settings.py defaults (and dream compose).
- local `.env` synced with `python scripts/sync_local_env_from_example.py orion-hub orion-dream`: yes, all 8 keys landed in the primary checkout's `.env`.
- skipped keys requiring operator action: none.
- `check_settings_defaults.py --example-drift orion-hub`: OK. orion-dream is not on that checker's allowlist (pre-existing); `check_service_env_compose_parity.py orion-dream`: N/A (env_file).

## Tests run

```text
tests/test_regulation_arousal_view.py                                  10 passed
services/orion-dream/tests (whole suite)                               216 passed, 2 skipped
services/orion-hub/tests/test_regulation_arousal_readers.py + test_regulation_must_not_readers.py
  + test_rest_drive_readers.py + test_endogenous_outreach.py
  + test_curiosity_investigation.py + test_curiosity_self_inquiry.py   534 passed
scripts/analysis/tests/test_measure_arousal_{reader,reducer}_replay.py 9 passed
services/orion-hub/tests (whole suite)                                 36 failed, 3487 passed, 86 skipped
origin/main, same suite, clean worktree                                36 failed, 3411 passed, 86 skipped
  -- identical failure sets (live Postgres / UI-fixture tests, none touching this PR)
check_metric_lineage --gate PASS; check_definition_drift --gate PASS (re-locked);
check_inner_state_registry OK; check_env_template_parity PASS; check_env_key_single_source OK;
tests/scripts/test_sync_local_env_from_example.py 19 passed
```

Per reader: each level's effect, flag off = today, absent / malformed / stale / future-stamped / producer-unknown / Redis error / Redis hang = today, forced runs override. Must-not, at branch level: every protected curiosity reason (disabled, daily cap, hour window, cooldown, rest-drive cooldown) and outreach reason (disabled, turn in flight, recent chat, quiet hours, daily cap, cooldown, with and without Door-A) gives the same decision under all four arousal values and none; an exhaustive grid over both gates shows arousal never turns a hold into a pass; the dream's six-hour minimum and forced sleeps never read arousal; idle arousal never authorizes a sleep the idle query refused; urgent curiosity runs do not read arousal.

## Evals run

`scripts/analysis/measure_arousal_reader_replay.py`, window 2026-10-09 07:00 to 2026-10-12 01:00 UTC (as far back as `gpu_pool_state_history` goes). Two lanes: the shipped reducer replayed over saved inputs, and the live `arousal_transition` rows (from 10-10 23:00). Decisions: real non-forced sleeps, non-urgent investigation and self-inquiry run starts, sent background outreach (forced debug sends included, since the shipped gate holds them too; there were none in the window). Evidence: `docs/superpowers/evidence/2026-10-12-temporal-self-arousal-readers/reader_replay.json`.

Held decisions per day (held / total), replay lane:

| Day (UTC) | Dream | Curiosity | Self-inquiry | Outreach | Arousal minutes (engaged / idle / strained) |
|---|---|---|---|---|---|
| 10-09 (from 07:00) | 0 / 0 | 5 / 5 | 6 / 6 | 4 / 4 | 0 / 0 / 1020 |
| 10-10 | 0 / 2 | 0 / 7 | 1 / 7 | 0 / 4 | 59 / 1071 / 310 |
| 10-11 | 0 / 2 | 0 / 7 | 0 / 7 | 0 / 4 | 182 / 1239 / 20 |

Live lane (10-11, the one full day with real transitions): 0 of 20 decisions would have changed. The live and replayed levels agree on that day (strained 06:16-06:36 both).

What the numbers say, plainly:

- **10-09 is the known fault-jammed GPU queue** (the replay's strained span runs 07:00 10-09 to 02:52 10-10, 593 of 598 steps citing `gpu_queue_sustained`). Every optional run that day would have been held. That is the caveat: S2 cannot tell a jam from real load, so a jam holds all optional work. During that jam those curiosity briefs actually sat 8-12 h in the queue, so holding them would have cost nothing real; but if a jam ever lasts days, curiosity, self-inquiry and outreach stop for days.
- **The dream would have slept through the jam anyway.** Its one sleep inside that strained span (10-10 01:01) came 18 h into the level, past the 6 h hold bound, so it is not held. Without the bound it would have been.
- **On normal days the readers change 0-1 decisions.** All held decisions were `arousal_strained`; no decision fell inside an `engaged` window. Curiosity and self-inquiry spend their daily cap from local midnight (06:00 UTC) through the morning, while Juniper talks in her evening (22:00-01:00 UTC), so the engaged dials (curiosity's doubled cooldown, self-inquiry's idle-only rule, the dream's engaged hold) would not have changed anything on this history. They are correct but currently near-inert; they matter if pacing or Juniper's hours change.
- The live lane holds a level until the next transition row; if durable-runs stopped writing, real readers would read unknown after 360 s while the lane keeps the last level, so it can overcount across an outage (none in this window: 0 holds).
- First-order only: a held run would have moved later runs, which is not simulated. Curiosity rows do not record forced runs (a forced run is never held), so curiosity counts are an upper bound.

## Docker/build/smoke checks

```text
UNVERIFIED -- not deployed from this branch (production deploys from the primary checkout on main).
```

## Review findings fixed

Review ran in a subagent against `origin/main...HEAD`. No must-fix bugs; it confirmed the gate order, Door-A exemption, urgent-run path, dream timing, stale-view decay, Redis timeouts, per-line flags and env parity.

- Finding: a strained reading also blocked the dream's overdue backstop with no limit, so a long GPU jam meant no sleep at all.
  - Fix: `DREAM_AROUSAL_MAX_HOLD_HOURS` (6): a level that has held that long (from its `since`) no longer holds a sleep.
  - Evidence: `test_a_level_held_past_the_max_hold_stops_holding_sleep`; replay: the 10-10 01:01 sleep, 18 h into the jam, is no longer held.
- Finding: manual `/dreams/cycle/run` returned `not_due` when arousal held the sleep.
  - Fix: `CycleDeps.hold_reason`; the endpoint returns `{"status": "held", "reason": "arousal_<level>"}`.
  - Evidence: `test_manual_run_says_held_not_not_due`.
- Finding: the eval left out forced outreach, which the shipped gate also holds.
  - Fix: export includes forced sends (none in window; totals unchanged).
  - Evidence: `docs/superpowers/evidence/2026-10-12-temporal-self-arousal-readers/export.sql`.
- Finding: the eval's live lane can overcount across a durable-runs outage.
  - Fix: stated in the script docstring and here (0 holds in the live lane, so no effect on the numbers).
  - Evidence: script docstring.
- Finding (nit): Hub settings / `.env_example` comments said forced runs never read arousal; true only for curiosity.
  - Fix: reworded (forced curiosity runs override it; a forced debug outreach is held like the cooldown).
- Not changed (nits, low cost): the urgent-run must-not test is a source-text check; a cancelled dream arousal read does not drop the cached bus (the half-built bus is never stored); the eval's curiosity cooldown base is the live 1800 s floor and must be passed by hand if a window is configured.

- Finding (found by the full Hub suite, not the reviewer): the first version of the Hub reader test file sorted before `test_curiosity_*` and imported `test_curiosity_investigation` early. Hub's conftest purges `scripts.*` per test, so the curiosity GPU-lease tests then patched a different module object than their loop used: 6 extra failures in the full suite only.
  - Fix: renamed to `test_regulation_arousal_readers.py` (collected after the curiosity tests).
  - Evidence: full Hub suite failure set now identical to `origin/main` (36 = 36, no branch-only failures).

## Restart required

From the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-dream up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

Live checks after deploy:

- `docker logs orion-athena-hub 2>&1 | grep arousal_view` shows one line per reader (`reader=curiosity`, `reader=outreach`) with the current level.
- `curl -s localhost:8080/api/debug/endogenous-outreach/status` includes `"arousal": {"level": ...}`.
- The one live `arousal_strained` refusal the spec asks for: wait for the next strained span (daily around 06:15 UTC when curiosity's morning burst backs up the GPU queue, `curl -s localhost:8124/regulation/state`), then `docker logs orion-athena-hub 2>&1 | grep "reason=arousal_strained"`.
- Dream: `docker logs orion-athena-dream 2>&1 | grep "dream_cycle held reason=arousal_"` appears only if a due sleep meets engaged/strained.
- Kill-switch smoke: `redis-cli -u "$ORION_BUS_URL" DEL orion:regulation:latest` and the dials keep today's behaviour (key reappears next step).

## Risks / concerns

- Severity: medium. Concern: a fault-jammed GPU queue reads as strained, so a long jam holds all optional work (replay: one whole day). Mitigation: each reader has its own off switch; `reasons` + `gpu_queue_depth` in the state show which it was; fixing S2's jam/load ambiguity belongs to the producer, not the readers.
- Severity: low. Concern: the engaged dials changed nothing on 3 days of history (schedule mismatch, not a bug). Mitigation: reported plainly; re-measure after two weeks of live transitions.
- Severity: low. Concern: spec names a repo-root `tests/test_regulation_must_not_readers.py`; Hub's `scripts` package collides with the root `scripts` package, so the Hub must-not tests live in `services/orion-hub/tests/` and the dream's in `services/orion-dream/tests/`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2624

🤖 Generated with [Claude Code](https://claude.com/claude-code)
