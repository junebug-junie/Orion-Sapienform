## Summary

Orion now tells Juniper when one of their faculties breaks, comes back, or stops being reliable. A faculty is one thing Orion does and gets graded on, like the walkway-grade journal or the office temperature forecast. This is effect E3 of the self-monitor organ (design: `docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md`, "Decisions" 2).

- A new Hub loop reads the self-monitor's hourly picture from Redis (`orion:self_monitor:latest`). Patch 2 publishes it. This loop only reads it.
- Only three changes send a message: a faculty becomes broken; a broken or unreliable faculty becomes reliable again; a reliable faculty becomes unreliable. Calibration changes and "too few tries to tell" never send anything.
- Each change is told once. The first check after deploy tells each faculty that is already broken once, because Juniper has never been told.
- The message goes through the same door as reading findings: Orion writes it in their own words (PASS allowed), then the existing gates and grounding guard run. Quiet hours and arousal `strained` still hold it.
- The prompt carries only the faculty record's own numbers. A message that names a faculty the finding is not about is refused.

## Outcome moved

Before: the self-monitor could see that walkway grading was broken (22 of 28 tries never ran at 03:51 UTC), but nobody told Juniper. After: every break, recovery or reliable-to-unreliable change produces exactly one message, and the first fresh state after deploy tells each faculty that is already broken once.

## Current architecture

- Patch 1 (PR #3041, merged to main) added the faculty record, the reducer, the read-only report and the reader contract `orion/self_monitor/view.py` (absent/stale state changes nothing).
- Patch 2 (parallel) makes orion-durable-runs publish the state hourly. Not merged yet.
- Hub already had a finding door: `services/orion-hub/scripts/reading_finding_outreach.py` (compose through `EndogenousOutreach.compose`, deliver with `offer_message`, dedupe by `finding_ids` in `endogenous_outreach_decisions`). It places no body-economy bid today.

## Architecture touched

- Hub: new loop `scripts/self_finding_outreach.py`, registered in `scripts/main.py` (start and stop only).
- Grounding vocabulary: `scripts/outreach_vocabulary.py` gets the faculty-name check. Only this door calls it; other doors are unchanged.
- Daily cap accounting: `self_finding_outreach` sends skip the cap, so `count_sent_on` (Hub) and `CAP_EXEMPT_SOURCES` (durable-runs physiology) exclude the source, the same way they exclude reading findings.
- Persisted cursor: Redis key `orion:hub:self_finding_outreach:state` (no TTL). The decision log stays the authority on what was told.

## Files changed

- `services/orion-hub/scripts/self_finding_outreach.py`: the loop, the transition rule, the prompt.
- `services/orion-hub/scripts/outreach_vocabulary.py`: `is_self_monitor_faculty`, `find_ungrounded_faculty_mentions`.
- `services/orion-hub/scripts/main.py`: start/stop registration.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: `count_sent_on` excludes the new source.
- `services/orion-durable-runs/app/physiology_store.py`: `CAP_EXEMPT_SOURCES` includes the new source.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`, `services/orion-hub/README.md`: flag and docs.
- `services/orion-hub/tests/test_self_finding_outreach.py`: gate tests.
- `services/orion-hub/evals/run_self_finding_replay.py`, `services/orion-hub/evals/test_self_finding_eval.py`: would-tell replay (live) and a planted 48-hour eval.
- `.github/workflows/orion-reading-tests.yml`: runs the new test and eval; triggers on `orion/self_monitor/**`.

## Schema / bus / API changes

- Added: none on the bus. The decision row gains `finding_ids`, `finding_kinds`, `faculties`, `evidence_refs`, `bundle_key` under `result_json` with `source='self_finding_outreach'` (same shape as reading findings).
- Removed: none.
- Renamed: none.
- Behavior changed: Hub reads one more Redis key and writes one cursor key.
- Compatibility notes: with patch 2 not deployed the key is absent, so the loop does nothing.

## Env/config changes

- Added keys: `HUB_SELF_FINDING_OUTREACH_ENABLED=true`, `HUB_SELF_FINDING_OUTREACH_TICK_SEC=300`, `HUB_SELF_FINDING_OUTREACH_MAX_ATTEMPTS=3`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`). The keys reach the container through `env_file: .env`, like the reading keys, so `docker-compose.yml` is unchanged.
- local `.env` synced: the primary checkout's `.env_example` does not have these keys until merge, so `sync_local_env_from_example.py` there reports no change. The three keys were appended to `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env` by hand with the example values (verified with grep, lines 990-992). The settings defaults are the same values.
- skipped keys requiring operator action: none.

## Design decisions

- **Faculty-name check.** A message may name a faculty only if the finding is about it. A shortened id of that faculty (`turn.self` for `turn.self.steps`), file names (`vision.py`) and words with capitals (`act.Then`) are not flagged. Catalog names are read live from `orion/self_monitor/catalog.py`, so a catalog rename (e.g. #3051) needs no change here.
- **Finding id is `self_finding:<faculty>:<to_status>@<status_since>`.** The brief suggested putting `from_status` in the id too. It is on the finding but not in the id: if the Redis cursor is lost, an episode first told as `reliable -> broken` would come back as `unseen -> broken` with a different id and be told twice. Without it, the decision log catches the repeat (test `test_same_episode_is_told_once_even_after_the_state_key_is_lost`).
- **`unknown` and `miscalibrated` never overwrite the last status that counts.** So `broken -> unknown -> reliable` is a recovery, `reliable -> miscalibrated -> unreliable` is told, and `broken -> unknown -> broken` is no news.
- **First evaluation tells current broken faculties once** (recommended in the brief). Unreliable faculties at first sight are not told: the bar is "becoming" broken or a change away from reliable.
- **A waiting finding is never dropped for age.** Quiet hours or days of strain only delay it. A newer finding for the same faculty replaces it (decision row `superseded`), so Orion never reports a break that is already over. Its numbers refresh from each new state. It leaves the queue untold only after 3 composition failures (row `gave_up`).
- **No decision log, no message.** With the decision log switched off or unreadable, nothing is told. The log is searched by finding id, with no time limit.
- **No body-economy bid.** The reading door places none today, so this door matches it.

## Tests run

```text
PYTHONPATH=<worktree> .venv/bin/python -m pytest -q services/orion-hub/tests/test_self_finding_outreach.py
25 passed

PYTHONPATH=<worktree> .venv/bin/python -m pytest -q test_reading_finding_outreach.py test_endogenous_outreach.py test_outreach_vocabulary.py test_strain_report_outreach.py
319 passed

orion-reading-tests main step, mirrored in a fresh venv from services/orion-hub/tests/requirements-reading.txt (64 targets, after rebase onto main)
1990 passed, 113 skipped

orion/self_monitor/tests: 25 passed

DECISIONS_SQL run read-only through asyncpg against the live endogenous_outreach_decisions (jsonb): valid, 0 rows

services/orion-durable-runs: tests/test_physiology_report.py test_physiology_node.py test_physiology_bid.py
56 passed, 6 skipped

Mutation check: making unknown/miscalibrated count -> 2 tests fail; ignoring the decision log -> 1 test fails.
```

## Evals run

```text
pytest -q services/orion-hub/evals/test_self_finding_eval.py
2 passed  (planted 48 hourly states through the real 3-evaluation hysteresis: exactly 3 messages --
           became_broken at hour 0, recovered at hour 32, became_unreliable at hour 42; a 2-hour wobble
           and an hourly miscalibrated/reliable flip send nothing)
```

Would-tell replay against the live database (read-only):

```text
$ PYTHONPATH=services/orion-hub:. POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
    .venv/bin/python -P services/orion-hub/evals/run_self_finding_replay.py
# run 1, 2026-11-03 03:51 UTC (before rebase)
self-monitor state 2026-11-03T03:51:22.529252+00:00  statuses {'broken': 1, 'miscalibrated': 8, 'reliable': 13, 'unknown': 23, 'unreliable': 1}
would tell 1 finding(s):
  self_finding:act.actions.journal.make.walkway_grade:broken@2026-11-03T03:51:22.529252+00:00  kind=became_broken  from=unseen
    In the last 7 days, 22 of its 28 tries never ran.
    Of 6 tries that ran and were checked, about 100% worked (95% interval 57% to 100%).
    Recent tries count more than older ones in the percentages; the counts are plain counts.

# run 2, after rebase onto main
self-monitor state 2026-11-03T08:05:14.566581+00:00  statuses {'miscalibrated': 9, 'reliable': 12, 'unknown': 24, 'unreliable': 1}
would tell 0 finding(s):
```

What this means: at 03:51 the first state after deploy would have sent one message, about walkway grading. By the second run, older failures had aged out of the 7-day window: walkway grading now has too few tries to judge (`unknown`, 21 bets), so nothing would be told. `act.prime.predict` is unreliable, but a faculty that is already unreliable at first sight is not told, by the bar. If walkway grading crosses the broken line again after patch 2 deploys, it is told once as `unseen -> broken`.

## Docker/build/smoke checks

```text
Not run. No Docker build in this PR: the change is Python inside the existing Hub image (the Dockerfile copies orion/ and services/orion-hub/ already).
```

## Review findings fixed

Review by an orion-repo-agent subagent on the diff.

- Finding (must-fix): the faculty-name check flagged a shortened id of the grounded faculty (`turn.self`), file names (`vision.py`, `body.json`) and `act.Then`. Three refusals use up the attempts, so a real finding could be dropped untold.
  - Fix: a dot-boundary prefix of a grounded faculty counts as grounded. Tokens with capitals and file suffixes are not faculty ids.
  - Evidence: `test_shortened_ids_file_names_and_capitals_are_not_flagged`.
- Finding: with the decision log switched off, "nothing found" read as "nothing told".
  - Fix: `decision_log_enabled()` false reads as unknown, and nothing is told.
  - Evidence: `test_decision_log_switched_off_tells_nothing`.
- Finding: a 60-day time limit on the decision-log query limited dedupe.
  - Fix: look up by finding id (`finding_ids ?| ids`) with no time limit. Ran read-only on the live table.
  - Evidence: the test fake now filters by source and id.
- Finding: the 7-day age cap dropped findings silently, and a held break and its later recovery were sent together.
  - Fix: no age cap. A newer finding for the same faculty supersedes the older one (decision row). Waiting numbers refresh. `gave_up` drops write a row.
  - Evidence: `test_a_newer_finding_replaces_an_untold_older_one`, `test_a_waiting_finding_gets_fresh_numbers`.
- Finding: a change without `status_since` was marked as counted and lost.
  - Fix: it is not marked as counted, so the next state evaluates it again.
  - Evidence: `test_a_change_without_status_since_is_evaluated_again_later`.
- Finding: an older snapshot could move the cursor backwards.
  - Fix: a `computed_at` that is not newer is skipped.
  - Evidence: `test_an_older_snapshot_is_ignored`.
- Finding: the docstring overclaimed what the name check catches.
  - Fix: reworded (raw dotted ids only).
- Finding: dedupe depends on patch 2 keeping `status_since` stable and keeping prior records across restarts.
  - Fix: stated as a producer contract in the module docstring. It is listed under risks for patch 2.
- Not fixed (same as the reading door): if a gate closes between compose and delivery, the next tick composes again. Left as-is to match the reading door.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run it from the clean primary `main` checkout after this PR merges. The loop acts only after patch 2 (durable-runs publishing `orion:self_monitor:latest`) is deployed; until then it logs `self_finding_view reason=absent` once and does nothing.

## Risks / concerns

- Severity: medium. Concern: dedupe and hysteresis depend on patch 2. `status_since` must stay the same while a status is held, and the host must keep prior records across restarts. Otherwise `advance(None, ...)` skips the 3-check hold. Mitigation: the producer contract is written in the module docstring. Patch 2 should pin it with a test.
- Severity: low. Concern: the decision-log lookup filters on `result_json->>'source'` with no index (582k rows), and it runs only while a finding waits. Mitigation: rare; add an expression index if it shows in `pg_stat_statements`.

- Severity: medium. Concern: UNVERIFIED that patch 2 publishes on the same Redis the Hub bus connects to (`ORION_BUS_URL`). Mitigation: the arousal reader already reads durable-runs' regulation state through the same `bus.redis`; check `self_finding_view reason=fresh` in Hub logs after patch 2 deploys.
- Severity: low. Concern: UNVERIFIED live send. No message has gone to Juniper yet. Mitigation: after deploy, look for `self_finding_outreach_result reason=sent` and a decision row with `source='self_finding_outreach'`.
- Severity: low. Concern: the first-evaluation message says "held as broken since <first check>", which is when the record began, not when walkway grading broke. The prompt says so in plain words.
- Severity: low. Concern: a faculty that goes `broken -> miscalibrated` (runs again, confidence off) is not told as a recovery until it reaches `reliable`. That is the bar as written.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3061

🤖 Generated with [Claude Code](https://claude.com/claude-code)
