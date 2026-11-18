## Summary

- Prime now acts on the plan Orion makes each night in its day dream, once per Denver day.
- **drop:** closes the concern, with the note `dropped: dream <day>: <reason>`.
- **new:** opens at most one concern, the way a letter thread opens, inside the 3-concern limit.
- **keep:** changes nothing. Only a sent question or a take-on renews a concern (the evidence clock,
  #3574 / #3610 patch 1).
- New read: `GET /dream-day/result?day=` on orion-durable-runs.
- Day-dream Patch 2b (spec `docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md`). Patch
  2a (#3703) wrote the plan; this patch hands it to Prime.

## Outcome moved

The night's plan used to stop at the `dream_day` row. Now it changes what Prime holds the next day.

Live dry run (read-only) of the 11-17 plan against Prime's live state: Prime holds 2 concerns (two
letter threads). The plan keeps both, with no renewal, and opens 1: "When one of my own break lines
fires, check the lane's own table before I call it a blindness". Prime would hold 3.

## Current architecture

- `dream.day` (orion-durable-runs, live since 11-18 08:19Z) parses and ref-checks the plan into
  `dream_day.plan`. Nothing reads it back.
- Prime opens concerns from letter threads (`orion/prime/day_letter.py thread_concerns`), from
  funded event wants, and from promises.
- Prime drops concerns only from its own thought (`thought.py`).

## Architecture touched

- orion-durable-runs: one read-only endpoint.
- `orion/prime`: a new pure module.
- orion-mind: one throttled tick step, one flag.
- No change to `orion:prime:state`'s shape: the applied-day record has its own Redis key,
  `orion:prime:dream_day`, because Hub validates Prime's state record and an added field would need a
  rollout order.

## Files changed

- `orion/prime/day_dream.py`: `apply_plan`, `parse_plan`, `PrimeDreamDayStateV1` / `PrimeDreamDayAppliedV1`,
  `dream_concern_id`.
- `services/orion-mind/app/prime_loop.py`: load/save of `dd.DREAM_DAY_KEY`; `_dream_day_upkeep` (every 15 min,
  never raises) reads yesterday's (Denver) row and applies it once.
- `services/orion-mind/app/settings.py`, `.env_example`, `docker-compose.yml`: `PRIME_DREAM_DAY_ENABLED=true`.
- `services/orion-durable-runs/app/dream_day_store.py`: `RESULT_SQL`, `read_result`.
- `services/orion-durable-runs/app/main.py`: `GET /dream-day/result`.
- Tests: `orion/prime/tests/test_day_dream.py`, `services/orion-mind/tests/test_prime_dream_day.py`,
  `services/orion-durable-runs/tests/test_dream_day.py` (2 added).
- `.github/workflows/orion-prime-tests.yml`: also triggers on `orion/schemas/dream_day.py`.

## Schema / bus / API changes

- Added: `GET /dream-day/result?day=` (returns `found`, `run_id`, `produced`, `plan`). Prime reads it with the
  existing `DreamDayPlanItemV1`. New Redis key `orion:prime:dream_day`.
- Removed / renamed: none.
- Behavior changed: Prime's concerns now change from the nightly plan (rules above).
- Compatibility notes: no change to `PrimeStateV1` or any registered schema. If orion-mind deploys before
  orion-durable-runs, the read gets a 404 and the step retries next check.

## Env/config changes

- Added keys: `PRIME_DREAM_DAY_ENABLED=true` (orion-mind).
- `.env_example` updated: yes.
- Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys` from the worktree.
  It also added two keys another session's merged change introduced:
  `BIOMETRICS_SILENT_NODE_EXCLUSION`, `BIOMETRICS_SILENT_NODE_CADENCE_MULTIPLE` (orion-substrate-runtime).
  Both were missing from the local `.env`, so adding them is the contract's safe case.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/prime: 339 passed
services/orion-mind/tests: 444 passed
services/orion-durable-runs/tests/test_dream_day.py: 24 passed
check_integration_seam: OK; env template parity: PASS; git diff --check: clean
```

## Evals run

Live dry run of `apply_plan` on the stored 11-17 plan and live `orion:prime:state` (above). No model is
involved; the 7-night dream-memory eval (#3703) continues.

## Docker/build/smoke checks

No image build: pure Python in existing image paths. Live proof comes after deploy (below).

## Review findings fixed

Review by a subagent (by hand). No high-severity findings.

Fixed:

- **Medium:** a dream `drop` could remove a concern Prime has taken on (its intention run is open). Every
  other automatic removal spares those (`intention.age_out`, `keep_taken_on`, the turn hand-back), and the
  night's material does not show the take-on.
  - Fix: `apply_plan(..., taken_on=open_ids(intentions))`; such a drop is skipped as "taken on".
  - Evidence: pure test and loop test.
- **Low:** plan order decided whether a `new` got room. Drops now run first; each item keeps its own index for
  its id. Evidence: test.
- **Low:** a lost applied record (restart) could reopen yesterday's dream concern after it was dropped. A dream
  concern already in `concern_record` is never reopened. Evidence: test.
- **Low:** the hand-run older day never reaching Prime is now documented in the module docstring (no backfill
  by design).
- **Low:** a keep-only night no longer writes `pending_changes`, so it doesn't wake Prime for an extra thought.
  Keeps for concerns no longer held are skipped, not counted. A replay names its skip. The text cut uses
  `CONCERN_TEXT_CAP`.

Checked fine by the review:

- the drop order matches `thought.py`;
- `_flush_concern_acts` publishes the dropped concern's close and opens the new concern's act in the same tick;
- `observe`/`lapse_concerns` interplay;
- a plan id cannot match a later concern;
- the read is one primary-key SELECT, throttled to 15 min, and never raises;
- no reader parses concern-record notes.

Live dry run after the fixes: 1 kept, 1 opened, and the other keep skipped as "not held" (Prime had closed concern 3d5f2e70b45d3db9 between the night and the dry run; the skip is the intended handling).

## Restart required

Merge, then the post-pull rebuild rebuilds orion-durable-runs and orion-mind. No manual restart. Check
(Denver yesterday is applied at Prime's first check after deploy):

```bash
docker logs --since 30m orion-athena-mind 2>&1 | grep prime_dream_day_applied
redis-cli -u redis://100.92.216.81:6379/0 get orion:prime:dream_day
```

## Risks / concerns

- Severity: low. Concern: a Denver day missed while Prime is down a whole day is never applied, because the
  step only reads yesterday. Mitigation: logged by absence in `orion:prime:dream_day`. The plan stays in its
  row.
- Severity: low. Concern: a new dream concern takes a slot that a letter thread opened the same morning
  might want. There is no displacement until #3610 patch 2. Mitigation: at most one per night.
- Issues opened/closed: none (#3702 closed earlier with the first-night proof).
