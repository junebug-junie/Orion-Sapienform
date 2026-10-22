## Summary

- Orion's "Is this <name>?" questions (yes/no on an unsure person guess) now have their own daily limit of 6.
- Before this patch they shared a limit of 2 a day with the "Who is this?" questions. "Who is this?" opens first and used both slots every day, so Orion never asked "Is this <name>?" (0 since #2807 deployed on 10-20).
- "Who is this?" keeps its own limit of 2, as before #2807.
- The replay eval now copies Orion's person guesses, so it shows the "Is this <name>?" questions too.

## Outcome moved

Juniper's yes/no answers are the fastest way to grow each person's picture album, which is how Orion learns to recognize people. Live replay: 0 questions a day before, 6 would open today after (Addy, Emerald, Maeve, all with pictures).

## Current architecture

`_open_asks` (services/orion-sql-writer/app/vision_individuals.py) runs three openers in order: face asks (own cap 6), the slow 10-sightings rule (`ORION_ASK_DAILY_CAP`=2), then "Is this <name>?". The last two were both counted by `ASKED_TODAY_SQL`.

Live evidence (orion_ask, charged to the shared cap per local day): 10-20: 2, 10-21: 2, 10-22: 2. `vision_person_confirm` rows ever: 0. Eligible candidates live: 27.

## Architecture touched

orion-sql-writer only: the ask budget SQL, `IndividualsConfig`, settings, loop wiring, README, tests, eval.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: `ASKED_TODAY_SQL` counts only `vision_individual`; new `PERSON_CONFIRM_ASKED_TODAY_SQL`; `person_confirm_ask_daily_cap`.
- `services/orion-sql-writer/app/settings.py`, `app/vision_individuals_loop.py`: new setting and its wiring.
- `services/orion-sql-writer/.env_example`, `README.md`: document the new cap.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: regression test (slow rule used the day, "Is this X?" still opens) and an own-cap test.
- `services/orion-sql-writer/tests/test_vision_individuals.py`: budget SQL contract.
- `services/orion-sql-writer/evals/run_vision_confirm_asks_replay_eval.py`: copies `vision_person_attribution`, reports the new cap.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: "Is this <name>?" asks are budgeted separately.
- Compatibility notes: no schema or bus change.

## Env/config changes

- Added keys: `VISION_PERSON_CONFIRM_ASK_DAILY_CAP=6` (orion-sql-writer)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced: yes. The worktree has no `.env`, so the missing key was appended to the primary checkout's `services/orion-sql-writer/.env`. `check_env_template_parity.py`: PASS.
- skipped keys requiring operator action: none

## Tests run

```text
ORION_TEST_PG_URI=<throwaway postgres:15 with walkway migrations> pytest
  tests/test_vision_person_model.py   38 passed
  tests/test_vision_individuals.py    27 passed
  tests/test_vision_confirm_asks.py   44 passed
  tests/test_vision_split.py          45 passed
Regression test against the old vision_individuals.py: 1 failed (as expected)
```

## Evals run

```text
evals/run_vision_confirm_asks_replay_eval.py --live <live, READ ONLY> --scratch <throwaway>
OK: 6 ask(s) would open   {'vision_person_confirm': 6}, every one with 1-3 pictures
```

## Docker/build/smoke checks

```text
None. The change is Python logic and one env key, with no new dependencies. The live path is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: the eval emptied the person-guess table on the scratch database without checking that it exists, so a scratch database without that migration would crash.
  - Fix: empty and copy the table only when it exists on scratch.
  - Evidence: eval rerun on live data: OK, 6 asks would open.
- Finding: the `_open_asks` docstring did not mention the third question type.
  - Fix: updated the docstring.
  - Evidence: commit 2.
- Review found no correctness bugs.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

After the deploy, check the live path: `SELECT count(*) FROM orion_ask WHERE source_kind='vision_person_confirm'` should be above 0 within a tick.

## Risks / concerns

- Severity: low
- Concern: up to 6 more cards a day on the Hub (14 walkway questions a day at most).
- Mitigation: set `VISION_PERSON_CONFIRM_ASK_DAILY_CAP` lower; 0 turns these questions off.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2841

🤖 Generated with [Claude Code](https://claude.com/claude-code)
