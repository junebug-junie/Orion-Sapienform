## Summary

- The slow "who is this?" rule (10 sightings over 5 days) no longer posts a card with no picture. The person is skipped and stays a candidate, so a card opens once a thumbnail exists.
- A skipped candidate no longer blocks everyone behind it. Both "who is this?" openers (the slow rule and the face rule) fetched only as many candidates as the day had slots. When the top one was skipped, that slot stayed empty on every tick. Both openers now look at every candidate and stop when the day's slots are used.
- Spec PR #2767, handoff item C.2.

## Outcome moved

Juniper no longer gets "who is this?" cards she cannot answer. Live 2026-10-20: 1 of 1 open `vision_individual` asks had `image_refs=[]` (`ask-9645cdc1bdd64660b80a07380b7c71ef`).

## Current architecture

`services/orion-sql-writer/app/vision_individuals.py` `_open_asks` runs every individuals tick. The face opener already skipped picture-less faces. The slow-rule opener posted the ask whatever `_images()` returned. Both opener queries used `LIMIT :b` (the remaining daily budget).

## Architecture touched

orion-sql-writer only. No schema, bus, env or migration change.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: skips the slow-rule ask when there is no picture; drops `LIMIT :b` and stops the loop at the budget, in both openers.
- `services/orion-sql-writer/tests/test_vision_confirm_asks.py`: two real-Postgres tests. A picture-less regular gets no card, does not block the next regular, and gets a card once thumbnails exist. A picture-less face does not block the next face.
- `services/orion-sql-writer/README.md`: the no-picture rule.
- `services/orion-sql-writer/evals/run_vision_confirm_asks_replay_eval.py`: the picture gate covers the slow rule.
- `services/orion-sql-writer/.env_example`: comment only.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: no `vision_individual` ask is created with empty `image_refs`.
- Compatibility notes: existing open asks are untouched. See C.3 below.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: comment only
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
ORION_TEST_PG_URI=<throwaway pg16> pytest tests/test_vision_confirm_asks.py tests/test_vision_split.py tests/test_vision_body_match.py -> 110 passed
Mutation: the old vision_individuals.py -> both new tests fail.
pytest tests -k "vision or ask" on the same database: 3 failed + 22 errors on main AND on this branch
(motion / re-anchor / story-strip / osnet migrations were not loaded); this branch adds 2 passing tests.
```

## Evals run

```text
evals/run_vision_confirm_asks_replay_eval.py --live <live> --scratch <throwaway pg16> -> exit 0 (pass), with the stricter all-rules picture gate
```

## Docker/build/smoke checks

```text
Not built. No dependency or compose change.
```

## Review findings fixed

Review (subagent): budget/break logic correct in both paths. Dropping LIMIT is cheap (live: 3 slow-rule and 0 face-rule candidates before the not-asked filter; one indexed lookup each). No other opener can post a picture-less open card. No material findings.

- Finding: the replay eval still exempted the slow rule from "every ask has a picture".
  - Fix: the exemption is removed, so the eval now enforces no picture, no ask for every rule.
  - Evidence: `evals/run_vision_confirm_asks_replay_eval.py` against live (scratch DB) -> exit 0.
- Finding: the `.env_example` comment said a too-short thumb retention means "asks carry no picture".
  - Fix: it now says no "who is this?" ask opens at all.
  - Evidence: diff; env parity PASS.

## Restart required

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d
```

## Risks / concerns

- Severity: low
  - Concern: the existing picture-less open ask (`ask-9645...`) stays open until it expires, and its 30-day cooldown then blocks a re-ask with pictures until about 2026-11-25. Do not dismiss it: a dismissed ask blocks a re-ask forever.
  - Mitigation: handoff C.3. A one-row status update. It is a production write and needs Juniper's OK.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2782

🤖 Generated with [Claude Code](https://claude.com/claude-code)
