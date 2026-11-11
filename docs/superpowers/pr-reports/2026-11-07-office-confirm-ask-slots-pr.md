# PR: office "Is this Juniper?" asks get their own daily slots

## Summary

- Orion never asked "Is this Juniper?" about an office visit. 74 office records carry the guess "Juniper (probably)", and none was asked about.
- Cause: every "Is this X?" ask shares one cap of 6 a day. Walkway guesses used all 6 on each of 11-04, 11-05 and 11-06, and office guesses are ranked last. So Juniper's office album never grew, and Orion can never move from "probably Juniper" to "Juniper (matched)".
- Fix: asks about a camera-prior guess (the office) carry `rule:camera_prior`, are left out of the shared count, and have their own cap, `VISION_PERSON_CONFIRM_PRIOR_ASK_DAILY_CAP=2`. There is still at most one open ask per name.
- Also fixed: if the first office record had no picture, its name was blocked for the rest of the cycle. The name is now taken only after the picture check passes.
- Found while checking: no CI workflow ran `tests/test_vision_person_model.py`, so its real-Postgres tests (from #3129 and this PR) had only ever run locally. It now runs in the sql-writer vision Postgres step, together with the `person_record_v1` migration it needs.

## Outcome moved

- Before: 0 office confirm asks ever. After deploy: up to 2 a day, without taking any walkway slot. UNVERIFIED until the next office visit after deploy.

## Files changed

- `services/orion-sql-writer/app/vision_individuals.py`: separate lane and cap for camera-prior confirm asks; `PRIOR_RULE_REF`, `PRIOR_CONFIRM_ASKED_TODAY_SQL`.
- `services/orion-sql-writer/app/settings.py`, `.env_example`, `app/vision_individuals_loop.py`: `VISION_PERSON_CONFIRM_PRIOR_ASK_DAILY_CAP` (default 2, ON).
- `services/orion-sql-writer/README.md`: documents the new slots.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: own-slots test (the walkway uses the shared cap; an earlier office ask must not use a shared slot) and a zero-cap test.
- `.github/workflows/orion-sql-writer-tests.yml`: runs `test_vision_person_model.py` and applies `person_record_v1` in the vision Postgres step.

## Env/config changes

- Added: `VISION_PERSON_CONFIRM_PRIOR_ASK_DAILY_CAP=2` (orion-sql-writer). Local `.env` updated by hand in the primary checkout.

## Tests run

```text
pytest services/orion-sql-writer/tests -q                                    1241 passed, 191 skipped
PG (CI's own migration procedure, throwaway postgres:16-alpine):
  test_vision_person_model.py + test_vision_confirm_asks.py + test_vision_office_sessions_pg.py   113 passed, 0 skipped
mutation: shared count without the camera_prior exclusion -> own-slots test fails
gates: check_personal_terms, check_definition_drift --gate, check_metric_lineage --gate,
       check_env_key_single_source, check_env_template_parity, git diff --check     ok
```

## Review findings fixed

The review ran in a subagent and found no correctness bugs.
- Finding (low): the new test did not prove that office asks stay out of the shared count. Fix: the test inserts an earlier office ask. Evidence: the mutation above.
- Finding (low): the README did not mention the new key. Fix: documented.
- Note: on deploy day only, office asks opened earlier that day carry no new ref, so they still count against the shared cap that day. This ends after one day.

## Restart required

```text
No manual step: the post-pull rebuild rebuilds orion-sql-writer.
```

## Risks / concerns

- Severity: low. Concern: up to 2 more asks a day go to Juniper. That is the intent: her answers are what teach Orion her look on the office camera.
