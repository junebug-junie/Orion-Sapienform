## Summary

- When a stroller or dog is right next to someone, Orion checks the name it is about to guess against that person's history. If that person almost never has that clue nearby, Orion drops the "Is this X?" question, or lowers "probably X" to asking.
- The guess is chosen exactly as before; the clue can only drop it or weaken it. It never changes the name and never names anyone.
- Clue rates are learned per person from album passages and compared with the street; only passages after the detector started boxing that kind count.
- Ships on (`VISION_PERSON_CLUE_VETO_ENABLED=true`); off = today's guesses exactly.
- Clue design patch 3 (#3811, ticket #3768).

## Outcome moved

Fewer wrong "Is this X?" questions. Evidence: on clusters seen since stroller tagging began, Juniper answered "no" to 17 of 18 questions about people with a stroller beside them, vs 46 of 73 otherwise.

## Current architecture

`attribute()` (orion-sql-writer person model, once a minute) picks a guess per unnamed camera cluster from album similarity, camera priors and the age veto. Stroller/dog boxes existed (`vision_crop_observation.label`) but were unused for identity.

## Architecture touched

- `services/orion-sql-writer/app/vision_person_model.py`: `clue_counts`, `clue_llr`, `load_crop_clues`, `clue_inputs`; `attribute(clues=, clue_trace=)`; wired into `run_person_model` in a fail-open savepoint.
- Settings / loop / `.env_example` / README: `VISION_PERSON_CLUE_VETO_ENABLED`.

## Files changed

- `services/orion-sql-writer/app/vision_person_model.py`: the veto.
- `services/orion-sql-writer/app/settings.py`, `app/vision_individuals_loop.py`, `.env_example`, `README.md`: flag + docs.
- `services/orion-sql-writer/evals/run_clue_veto_replay_eval.py`: replay against her answered "Is this X?" questions.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: counts, llr, drop, weaken, never-promote, never-other-name, exemptions, settings.

## Schema / bus / API changes

- Added: none stored. Guess dicts carry `clue_veto` {name: llr} in memory; log line `person_clue_veto`; cycle stats `person_clue_vetoes`, `person_clue_veto_dropped`, `person_clue_veto_error`.
- Removed / Renamed: none.
- Behavior changed: some unsure guesses are not written (no question); some accepts become unsure.

## Env/config changes

- Added keys: `VISION_PERSON_CLUE_VETO_ENABLED=true` (orion-sql-writer).
- `.env_example` updated: yes.
- local `.env` synced: key added to `services/orion-sql-writer/.env` by hand (sync script runs from the main checkout, which does not have the template yet).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-sql-writer/tests/test_vision_person_model.py   55 passed, 27 skipped (pg)
pytest tests/test_no_personal_terms.py                                7 passed
python scripts/check_env_template_parity.py                           PASS
```

## Evals run

```text
PYTHONPATH=. python services/orion-sql-writer/evals/run_clue_veto_replay_eval.py   (live, read-only)
answered 216; dropped 2 (both "no"); "yes" lost 0; gate passed
street: stroller 26/301 passages, dog 44/376
Before the review fixes (per-candidate, summed per passage): 6 dropped, all "no".
Sensitivity (old rule): cutoff 1.5 -> 7, 1.0 -> 8, all "no". Cutoff kept at the pre-set 2.3.
```

## Docker/build/smoke checks

```text
Live timing of the new lookup: 10,385 crops in 0.51 s per cycle (cycle every 60 s). No image/dependency change.
```

## Review findings fixed

- Finding: demoting one of two accepts promoted the other to accept.
  - Fix: the veto now runs after the guess is chosen, on that guess only.
  - Evidence: `test_clue_veto_never_promotes_the_other_of_two_accepts`.
- Finding: a vetoed top guess could switch the question to another name.
  - Fix: same; a vetoed unsure guess is dropped, not replaced.
  - Evidence: `test_clue_veto_drops_the_question_and_never_changes_the_name`.
- Finding: summing per stroller passage let one mixed cluster veto someone who does push a stroller.
  - Fix: each clue counts once per cluster.
  - Evidence: `test_clue_llr_is_negative_when_the_person_never_shows_the_clue` (7 passages = 1; a 2/40 stroller pusher stays above -1).
- Finding: the "starts" lookup ran once per crop (timed out).
  - Fix: computed once per cycle.
- Noted, not changed: a walk past crossing a minute boundary counts as two passages (README).

## Restart required

```text
No manual restart: post-pull rebuild rebuilds orion-sql-writer after merge.
```

## Risks / concerns

- Severity: low. Small effect today (2 of 216 past questions); it grows as stroller history builds. Refs #3768.
- Severity: low. A mis-clicked "yes" on a stroller story puts stroller passages in that person's album, which weakens the veto for them.
- Live path UNVERIFIED until merged: check sql-writer logs for `person_clue_vetoes` in the cycle summary and any `person_clue_veto` lines.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3862

🤖 Generated with [Claude Code](https://claude.com/claude-code)
