## Summary

- The clue veto now counts a stroller/dog only on cameras that were already tagging that kind. Before, the office (never tags strollers) and the doorbell before Nov 24 counted every walk past as "no stroller", which diluted the rates.
- Spec: says plainly that patch 3 shipped as a drop-only veto with its own replay gate (it cannot raise a right-name rate), how few cells can veto, box precision, and that nothing durable is stored.
- Answers Orion's four questions on #3835.

## Outcome moved

Street stroller rate is now measured on cameras that can see strollers: 26/301 -> 35/192 passages. The veto's evidence is honest per camera.

## Current architecture

`load_crop_clues` marked a kind eligible from its first box on ANY camera.

## Architecture touched

- `services/orion-sql-writer/app/vision_person_model.py`: `_CLUE_STARTS_SQL` groups by (label, stream); eligibility matches the crop's own stream.

## Files changed

- `services/orion-sql-writer/app/vision_person_model.py`: per-camera start.
- `services/orion-sql-writer/tests/test_vision_person_model.py`: real-Postgres test (walkway before first stroller, near vs far box, doorbell dog yes / stroller not eligible).
- `services/orion-sql-writer/README.md`: one line.
- `docs/superpowers/specs/2026-11-24-walkway-clue-evidence-design.md`: "Patch 3 as shipped".

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
ORION_TEST_PG_URI=<throwaway postgres:16 with the walkway + person migrations> \
  pytest services/orion-sql-writer/tests/test_vision_person_model.py     92 passed (pg lane included, locally)
```

## Evals run

```text
run_clue_veto_replay_eval.py (live, read-only): 222 answered; 3 dropped, all "no"; 0 "yes" lost; gate passed
cells clearing 10 album passages: 10; only one can veto on one sighting (stroller, 0/58, llr -3.08); dog -1.95..+1.06
stroller boxes checked by eye: 31 most recent, >= 29 clearly strollers, 2 unjudgeable
```

## Docker/build/smoke checks

```text
Not run; SQL change exercised live by the replay (read-only). Post-pull rebuild redeploys orion-sql-writer.
```

## Review findings fixed

- Finding: spec placeholder "#this PR".
  - Fix: replaced with the PR number.
- Finding: the new test is pg-only and skips without ORION_TEST_PG_URI.
  - Fix: ran it locally against a throwaway Postgres (92 passed, above).
- Noted, not changed: a camera becomes eligible from its first box ever; one early false box would start it early (pre-existing).

## Restart required

```text
No manual restart: post-pull rebuild rebuilds orion-sql-writer after merge.
```

## Risks / concerns

- Issues opened: #3903 (verify-live: yes-rate before/after, ~12-09, needs-juniper), #3904 (veto leaves no durable record). Refs #3768.
- Severity: low. The veto is thin: one cell can act alone today.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3905

🤖 Generated with [Claude Code](https://claude.com/claude-code)
