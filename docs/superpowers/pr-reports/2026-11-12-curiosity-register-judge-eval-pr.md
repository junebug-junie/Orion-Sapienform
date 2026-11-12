## Summary

- Adds the judge-agreement eval for the curiosity finding register and the pure module it runs (`orion/curiosity/register.py`: pinned prompt v2, strict reply parser, novelty, input assembly).
- Result: **no local lane passes the gate.** Nothing in this PR labels live findings, and no table, flag or service changes. Steps 2 and 3 of the curiosity event-wants stack stay blocked.
- DRAFT: kept so the eval can be re-run the day a new lane exists. Close it if that is not wanted.

## Outcome moved

- The judge question is answered with numbers instead of a guess: kappa 0.58 (metacog), 0.59 (metacog, thinking on), 0.60 (quick), against the >= 0.8 bar.

## Current architecture

- Spec: `docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md` ("The judge, as specified for the live patch"). Reference labels: `scripts/analysis/fixtures/curiosity_register/reference_labels_2026-10-20.jsonl` (263 runs, Sonnet judge).

## Architecture touched

- None at runtime. One shared pure module, one analysis script, unit tests.

## Files changed

- `orion/curiosity/register.py` (new): prompt v2 byte-identical to the fixture (sha `7dcaa667...1222` pinned), `parse_judge_reply`, `novelty`, `run_question_and_answer` (finish row + graph Finding texts, as the reference labels were made).
- `orion/curiosity/tests/test_register.py` (new): 20 tests.
- `scripts/analysis/eval_curiosity_register_judge.py` (new): runs a lane against the reference labels; results in its docstring.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none. Behavior changed: none. Compatibility notes: none.

## Env/config changes

- None.

## Tests run

```text
pytest orion/curiosity/tests/test_register.py -> 20 passed
```

## Evals run

```text
eval_curiosity_register_judge.py, 263 reference runs, 0 missing inputs, 0 judge errors
lane                    model            kappa  agree  phil-agree  6 winners phil
metacog                 Qwen3-8B Q5_K_M  0.580  0.772  0.96        5/6
metacog (thinking on)   Qwen3-8B Q5_K_M  0.586  0.776  0.96        5/6
quick                   Qwen3-8B Q4_K_M  0.604  0.787  0.95        5/6
Gate: kappa >= 0.8, phil-agree >= 0.90, 6/6 winners. All three FAIL on kappa and winners.
Failure is one-sided: 34-39 of 161 operational findings called philosophical.
```

## Docker/build/smoke checks

```text
Not applicable (no runtime change).
```

## Review findings fixed

- The code review for the stack ran on step 1 (PR #3220). This draft adds no runtime path.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: info. Concern: a pure module with only an eval consumer. Mitigation: draft; it becomes the live writer's core once a lane passes, or this PR is closed.

## PR link

(this PR)
