## Summary

- CI now runs the office person eval (`services/orion-vision-host/evals/office_person_cutoff_eval.py`). It runs in the existing vision council/window workflow.
- That workflow now also triggers on `config/vision_profiles.yaml` and on the eval's script and data. No workflow watched that config before.

## Outcome moved

Before, a PR could change the office "person" cut-off (0.55) or the room prop rule, and nothing would check it. Now the eval fails the PR.

## Current architecture

No workflow covered `config/vision_profiles.yaml` or `services/orion-vision-host` (#3615). The eval ran by hand only.

## Architecture touched

- CI only: `.github/workflows/orion-vision-council-window-tests.yml`.

## Files changed

- `.github/workflows/orion-vision-council-window-tests.yml`: adds the new paths and one eval step.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- None.

## Tests run

```text
Clean venv with only PyYAML installed:
  office_person_cutoff_eval.py -> PASS, exit 0
With the cut-off lowered to 0.35 (local mutation, reverted):
  office_person_cutoff_eval.py -> FAIL, exit 1
```

## Evals run

```text
office_person_cutoff_eval.py PASS
```

## Docker/build/smoke checks

```text
Not applicable. This is a CI change only.
```

## Review findings fixed

- Finding: only the one fixture CSV triggered the workflow; a new fixture file would be missed.
  - Fix: trigger on `services/orion-vision-host/evals/fixtures/**`.
  - Evidence: review found no other problems (step exits 1 on FAIL, paths cover every file the eval reads).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
  - Concern: the vision-host unit tests (for example `test_box_thresholds.py`) still don't run in CI. They import torch.
  - Mitigation: the eval covers the office cut-off. Those other tests are mostly torch-bound runner tests.
- Fixes #3615.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
