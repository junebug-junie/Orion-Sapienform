## Summary

- PR #3107 merged before its last two commits were pushed. This PR carries those two commits onto main.
- Capacities stop telling Orion to claim S5 (identity) "without asking first" while S5 is shared with Juniper. Orion may still write an S5 claim if they choose.
- The one-part view (`body <part>`) no longer prints the strength twice ("metabolism claimed (claimed)" → "metabolism claimed").

## Outcome moved

On main after #3107, the capacities tail ends with a ready `capacity claim '<part>' S5` line, which pushes Orion to claim the one job Juniper decided to share. On turn 00655f13, Orion deliberately declined to claim S5. After this PR, the tail ends with the generic `to claim a job:` line.

## Current architecture

`open_jobs` in `orion/body_atlas/capacities.py` lists jobs with no holder. The dig tail prints a ready claim line per open job. `vsm_confirmed.yaml` sets `s5_shared_with: Juniper`.

## Architecture touched

- `orion/body_atlas/capacities.py`: `open_jobs` counts S5 as held while `s5_shared_with` is set.
- `orion/dig/__main__.py`: the part view prints a rule name only when it adds something.
- Docs: the design doc's "Decision change 2026-11-04" section, both READMEs.

## Files changed

- `orion/body_atlas/capacities.py`, `orion/dig/__main__.py`: the two fixes.
- `orion/body_atlas/tests/test_capacities.py`, `test_claims.py`, `orion/dig/tests/test_capacities_cmd.py`, `test_body_part.py`: tests.
- `orion/body_atlas/README.md`, `orion/dig/README.md`, `docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`: wording.
- `docs/superpowers/pr-reports/2026-11-04-claims-hold-when-evidence-checks-pr.md`: #3107's report, its follow-up section.

## Schema / bus / API changes

- None.

## Env/config changes

- None.

## Tests run

```text
pytest orion/dig/tests orion/body_atlas/tests tests/test_organ_names_partition.py tests/test_report_dig_kit_usage.py -> all passed
```

New tests: `test_s5_shared_with_juniper_is_never_offered_as_an_open_job`, `test_a_claim_on_s5_is_allowed_while_it_is_shared_with_juniper`, `test_s5_shared_with_juniper_gets_no_ready_claim_line`.

## Evals run

```text
orion/body_atlas/evals/run_*.py -> all exit 0
```

## Docker/build/smoke checks

The live read-only smoke on the #3107 branch (same commits) showed the tail ending with the generic `to claim a job:` line and `jobs: metabolism claimed` in the part view.

## Review findings fixed

- These are the coordinator's follow-ups on #3107, made after its review. The detail is in that PR's report.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild covers `orion-harness-governor`.

## Risks / concerns

- Severity: low. Orion can still claim S5 unprompted. That claim holds like any other, and Juniper can reject it.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
