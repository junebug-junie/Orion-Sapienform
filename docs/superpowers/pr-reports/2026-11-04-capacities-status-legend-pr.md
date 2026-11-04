## Summary

- The `dig capacities` printout now has one `status:` line under the job lines. It says what each job status shown means.
- `BY RULE` reads "the atlas rule found it; not Juniper's confirmation (a confirmation is named on its line)". This keeps a rule finding apart from Juniper's sign-off.
- Regression tests fail if any allowed status has no meaning, or if the `BY RULE` meaning stops separating the rule from Juniper.

## Outcome moved

On 2026-11-04 01:45Z (turn `1ee0e35d-9d49-4564-8aa9-e321a9ae3ac2`), Juniper asked Orion which parts do which jobs. Orion ran the dig kit correctly. In the reply, Orion called S1 and S2 "Confirmed". The printout said `BY RULE`, and `config/body_atlas/vsm_confirmed.yaml` has `confirmed: []`. The printout never said what `BY RULE` means, so Orion guessed. Now the meaning sits next to the status.

## Current architecture

`orion/body_atlas/capacities.py` `render()` prints one line per Viable System Model job: status plus evidence. Status is one of `BY RULE`, `CANDIDATE`, `CONFIRMED`, `NO PART FOUND`, `NO RULE`. The meanings lived only in the design doc (`docs/superpowers/specs/2026-10-30-self-now-capacities-vsm-design.md`) and code comments. Orion cannot see either during a turn unless it opens them.

## Architecture touched

- `orion/body_atlas/capacities.py`: `STATUS_MEANINGS` plus one line in `render()`.
- No bus, schema, env or API change.

## Files changed

- `orion/body_atlas/capacities.py`: status meanings and the legend line.
- `orion/body_atlas/tests/test_capacities.py`: regression test.
- `docs/superpowers/pr-reports/2026-11-04-capacities-status-legend-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the `dig capacities` printout has one more line.
- Compatibility notes: the full-result JSON file is unchanged.

## Env/config changes

- None. No `.env_example` changed.

## Tests run

```text
.venv/bin/python -m pytest orion/body_atlas/tests -q   -> 173 passed
.venv/bin/python -m pytest tests/test_organ_names_partition.py tests/test_report_dig_kit_usage.py -q   -> 11 passed
```

Rendered on the live atlas file `/mnt/orion-fcc/dig/2026-11-03/body-now-7.json`:

```text
status: BY RULE = the atlas rule found it; not Juniper's confirmation (a confirmation is named on its line) · CANDIDATE = the rule fits but cannot show the whole job; Juniper must confirm · NO RULE = no rule tests this job; a claim can name a part, Juniper confirms it
```

## Evals run

No eval change. The printout is not scored by `run_capacity_roles_eval.py`, which scores roles, not wording. The live check is Orion's next capacities turn: the reply must not call a `BY RULE` job confirmed.

## Docker/build/smoke checks

The dig code is baked into the harness image. The post-pull rebuild covers it through `scripts/service_rebuild_paths.yaml`.

## Review findings fixed

- Finding (must): when Juniper confirms a part for a job the rule already holds (S1 or S2), the code keeps the status `BY RULE` and adds "confirmed by Juniper: X" to the line. The first legend text ("not confirmed by Juniper") would then contradict that line.
  - Fix: `BY RULE` now means "not Juniper's confirmation (a confirmation is named on its line)".
  - Evidence: `test_by_rule_line_names_juniper_confirmation_and_legend_allows_it`.
- Finding (should): the test checked `BY RULE` only under an `if`, and checked meanings only for the statuses one fixture shows.
  - Fix: the test now asserts `BY RULE` unconditionally. It also asserts that `STATUS_MEANINGS` equals the full status Literal on `JobLineV1`.
  - Evidence: `test_printout_says_by_rule_is_not_juniper_confirmed`.
- Finding (nit): "NO RULE = a claim may" could read as a claim holding the job.
  - Fix: "no rule tests this job; a claim can name a part, Juniper confirms it".
  - Evidence: rendered line above.
- Reviewer checked live: 26 lines, 2828 of 4000 chars. No cut.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

The post-pull rebuild rebuilds `orion-harness-governor`. If it does not run: `scripts/safe_docker_build.sh orion-harness-governor up -d --build`

## Risks / concerns

- Severity: low
- Concern: Orion can still paraphrase wrongly.
- Mitigation: the meaning now sits on the same screen as the status. Check the next capacities turn.

## Turn follow-up (not changed here)

- The fix for text tool calls (#3060) held: this turn ran 23 steps, with no text tool call and no repair. The retry path itself is still UNVERIFIED live, because no turn has needed it since the deploy.
- Two organs are unnamed since 2026-11-01. Rules v2 added HTTP and Redis wires on 10-31. That split Orion's old "cortex" and "memory" organs. The 41-part organ that holds `orion-cortex-exec` and `orion-llm-gateway` now overlaps "memory" by only 0.29. The 2-part "climate" organ merged into a 46-part organ. The carry rule (overlap ≥ 0.5) refused both names, as designed. Orion can name them with `dig name-organs`.
- Orion described the jobs but did not write claims. Orion offered to write them. Juniper asked Orion to "tell", so this is not a bug.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3091

🤖 Generated with [Claude Code](https://claude.com/claude-code)
