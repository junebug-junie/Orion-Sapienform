## Summary

- Self-inquiry runs did test beliefs Orion already holds about itself. The score just never counted those tests.
- Cause: the self-inquiry prompt listed Orion's self-priors but never showed the one command that records a test with this run's id (`last_run_id`). Orion improvised the write without the id, and the scorer filed each test as "someone else's change".
- Fix: the testing command now lives in one shared helper, `prior_testing_lines()` in `orion/curiosity/kickoff_prompt.py`. The investigation kickoff and the self-inquiry prompt both render it.
- The kickoff prompt text does not change (pure extraction).
- Fixes #3375.

## Outcome moved

Self-inquiry tests of existing self-priors become countable as `n_tested` / `realized_nats` in `curiosity_run_outcomes`. Before: 1 of 45 self-inquiry runs (2026-11-12..24) had `n_tested > 0`.

## Current architecture

Hub snapshots every prior at turn start and turn end and diffs them (`orion/curiosity/value.py` `diff_snapshots`). A change counts as this run's test only when the prior carries `last_run_id == run_id`. Otherwise it counts as `n_unattributed`. The investigation kickoff shows a one-command test that sets the count, the confidence, and `last_run_id` together. The self-inquiry prompt showed only the MERGE that forms a new self-prior.

## Cause evidence (step 1 of #3375)

1. Outcomes, last 7 days, by arm (`curiosity_offer_decisions` JOIN `curiosity_run_outcomes`):
   self_inquiry 45 runs, 1 with n_tested>0, **50 unattributed changes**. The only counted run (891ce6791bdc) is the only one whose priors carry its own `last_run_id` in FalkorDB.
2. `GRAPH.RO_QUERY orion_worldview` for `:PriorRevision` with run_id in the 54 self-inquiry run ids of the last 8 days: **0 rows**.
3. Start snapshot of each self-inquiry run, diffed against the next run's start snapshot (taken after it ended): 16 changes to existing `self:` priors had `times_tested` go up (and often confidence move) with **no `last_run_id`**. Every one of them happened while a self-inquiry run was active. Four windows had only a self-inquiry run active (0bce71c76e26, 5de65d80071b, c1d958a9e31a, 917417439ab0/32a398e21af4). Examples: `self:organ_ids_are_content_hashes_configs_recur_20261114` 0.9 -> 0.2, tested 1 -> 2, no stamp, during 5754ca44282e; `self:dream_pair_judge_flipped_at_the_metacog_lane_retirement_20261118` 0.55 -> 0.4 then 0.4 -> 0.5 across two self-inquiry runs, no stamp.
4. `orion/curiosity/self_inquiry_prompt.py` imported `_priors_section` (the prior list) but not the TESTING block in `kickoff_prompt._write_section`.

So the candidate "the prompt never asks Orion to test" is wrong: Orion tests. The loss is in recording, caused by the missing command. The measurement code is correct and unchanged.

Not fixed here, seen while diffing (filed as #3855): on 11-24 an investigation run (a32417ee8a57, value_order) moved two self-priors' confidence with `times_tested` unchanged and no stamp.

## Architecture touched

Curiosity prompt text only. No schema, bus, env, or service code change.

## Files changed

- `orion/curiosity/kickoff_prompt.py`: TESTING block extracted into `prior_testing_lines()`; `_write_section` renders it unchanged.
- `orion/curiosity/self_inquiry_prompt.py`: `_self_write_section` adds one lead-in sentence plus `prior_testing_lines()` after the self-prior shape.
- `orion/curiosity/tests/test_self_inquiry_prompt_prior_testing.py`: regression tests.
- `docs/superpowers/pr-reports/2026-11-24-self-inquiry-tested-count-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the self-inquiry prompt (writable runs only) now shows the stamped testing command and the `:PriorRevision` record.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/curiosity/tests/test_self_inquiry_prompt_prior_testing.py  -> 5 passed
pytest orion/curiosity/tests tests/test_curiosity_peer_kickoff.py tests/test_curiosity_peer_patch0_acceptance.py -> 315 passed, 29 skipped
pytest services/orion-hub/tests -k "curiosity or kickoff or self_inquiry" -> 723 passed, 4 skipped
```

## Evals run

```text
None. No eval harness covers prompt-to-graph-write behavior. The live check below is the eval.
```

## Docker/build/smoke checks

```text
None run. Prompt text only; Hub builds the prompt at kickoff.
Live check after deploy (verify-live): self-inquiry rows in curiosity_run_outcomes with n_tested > 0, and :PriorRevision rows carrying self-inquiry run ids.
```

## Review findings fixed

- Review (subagent) found no material issues. It verified that the kickoff prompt renders byte-identical before and after the change. It also checked that a mutation dropping the new lines fails 3 of the 5 tests.
- Finding (minor): self-inquiry had no "inconclusive still counts" line, so a test that did not move the confidence could go uncounted.
  - Fix: added a self-inquiry inconclusive line after the testing command. The kickoff text is unchanged.
  - Evidence: asserted in `test_writable_self_inquiry_prompt_carries_the_stamped_testing_command`.
- Finding (minor): "(the list above)" was wrong for stale self-priors. Those are kept out of the main list.
  - Fix: reworded to "(listed above)".
- Finding (minor): the scorer test passes on the parent commit too.
  - Fix: its docstring now says it pins scorer behaviour and is not the regression guard.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build
```

(Run from the clean primary main checkout after merge. The FCC sandbox sync after the post-pull rebuild also picks up `orion/`.)

## Risks / concerns

- Severity: low. Concern: the self-inquiry prompt grows by ~30 lines. Mitigation: the same text already ships in every investigation prompt.
- Severity: low. Concern: live effect is UNVERIFIED until the next self-inquiry runs after deploy. Mitigation: check query above.
- Issues: closes #3375; opened #3855 (unstamped investigation move).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3860

🤖 Generated with [Claude Code](https://claude.com/claude-code)
