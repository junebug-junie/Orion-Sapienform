# feat: re-check a reply that was rewritten to fix a false perception claim

## Summary

- When the reply check (finalize 5b) finds that a draft says something about a named person or thing that the Situation notes don't support, the rewrite (5c) now gets checked once more.
- If the re-check still finds such a claim, the reply is rewritten once more, from the re-check's notes plus the first check's other notes. There is no third check.
- The re-check stays inside a turn's reply budget. It is skipped when less than 30 s would remain (after a 15 s margin), and cut if it runs over. Either way the first rewrite is sent, never the original draft.
- New repair reasons, shown in Hub as "revised: …": `unsupported_claim` (re-check passed), `unsupported_claim_repaired_twice`, and `unsupported_claim_unchecked` (re-check failed, degraded, or had no budget).
- Follow-up to PR #3086.

## Outcome moved

Replay eval on the 11-03 invented-name-story turn. The text that would be sent is now judged correct 6/6 (2 runs needed the second rewrite). After #3086 alone it was 3/9. Reading the six final texts by hand: all six drop the false arrival and the "I started calling that slot" story. Four of the six keep a vague "vehicle after vehicle I've clocked" sentence next to her name, but none says she is a vehicle.

## Current architecture

`run_harness_finalize_chain` ran 5b once, then 5c once when 5b asked for it, and sent 5c's text unchecked. PR #3086 made 5b compare perception claims with the Situation notes, in the format `unsupported claim: …; notes say: …`.

## Architecture touched

- `orion/harness/finalize.py`: the 5c branch of `run_harness_finalize_chain`, plus new `_recheck_unsupported_claim_repair` and `_recheck_and_maybe_repair`.
- `services/orion-harness-governor/app/bus_listener.py`: passes the reply deadline to the chain.
- Hub `draft-revision.js`: labels for the three new reasons.

## Files changed

- `orion/harness/finalize.py`: bounded re-check and second repair; `has_unsupported_claim(_note)`; reason constants; budget guard; `deadline_monotonic` parameter.
- `orion/harness/tests/test_perception_claim_recheck.py`: 9 tests (clean re-check, second repair, bounded, no re-check for other misalignment or without notes, degraded re-check, budget skip, budget cut, notes carried, `finalize_changed` kept).
- `services/orion-harness-governor/app/bus_listener.py`, `services/orion-harness-governor/tests/test_finalize_reply_budget.py`: deadline passed through, and a test for it.
- `services/orion-hub/static/js/draft-revision.js` + `.test.js`: labels.
- `orion/harness/evals/perception_claims_replay_eval.py`: follows the live 5c path.
- `docs/superpowers/specs/2026-09-12-kill-voice-finalize-design.md`: documents the new `response_repair_reason` values.

## Schema / bus / API changes

- Added: none (no schema fields).
- Behavior changed: `response_repair_reason` can now be `unsupported_claim`, `unsupported_claim_repaired_twice` or `unsupported_claim_unchecked`, instead of the verdict reason, for these turns. It is a free-form string; nothing compares it to `misaligned` (checked across Python, SQL, YAML, JS and j2). When counting 5b rejections, count all four values together with `misaligned`.
- `run_harness_finalize_chain(deadline_monotonic=None)`: new optional keyword argument.

## Env/config changes

- None.

## Tests run

```text
pytest tests/test_no_personal_terms.py orion/harness orion/cognition orion/situational services/orion-harness-governor/tests   1199 passed
node --test services/orion-hub/static/js/*.test.js                                                                             403 pass, 0 fail
```

## Evals run

```text
python -m orion.harness.evals.perception_claims_replay_eval --repeats 6   (live chat lane, before the review fixes)
fabricated_arrival        misaligned 6/6, sent text aligned 6/6
invented_name_origin      misaligned 6/6, sent text aligned 6/6 (2 needed the second repair)
honest_control            aligned 6/6
supported_street_control  aligned 6/6
After review fixes, --repeats 4: fabricated_arrival 4/4 -> 4/4; invented_name_origin 4/4 -> 4/4; honest_control 4/4; supported_street_control 4/4. PASS
```

## Docker/build/smoke checks

```text
Not run: no compose, dependency or env change.
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent) on 969b4c088.

- Finding (PLAUSIBLE, high): on a turn with a reply budget, the two extra model calls could run out the budget. The governor would then send the ORIGINAL draft, the one with the false claim.
  - Fix: `deadline_monotonic` from the governor. The re-check is skipped below 30 s left (after a 15 s margin) and runs under its own `asyncio.timeout`. On either, the first repair is sent.
  - Evidence: `test_too_little_budget_skips_the_recheck`, `test_a_recheck_that_would_run_out_the_budget_is_cut_and_sends_the_first_repair`, governor `test_finalize_gets_the_reply_deadline_for_the_perception_recheck`.
- Finding (CONFIRMED, medium): a failed re-check degrades instead of raising, so it read as a clean pass.
  - Fix: `reflection_source == "degraded_llm_failure_fallback"` returns `unsupported_claim_unchecked`.
  - Evidence: `test_a_failed_recheck_keeps_the_first_repair_and_says_unchecked`.
- Finding (CONFIRMED, medium): these turns drop out of a `response_repair_reason=misaligned` count.
  - Fix: kept the distinct values (they are the only durable record of what the re-check did, and Hub shows them). Documented them in the design doc, with the rule to count them together with `misaligned`.
- Finding (CONFIRMED, low): `finalize_changed` lost the "misaligned always counts as changed" rule.
  - Fix: OR with the first repair's value. Evidence: `test_misaligned_repair_still_counts_as_changed_when_text_echoes`.
- Finding (PLAUSIBLE, low): the second repair lost the first check's non-perception notes.
  - Fix: carried them in. Evidence: `test_second_repair_keeps_the_first_checks_other_notes`.
- Finding (PLAUSIBLE, low): the eval only repaired on `misaligned`; live repairs on more.
  - Fix: the eval now follows the live condition.
- Not fixed: the re-check's own reflection is recorded only as the reason value and a log line (`perception_claim_recheck corr=… result=…`), not in the outcome molecule. That would be a schema change. The reason value carries the outcome.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: up to two extra model calls on a turn whose draft made an unsupported perception claim. Live rate so far: 0 such turns since #3086 deployed. Mitigation: the calls only fire in that case, and they stay inside the reply budget.
- Severity: low. Concern: 5b grades its own rewrite (the same model), so "aligned" after the re-check is still that model's judgement. The hand read is in Outcome moved.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3098

🤖 Generated with [Claude Code](https://claude.com/claude-code)
