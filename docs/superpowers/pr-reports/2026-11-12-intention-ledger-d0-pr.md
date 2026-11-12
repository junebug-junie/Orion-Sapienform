## Summary

Intention ledger step D0 (design PR #3188, approved by Juniper 2026-11-12). The code shipped in PR #3217. That PR was merged with a stub report. This report covers #3217 and its follow-up.

- The 5b finalize judge now also lists the promises Orion makes in a reply. Each promise is an exact quote. Offers ("if you want, I'll...") are marked.
- Plain code (`orion/harness/promise_gate.py`) drops any quote that is not literally in the reply. It checks the draft the judge read, then the final text Juniper saw.
- Offers count only after Juniper accepts them. A later judge call decides acceptance from Orion's reply plus Juniper's next message. There is no word list.
- Flag `ORION_PROMISE_LEDGER_ENABLED`, shipped ON (orion-harness-governor). D0 only records. No acts open, and no cue reaches stance or the motor.
- Replay eval over 30 days of Juniper's turns. **Precision is 0.72, which is below the 0.8 bar. Per the design, D1 does not start. The design goes back to Juniper.**
- Follow-up PR: the judge-crash eval now follows the flag, the gate strips code backticks, and the hand labels are committed (hashes only).

## Outcome moved

We now know how well the judge finds promises. It finds all of the real ones (6 of 6). It also lists too many things that are not promises: 8 of 29 items, mostly forecasts about a system ("the next deploy makes the command native").

## Replay eval (acceptance check 1)

Window: 2026-10-13 03:15 to 2026-11-12 03:15 UTC. 185 of Juniper's turns. 175 judged, 5 skipped (no stance sighting), 5 errored (Strata 502, see below). Lanes: Flash-Next chat (GPU 0) and agent (GPU 1). Thinking off. The run used the review-fixed prompt.

| measure | result | bar |
|---|---|---|
| precision (29 extracted items, hand-labelled) | **21 / 29 = 0.72** | ≥ 0.8: **FAIL** |
| precision of counted promises (what D1 would open as acts) | 12 / 17 = 0.71 | |
| recall against the 6 hand-found statements | **6 / 6**, offer flag right on all 6 | report |
| gate: non-quotes dropped | 2 of 2 quotes not in the text were dropped (100%) | 100%: PASS |
| gate: real promises lost | 1. The judge dropped the code backticks from a real offer. The follow-up now strips backticks before matching. | |
| offers accepted (judge, from Juniper's next message) | 2 | |

The 8 false items, by kind:
- forecasts about a system or a queue ("should be processed within the next hour", "the next deploy makes the command native", "settled in my records by Monday"): 3
- asks Juniper to act ("text when you're parked") or hands the task to others: 2
- a wish, a readiness line, or planned work that is not Orion's commitment: 3

**Baseline keep rate.** The same judge was shown the open promises (oldest 3, under 72 h) on each later turn. It marked them kept when a reply delivered them.
- Over all 17 counted promises: 1 kept, 16 expired, 0 censored, 0 never shown. Beta(1, 1) posterior: mean 0.11, 90% interval 0.02 to 0.24.
- Over only the 12 hand-confirmed real promises: 1 kept, 11 expired. Mean 0.14, 90% interval 0.03 to 0.32.
- This is a thin-data result. It is reported as it is, and it is not a veto. Follow-through outside chat (for example a read that really ran) is not visible to this judge: `UNVERIFIED`.

**Strata 502s.** 5 of 180 calls returned `structured_output_failed`. The live judge had passed all 5 of these turns. A/B rerun of 4 of those turns on the chat lane, 3 tries each. Flag off: 1 of 12 calls returned 502 ("model did not return valid JSON: Invalid \\escape"). Flag on: 0 of 12. The 502s are an intermittent JSON-escape failure that happens without the promise keys too. The promise keys did not cause them. (The agent lane was busy with live work, so it was left out of the A/B.)

Commands:
```text
ORION_PROMISE_LEDGER_ENABLED=true python -m orion.harness.evals.promise_ledger_replay_eval run --until 2026-11-12T03:15:00
python -m orion.harness.evals.promise_ledger_replay_eval score
```
The report (it holds Orion's quotes) is local only: `/tmp/promise-ledger-replay/report.json`. The labels are in `orion/harness/evals/fixtures/promise_ledger_labels.json`: correlation id, quote hash and a short reason, with no text.

## Recommendation for Juniper

Precision fails on forecasts and on lines that ask Juniper to act. One concrete next try: add two exclusions to the PROMISES section. First, "a forecast of what a system, queue or deploy will do". Second, "anything that asks Juniper to act". Then rerun this eval with no other change. The labels are committed, so the rerun is scored the same way. This is your call. Nothing past D0 is built.

## Current architecture

Before #3217, the judge returned 9 judgement keys. Nothing on the unified turn recorded promises. The social-memory word list was retired separately in #3216.

## Architecture touched

- `orion/schemas/harness_finalize.py`: `PromiseV1`, `OpenPromiseRefV1`, and 3 list fields on `FinalizeReflectionV1`. No consumer validates it with `extra="forbid"`; checked in `services/` and `orion/`. The run artifact is the source of truth; the verdict-molecule copy is gated only against the draft (documented on the model).
- `orion/harness/finalize.py`: the schema and grade turn follow the flag. The promise keys are parsed apart and leniently, so a bad promise list cannot fail the verdict. The gate runs after the judge, then again on the final text in `run_harness_finalize_chain`. `resolve_finalize_llm_lane` is untouched.
- `orion/cognition/prompts/harness_finalize_reflect.j2`: a PROMISES section after TOOL RECOMMENDATION, only when the flag is on.
- `orion/harness/promise_gate.py` (new): the plain-code gate.
- `orion/harness/evals/promise_ledger_replay_eval.py` (new): the eval.

## Files changed

- `orion/harness/promise_gate.py`, `orion/schemas/harness_finalize.py`, `orion/harness/finalize.py`, `orion/cognition/prompts/harness_finalize_reflect.j2`: the feature.
- `orion/harness/tests/test_promise_gate.py`, `orion/harness/tests/test_finalize_reflect_judge_contract.py`: the tests.
- `orion/harness/evals/promise_ledger_replay_eval.py`, `test_promise_ledger_replay_eval.py`, `fixtures/promise_ledger_labels.json`: the eval.
- `orion/harness/evals/judge_crash_replay_eval.py`: the grade-turn assert now follows the flag (follow-up).
- `services/orion-harness-governor/.env_example`, `app/settings.py`, `README.md`: the flag.

## Schema / bus / API changes

- Added: `PromiseV1`, `OpenPromiseRefV1`, and `FinalizeReflectionV1.promises_made / offers_accepted / promises_kept` (default empty).
- Removed / renamed: none.
- Behavior changed: the judge's structured-output schema has 3 more required keys when the flag is on. The schema has no caps (Strata returns 502 on a broken cap, which would sink the verdict). The gate applies the caps.
- Compatibility: additive. Readers ignore unknown fields. No bus channel changed.

## Env/config changes

- Added keys: `ORION_PROMISE_LEDGER_ENABLED=true` (orion-harness-governor).
- `.env_example` updated: yes.
- local `.env` synced: `python scripts/sync_local_env_from_example.py --all-keys orion-harness-governor` added the key to the primary checkout's `services/orion-harness-governor/.env`. The default sync run skips non-routine keys, so `--all-keys` was needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/harness services/orion-cortex-exec/tests/test_finalize_reflect_request_shape.py tests/test_no_personal_terms.py
  837 passed
pytest orion/harness/tests/test_promise_gate.py   20 passed (checks 2 and 5, offer rule, flag off, chain re-gate)
mutation: chain re-gate pointed back at the draft -> test_chain_regates_promises_against_the_final_text fails
python scripts/check_env_template_parity.py   PASS
python scripts/check_env_key_single_source.py OK
python scripts/check_service_env_compose_parity.py orion-harness-governor  N/A (env_file)
services/orion-harness-governor/tests: 100 passed, 1 failed (test_fcc_warm_pool_wiring: PermissionError reading ~/.fcc/.env on this host; unrelated)
```

## Evals run

```text
promise_ledger_replay_eval run/score: see the Replay eval section above.
```

## Docker/build/smoke checks

```text
No manual build. #3217 merged at 03:56 UTC and the post-pull rebuild recreated orion-athena-harness-governor at 04:03 UTC.
Live check (harness_turn_trace since 04:03): 3 turns. 1 judged turn has reflection.promises_made = [] (the field is live).
2 turns degraded on transport (gpu_pool_unavailable, RPC timeout), not on the schema. No live promise has been recorded yet: UNVERIFIED.
```

## Review findings fixed

- Finding: the PROMISES block split the prompt's first line ("described under REQUIRED JSON FIELDS").
  - Fix: restored the line. The section moved after TOOL RECOMMENDATION.
  - Evidence: `test_opening_instruction_is_intact_with_flag_on_and_off`. The first eval run used the broken prompt. It was stopped and fully rerun.
- Finding: `maxItems` / `maxLength` in the schema would let Strata 502 the whole verdict.
  - Fix: removed from the schema. The gate caps instead.
  - Evidence: the schema test now accepts 4 promises and a 241-character quote.
- Finding: a non-list promise value or `"offer": "false"` sank the verdict or flipped the flag.
  - Fix: lenient list coercion, and `offer is True`.
  - Evidence: parametrized malformed-payload test.
- Finding: no test proved the chain re-gate.
  - Fix: a chain test, plus a mutation check.
- Finding: the verdict molecule and the run artifact disagree on promises.
  - Fix: documented on the model that the run artifact is the source of truth.
- Finding: the eval's keep rate was biased down. An offer done in the same reply could not be marked kept, and promises never shown were counted as broken.
  - Fix: pending offers are keepable. Never-shown promises are their own bucket.
- Finding: unused promise params leaked onto the response-repair builders.
  - Fix: removed.
- Finding: short quotes were counted as "not in text"; dashes and ellipses were not normalized; the eval did not filter on source.
  - Fix: a `dropped_too_short` bucket, dash/ellipsis normalization, and `source = 'hub_orion'`.
- Found after review (follow-up PR): the judge-crash eval asserted the flag-on grade turn with the flag off. A real offer was lost to stripped backticks. Both are fixed.

## Restart required

```text
No restart required. The governor was already rebuilt from main at 04:03 UTC with #3217.
The follow-up PR changes only eval code, labels and docs, plus the gate's backtick
normalization (it reaches the governor with the next post-pull rebuild).
```

## Risks / concerns

- Severity: medium. Concern: precision is 0.72. Anything that reads `promises_made` today would get about 1 false promise in 4. Mitigation: D0 has no reader. D1 (acts and the cue) must not start until Juniper decides.
- Severity: low. Concern: the live judge now writes 3 more keys. 5 of 180 replay calls hit Strata 502, but the A/B shows the same failure with the flag off. Mitigation: `ORION_PROMISE_LEDGER_ENABLED=false` removes the keys from the schema.
- Severity: low. Concern: the replay eval bypassed the GPU pool on the live chat and agent lanes for about an hour (03:20-04:30 UTC). One live judge call at 04:10 timed out (RPC timeout), and my load may have contributed. Mitigation: the next run should go when the lanes are quiet.
- Severity: low. Concern: #3217 merged with a stub report. Mitigation: this report replaces the stub. The merged PR's body is updated to match.

## PR link

- Code: https://github.com/junebug-junie/Orion-Sapienform/pull/3217 (merged)
- Follow-up: https://github.com/junebug-junie/Orion-Sapienform/pull/3228

🤖 Generated with [Claude Code](https://claude.com/claude-code)
