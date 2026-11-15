## Summary

The promise finder (the 5b judge's PROMISES section) listed things that are not Orion's promises. The two biggest kinds were forecasts about a system ("should be processed within the next hour") and asks for Juniper to act ("text when you're parked"). Juniper approved excluding both on 2026-11-14.

- Each promise the judge lists now says who does the act: `orion`, `juniper`, `someone_else`, or `a_system`. The judge must fill it (JSON schema enum).
- Plain code keeps only items where Orion does the act. It drops the rest before the 3-item cap, so they cannot push a real promise out. A missing or unknown doer is dropped (fail closed).
- No word list. The judge reads the grammar ("who is the subject of the future verb?"); code reads one closed field. Prompt examples are invented, not taken from the labelled set.
- The replay eval now reports Wilson 95% intervals, pooled recall by promise, an ablation of the doer rule, and can resume after the GPU pool takes the seat back (`--skip-done`, multi-report `score`).
- First live proof that a leased eval seat works end to end (grant, gateway calls under the hold, recall, partial report, exit 3).

## Outcome moved

The doer rule works as a filter: on the replay it dropped 6 items, all false (4 forecasts, 2 asks), and **no real promise**.

Precision did **not** rise on the turns measured so far. The new prompt also makes the judge list more "presence" lines ("I'll be here when you wake"), and the doer rule does not catch those because Orion is the subject. The 0.8 bar is still not met. D1 must stay closed.

### Replay eval (same window and labels as D0: 2026-10-13 03:15 .. 11-12 03:15 UTC)

Only 123 of 185 turns ran (see Risks). Numbers below are on the **104 turns both runs judged**, scored with the same hand labels.

| measure | before (D0 run, 11-12) | after (doer, 11-15) |
|---|---|---|
| items listed | 12 | 15 (21 before the doer rule) |
| precision | 8/12 = 0.67 (95% Wilson 0.39-0.86) | 9/15 = 0.60 (0.36-0.80) |
| precision without the doer rule (ablation, new prompt) | n/a | 9/21 = 0.43 (0.25-0.64) |
| pooled recall (real promises any run found, by promise) | 8/11 = 0.73 (0.43-0.90) | 9/11 = 0.82 (0.52-0.95) |
| items the doer rule dropped | n/a | 6: a_system 4, juniper 2; real promises dropped 0 |

How to read it: the intervals overlap almost completely. With 12-15 items, a change of 0.1 is noise. The two things that are not noise: the doer rule removed only false items, and the after run still lists presence lines (3 of the 6 remaining false items).

Remaining false items after the rule, by kind: presence/readiness line with no deliverable 3 ("I'll be here when you wake"), a fragment of one 1, a quip ("I'll be the first to know") 1, a hand-off phrased with Orion as subject ("I'll leave that question for the morning crew") 1.

Of the 123 turns that ran: 107 judged, 5 skipped (no stance sighting, same as D0), 2 Strata `structured_output_failed` 502s (D0: 5 of 180), and 9 gateway `gpu_pool_unavailable` 503s on calls made while a hold was being recalled. My ad hoc resume treated those 9 as done, so they were not retried; the committed `--skip-done` now retries error rows.

Not measured: recall on the design's 6 hand-found promises (all 6 are in the 62 turns that never got a seat), and the held-out windows (09-13..10-13, 11-12..11-15). The ablation is the new prompt with the rule removed; it is not the old prompt.

Overfitting note: the fix was designed from the D0 false items, which are in this labelled set. The prompt examples are invented. A held-out window was planned and not reached (issue #3464).

Labelling: 17 new items were hand-labelled with the D0 rule (a real forward commitment by Orion, offers included; readiness/presence lines with no deliverable are false). I saw each item's doer while labelling, which is a possible bias. Labels are committed as hashes only.

## Leased seat (first live proof, PR #3242 helper)

- Smoke 2026-11-15: hold `0c913a1d...` (holder `operator:promise-ledger-replay`, class `operator_agent`, priority background) admitted 00:21:58Z, queued behind Orion's own shed (`shed:orion_self_shed`), **granted 00:50:55Z** on gpu1 (Flash-Next agent worker, waited 29 min), released `ok` 00:51:40Z.
- Gateway, same window: `gpu_pool_lease_granted route=operator_agent class=operator_agent priority=background ... hold=0c913a1d030d4a78a7ba204e09187ac6` then `openai_chat_passthrough route=operator_agent upstream=http://100.112.254.99:8015/v1/chat/completions served_by=circe-worker-agent`, once per call.
- Recall path: 7 holds were recalled (`owner_waiting`); each time the eval wrote a partial report and exited 3, and the hold settled `cancelled operator_lane_recalled`. Example: lease `73844e78...` granted 00:57:14Z, recalled 01:05:32Z, cancelled 01:05:46Z.
- Totals: 9 grants (the smoke, 7 replay parts, and 1 grant to an already-stopped client, #3407); 7 recalls (one per replay part); 121 judge calls under holds; mean queue wait 31 min. One 6 h queue wait ran out with no seat (12:18-18:18Z).
- Found a pool bug: a hold whose eval was stopped while still queued was granted 45 min later to a dead client and held the seat 90 s until `heartbeat_lost` (issue #3407).

## Current architecture

The 5b finalize judge returns `promises_made: [{quote, offer}]` when `ORION_PROMISE_LEDGER_ENABLED` is on. `orion/harness/promise_gate.py` keeps a quote only if it is literally in the draft and in the final text. The gated list is stored in `harness_turn_trace.run_artifact.reflection.promises_made`, and Prime's promise concerns read it (`orion/prime/intention_sources/turn_promise.py`, `orion/prime/handback.py`).

## Architecture touched

- Judge contract: one more required key per promise item (`doer`, closed enum).
- Gate: one more rule (rule 0), before the existing rules and the cap.
- Readers: unchanged. They read stored JSON and never look at `doer`; stored items are all Orion's now. Old rows without `doer` read as before.

## Files changed

- `orion/schemas/harness_finalize.py`: `PromiseDoer`, `PROMISE_DOERS`, `PromiseV1.doer`.
- `orion/harness/finalize.py`: `doer` in the judge's JSON schema.
- `orion/cognition/prompts/harness_finalize_reflect.j2`: the doer rule, with invented examples.
- `orion/harness/promise_gate.py`: rule 0, `dropped_not_orion`, lenient parse to `None`.
- `orion/harness/evals/promise_ledger_replay_eval.py`: Wilson intervals, pooled recall, `ablate_doer_rule`, `--skip-done`, merged scoring.
- `orion/harness/evals/judge_crash_replay_eval.py`: parses like production (promise fields apart).
- `orion/harness/evals/fixtures/promise_ledger_labels.json`: `key` on every real promise, 17 new labels.
- Tests: `orion/harness/tests/test_promise_gate.py`, `orion/harness/evals/test_promise_ledger_replay_eval.py`.
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: rule 0.

## Schema / bus / API changes

- Added: `PromiseV1.doer` (optional on the model, required in the judge's JSON schema).
- Removed / renamed: none.
- Behavior changed: the gate drops items whose doer is not `orion`.
- Compatibility: additive. No bus channel changed. No registry entry (the model is not a bus payload).

## Env/config changes

- None. No `.env_example` changed. The flag stays `ORION_PROMISE_LEDGER_ENABLED=true`.

## Tests run

```text
pytest orion/harness orion/prime/tests orion/evals/tests services/orion-cortex-exec/tests/test_finalize_reflect_request_shape.py tests/test_no_personal_terms.py
  1194 passed
pytest services/orion-durable-runs/tests/test_intention_carry.py   6 passed
pytest services/orion-mind/tests/test_prime_turn_handback.py      10 passed
pytest services/orion-hub/tests/test_intention_order.py           11 passed
mutation: doer check removed in promise_gate.py -> 7 tests fail
scripts/check_personal_terms.py   no personal terms in tracked files
CI on #3408: all checks green before the final commits
```

## Evals run

```text
promise_ledger_replay_eval run (leased, operator_agent), parts 1-7, --until 2026-11-12T03:15:00 --days 30: 123/185 turns
promise_ledger_replay_eval score over the merged parts vs /tmp/promise-ledger-replay/report.json (D0): see the table above
```

## Docker/build/smoke checks

```text
No build. The change reaches orion-harness-governor at the next post-pull rebuild after merge.
Leased smoke: 3 turns, 3 calls, see "Leased seat".
```

## Review findings fixed

- Finding: requiring `doer` adds a way for Strata to 502 the whole verdict.
  - Fix: kept required (the enum guides the model) and measured it.
  - Evidence: 2 schema 502s in the 109 calls that reached Strata (1.8%) against the D0 baseline of 5 in 180 (2.8%, unrelated JSON escapes). No sign the new key adds failures.
- Finding: a sentence with two subjects could lose a real promise.
  - Fix: prompt says to quote only Orion's clause, and that a promise waiting on Juniper is still Orion's.
- Finding: unlabelled items counted as false in precision.
  - Fix: precision counts labelled items only and reports `unlabelled` beside it.
- Finding: the ablation did not copy the gate's length and cap rules.
  - Fix: `ablate_doer_rule` runs the gate itself. Unit test added.
- Finding: the ablation is not the old prompt's precision.
  - Fix: said in the docstring and here.
- Finding: the judge-crash eval would count an unknown doer as a verdict crash.
  - Fix: it now parses like production.
- Nits: docstring "four rules", per-row `dropped_not_orion`, spec rule 0. Fixed.
- Second review (resume tooling): a later error row could overwrite a good row in `merge_reports`.
  - Fix: a row without an error always wins. Test added.
- Finding: `--out` equal to a `--skip-done` input would erase it.
  - Fix: refused at start.
- Finding: a merged report showed 0 promises counted.
  - Fix: `None` for the ledger fields on merged reports.
- Finding: a resumed part starts with no pending offers or open promises, so `offers_accepted` / `promises_kept` in a merged report are not comparable to an unbroken run.
  - Fix: documented on `merge_reports`. `promises_made` (what this PR measures) is not affected.
- Finding: the commit message for the resume commit says "9 recalls"; the pool shows 7.
  - Fix: corrected here (7 recalls, 9 grants).

## Restart required

```text
No restart required. After merge, the post-pull rebuild picks up orion-harness-governor (the 5b judge prompt and gate run there).
```

## Risks / concerns

- Severity: medium. Concern: precision is still below 0.8 (0.60 on 104 turns). Mitigation: nothing reads `promises_made` as an act yet beyond Prime's concerns; D1 stays closed. Next step is a design question for Juniper (#3464).
- Severity: medium. Concern: the eval is incomplete. 62 turns, including all 6 hand-found promises, and both held-out windows did not run, because the agent lane was busy with Orion's own work. Mitigation: `--skip-done` resumes it; exact command in #3464.
- Severity: low. Concern: 9 calls got `gpu_pool_unavailable` 503 during a recall and were recorded as errors, not recalls, so this run's resume did not retry them. Mitigation: `--skip-done` now retries error rows; the #3464 rerun picks them up.
- Severity: low. Concern: the new prompt lists more items before the rule (21 vs 12). Mitigation: the gate drops non-Orion ones before the cap.
- Issues opened: #3407 (dead queued hold gets granted), #3464 (finish the rerun; readiness class). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3408

🤖 Generated with [Claude Code](https://claude.com/claude-code)
