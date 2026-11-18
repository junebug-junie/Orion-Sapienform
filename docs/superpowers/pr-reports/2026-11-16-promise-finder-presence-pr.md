## Summary

The promise finder (the 5b judge's PROMISES section) still listed lines that are not promises. After the doer rule (#3408), the biggest leftover kind was presence lines like "I'll be here when you wake": Orion is the subject, but nothing Orion does later could break the line. Juniper approved a second round on 2026-11-16.

- Each promise item now also carries **`act`** (what Orion will do) and **`done_when`** (what a later reply or run would show once it is done, written so it could also show it was not done). The judge writes `done_when` null when nothing could break the line. Code drops a null, missing or placeholder (< 12 chars) `done_when` (rule 0b).
- Each item also carries **`force`**, the line's speech act: `commits`, `reports`, `predicts`, `suggests`, `wishes`. Code keeps only `commits` (rule 0a). Offers count as `commits`, as before.
- No word list. The judge reads the line; code reads one closed enum and one null check. Both rules run before the 3-item cap, so a dropped line cannot crowd out a real promise.
- The replay eval records each rule's drops, ablates each rule, and matches hand-found promises by key (a different span of the same promise now counts).
- `done_when` gives D1 the thing to check when it settles a promise.

## Why this should generalize

Both rules come from Searle's conditions for a promise, not from the false lines we saw:
- A promise is a **commissive**: it binds the speaker to a future act. Reports, predictions, suggestions and wishes have other illocutionary points (`force`).
- Its **preparatory condition**: the act is not one the speaker would do anyway in the normal course of events. "I'll be here when you wake" fails it. Orion is here anyway, so no later turn could show it broken (`done_when` null).

The prompt examples are invented. The reviewer checked: none of the 7 example phrases appear in 90 days of `harness_turn_trace`.

## Outcome moved

Precision rose on both the dev window and the held-out windows. It is at or above 0.8 as a point estimate in both. The held-out interval alone cannot yet rule out 0.6.

Replay, same model and lane as live (Flash-Next, leased operator seat, temperature 0.7). "Before" is the current main prompt (doer rule). "After" is this branch. Precision counts labelled items. Pooled recall counts real promises that any labelled run found, by promise.

| window | measure | before (main) | after (this PR) |
|---|---|---|---|
| dev, 10-13..11-12 (185 turns) | precision | 24/39 = 0.62 (95% Wilson 0.46-0.75) | **26/31 = 0.84 (0.67-0.93)** |
| dev | pooled recall | 23/34 = 0.68 (0.51-0.81) | 25/34 = 0.74 (0.57-0.85) |
| dev | the design's 6 hand-found promises | 3/6 | 2/6 |
| held-out, 09-13..10-13 (87 turns) + 11-12..11-16 03:15 (24 turns) | precision | 7/14 = 0.50 (0.27-0.73) | **10/12 = 0.83 (0.55-0.95)** |
| held-out | pooled recall | 7/10 = 0.70 (0.40-0.89) | 10/10 = 1.00 (0.72-1.00) |
| all 296 turns | precision | 31/53 = 0.58 (0.45-0.71) | **36/43 = 0.84 (0.70-0.92)** |
| all | pooled recall | 30/44 = 0.68 (0.53-0.80) | 35/44 = 0.80 (0.65-0.89) |

How to read it:
- The before and after intervals no longer overlap on all 296 turns: 0.45-0.71 vs 0.70-0.92, touching only at the edge. On held-out alone (12 items), the after interval still reaches down to 0.55.
- Most of the gain comes from the judge listing fewer non-promises, not from code drops. On the held-out run the code dropped 4 items (force 3, done_when 1), all false. On dev it dropped 3. Asking for `force` and `done_when` makes the judge think about each line before it lists it.
- Rule drops against labels: on held-out, 0 real promises lost. On dev, rule 0b dropped 1 real promise (a conditional joint plan given a null `done_when`); a second sample of the same turn kept it. That is sampling noise at temperature 0.7, and it can cost a real promise.
- Hand-found recall 3/6 to 2/6 is one promise: `hold-deep-read` was quoted at 330 chars, and the existing 240-char quote cap dropped it (filed #3530). With 6 items this number moves by noise.
- Intermediate dev step (decided on dev, not shipped alone): `done_when` alone gave 20/28 = 0.71 (0.53-0.85). The leftover false items were a recommendation, a wish, a hypothetical, a report of a done act, and a hand-off. That is why `force` was added. The rule was then frozen before any held-out "after" run.

### Is the 0.8 bar met (may D1 open)?

On the point estimate, yes: 0.84 dev, 0.83 held-out, 0.84 overall. On held-out alone the interval cannot yet tell 0.8 from 0.6 (12 items, lower bound 0.55). Telling them apart (interval half-width about 0.1) needs about 60 labelled items. At the after run's held-out rate (about 1 item per 9 turns) that is about 550 held-out turns, and only 111 exist after a stance sighting in these windows. So I stopped at the honest number. My recommendation: D1 may open in shadow, and #3531 measures live precision on the first ~60 stored items. That is Juniper's call.

### Labelling

94 new items were hand-labelled with the same rule as before: a real forward commitment by Orion, offers included; presence/readiness lines with no deliverable are false. Labels are hashes only. I saw `force`/`done_when` on the after items while labelling, which is a possible bias. Borderline calls say "borderline" in `why`: 5 of them, all labelled true (a worktree kept until Juniper decides, a queued curiosity question, "I won't float anything back up while you're asleep", "I'll stop muttering about bus error", a conditional work arrangement). Two of them are in the after list on dev and two on held-out. If all 5 were false, after precision would be 24/31 = 0.77 on dev and 8/12 = 0.67 on held-out, and before would be 23/39 = 0.59 and 6/14 = 0.43. The held-out number depends on these calls.

## Current architecture

The 5b finalize judge returns `promises_made: [{quote, offer, doer}]` when `ORION_PROMISE_LEDGER_ENABLED` is on. `orion/harness/promise_gate.py` keeps `doer == "orion"`, then quotes that are literally in the draft and the final text, up to 3. The gated list is stored in `harness_turn_trace.run_artifact.reflection.promises_made`. Prime's promise concerns read it (`orion/prime/intention_sources/turn_promise.py`, `orion/prime/handback.py`).

## Architecture touched

- Judge contract: three more required keys per promise item (`force` closed enum, `act` string, `done_when` string or null).
- Gate: rules 0a (force) and 0b (done_when), after the doer rule and before the quote/length/cap rules. Missing or unknown values are dropped (fail closed). `act`/`done_when` are cut to 240 chars at parse, because they are stored.
- Readers: unchanged. They read stored JSON and use only `quote`/`offer`. Old rows without the new keys read as before. Nothing re-gates stored rows.

## Files changed

- `orion/schemas/harness_finalize.py`: `PromiseForce`, `PROMISE_FORCES`, `PROMISE_FORCE_COMMITS`, `PROMISE_DONE_WHEN_MIN_CHARS`; `PromiseV1.force/act/done_when`.
- `orion/harness/finalize.py`: the new keys in the judge's JSON schema.
- `orion/cognition/prompts/harness_finalize_reflect.j2`: force, act, done_when, with invented examples.
- `orion/harness/promise_gate.py`: rules 0a/0b, `dropped_not_commit`, `dropped_no_completion`, `has_completion`, lenient parse.
- `orion/harness/evals/promise_ledger_replay_eval.py`: per-rule drops, `ablate_force_rule`, `ablate_done_when_rule`, hand-found recall by key, drop lists deduped.
- `orion/harness/evals/fixtures/promise_ledger_labels.json`: 94 new labels (hashes).
- Tests: `orion/harness/tests/test_promise_gate.py`, `orion/harness/evals/test_promise_ledger_replay_eval.py`.
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: completion and force bullets.

## Schema / bus / API changes

- Added: `PromiseV1.force`, `PromiseV1.act`, `PromiseV1.done_when` (optional on the model, required in the judge's JSON schema).
- Removed / renamed: none.
- Behavior changed: the gate drops items that do not commit or have no completion that could fail.
- Compatibility: additive. No bus channel. No registry entry (not a bus payload). `PromiseV1` has no `extra="forbid"` reader.

## Env/config changes

- None. No `.env_example` changed. The flag stays `ORION_PROMISE_LEDGER_ENABLED=true`.

## Tests run

```text
pytest orion/harness orion/prime/tests orion/evals/tests services/orion-cortex-exec/tests/test_finalize_reflect_request_shape.py tests/test_no_personal_terms.py
  1229 passed
pytest services/orion-durable-runs/tests/test_intention_carry.py   6 passed
pytest services/orion-mind/tests/test_prime_turn_handback.py      10 passed
pytest services/orion-hub/tests/test_intention_order.py           11 passed
scripts/check_personal_terms.py   no personal terms in tracked files
orion-static-gates.yml run steps, locally: all pass (on the pre-rebase base, one gate failed because main lacked #3493; it passes after the rebase)
mutation: rule 0b removed -> 7 tests fail; rule 0a removed -> the force test fails (reviewer-checked); 0b moved after the cap -> the cap test fails
New tests: a presence line dropped, a real promise kept (with act/done_when stored), a forecast and an ask still dropped; a report, a suggestion and a wish dropped; missing/unknown force and missing/placeholder done_when fail closed; schema requires the keys and rejects a bad force.
```

## Evals run

```text
promise_ledger_replay_eval run (leased operator_agent seat), resumed with --skip-done through pool recalls:
  dev 10-13..11-12   after (done_when only, intermediate)  185 turns, 180 judged
  dev                after (this PR)                       185 turns, 179 judged, 1 error
  dev                before (main), merged with the 11-15 doer-run parts, filling the 62 missing turns
  held-out 09-13..10-13 and 11-12..11-16 03:15   before and after, 111 turns each, 110 judged
promise_ledger_replay_eval score per window: see the table above
Reports are local only: /tmp/promise-ledger-replay-v3/
```

Judge failure rate with three more required keys: Strata `structured_output_failed` 502s were 4 of 253 calls after (1.6%) and 3 of 186 before (1.6%). No rise. Two other errors were `recommended_tool_reason` over 300 chars (unrelated).

## Docker/build/smoke checks

```text
No build. The change reaches orion-harness-governor at the next post-pull rebuild after merge.
Leased smoke: 4 turns on the agent lane, granted at once; the judge returned force/act/done_when in valid JSON.
```

## Review findings fixed

The code review subagent found no material issues. Fixed:
- Finding: the doer-enum negative schema test failed on the missing new keys, not on the enum. There was no bad-force schema test.
  - Fix: the item now has all keys. Added a `force: "promise"` case that must fail.
  - Evidence: `test_schema_requires_a_force_from_the_closed_set_and_the_prompt_defines_each`.
- Finding: the ablations reported the last copy of a duplicated quote, but the gate keeps the first.
  - Fix: first copy wins.
  - Evidence: `test_ablation_reports_the_first_copy_of_a_duplicated_quote`.
- Finding: the doer ablation now runs rules 0a/0b, so its number is not comparable to #3408's. Its items also lacked force/act/done_when.
  - Fix: docstring note, and the fields added.
- Finding: the spec did not describe `force`.
  - Fix: spec bullet added.
- Finding: `act`/`done_when` are stored with no length cap.
  - Fix: cut to 240 at parse, not in the schema (a schema cap would 502 the verdict).
  - Evidence: `test_act_and_done_when_are_cut_to_the_quote_cap`.
- Finding: the per-rule drop lists were not deduped per turn and quote.
  - Fix: deduped.
- Finding: `_opt_str` precedence was hard to read.
  - Fix: parenthesized.
- Finding: three more required keys could raise Strata 502s.
  - Fix: measured. No rise (see Evals).
- Not changed: an optional prompt lead-in clause. Changing the prompt would invalidate the held-out runs, and "when you are unsure, list it" already covers it.

## Restart required

```text
No restart required. After merge, the post-pull rebuild picks up orion-harness-governor (the 5b judge prompt and gate run there).
```

## Risks / concerns

- Severity: medium. Concern: held-out precision 10/12 cannot yet tell 0.8 from 0.6. Mitigation: #3531 measures live precision on ~60 stored items before D1 bets count.
- Severity: low. Concern: the judge samples at temperature 0.7, so a real promise is sometimes given null `done_when` or not listed. Seen once on dev. Mitigation: recall did not fall (0.68 to 0.80 pooled), and ablations are recorded per run.
- Severity: low. Concern: the 240-char quote cap drops long real promises. Mitigation: #3530.
- Severity: low. Concern: labeller bias (the after items showed force/done_when). Mitigation: borderline calls are marked; worst case stated above.
- Issues opened: #3530 (quote cap drops real promises), #3531 (verify live precision). Issues closed: #3464 (the doer rerun is finished on all 185 turns, and the readiness class is built).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3532

Fixes #3464

🤖 Generated with [Claude Code](https://claude.com/claude-code)
