# fix(promises): settle classes, not-due promises, offer window, stable refs, long quotes, go-live defaults

Juniper (2026-11-18): "we need to fix the promises Orion found." This is PR 2 of 2, the ledger side. It is
**stacked on PR #3697** (Prime side, branch `fix/promise-orion-findings-prime`): merge #3697 first. Both PRs edit
the promise report, so they were stacked rather than left to conflict.

## Summary

- A promise that can't be shown kept by a chat reply no longer counts as broken. Each promise the judge finds now
  says where its keep could be seen (`chat`, `run`, or `absence`) and what it waits on, if anything. At the 72 h
  deadline an unkept promise that was waiting on something that never happened, or that a run delivers, or that
  is an "I won't..." closes as "not graded" instead of a 0. (#3595)
- The judge can now say "the thing this promise was waiting for just happened" for an open conditional promise.
  Only then can that promise count as broken. (#3595, option b on the ticket)
- An offer ("want me to ...?") can be accepted only within 72 h, and only from Orion's previous reply. (#3595)
- A turn's stored promise list can no longer be swapped out by a re-published run record, so a promise's
  reference always points at the same words. (#3596)
- A real promise the judge quoted in full past 240 characters is now kept as its leading part instead of
  dropped. The replay case (turn d05f08e3, "I'll hold the deep read ...", 243 characters) now passes. (#3530)
- Two missing tests: the turn request carries the promise view only on Juniper's viewed turns, and the tool-retry
  path passes it on. (#3560)
- The report now splits "not graded" closes by kind, lists not-graded promises a carry run delivered later, and
  counts where Prime's promise concerns and the ledger's acts disagree. Spec amendments record every default
  and the #3531 go-live defaults. Go-live is NOT flipped; the open-promises cue stays off.

## Outcome moved

Before: a conditional, run-delivered or abstention promise could only close kept (by a quoting chat reply) or
expired 0, so at go-live it would drag the keep rate down for reasons that are not broken promises (#3595: about
14 of the 23 firm replay promises). After: those close censored with their reason, and the headline keep rate is
over the graded class only, with every other class printed beside it.

## Current architecture

- `orion/grammar/act_ledger.py` resolver: at the horizon, an unkept promise act closed expired 0 if Juniper wrote
  in the window, else `void:` (censored). It never read the promise itself.
- The judge (`orion/cognition/prompts/harness_finalize_reflect.j2`, schema in `orion/harness/finalize.py`) gave
  quote, offer, force, doer, act, done_when per promise.
- The offer view showed the previous reply's offers with no age limit.
- sql-writer upserted `harness_turn_trace.run_artifact` wholesale on conflict.
- The gate dropped any quote over 240 characters (`dropped_too_long`).

## Architecture touched

- Judge contract: `PromiseV1.settles_in`, `PromiseV1.condition`, `OpenPromiseRefV1.condition`,
  `FinalizeReflectionV1.promise_conditions_met` (all optional with defaults; the models ignore unknown keys).
- Resolver: reads the item behind each due promise act (one primary-key probe) and, for a conditional one, scans
  that act's 72 h window of judged turns for its id.
- sql-writer `harness_turn_trace` upsert (first stored `promises_made` wins).
- Hub promise view (offer TTL). Report. Spec.

## Files changed

- `orion/schemas/harness_finalize.py`: new optional fields and `PromiseSettlesIn`.
- `orion/cognition/prompts/harness_finalize_reflect.j2`, `orion/harness/finalize.py`: judge asked for the new
  fields; conditions-met ids survive the final-text re-gate.
- `orion/harness/promise_gate.py`: `leading_quote` (`trimmed_long`), `promise_conditions_met` gated to shown ids
  (`dropped_unknown_condition`), lenient parsing of the new fields.
- `orion/grammar/promise_acts.py`: void notes per kind, `horizon_void_note`, `settle_class`, `OFFER_ACCEPT_TTL_SEC`,
  docs on the offer counters.
- `orion/grammar/promise_view.py`: offer TTL (`within_ttl`), condition shown with each open promise.
- `orion/grammar/act_ledger.py`: settle-class closes in `resolve_due_acts`, counts in `ResolveSummary`.
- `services/orion-sql-writer/app/harness_turn_trace_persist.py`: first stored promise list wins.
- `scripts/report_promise_ledger.py`: void by kind, later carry deliveries, Prime vs act counts, headline label.
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: 2026-11-18 amendments.
- Tests: `orion/harness/tests/test_promise_gate.py`, `orion/harness/tests/test_finalize_tool_loop.py`,
  `orion/harness/tests/test_finalize_reflect_judge_contract.py`, `orion/grammar/tests/test_promise_acts.py`,
  `orion/harness/evals/test_promise_ledger_replay_eval.py`, `services/orion-hub/tests/test_promise_ledger_turn_gate.py`,
  `services/orion-sql-writer/tests/test_promise_acts_integration.py`, `tests/test_report_promise_ledger.py`.

## Schema / bus / API changes

- Added: `PromiseV1.settles_in`, `PromiseV1.condition`, `OpenPromiseRefV1.condition`,
  `FinalizeReflectionV1.promise_conditions_met`; void notes `void:not_due`, `void:not_chat`,
  `void:absence_unchecked`.
- Removed: none. Renamed: none. No bus channel or registry entry changes (same events, same payload models).
- Behavior changed: the judge's JSON schema now REQUIRES `settles_in` and `condition` per promise and
  `promise_conditions_met` on the reflection.
- Compatibility notes: every new field is optional on the pydantic models (default None / []), and none of the
  models is `extra="forbid"`, so an older Hub or sql-writer reading a newer artifact, or the reverse, does not
  fail. Rows stored before this read as `chat`, unconditional (the v1 rule). Every new void keeps the `void:`
  prefix, so every existing reader (self-monitor, act-ledger posterior skip, report) already treats it as censored.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Defaults applied that Juniper may want to reverse

1. **Conditional promises (option b):** the judge marks when a waited-on condition arose; a conditional promise
   whose condition no judged reply saw arise closes `void:not_due`, not 0. A condition seen only after the
   deadline still counts as not due. Reverse: drop the `condition` branch in `horizon_void_note`.
2. **Run-delivered promises close `void:not_chat` when unkept** (never 0). A carry run's later delivery is a
   report note, never a keep (#3595 rec 2: the act is the record of truth). Reverse: grade them, once a run-evidence
   keep exists.
3. **Abstentions close `void:absence_unchecked`** until a breach check exists (#3595 rec 1). Reverse: treat
   them as chat promises.
4. **Offer window = 72 h** (the same clock as the promise), on top of "only Orion's immediately previous reply".
   A miss shows as the gate's `dropped_unknown_offer`, or the ledger's `accepted_unmatched` (should stay 0).
5. **Stable refs by "first stored list wins"**, not by changing the ref format (Prime uses the same string).
6. **Long quotes kept as their leading part** (cut at a word boundary, at most 240 characters), only when the
   whole quote is literal in the reply.
7. **Headline keep rate = the graded class** (chat-showable, plus conditional ones whose condition arose), with
   each void kind and the 9/23 ceiling beside it. This replaces the 11-17 default "over all firm acts".
8. **#3531 go-live defaults recorded, not applied:** shadow ends 2027-01-12; only Juniper-browser promises
   count toward the gate; duplicates footnoted.
9. **Prime's concern does not close with the act yet** (in shadow the two may disagree; the report counts it).
   The go-live PR wires it.

## Tests run

```text
pytest orion/harness/tests orion/harness/evals/test_promise_ledger_replay_eval.py orion/grammar/tests
  orion/self_monitor tests/test_report_promise_ledger.py tests/test_juniper_browser_surface_rule.py
  orion/prime orion/mindloop                                                          all passed
(services/orion-sql-writer) pytest tests                                              1334 passed, 248 skipped
(throwaway postgres:16) pytest -m integration test_act_ledger_integration.py
  test_promise_acts_integration.py test_camera_ask_acts_integration.py               all passed, 0 skipped
(services/orion-hub) pytest tests/test_promise_ledger_hub.py tests/test_promise_ledger_turn_gate.py   36 passed
(services/orion-mind) pytest tests                                                    438 passed
(services/orion-thought) pytest tests/test_open_promises_cue.py                       5 passed
#3560 mutation check: dropping open_promises in the retry call, or the request's open_promises in the Hub,
  fails the new test each time.
python scripts/check_integration_seam.py --require-base                               OK
git diff --check                                                                      clean
```

## Evals run

```text
python services/orion-sql-writer/evals/promise_acts_mutations.py   all 14 mutants CAUGHT (4 new for this PR)
Gate replay on the real judge output for d05f08e3 (11-16 replay, dev2.part5) against its stored final_text:
  trimmed_long=1, dropped_too_long=0; kept quote 235 chars "I'll hold the deep read until the pipeline digests it ..."
promise_ledger_replay_eval score on the dev window: NOT re-run (needs a GPU lease for the judge).
Live report (read-only, 7 days): runs; 0 hub.promise acts exist, so every new line prints 0.
```

## Docker/build/smoke checks

```text
Not built here. Python/template changes copied into the images of orion-harness-governor (judge), orion-hub
(promise view), orion-sql-writer (resolver, upsert).
```

## Review findings fixed

Review ran in a subagent on `git diff origin/fix/promise-orion-findings-prime...HEAD`: no material findings
(SQL, timezones, ON CONFLICT, consumer compatibility, void readers and resolver all checked clean); 8 minor/nit,
all handled.

- Finding: a re-publish with no reflection got a synthesized reflection holding only the promise list (no
  `reflection_source`), skewing judged-turn counts.
  - Fix: such a re-publish keeps the stored reflection whole.
  - Evidence: `test_a_republished_run_artifact_keeps_the_first_stored_promises` (asserts the stored tone survives).
- Finding: first-stored-wins pins only the first run's refs.
  - Fix: limit written into the spec amendment and the persist docstring; the log line counts it if it happens.
  - Evidence: spec "Stable promise refs (#3596)".
- Finding: the judge's labels can shrink the graded set unseen.
  - Fix: the report prints the settle-class void share of closed acts.
  - Evidence: `test_close_summary_splits_void_by_settle_class_and_keeps_them_out_of_the_keep_rate`.
- Finding: view cap and read-error fallbacks bias conditional promises, uncounted.
  - Fix: `ResolveSummary.promise_item_unread` / `promise_condition_unread`; the cap bias written into the spec.
  - Evidence: `test_settle_class_read_errors_fall_back_and_are_counted`.
- Finding: a misleading test line; missing tests for the promising turn's own mark and read errors.
  - Fix: line removed; both tests added.
  - Evidence: `test_the_promising_turns_own_condition_mark_does_not_count` (+ its mutant caught).
- Finding (nit): "delivered later" label compared with open time; Prime-vs-act window mismatch.
  - Fix: labels say exactly what is compared.
  - Evidence: report text.
- Finding (nit): no mutants for the new branches.
  - Fix: 4 mutants added (conditional never not-due, own-turn mark counts, run promise graded 0, re-publish
    replaces the list).
  - Evidence: `promise_acts_mutations.py`: all 14 mutants CAUGHT.

## Restart required

Run from the clean primary `main` checkout after both PRs merge (the post-pull rebuild does this):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Order: sql-writer first (it reads the new fields as optional), then Hub, then the governor (the judge that starts
writing them).

## Risks / concerns

- Severity: medium. Concern: the judge must now emit three more required keys. More required keys can raise the
  judge-failure rate (`reflection_source = degraded_llm_failure_fallback`). Mitigation: the fields are short enums
  and a nullable string; watch the failure rate after the governor rebuild against the 1.6% replay rate (#3531).
- Severity: low. Concern: `settles_in` is the judge's call, one model, unmeasured. A promise wrongly labelled `run`
  closes uncounted instead of 0. Mitigation: the report prints each void kind; label precision belongs in the
  #3531 blind pass.
- Severity: low. Concern: UNVERIFIED live. 0 promise acts exist today, so no settle-class close has run on a real
  row. Check: the report's "void by kind" line at the first live promise act.
- Severity: low. Concern: whether any Juniper turn's corr ever gets a second `harness.run.v1` is UNVERIFIED (no
  store keeps bus history for that channel). The upsert now protects the list either way and logs
  `harness_turn_trace_promises_kept_first` if it ever happens.
- Issues: closes #3595, #3596, #3530, #3560. #3531 stays open (go-live criteria only recorded).

Fixes #3595
Fixes #3596
Fixes #3530
Fixes #3560
Refs #3531

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3700

🤖 Generated with [Claude Code](https://claude.com/claude-code)
