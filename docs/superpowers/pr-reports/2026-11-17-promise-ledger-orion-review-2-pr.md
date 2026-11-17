## Summary

Orion asked a third set of questions on three session threads (PRs #3561, #3598, #3532). This PR fixes the small, clear findings. The rest went into #3531, #3595 and #3602 for Juniper.

- The self-reinforcement guard cannot go red before 60 judged turns. Before this, 3 firm promises in 10 turns could read red.
- The guard counts only turns sent from a Hub browser surface. A coding-session message posted to the Hub API is not Juniper's turn (#3602).
- The report names a keep-rate ceiling (the chat-showable share, 9/23), lists duplicate-promise candidates, and prints accepted offers with `accepted_unmatched`.
- The blinded labelling sheet shows Juniper's message that the reply answered. It still hides the judge's reading.
- The quote match removes markdown asterisks and zero-width characters, and folds two more apostrophe forms. Replay turn 868ed31f lost a real line to `*...*`.
- A test pins that Prime never reads the promise acts while they are in shadow.

## Outcome moved

- Before 60 judged turns, the guard prints "not armed" and never shows red on noise. At 60 turns, red needs 10 or more firm promises, which noise alone gives about 2.4% of the time (Poisson). At 50 turns it would need 8 (about 5%), and at 20 turns 4 (about 8%).
- Agent-posted turns no longer count toward the guard. The frozen baseline holds 2 of them (2 of its 23 firm promises). Clean, it would be 21/287 = 0.073 instead of 0.080. It stays frozen; re-freezing it is Juniper's call.
- A real promise wrapped in markdown emphasis is no longer refused.

## Current architecture

`scripts/report_promise_ledger.py` reads `grammar_acts`, `harness_turn_trace` and `chat_history_log`, read-only. Its guard counted every judged Juniper turn under `JUNIPER_TURN_PREDICATE` from `FINDER_LIVE_AT`, with no minimum n. `normalize_for_quote_match` (`orion/harness/promise_gate.py`) folded curly quotes, dashes, ellipsis and backticks. Everything is in shadow until #3531.

## Architecture touched

- Report only, plus one normalization function in the promise gate. No bus, schema or env change.
- The report's new JSON fields: `guard.armed`, `guard.min_judged_turns`, `guard.turns_not_from_browser`, `ledger.keep_rate_ceiling`, `offers`, `duplicate_candidates`. The sheet adds `juniper_said`.

## Files changed

- `scripts/report_promise_ledger.py`: the guard arms at 60 and counts browser turns only, plus the ceiling, `duplicate_candidates`, `offers_summary` (against the previous chat row, the Hub's rule) and `juniper_said`. Both predicates are now column-qualified by `qualify()`, which fails loudly on a mismatch.
- `orion/harness/promise_gate.py`: the quote normalization (asterisks, zero-width characters, U+02BC/U+FF07/U+FF02).
- `orion/harness/evals/fixtures/promise_ledger_labels.json`: 3 label hashes re-keyed for the asterisk change (same labels, same quotes).
- `orion/grammar/tests/test_promise_acts.py`: the test that Prime never reads the promise acts.
- `orion/harness/tests/test_promise_gate.py`, `tests/test_report_promise_ledger.py`: new tests.
- `.github/workflows/promise-ledger-tests.yml`: the workflow now also runs on `orion/prime/**` and `services/orion-mind/app/**`, the two trees the new test scans.
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: the round-2 amendments.

## Schema / bus / API changes

- Added: none on the bus. Only report JSON fields were added.
- Removed: none
- Renamed: none
- Behavior changed: the quote match also ignores `*` and zero-width characters, and folds ʼ ＇ ＂. A promise whose quote differs from the reply only by these now passes the gate.
- Compatibility notes: any committed label keyed on a quote containing `*` changes its hash. Exactly 3 exist, and they are re-keyed.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/harness/tests/test_promise_gate.py orion/harness/evals/test_promise_ledger_replay_eval.py -> 68 passed
pytest orion/grammar/tests/test_promise_acts.py orion/self_monitor/tests/test_self_monitor_promise_shadow.py tests/test_report_promise_ledger.py -> 35 passed
pytest services/orion-hub/tests/test_promise_ledger_hub.py -> 15 passed
cd services/orion-thought && pytest tests/test_open_promises_cue.py -> 5 passed
pytest orion/harness orion/prime/tests -> 1230 passed
pytest tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py tests/test_grammar_event_producer_catalog.py -> 13 passed
static: git diff --check, check_scripts_dir_no_stdlib_shadow, check_metric_lineage --gate, check_personal_terms,
        check_env_key_single_source, check_definition_drift --gate -> all pass
mutation: '# hub.promise' added to orion/prime/tending.py -> the Prime test fails; the asterisk strip removed -> the 868ed31f case fails;
          zero-width mapped to deletion again -> the space-substitute case fails
```

## Evals run

```text
No live eval was rerun. The finder is unchanged, and only the quote match gets wider.
Replay data (read-only, /tmp/promise-ledger-replay-v3): baseline reproduced at 23/289 (dev 15/179, held-out 0/86 and 8/24).
Duplicate check on the 23: 0 pairs at 24 shared characters (1 spurious pair at 12, which is why the bar is 24).
Quote-refusal scan of every replay report: 4 refusals. 2 are correct, because repair removed the line. 1 was asterisks (fixed here). 1 was backticks (already fixed on main).
```

## Docker/build/smoke checks

```text
No build. The report is a script; the gate change reaches orion-harness-governor at its next post-pull rebuild.
Live report (read-only): scripts/report_promise_ledger.py --days 28
  -> 0 promise acts; ceiling 0.39 (9/23); guard 0 firm over 4 judged turns, not armed (needs 60); runs in ~1 s;
     turns not from a Hub browser surface left out 0; offers accepted 0; duplicate candidates 0
```

## Review findings fixed

A subagent reviewed commit b177ff9e2. It found no must-fix issues.

- Finding (should): the comment and spec said red at 50 turns needs 9 firm promises, about 2% on noise. It actually needs 8, about 5%.
  - Fix: the guard arms at 60 turns, where red needs 10 promises, about 2.4% on noise. The comment, the spec and the #3531 comment are corrected.
  - Evidence: `test_guard_red_boundary_at_the_arming_point` (9 firm promises are not red, 10 are red).
- Finding (should): the offer and duplicate lines still counted coding-session turns.
  - Fix: both now use the same browser-only turns as the guard.
  - Evidence: `test_report_leaves_session_turns_out_of_offers_and_duplicates`.
- Finding (should): `offers_summary` compared against the previous Juniper turn. The Hub compares against the previous chat row of any kind.
  - Fix: the SQL selects the previous row the Hub's way (LATERAL, same order). A previous row that is not Juniper's is counted apart.
  - Evidence: the `c5` case in `test_offers_summary_counts_accepted_and_unmatched_against_the_previous_turn`.
- Finding (should): `.replace(" AND ", " AND k.")` was a fragile way to qualify columns.
  - Fix: `qualify()` matches whole words, skips quoted strings, and raises if a column is missing.
  - Evidence: `test_judged_sql_qualifies_both_predicates_by_alias`.
- Finding (nit): a zero-width space was deleted. When it stands in for a real space, deleting it joins two words.
  - Fix: it now maps to a space.
  - Evidence: a new parametrize case, mutation-checked.
- Finding (nit): a duplicate-check item with no timestamp paired with every other item.
  - Fix: it is skipped. Sorted items also stop the inner loop early.
  - Evidence: `test_duplicate_candidates_skip_items_with_no_time`.
- Finding (nit): the "~40 a week" claim had no evidence attached.
  - Fix: the spec cites 26-56 a week (10-05..11-15) and the query behind it.
- Finding (nit): the blinded-sheet key changes for quotes that contain `*`.
  - Fix: one spec line. Scoring does not recompute keys, so it is unaffected.
- Not changed (nit): the static Prime test matches substrings. It is a tripwire, and it is mutation-checked.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build orion-harness-governor
```

Run it from the clean primary main checkout after merge. The post-pull rebuild does the same thing. The report needs no restart.

## Risks / concerns

- Severity: low. Concern: the browser filter depends on the Hub client's `surface_context`. A known false positive is `scripts/trace_hub_skill_runner_e2e.py`, which sends it. Mitigation: `turns_not_from_browser` is printed, so a drop shows.
- Severity: low. Concern: the keep-rate ceiling is a constant from one reader's hand count, and it was not blind. Mitigation: the report labels it that way. #3595 replaces it with a judge field.
- Severity: low. Concern: the frozen baseline holds 2 agent turns. Mitigation: documented. Re-freezing is Juniper's call (#3531).
- UNVERIFIED: the per-turn `promise_ledger` line from #3598. No Juniper turn has happened since the 09:13Z rebuild.
- Issues opened: none.
- Issues updated: #3531 (guard power, one blind pass, labeller rule, official number and the 0.65 bound), #3595 (void promises delivered unseen), #3602 (session-turn count and baseline contamination).
- Issues closed: none.
- Waiting on Juniper: the 4 #3561 defaults, #3531, #3595.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3619

🤖 Generated with [Claude Code](https://claude.com/claude-code)
