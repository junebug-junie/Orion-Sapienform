## Summary

Orion reviewed PR #3532 (promise finder) and PR #3561 (promise ledger in shadow) and asked 14 questions. This PR fixes the small, clear findings. The rest went to tickets, and Orion got an answer on both threads.

- Hub now logs one `promise_ledger` line on every Juniper turn the ledger sees. Before, it logged only when a count was not zero, so "found nothing" and "never ran" looked the same.
- If Orion says an open promise again, word for word, it no longer opens a second act. (A reworded repeat still does.)
- The report now prints the kept, expired, void and un-closeable counts next to the keep rate. It lists the act ids of expired promises that got a carry reply. It also prints a 90% interval on the guard rate.
- The guard baseline (23/289) is frozen for the whole shadow period, and a test pins it.
- Spec fixes: the offers wording (an offer opens no act until Juniper accepts it), the recall and precision counts (both were 2x to 6x too low), and a 2026-11-17 amendments section.

## Outcome moved

- From the logs alone, you can now tell whether the ledger ran on a turn. Live evidence that this was needed: the 2 Juniper turns since #3561 deployed (06:34Z and 06:48Z on 11-17) left no `promise_ledger` line and no `promise_view` line.
- The go-live criteria in #3531 now use counts that can actually be reached.

## Current architecture

`orion/hub/promise_ledger.py` reads the open promises before each Juniper turn and publishes the promise acts after delivery. `plan_promise_events` (`orion/grammar/promise_acts.py`) decides what opens and what closes. `scripts/report_promise_ledger.py` reads `grammar_acts`, `harness_turn_trace` and `chat_history_log`. Everything is in shadow until #3531.

## Architecture touched

- Hub log lines: `promise_view` (logs every read) and `promise_ledger` (one per turn, with an `outcome=` word).
- `plan_promise_events` takes an optional new argument, `open_promises` (pairs of act id and quote). It has 2 new counts, `items` and `repeats_open`; the matched act ids are logged.
- The report's JSON has new fields: `expired_with_carry_reply_acts`, `guard.rate_poisson90` and `guard.turns_before_finder`.

## Files changed

- `orion/grammar/promise_acts.py`: the `open_promises` argument and the repeat skip (only for quotes of 40 characters or more, `REPEAT_MIN_CHARS`), the `items` and `repeats_open` counts, and `PromiseLedgerPlan.outcome()`.
- `orion/hub/promise_ledger.py`: always logs `promise_view` (including `outcome=unread`) and `promise_ledger`, and passes `open_promises` through.
- `orion/hub/turn_orchestrator.py`: `_publish_promise_acts` logs `outcome=skipped reason=...` and passes the quotes of the open promises the judge was shown.
- `scripts/report_promise_ledger.py`: the frozen-baseline note, a 90% Poisson interval on the guard rate, the guard now counting only turns from when the #3532 finder went live (`FINDER_LIVE_AT`, 2026-11-16 18:05Z), the raw counts on the keep-rate line, and the carry act ids (capped at 10).
- `orion/harness/promise_gate.py`: docstring for rule 0a (offers).
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: offers wording, corrected counts, frozen baseline, and the 11-17 amendments.
- Tests: `orion/grammar/tests/test_promise_acts.py`, `services/orion-hub/tests/test_promise_ledger_hub.py`, `tests/test_report_promise_ledger.py`.

## Schema / bus / API changes

- Added: none on the bus. Only report JSON fields and log fields were added.
- Removed: none
- Renamed: none
- Behavior changed: a firm promise whose normalized quote equals an open promise the judge was shown opens no act.
- Compatibility notes: `open_promises` defaults to empty, so existing callers behave as before. The guard's denominator changed: it now leaves out turns from before the finder went live (172 of them in the 28-day window). Those turns pulled the live rate toward 0.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/harness/tests/test_promise_gate.py orion/harness/evals/test_promise_ledger_replay_eval.py \
  orion/grammar/tests/test_promise_acts.py orion/self_monitor/tests/test_self_monitor_promise_shadow.py \
  tests/test_report_promise_ledger.py services/orion-hub/tests/test_promise_ledger_hub.py -> 102 passed
pytest tests/test_no_personal_terms.py tests/test_grammar_event_producer_catalog.py \
  tests/test_no_private_eval_artifacts.py                 -> 13 passed
```

## Evals run

```text
orion/harness/evals/test_promise_ledger_replay_eval.py (in the run above): passed.
No live eval was rerun. The live finder output is unchanged by this PR.
```

## Docker/build/smoke checks

```text
No Docker build was run. The change is Python only, and the Hub image picks it up at its next rebuild.
Live report (read-only): scripts/report_promise_ledger.py --days 28
  -> 0 promise acts; guard: 0 firm promises over 2 judged Juniper turns (90% 0.0-1.353);
     172 turns from before the finder went live left out; 12 pre-#3532 items skipped.
```

## Review findings fixed

A subagent ran the code review on the branch diff.

- Finding (must): the guard interval used Wilson. The guard rate is a count rate, so with more firm promises than turns it took the square root of a negative number and crashed the report.
  - Fix: a Poisson score interval (`poisson_rate_interval`). The field is renamed `rate_poisson90`.
  - Evidence: `test_guard_interval_handles_more_promises_than_turns` (2 promises over 1 turn).
- Finding (should): a view that came back `None` (no `POSTGRES_URI`) logged no `promise_view` line.
  - Fix: it now logs `promise_view corr=... outcome=unread`.
  - Evidence: `test_an_unread_view_still_logs_a_promise_view_line`.
- Finding (should): the repeat skip could swallow a short, generic promise made on a new topic, and the log did not say which act it matched.
  - Fix: a 40-character floor (`REPEAT_MIN_CHARS`), and the matched act ids go in the log (`repeats_open_acts=`).
  - Evidence: `test_a_short_generic_line_said_again_is_a_new_promise`.
- Finding (nit): the spec's list of skip reasons left some out.
  - Fix: the spec now names `no_bus` and `no_view` as defensive reasons.
- Finding (nit): a test changed its shared parametrize dict.
  - Fix: the test copies the dict first.
- Finding (nit): the list of carry act ids had no cap, and nothing tested the text output.
  - Fix: the list stops at 10 ids and adds "+N more".
  - Evidence: `test_carry_act_ids_are_capped_in_the_text_report`, plus a check of the text line.
- Finding (nit): the `outcome()` docstring did not say that a keep-only turn counts as `acted`.
  - Fix: the docstring now says so.
- Found by me during the live check, after the review: the guard's denominator held 172 judged turns from before the #3532 finder went live. None of those turns could store a firm item.
  - Fix: `FINDER_LIVE_AT` (the governor rebuild at head d198f79ed, 2026-11-16 18:05Z, `~/.orion/post_pull_rebuild.log`).
  - Evidence: `test_guard_leaves_out_turns_before_the_finder_went_live`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build orion-hub
```

Run it from the clean primary main checkout after merge. The post-pull rebuild does the same thing.

## Risks / concerns

- Severity: low. Concern: one more INFO line per Juniper turn (about 6 a day). Mitigation: none needed.
- Severity: low. Concern: only word-for-word repeats are caught, so a reworded repeat still opens a second act. Mitigation: documented in the spec amendment and in #3531.
- Issues opened: #3595 (conditional promises and promises a chat reply cannot show kept; design, needs-juniper) and #3596 (the promise index can drift if a run artifact is re-published).
- Issues updated: #3531 (corrected go-live criteria; `needs-juniper` added).
- Issues closed: none.
- Juniper's 4 proposed defaults for #3561 are still unanswered. Nothing in this PR assumes them.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3598

🤖 Generated with [Claude Code](https://claude.com/claude-code)
