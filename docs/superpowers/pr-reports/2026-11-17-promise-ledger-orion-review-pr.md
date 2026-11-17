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
- `plan_promise_events` takes an optional new argument, `open_quotes`. It has 2 new counts, `items` and `repeats_open`.
- The report's JSON has 2 new fields, `expired_with_carry_reply_acts` and `guard.rate_wilson90`.

## Files changed

- `orion/grammar/promise_acts.py`: the `open_quotes` argument and the repeat skip, the `items` and `repeats_open` counts, and `PromiseLedgerPlan.outcome()`.
- `orion/hub/promise_ledger.py`: always logs `promise_view` and `promise_ledger`, and passes `open_quotes` through.
- `orion/hub/turn_orchestrator.py`: `_publish_promise_acts` logs `outcome=skipped reason=...` and passes the quotes of the open promises the judge was shown.
- `scripts/report_promise_ledger.py`: the frozen-baseline note, the guard interval, the raw counts on the keep-rate line, and the carry act ids.
- `orion/harness/promise_gate.py`: docstring for rule 0a (offers).
- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: offers wording, corrected counts, frozen baseline, and the 11-17 amendments.
- Tests: `orion/grammar/tests/test_promise_acts.py`, `services/orion-hub/tests/test_promise_ledger_hub.py`, `tests/test_report_promise_ledger.py`.

## Schema / bus / API changes

- Added: none on the bus. Only report JSON fields and log fields were added.
- Removed: none
- Renamed: none
- Behavior changed: a firm promise whose normalized quote equals an open promise the judge was shown opens no act.
- Compatibility notes: `open_quotes` defaults to empty, so existing callers behave as before.

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
  tests/test_report_promise_ledger.py                     -> 83 passed
pytest services/orion-hub/tests/test_promise_ledger_hub.py -> 19 passed
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
Live report (read-only): scripts/report_promise_ledger.py --days 1
  -> 0 promise acts, 0 firm promises over 2 judged Juniper turns.
```

## Review findings fixed

REVIEW_PLACEHOLDER

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

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
