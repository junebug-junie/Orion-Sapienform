## Summary

- Orion's day letter is now tier 0. Juniper, 2026-11-12: "It needs to happen each day." Prime no longer bids for it, the market never prices or defers it, and the self-monitor gate is not asked. When yesterday closes with no letter, Prime opens the `make` act and orders the run on the same tick.
- A run that writes nothing (failed, cancelled, or completed with no row) is ordered again as the next attempt (`orion-day-<date>-2`, `-3`). The limit is 3 runs (`PRIME_DAY_LETTER_MAX_ATTEMPTS`), and all of them must finish inside the first order's 12 h deadline. One act covers the whole day. The day is "missed" only when every attempt fails or Hub refuses the order.
- The letter run asks the GPU pool for priority `system` instead of `background`. That puts it ahead of background work and behind Juniper's own chat (`interactive`) and perception.
- This reverses the #3224 answer that "a refused letter is acceptable".

## Outcome moved

On 11-11, the letter was lost in two places: a tier-2 bid deferred for 10 h, then a single failed run (E2BIG, fixed in #3248). Neither can end the day on its own now.

## Current architecture

Prime bid a `make` want (family `prime`, tier 2, priced at the turn ceiling). It re-bid inside an 18 h window, and ordered one run (`-1`) only when the bid was funded or a worth-it capacity deferral. Any terminal run closed the day. The run queued at GPU priority `background`.

## Architecture touched

- `orion/prime/day_letter.py`: removed bid parts (`want_id`, `letter_cost`, `deferred`, `window_over`, `REBID_MIN_SEC`, and the bid fields). Added `attempt` and `failed_attempts`, `can_retry`, `next_attempt`, and per-attempt run and correlation ids (attempt 1 keeps the old ids). A record saved by the old image still loads, because its bid fields are dropped on load.
- `services/orion-mind/app/prime_loop.py`: `_bid_letter` was replaced by `_open_letter` (no bid, no gate). `_settle_letter` orders the next attempt. `_close_letter` has no bid to release.
- `orion/orion_day/brief.py`: admission priority is now `system`.

## Files changed

- `orion/prime/day_letter.py`, `services/orion-mind/app/prime_loop.py`: tier 0 and retry.
- `orion/orion_day/brief.py`: GPU priority.
- `services/orion-mind/{settings.py,.env_example,docker-compose.yml,README.md}`: key swap and docs.
- Tests: `orion/prime/tests/test_day_letter.py`, `services/orion-mind/tests/test_prime_day_letter.py`, `orion/orion_day/tests/test_contracts.py`.

## Schema / bus / API changes

- Added: `PrimeDayLetterOpenV1.attempt` and `.failed_attempts` (Redis `orion:prime:day_letter`, Prime-internal).
- Removed: the `PrimeDayLetterOpenV1` bid fields (`bid_until`, `bids`, `next_bid_at`, `last_refusal`, `want_id`). They are dropped when an old record loads.
- Behavior changed: the durable request's `admission.priority` for `orion_day.letter` is now `system`.
- Compatibility: Hub's order API is unchanged. It already accepts any `orion-day-<date>-<n>` run id.

## Env/config changes

- Added keys: `PRIME_DAY_LETTER_MAX_ATTEMPTS=3`.
- Removed keys: `PRIME_DAY_LETTER_BID_WINDOW_HOURS`.
- `.env_example` updated: yes.
- Local `.env`: the key was swapped by hand in `services/orion-mind/.env`, because the sync script syncs its own checkout. A backup is in the session scratchpad. Both images run with either key, since each has a default.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/prime/tests orion/orion_day/tests -q     285 passed
pytest services/orion-mind/tests -q                   273 passed
pytest services/orion-durable-runs/tests -k 'orion_day or admission'   42 passed, 29 skipped (Postgres-only)
pyflakes (changed files)                              clean
check_env_template_parity.py                          PASS
```

New tests cover:
- a retry under the same act, then success;
- 3 failures, then missed and named in the next order;
- no retry past the deadline;
- a failed act open tried again with no bid;
- no bid and no gate ever;
- an old bidding-era record still loads.

## Evals run

No eval: this is control flow, and the runtime tests drive the full tick. The live check is the next real letter (UNVERIFIED until then).

## Docker/build/smoke checks

The post-pull rebuild covers orion-mind, plus Hub and durable-runs for `orion/orion_day`.

## Review findings fixed

- Finding (should-fix): a run that wrote the row and then ended failed or cancelled (the journal-publish retry, the driver deadline, an operator cancel) read as "nothing written". With retries, Hub then answers `already_written` naming our own run: the act closed at 0, the threads were lost, and the missed days that letter carried were reported again.
  - Fix: `run_outcome` treats `persist_outcome == "written"` as written in any terminal state. An `already_written` naming one of this day's own attempts (`own_run`) closes the day written, with that run's threads. `next_attempt` keeps `missed_sent`.
  - Evidence: `test_a_run_that_wrote_the_row_then_failed_is_written_and_not_retried`, `test_hub_naming_our_own_earlier_attempt_closes_the_day_written_and_tells_the_missed_days`, `test_a_row_written_before_the_run_failed_is_a_written_letter`.
- Finding: retrying `cancelled` would re-order a run someone cancelled on purpose.
  - Fix: only failed runs and completed-with-no-row runs are retried. Evidence: `test_a_cancelled_run_is_not_retried`, `test_a_completed_run_with_no_row_is_retried`.
- Finding: the 300-character reason cap could cut off the attempt history.
  - Fix: `attempts_note` puts the history first and applies it on every not-written close, including the deadline paths. Evidence: `test_the_attempt_history_leads_the_reason_so_the_cap_keeps_it`.
- Finding: stale text. Updated: orion-mind README log names, durable-runs README (no bid, priority `system`), the `orion_day_order.py` docstring, the `MissedLetterV1`/`OrionDayLetterOrderV1` docstrings, and the system-priority producer comments in `resource_admission.py` and `admission_runtime.py`.
- Finding: the local `services/orion-mind/.env` had a second copy of `PRIME_DAY_LETTER_BID_WINDOW_HOURS`. Fix: removed.
- Confirmed clean by review: no Hub, durable-runs or mailer code assumes run `-1`; the `orion_day_letter` row stays one per day (`ON CONFLICT DO NOTHING`); `system` stays preemptible and sheddable under heat, and the letter does not hold the one system driver slot while it queues.

## Restart required

```text
No manual restart: git pull rebuilds orion-mind / orion-hub.
```

## Risks / concerns

- Severity: low. Concern: the letter's make act no longer passes the self-monitor gate, so a broken `act.prime.make` faculty cannot stop it. Mitigation: this is intended for tier 0. The act is still graded.
- Severity: low. Concern: a deterministic failure (one that fails the same way every time) spends 3 runs. Mitigation: each run is bounded by its turn timeout and the shared deadline.
- Severity: low. Concern: `system` priority can delay other background runs on the agent lane by one letter turn per day. Mitigation: it is still below interactive chat.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
