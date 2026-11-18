# PR report: promise ledger report shows lanes, the browser window and Prime by exact key

## Summary

All of the finder's live promise items so far come from lanes that can never open a ledger act. Since the finder went live (2026-11-16 18:05Z), 18 turns stored 19 promise items, and none of those turns came from Juniper's browser. On her lane, 6 judged turns stored 0 items. At the frozen rate, 6 turns expect 0.48 firm promises, and a count of 0 happens 62% of the time. So this is low traffic on her lane, not a blind finder. The report now shows all of this:

- **Browser window.** It prints Juniper's browser turns since the finder went live, plus the last one's correlation id and age. A stretch with no `promise_view`/`promise_ledger` log line then reads as "she sent no browser turn", not as "the ledger was quiet".
- **Lane split.** It lists every turn that stored promise items, by lane: browser versus each other session.
- **Expected versus observed.** It prints judged browser turns × 0.080 beside the observed count, with the chance of a count that low and of a count that high.
- **Void share.** It prints the void share of closed acts and says what void is computed from.
- **Prime by exact key.** For each item-bearing turn it shows Prime's concern, found by `turn_concern_id(corr)`: open, closed, refused, aged out, or never handed back. It also shows Prime's `classify_who` label and the `intention.carry` replies on the same key.

## Outcome moved

Before this patch, a reader could misread a zero two ways: as "the ledger was quiet", or as "the finder cannot see Juniper". Now the report prints the cause of each zero.

## Current architecture

`scripts/report_promise_ledger.py` read the promise acts, Juniper's judged chat turns, and carry rows. Its guard counted only browser turns. It never looked at item-bearing turns on other lanes or at Prime state. `expired_with_carry_reply` already joined on the exact concern id (`turn_concern_id(corr)`) inside the act's window.

## Architecture touched

This is a read-only report. It adds one Postgres query (`STORED_SQL`). That query looks up `cockpit_turn_sighting` by primary key only, and runs in 1.3 s live. It also adds an optional Prime read: from Redis when the `redis` package is installed, otherwise from `--prime-state`/`--prime-turns` files. No service, schema, bus or env change.

## Files changed

- `scripts/report_promise_ledger.py`: the new lines and functions (`lane_split`, `browser_window`, `poisson_cdf`, `prime_by_ref`, `read_prime`), plus the void share.
- `tests/test_report_promise_ledger.py`: 9 new tests; the module never touches Redis.
- `docs/superpowers/pr-reports/2026-11-18-promise-ledger-report-lanes-pr.md`: this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the report prints more lines and adds JSON keys: `browser_window`, `lanes`, `prime`, `guard.expected_firm`, `guard.p_at_most_observed`, `guard.p_at_least_observed`, `ledger.void_share`.
- Compatibility notes: none.

## Env/config changes

- None. The report reads the existing `ORION_BUS_URL` only when the `redis` package is present.
- local `.env` synced: not needed (no template changed).

## Tests run

```text
pytest tests/test_report_promise_ledger.py -q                      32 passed
pytest tests/test_no_personal_terms.py orion/grammar/tests/test_promise_acts.py \
  orion/self_monitor/tests/test_self_monitor_promise_shadow.py orion/prime -q   323 passed
python scripts/check_scripts_dir_no_stdlib_shadow.py               clean
python scripts/check_metric_lineage.py --gate                      PASS
git diff --check                                                   clean
```

## Evals run

```text
None. This is a report script. Its "eval" is the live run below.
```

## Docker/build/smoke checks

Live run, read-only, against conjourney on 2026-11-18 at about 00:35Z:

```text
expected vs observed: 0.48 firm promises expected over 6 judged browser turns at the frozen rate 0.0796; observed 0; P(a count this low by chance) 0.62, P(a count this high by chance) 1.0
browser window: 7 Juniper browser turns since 2026-11-16 18:05Z (6 judged); last 2026-11-17T18:12:33+00:00 (e3e20e97), 6.4 h before this report. ...
stored promise items since 2026-11-16 18:05Z: 19 on 18 turns, 0 of them Juniper browser turns ...; by lane: memory-scoreboard-eval 11t/12i/2f, orion_curiosity 4t/4i/4f, self-sense-eval 3t/3i/0f
Prime by exact key (turn_concern_id) on 6 item-bearing turns with a non-offer promise [...]: {'no_concern_or_aged_out': 4, 'refused': 2}; intention.carry replies on those keys 0 (... d391d241 memory-scoreboard-eval who=chat_juniper refused, ...)
```

## Review findings fixed

- Finding: `jsonb_typeof = 'array'` did not guard `jsonb_array_length`, because Postgres can evaluate the AND terms in any order. One non-array row would abort the report.
  - Fix: filter with `promises_made @> '[{}]'` instead.
  - Evidence: the live run passes, and the filter returns the same 18 rows.
- Finding: `no_concern` mixed three cases: Prime never read the session, the concern aged out of `concern_record` (Prime keeps 20), or Prime saw the turn and opened nothing.
  - Fix: the report now uses Prime's own `classify_who` (the ingress `mode` flag joins by primary key). It shows `not_handed_back` and `no_concern_or_aged_out` as separate states.
  - Evidence: `test_prime_by_ref_joins_on_the_exact_turn_concern_id`.
- Finding: the report's promise count differed from Prime's count when a quote was empty.
  - Fix: the report reuses `handback._counted_promises`.
  - Evidence: the same test (its `no-quote` case).
- Finding: the report printed only the lower tail; it printed `1.0` when there were 0 turns; it said "since the finder" when `--days` started later; `--prime-turns` given alone was silently ignored.
  - Fix: the upper tail prints too; `-` prints when there are no judged turns; the line prints the real start time; `ap.error` stops a lone `--prime-turns`.
  - Evidence: tests and the live run.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: Prime labels eval-script turns sent over `/api/chat` (session `memory-scoreboard-eval`) as `chat_juniper`. Live example: d391d241. Prime refused it only because all 3 concern slots were full. With a free slot, Prime would hold "a promise I made in a chat with Juniper" that came from an eval. Mitigation: recorded on #3602 with the fix (`classify_who` should require a browser `surface_context`). Not fixed here.
- Severity: low. Concern: the Prime column reads Redis only when the `redis` package is installed, and the host Python has no `redis` package. Mitigation: pass the files from `redis-cli GET` (used for the live run above).
- Issues: commented on #3531 (the open points from thread 0c1e54f0ee1f and the question on thread 0a4514841253) and on #3602 (Prime eval-turn label). None opened or closed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3679
