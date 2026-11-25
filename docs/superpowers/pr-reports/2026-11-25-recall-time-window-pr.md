## Summary

- Lived-memory chat recall now reads time words in the message ("last week", "in October", "a month or so ago", "yesterday", a date) and favours memories from that time. A small date grammar does this, with no model involved (`services/orion-recall/app/lived_time.py`).
- The window applies before the rerank pool is cut. In-window candidates fill the pool first, and at most 4 from outside the window stay. The best 2 of those 4 go into the first rerank chunk. After the rerank, in-window memories get +1 logit, and when one of them fits, at most 2 lines from outside the window are served. A window with nothing in it changes nothing.
- Past-tense questions ("where was I in early October?") also read facts that an update has retired since. They come from a new interval as-of reader, `fetch_valid_during` (`orion/memory/lived_memory.py`). They are labelled `true then, not now: changed <date>` and are re-checked through the same reader before they render.
- The pre-turn query rewrite is unchanged. Asking it for a time line was measured, and every wording tried made it stop searching or was ignored, so the window comes from the parser alone and never waits on the model. Rewrite timeouts were measured on live traffic; the cap is unchanged (findings below).
- Scoreboard: 41/55 in all 4 baseline runs, 45/55 and 46/55 after. The time category went from 10/17 to 14/17. The #3881 event item hits in both after-runs, and so do the 3 backfill items in the run without a rewrite timeout.

Fixes #3674

## Outcome moved

Retrieval scoreboard, `make eval-memory-scoreboard ARGS="--phases retrieval --lived-local"`, private bank (65 items, 55 scored). 6 time-scoped items were added for this PR: last week, a named month, a month or so ago, a past-state question, yesterday, and a date.

| run | code | all | fact | event | time | update | lived p50 / p90 ms | rewrite timeouts |
|---|---|---|---|---|---|---|---|---|
| before 1 (05:47Z) | main | 41/55 (75%) | 12/13 | 8/13 | 10/17 | 11/12 | 1166 / 2958 | not recorded |
| before 2 (05:52Z) | main | 41/55 (75%) | 12/13 | 8/13 | 10/17 | 11/12 | 1313 / 2922 | not recorded |
| before 3 (05:59Z) | main | 41/55 (75%) | 12/13 | 8/13 | 10/17 | 11/12 | 1030 / 2972 | 5 of 65 |
| before 4 (06:04Z) | main | 41/55 (75%) | 12/13 | 8/13 | 10/17 | 11/12 | 1098 / 2798 | 5 of 65 |
| **after 1 (06:49Z)** | branch | **45/55 (82%)** | 11/13 | 9/13 | 14/17 | 11/12 | 1506 / 2867 | 8 of 65 |
| **after 2 (06:54Z)** | branch | **46/55 (84%)** | 12/13 | 9/13 | 14/17 | 11/12 | 1505 / 3116 | 6 of 65 |
| window off (07:00Z) | branch, `RECALL_LIVED_MEMORY_TIME_WINDOW_ENABLED=false` | 41/55 (75%) | 11/13 | 8/13 | 11/17 | 11/12 | 1522 / 3207 | 10 of 65 |

Bars:
- **The #3881 event item hits:** yes in both after-runs. It missed in all 4 baseline runs and in the window-off run.
- **The time category improves:** 10/17 to 14/17 in both after-runs. All 6 new items hit in both runs; before, 2 of 6 hit. The 11 older time items stay at 8/11 (see "Remaining gaps").
- **Overall at least 80%:** 82% and 84%.
- **No drop in facts or updates:** updates 11/12 in every run. Facts were 12/13 in after 2 and 11/13 in after 1. The one miss in after 1 is the short-message backfill item. Its search runs only when the short-message rewrite answers, and that rewrite timed out (`gate=short_none`, `rewrite=timeout`, 0 candidates). No window was involved. It also missed in the window-off run for the same reason.
- **The 3 backfill items still hit:** all 3 in after 2. In after 1, 2 of 3 hit, and the miss is the rewrite timeout above.
- **Latency:** measured in the same hour, lived recall p50 is 1505-1506 ms with the window and 1522 ms with it off. p90 is 2867-3116 ms on and 3207 ms off. The window itself adds no measurable time. The earlier baseline hour was quieter (p50 1030-1313 ms), and the window-off control shows that difference is load, not this change. A first version used a Chroma `where` filter on `occurred_ts`. It cost about 1 s a query (a plain 256-deep query takes about 20 ms), so it was replaced before measuring.

Small-talk injection (`run_lived_memory_replay_eval.py`, variant step2, live bar 0.004, rewrite on with a 2.5 s cap, links 2, Juniper's last 150 real messages plus the 22-message referring bank; run twice each, alternating):

| replay | small talk that injects anything | any line on her last 150 messages | mean lines | referring hits | lived total p50 / p90 ms | rewrite timeouts |
|---|---|---|---|---|---|---|
| before 1 | 3/24 | 77.3% | 2.57 | 12/22 | 2502 / 3810 | 32 |
| after 1 | 3/24 | 79.3% | 2.59 | 11/22 | 2613 / 3732 | 29 |
| before 2 | 3/24 | 78.0% | 2.60 | 10/22 | 2621 / 4003 | 39 |
| after 2 | 2/24 | 73.3% | 2.31 | 11/22 | 2895 / 4078 | 67 |

Small-talk injection is no worse: 3 and 3 of 24 before, 3 and 2 of 24 after. The other columns move within run-to-run rewrite noise. The last run had 67 rewrite timeouts, which is lane contention, and its latency reflects that.

## Rewrite timeout findings (step 3)

- **Live rate.** Over the last 7 days of `recall_telemetry.lived_trace`, the rewrite ran 1,400 times and timed out 171 times (12%). Outside the scoreboard session the rate was 91 of 821 (11%); inside it, 80 of 579 (14%).
- **Distribution.** Rewrites that finished took p50 0.95 s, p90 1.5 s and p95 1.8 s, measured at recall. Counts per 250 ms bucket fall from 428 to 54, then flatten at 25, 27 and 23 between 2.0 and 2.5 s, and 171 sit at the cap. That is a fat tail on top of a normal latency curve: requests that queued, not a slow model.
- **Cause, from the gateway logs** (48 h, 876 rewrite requests from orion-recall): 71 got no model answer.
  - 48 never got a GPU lease. The pool's acquire RPC timed out at about 2.5 s (`gpu_pool_unreachable ... (acquire RPC timed out)`).
  - 23 got a lease and were cancelled mid-generation (`upstream_cancelled reason=caller_budget_exhausted`).
  - 66 of the 71 fell between 00:00Z and 04:00Z on 11-25, during the #3881 scoreboard reruns and diagnostics. Outside that window, 5 failed in about 40 h.
  - For rewrites that reached the model, model time was p50 0.68 s and p95 1.78 s. Only 6 of 805 took over 2.3 s.
- **Verdict.** This is mostly contention: the lease service and the fast seat under burst load, much of it from eval runs themselves. Raising the cap would add wait to every turn and would not fix lease RPCs that go unanswered. The cap stays at 2.5 s. The time window now comes from the deterministic parser, so it never depends on the model. The lease RPC finding is filed as #3916.
- **Model time line, measured and dropped.** Every prompt was run once on the 57 bank questions:

| prompt | searches written | WHEN lines written |
|---|---|---|
| main's prompt (unchanged) | 11 / 57 | n/a |
| time line inline with the query rule | 0 / 57 | 0 |
| time line as an optional note | 3 / 57 | 1 |
| time line after the rule | 1 / 57 | 0 |
| worked example | 18 / 57 | 0 |

  A full scoreboard run with the first wording showed the same thing: searches written fell from about 20 of 65 to 11 of 65.

## Current architecture

- Lived recall (`services/orion-recall/app/lived_memory.py`) ranks candidates from meaning search (top 16) and exact-word search (top 10), merged by reciprocal-rank fusion and cut to 24, plus rewrite and link hits up to 32. Then a cross-encoder rerank, a fit bar, and `logit + recency + importance`.
- No step read time words. Recency (half-life 30 days) applied only after the cut, so a record from the named month that the cut dropped could never come back (#3881).
- Chat recall served only currently-true rows (`EPISODE_SQL`). The as-of reader `EPISODE_VALID_AT_SQL` (#3659) read one instant and had no chat caller.

## Architecture touched

- orion-recall lived recall: time window, pool split, rank boost and cap, as-of candidates, trace field `lived_trace.window`.
- `orion/memory/lived_memory.py`: `EPISODE_VALID_DURING_SQL`, `fetch_valid_during`, and `LivedRecord.invalid_at` (default None, set only by the as-of readers).
- Config: two new orion-recall env keys. Plus the env sync script's prefix list.

## Files changed

- `services/orion-recall/app/lived_time.py`: new. The date grammar: 11 shapes, the earliest phrase wins, padded windows, and the past-state rule.
- `services/orion-recall/app/lived_memory.py`: window parse, deep meaning search filtered to the window, in-window exact-word search, pool split before the cut, first-chunk interleave, `rank(in_window=...)`, as-of candidates (run alongside the meaning search, deadline-bound), as-of re-check, the `true then, not now` label, and the trace.
- `orion/memory/lived_memory.py`: the interval as-of reader (`LIMIT` for wide windows) and `LivedRecord.invalid_at`.
- `services/orion-recall/app/settings.py`, `.env_example`, `docker-compose.yml`, `app/main.py`: `RECALL_LIVED_MEMORY_TIME_WINDOW_ENABLED`, `RECALL_LIVED_MEMORY_TIMEZONE`.
- `scripts/sync_local_env_from_example.py`: adds the `RECALL_LIVED_MEMORY_` prefix. Without it, default sync silently skipped the new keys.
- `services/orion-recall/evals/run_memory_scoreboard.py`: time items resolve against their `as_of` day (`asked_at`, local runs only), the window flag reaches the local recall, and the report keeps rewrite outcomes and per-item traces (ids and dates only).
- `services/orion-recall/tests/test_lived_time_window.py`: new. 66 tests: the parser (including negative cases for ordinary chat), window filtering, the out-of-window caps, the first-chunk interleave, the as-of label, the re-check, deadline, replay and undo cases, and trace privacy.
- `orion/memory/episode/tests/test_supersede_pg.py`: the interval reader against real Postgres (update retired: served for the window, not before it; a correction: never served; ids and limit).
- `services/orion-recall/evals/test_memory_scoreboard.py`: an `asked_at` test.
- `.github/workflows/lived-memory-recall-tests.yml`: runs the new test file.
- `services/orion-recall/README.md`: section 17a2, plus the kill-switch list.

## Schema / bus / API changes

- Added: `recall_telemetry.lived_trace.window`, a JSON field inside an existing JSONB column. It holds local start and end dates, shape, `past_state`, and counts. No column changes and no migration.
- Removed: none.
- Renamed: none.
- Behavior changed: a message with a time phrase gets a pool and ranking that favour that time. Past-tense questions can serve update-retired facts with a label.
- Compatibility notes: `LivedRecord` gains an optional field with a default. Existing readers never set it.

## Env/config changes

- Added keys: `RECALL_LIVED_MEMORY_TIME_WINDOW_ENABLED=true`, `RECALL_LIVED_MEMORY_TIMEZONE=America/Denver`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (orion-recall).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys were added to the primary checkout's `services/orion-recall/.env`.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-recall: pytest tests evals -> 624 passed, 2 failed
  (the 2 failures are pre-existing on main e1e524674: test_recall_policy_harness diagnostic,
   test_recall_vector_amputation import; same result in an unchanged main worktree)
services/orion-recall: pytest tests/test_lived_time_window.py -> 66 passed
orion/memory: pytest -> 3355 passed, 7 skipped
orion/memory/episode/tests/test_supersede_pg.py against a throwaway postgres:16 -> 12 passed
tests/test_no_personal_terms.py -> 7 passed
scripts/check_env_template_parity.py -> PASS; scripts/check_integration_seam.py -> OK
```

## Evals run

```text
memory scoreboard, retrieval, --lived-local: 4 baseline runs (main), 2 after-runs, 1 window-off run (table above)
lived-memory replay, step2, small-talk injection: 2 runs each (before/after), table above
rewrite prompt probe, 57 bank questions x 5 prompts (table above)
```

## Docker/build/smoke checks

```text
Not deployed (agents do not deploy). No Docker build: no new dependency. zoneinfo/tzdata checked inside
the running orion-athena-recall container: ZoneInfo('America/Denver') resolves.
Live path after deploy: UNVERIFIED until orion-recall is rebuilt.
```

## Review findings fixed

- Finding: the bare words "today" and "tonight" narrowed recall to two days for ordinary chat. Also, the first matching shape won, not the first phrase in the message.
  - Fix: "today" and "tonight" are never a window, and "this morning" counts only in the past tense. The earliest phrase across all shapes wins.
  - Evidence: negative tests `How are you today?` and `what should we cook tonight`, and a test that "Today ... two weeks ago" gives the two-weeks-ago window.
- Finding: fractions and ratings parsed as US dates ("8/10", "3/4", "1/2 cup").
  - Fix: a bare M/D needs a year or a lead word, and N/10 is never a date.
  - Evidence: negative tests.
- Finding: month abbreviations had no word boundary ("3 of mark's" read as March 3), and a future month became last year's.
  - Fix: word boundary added. A month still ahead this year gives a window only in the past tense.
  - Evidence: negative tests.
- Finding: the 4 best outside candidates landed in the last rerank chunk, where a short deadline would never score them.
  - Fix: the best 2 ride in the first chunk.
  - Evidence: `test_the_best_outside_hits_ride_in_the_first_rerank_chunk`.
- Finding: the as-of read ran serially, had no deadline and no row limit.
  - Fix: it starts alongside the meaning search, its wait is bounded by `min(0.8 s, deadline)`, the read has `LIMIT 500`, and the re-check has a timeout.
  - Evidence: `test_a_slow_as_of_read_is_cut_off_and_recall_still_serves`, and the pg test with `limit`.
- Nits fixed:
  - the model PAST rule used the midnight rule (later removed with the model line);
  - the label uses the local date;
  - an undone retirement is served without the label;
  - a replay never labels a row retired after its `as_of`;
  - "the last day of" is not a span.
- Found during measurement, not review:
  - The model time line suppressed searches. It was removed (prompt probe above).
  - The Chroma `where` filter cost about 1 s. It was replaced with a deep plain query.
  - The privacy gate caught real names in the test fixtures. They were replaced with neutral ones.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-recall up -d --build
```

Run it from the clean primary `main` checkout after merge. The post-pull rebuild does this on its own.

## Risks / concerns

- Severity: medium. Concern: the past-state path has never run on real data. No row has `invalid_reason = 'update'` yet (live counts: 1,323 active, 55 rejected, 5 superseded, all by the confirmation loop). So the issue's "answers with the old value for the old date" can only be shown in unit and Postgres tests today. Mitigation: #3914 (`verify-live`).
- Severity: low. Concern: a time word in a statement rather than a question ("it went well yesterday") narrows the pool for that turn. Mitigation: at most 2 outside lines are still served, the best 2 outside hits are always reranked first, and a window with nothing in it changes nothing. The small-talk replay above checks the injection side.
- Severity: low. Concern: two older date questions still miss. Their target ranks first, but its fit is below the bar. Mitigation: #3925.
- Severity: low. Concern: relative-time bank items drift on deployed (non-local) scoreboard runs. Mitigation: #3915.
- Issues opened: #3914, #3915, #3916, #3925. Issues closed: #3674 (by this PR).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3923

🤖 Generated with [Claude Code](https://claude.com/claude-code)
