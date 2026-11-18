## Summary

Fixes #3536.

- One command, `make eval-memory-scoreboard`, now scores how well Orion remembers Juniper's real history. It gives two scores: retrieval (is the right memory in the prompt?) and answer (does Orion's reply get it right?).
- Retrieval replays orion-recall's two chat recall phases over the real bus, with no model. It counts a hit when a target memory id is in the prompt's DURABLE MEMORY section or is cited by the core page.
- Answer sends each bank question as a real Hub chat turn that writes nothing (`no_write`), on its own eval session. A local judge, held through the GPU pool, grades each reply.
- New in this round: the scoreboard reads recall's own verdict on each search (`recall_telemetry.lived_status` / `lived_trace`, #3628). A "no memory" answer on a stored item is always a false denial. It is also reported under its search cause (too short, degraded, rerank timeout, skip gate, and so on), as asked in the #3536 comment.
- Questions too short for the lived search (`skipped_short`) are now scored, with their own column. A real chat turn with the same message does the same thing, so these are not left out as "degraded".
- Implicit items (real past messages resent, with no direct question) get their own row in both tables.

## Outcome moved

There is now a measured baseline for every memory change after this one. The bank (59 items) and every reply stay host-side under `~/.orion/evals/` (mode 600). This report holds counts only.

### Baseline, 2026-11-18 (live: recall steps 1 and 2, #3676, #3645, #3677 deployed)

**Retrieval** (all 59 items, bar 80% of stored items)

| category | stored | scored | hits | hit rate | memory section only | too short for lived search (of them hit) | degraded |
|---|---|---|---|---|---|---|---|
| facts | 12 | 12 | 9 | 75% | 0% | 9 (8) | 0 |
| events | 11 | 11 | 7 | 64% | 64% | 2 (0) | 0 |
| time | 11 | 11 | 7 | 64% | 46% | 3 (1) | 0 |
| updates | 12 | 12 | 10 | 83% | 58% | 5 (3) | 0 |
| abstention | 0 (10 items, nothing to find) | - | - | - | - | - | - |
| implicit (subset) | 8 | 8 | 5 | 62% | 50% | 0 | 0 |
| **all** | **46** | **46** | **33** | **72% - FAIL** | 41% | 19 (12) | 0 |

- Pre-October items, not in the store yet (#3537): 3, reported apart, no bar.
- Degraded recalls: 0 this run. No deadline hits, no rerank timeouts, and no lived-search errors on the scoreboard's calls.
- Superseded (pre-correction) rows served on update items: 0.
- 7 of the 13 misses are questions too short for the lived search. Only the core page could catch them. This is #3613; counts added there.

**Answer** (stratified subset: 26 real Hub turns, 4 per category plus 3 implicit plus the 3 pre-October items. The full 59 would take about 3.5 hours of chat-lane time; the subset took about 2 hours with retries.)

| category | asked | correct | don't know | invented | false "no memory" | correct rate |
|---|---|---|---|---|---|---|
| facts | 5 | 5 | 0 | 0 | 0 | 100% |
| events | 5 | 3 | 0 | 2 | 0 | 60% |
| time | 4 | 4 | 0 | 0 | 0 | 100% |
| updates | 5 | 5 | 0 | 0 | 0 | 100% |
| abstention | 4 | 4 | 0 | 0 | 0 | 100% |
| implicit (subset) | 3 | 3 | 0 | 0 | 0 | 100% |
| **all scored** | **23** | **21** | 0 | **2** | **0** | **91%** |

- Bars: correct 91% vs 70% PASS. Invented 2 vs 0 FAIL. False "no memory" 0 PASS. Coverage 100%: two Hub failures (one 900 s timeout and one dropped connection) were redone with `--resume`.
- False denials by search cause: none to split this run. 12 of the 26 turns had a lived search too short to run, and none of them denied a stored memory.
- Both "invented" answers are event items whose target was not in the prompt. The judge's own reasons describe the reply leaving out the recorded events and adding other details. The rubric says that alone is not invention. These two labels need a hand check: #3701.
- Pre-October items: 3 of 3 answered "no memory". That is honest, since they are not stored yet.
- Target in the prompt, read from each turn's own recall trace: 14 of the 19 turns where it could be known.

## Current architecture

Before this round, the branch scored retrieval and answers. It guessed "degraded" from timing, because recall did not record why a search fell short. Every short question then read as degraded, which left 19 of 46 stored items out of the score.

## Architecture touched

- `services/orion-recall/evals/` only (the eval, its runner and its tests), plus the Makefile target, the CI workflow step and the README section. No service code, schema, bus or env change.

## Files changed

- `services/orion-recall/evals/memory_scoreboard.py`: pure scoring. Reads `lived_status`/`lived_trace`; adds `search_causes` and `primary_cause` (false denials by their most specific cause); scores skipped_short; adds the implicit row.
- `services/orion-recall/evals/run_memory_scoreboard.py`: runner. Selects the #3628 columns. Each Hub turn records its search causes, and a local phase-0 skip-gate check names `recall:skip_gate`. Waits longer for the trace row.
- `services/orion-recall/evals/test_memory_scoreboard.py`: fixture tests. Includes the #3536 comment check (skipped_short + stored target + denial = false denial, cause skipped_short) and runner tests through `run_retrieval` and `run_answers` with real row shapes.
- `services/orion-recall/evals/fixtures/memory_scoreboard_fixture.json`: made-up fixture, not real history.
- `services/orion-recall/README.md`: scoreboard section.
- `Makefile`: `eval-memory-scoreboard`.
- `.github/workflows/lived-memory-recall-tests.yml`: runs the fixture test. The merge conflict with #3622/#3676 is resolved, and both sides' test lists are kept.
- `docs/superpowers/pr-reports/2026-11-18-memory-scoreboard-pr.md`: this report.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: none in services. The eval reads `recall_telemetry.lived_status` and `lived_trace`, which were added in #3628.
- Compatibility notes: older telemetry rows without `lived_status` fall back to the timing checks.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-recall && PYTHONPATH=.:<repo> python -m pytest -q evals/test_memory_scoreboard.py
56 passed
(after the merge, also tests/test_lived_know_and_status.py tests/test_recall_telemetry_persist.py: pass)
```

## Evals run

```text
make eval-memory-scoreboard ARGS="--phases retrieval"            # 59 items, live bus + DB
make eval-memory-scoreboard ARGS="--phases answer --only <26 qids>"   # 26 live Hub turns, local judge
  ... --resume <run>.json                                           # redid 2 Hub failures + 1 unjudged
Results host-side: ~/.orion/evals/memory_scoreboard_runs/2026-11-18T023718.partial.* (retrieval),
2026-11-18T042643.partial.* (answer). Tables above.
```

## Docker/build/smoke checks

```text
No image or runtime change. The live smoke is the eval above (real bus, real recall, real Hub).
```

## Review findings fixed

- Finding: a test used a trace shape the real code never writes (`ok` + `degraded: [rerank_partial]`).
  - Fix: tests now use the two real shapes (`ok` + partial dict, `degraded` + reason + dict).
  - Evidence: `test_degraded_reasons_read_lived_status`.
- Finding: false denials were bucketed under the bare status, so the README's `rerank_timeout` / `embedder_*` buckets never appeared.
  - Fix: `primary_cause` picks the most specific cause, and every qid keeps its full cause list.
  - Evidence: `test_primary_cause_is_the_most_specific`.
- Finding: a turn with no cause data read as `search_full` (claimed a full search without evidence).
  - Fix: no data is `cause:unknown`. `search_full` needs an explicit empty list.
  - Evidence: same test.
- Finding: "no trace rows" mixed up the skip gate and a late write.
  - Fix: the runner runs the skip gate locally (`recall:skip_gate`) and polls 5 x 3 s before `recall:no_trace`.
  - Evidence: runner code; `test_runner_answers_report_false_denials_by_their_turn_cause`.
- Finding: no test covered the runner reading the #3628 columns.
  - Fix: added `run_retrieval` and `run_answers` tests whose fake pool requires `lived_status` and `lived_trace` in the SELECT.
  - Evidence: `test_runner_retrieval_reads_lived_status_rows`, `test_runner_answers_report_false_denials_by_their_turn_cause`.
- Finding (nit): a partial rerank was tagged twice.
  - Fix: one tag, with counts.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: the answer score is a 26-item subset, not the whole bank. Mitigation: the subset is stratified (at least 4 per category, 3 implicit) and every count says so. A whole-bank run is `make eval-memory-scoreboard`, about 3.5 h.
- Severity: medium. Concern: the 2 "invented" labels may be judge strictness, not real invention. Mitigation: #3701 (hand check, then tighten the judge and `--rejudge`).
- Severity: low. Concern: retrieval misses are dominated by short questions skipping the lived search (7 of 13). Mitigation: #3613, counts commented there.
- Issues opened: #3701. Commented: #3613. Closed by this PR: #3536.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3603

🤖 Generated with [Claude Code](https://claude.com/claude-code)
