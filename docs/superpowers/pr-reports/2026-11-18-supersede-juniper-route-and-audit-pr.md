## Summary

This answers Orion's five questions on #3735. Investigating them turned up one live problem: tonight's write-time check never asked a single pair, because the meaning index stalled once.

- **Juniper's words are no longer blocked by the number rule.** v3.4 blocked every pair where the old row named a PR or issue number that the new row did not repeat. That included Juniper's own verified corrections. Now her path may act without the number, but still only when every vote says update or correction with "same referent: yes". The model-only path keeps the rule.
- **The run audit counts why pairs abstained.** The plan and each `memory_supersede_run.counts` row carry an `abstained` count by reason, and linked pairs are not counted. Each pair's reason was already in `decisions[].abstain`. Juniper-only numbered pairs get the reason `numbered_item_juniper_path_only`.
- **The meaning-index lookup gets one retry.** At 11:10Z the nightly check refused as `dense_unavailable`: Chroma's collection lookup took 25 s once, against a 20 s timeout (#3747). There is now one retry per plan, for transient faults only, and each attempt is bounded at 40 s in total. If the retry fails, it still refuses.
- **Spec status lines are updated.** The #3593 and #3611 grants are applied (checked live; #3611 closed). The spec now separates write time (automatic, behind the eval) from the hand-run pass `apply`; the #3721 approval mark applies only to the hand-run command.
- **The audit records the meaning-index lookups.** `memory_supersede_run.counts` gains `dense_calls`, `dense_ms` (total wall time) and `dense_retries`. `counts.abstained` is now derived from `decisions` on every path, so the nightly distill, the pass dry-run, and the hand-run pass apply (whose per-pair decisions #3751 adds) all land the same count in one table.
- **The eval scores the clear-target rule.** A new item type is a group: one new row against several older rows, run through `check_pairs`. The made-up case `syn:juniper-two-targets` checks that Juniper's words with two qualifying targets retire nothing.
- **The check version is now `supersede_classify.v3.5`.** One made-up fixture was added: Juniper corrects a numbered fix without restating its number.

## Outcome moved

Before this PR, a Juniper correction that did not restate a PR number could never apply. Now it can. A one-off Chroma stall no longer costs a whole night's check. The audit now shows which rule stopped each pair.

## Current architecture

In v3.4, `classify_pair` returned `numbered_item_not_in_b` before any vote for every new row, including Juniper's. `index_dense` made one 20 s attempt.

## Architecture touched

`orion/memory/episode/supersede.py` (`classify_pair`, `plan_for_new_rows`, `record_run`, `index_dense`), one fixture, and the spec.

## Files changed

- `orion/memory/episode/supersede.py`: the Juniper route, abstain counts, the bounded retry, and v3.5.
- `orion/memory/episode/tests/test_supersede.py`, `test_supersede_pg.py`: tests for the route, labels, clear target, split votes, retry, and audit counts.
- `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: one made-up pair.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: status lines and who approves what.

## Schema / bus / API changes

- Added: the plan keys `abstained`, `dense_calls`, `dense_ms` and `dense_retries`. All four are copied into `memory_supersede_run.counts` (JSONB, no migration). `abstained` is derived from `decisions` when a caller does not give it. The eval item field `others` holds the group items.
- Removed: none. Renamed: none.
- Behavior changed: v3.5. The gate is per version and needs a recorded v3.5 pass.
- Compatibility notes: none.

## Env/config changes

- None.

## Tests run

```text
ORION_MEMORY_EPISODE_TEST_DATABASE_URL=<throwaway postgres:16> pytest -q orion/memory/episode/tests   399 passed (Postgres-backed included)
```

## Evals run

```text
scripts/memory_supersede_eval.py run --record-db   (v3.5, 2 repeats)  PASSED and recorded (run bb7d5b9f)
  precision 1.00 (24/24), recall 0.71 (24/34), 0 false acts, 0 errors over 122 item-runs
  Juniper's rejected pairs: 65590f70->4da33de0 compatible x2; af9d823c->06c7cf45 compatible x2 (her quote no longer
  short-circuits it, and the model now says compatible)
  made-up Juniper fix correction without the PR number: acted via her path, 3/3, both repeats
  10 misses = Juniper's 5 real confirmation revisions x 2: compatible 5, refine 4, checklist:correction 1
```

## Docker/build/smoke checks

```text
From inside orion-athena-durable-runs (read-only probes): Chroma collection lookup took 25.2 s once, then 0.01 s on three later tries. The query took 0.02-0.9 s, and the embedder 0.45 s.
```

## Review findings fixed

- Finding: a Juniper-only numbered pair was labelled `low_confidence`.
  - Fix: it now gets its own reason, `numbered_item_juniper_path_only`. This is not the rule-stop reason, so the eval still counts her acts.
  - Evidence: an assertion in the test.
- Finding: linked pairs were counted as abstained.
  - Fix: linked pairs are excluded from the count.
  - Evidence: a plan test, and a PG `record_run` assertion.
- Finding: the retry could exceed the 300 s node budget, because httpx's timeout applies to each phase.
  - Fix: `asyncio.timeout(40)` now covers the whole attempt, with one retry per plan, and only for a transient fault.
  - Evidence: `test_index_dense_retries_once_then_refuses`.
- Finding: there were test gaps.
  - Fix: added tests for two Juniper candidates (no clear target), a split vote, and the abstain label.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: one of Juniper's rejected pairs has a Juniper-quoted new row. With the number rule bypassed for her words, that pair now depends on the model's same-referent answer. Four probes on v3.5's prompt gave unrelated or compatible every time, and the eval keeps it as a hard negative, run twice.
- Severity: low.
  - Concern: Juniper-quoted numbered pairs now spend model calls that v3.4 skipped.
- Issues: opened #3747 (Chroma stall). Closed #3611 (grants applied). #3721 is unchanged; its approval mark covers the hand-run pass only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3750
