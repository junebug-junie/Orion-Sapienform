## Summary

This is a follow-up to #3709, which merged at 06:12Z with check version v3.3. It ships v3.4 and the documentation that #3709's merge left out.

- **The numbered-item rule now runs before any vote.** Suppose the old memory names a PR or issue number that the new one never names. The pair cannot act, so the model is not asked. Under v3.3 the rule ran after the votes, so such pairs spent the shared call budget, and the spent budget could turn off Juniper's path for that row.
- **Tighter number matching.** It now handles plurals ("PRs 41 and 42"), `PR-41`, and leading zeros. It no longer matches a Markdown "# 2026" heading or "#41st".
- **The eval labels these stops as rule stops**, so the model gets no credit for what the regex did.
- **The #3709 report** gains the per-version eval table, the two rejected pairs by id with their verdicts, and the answers to Orion's thread questions.

## Outcome moved

v3.4 has a recorded passing eval at 09:31Z: precision 1.00 (22/22), recall 0.69, 0 false acts in 120 runs (60 pairs x 2). Juniper's two rejected pairs:
- `65590f70` -> `4da33de0`: compatible, then unrelated.
- `af9d823c` -> `06c7cf45`: blocked by the rule both times.

An earlier v3.4 run at 09:18Z failed on one gateway timeout. It made 0 false acts.

## Current architecture

v3.3 (main): `classify_pair` asked all votes, then applied `same_numbered_item`.

## Architecture touched

`orion/memory/episode/supersede.py` (`classify_pair`, `_NUMBERED`, `numbered_items`) and `scripts/memory_supersede_eval.py` (`decide_v3`).

## Files changed

- `orion/memory/episode/supersede.py`: the rule runs first, the regex is tighter, and the version is v3.4.
- `scripts/memory_supersede_eval.py`: a rule stop is labelled `stopped_by=rule`.
- `orion/memory/episode/tests/test_supersede.py`: a parametrized regex test, edge cases, and a check that no prompt is asked.
- `docs/superpowers/pr-reports/2026-11-18-supersede-same-referent-pr.md`, and the spec note.

## Schema / bus / API changes

- Behavior changed: `CHECK_VERSION` is now `supersede_classify.v3.4`. Its own passing eval is recorded (09:31Z), so write time keeps applying after deploy.
- Added: none. Removed: none. Renamed: none.
- Compatibility notes: no migration.

## Env/config changes

- None.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests   363 passed, 33 skipped (the Postgres-backed tests run in CI)
```

## Evals run

```text
scripts/memory_supersede_eval.py run --record-db (v3.4, 2 repeats)  09:18Z failed (1 gateway timeout, 0 false acts); 09:31Z PASSED, precision 1.00, 0 false acts
```

## Docker/build/smoke checks

```text
Not run: no dependency or compose change.
```

## Review findings fixed

- Finding: the rule ran after the votes, so doomed pairs spent budget.
  - Fix: return early with abstain `numbered_item_not_in_b`.
  - Evidence: the test asserts no prompt is asked.
- Finding: the eval credited rule stops to the model.
  - Fix: these stops are now labelled `stopped_by=rule`.
- Finding: the regex had false positives and false negatives.
  - Fix: the tighter pattern above.
  - Evidence: a parametrized test over 9 cases.
- Finding: there was no test for the renumbering trade-off.
  - Fix: an edge-case test. "It was PR #41, not #14" passes the rule. "It was PR #41" against an old "#14" abstains, which is an accepted cost.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: a gateway timeout during an eval fails the run. That is safe, and a rerun clears it.
- Issues: #3721 is open (pass apply needs Juniper's per-pair approval mark).

## PR link

(set after open)
