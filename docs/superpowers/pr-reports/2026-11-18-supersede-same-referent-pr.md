## Summary

Juniper reviewed the two memory corrections that the v3.1 checker proposed in the dry run. She rejected both. In each case all 3 votes said "correction", and in each case the two memories only shared a project word:

- The first pair: the old memory measured one slice (overnight GPU overhead) and the new one gave whole-host totals.
- The second pair: the old memory said one fix was deployed, and the new one said nothing was deployed. That later note was about a different fix.

This PR makes the checker ask the missing question.

- **The checker now asks whether both memories are about the same specific thing.** Every vote must also answer `same_referent`: is B about the same PR, fix, pet, or measured quantity over the same scope as A? A shared project or host word does not count. A vote acts only when the answer is "yes". A missing or malformed answer abstains. The check version is now `supersede_classify.v3.2`.
- **The eval learns from undos.** Every pair someone undid after an apply (`supersede_undone`) becomes a hard negative, and the undoing actor is kept in the item's source. Juniper's two rejected pairs are in the labelled set this way.
- **The eval runs every item twice.** Sampled votes vary: one held-out negative passed at 00:44 and acted at 05:25. Scoring is pooled, so one false act in any repeat fails the run. `--record-db` requires `--repeats >= 2`.
- **The labelled set no longer depends on the live manifest.** The live manifest had lost its watch list, so the six watched row ids are now fixed in the script (ids only, already public in the issues). The set is deduped by pair, item ids are checked to be unique, and coverage floors count distinct pairs.
- **New made-up fixtures:** a measurement slice against a whole total, one fix against a different fix, and a true correction of the same fix.

## Outcome moved

The gate was open on a checker that was confidently wrong. It is now closed, and the fixed checker passes.

| Check version | Labelled set | Precision | Recall | False acts | Gate |
|---|---|---|---|---|---|
| v3.1 (recorded 04:18Z) | before Juniper's 2 rejections | 1.00 | 0.62 | 0 | open |
| v3.1 (recorded 05:25Z) | + her 2 rejections, + 3 made-up | **0.77** (10/13) | 0.67 | **3**: one of her rejected pairs, plus a held-out negative that flipped between runs | **closed** |
| v3.2 (this PR, 2 repeats) | same set | **1.00** (22/22) | 0.69 | **0** in 120 runs (60 items x 2) | passes |

On both of Juniper's rejected pairs, v3.2 voted compatible or unrelated with `same_referent: no` in every repeat. The true correction of the same fix still acts, 3/3 votes in both repeats.

## Current architecture

The v3.1 checklist asked for the same attribute but not the same specific item. The model said "same attribute: yes" when both memories talked about energy numbers or "deployed".

## Architecture touched

- `orion/memory/episode/supersede.py`: the prompt, `Vote.same_referent`, the `acting` rule, and the version bump.
- `orion/memory/episode/supersede_eval.py`: coverage floors count distinct pairs.
- `scripts/memory_supersede_eval.py`: undone pairs, fixed watch ids, dedupe, `--repeats`, and the record guard.

## Files changed

- `orion/memory/episode/supersede.py`: the same-referent checklist question; v3.2.
- `orion/memory/episode/supersede_eval.py`: `base_id` and distinct-pair floors.
- `scripts/memory_supersede_eval.py`: as above.
- `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: 3 made-up pairs.
- `orion/memory/episode/tests/test_supersede.py`, `test_supersede_eval.py`, `test_supersede_pass_script.py`: tests.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: v3.2 note.

## Schema / bus / API changes

- Added: the vote field `same_referent` (in the model's JSON answer and the audit rows).
- Removed: none. Renamed: none.
- Behavior changed: `CHECK_VERSION` is now v3.2. The eval gate is per version, so write time needs a recorded v3.2 pass before it applies anything.
- Compatibility notes: no DB migration.

## Env/config changes

- None.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests   352 passed, 33 skipped (the Postgres-backed tests run in CI)
```

## Evals run

```text
scripts/memory_supersede_eval.py build                      23 real items (5 positives, 18 hard negatives incl. the 2 undone pairs)
scripts/memory_supersede_eval.py run --record-db   (v3.1)  precision 0.77, 3 false acts  -> recorded FAILED, gate closed
scripts/memory_supersede_eval.py run               (v3.2)  precision 1.00, 0 false acts / 120 runs, recall 0.69  PASS
scripts/memory_supersede_eval.py run --record-db   (v3.2)  RECORD_PLACEHOLDER
```

## Docker/build/smoke checks

```text
Not run: no dependency or compose change.
```

## Review findings fixed

- Finding: repeats halved the real-data coverage floors.
  - Fix: the floors count distinct pairs (`base_id`). `repeats` is now recorded in the summary.
  - Evidence: `test_repeats_do_not_halve_the_real_coverage_floors`.
- Finding: "undone" does not always mean "Juniper rejected it".
  - Fix: the undoing actor is now in the item's source (`undone_by_<actor>`). These two were undone by the operator after her review.
- Finding: deduping by pair allowed duplicate item ids. A revision id used only the old row's id, so one decision could overwrite another.
  - Fix: the revision id now uses both rows, and the build refuses duplicate ids.
- Finding: malformed-vote audit rows dropped `same_referent`.
  - Fix: every vote now records it.
- Finding: there were test gaps.
  - Fix: added tests for the record guard and the repeat floors.
- Nits fixed: comments and docstrings now name the same-referent rule.
- Kept: the made-up "other fix" negative stays deliberately hard. It is the same trap shape Juniper caught.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Until this deploys, the running v3.1 code finds its gate closed (05:25Z failed run), so it proposes only.

## Risks / concerns

- Severity: medium.
  - Concern: the v3.2 numbers are partly in-sample, because Juniper's 2 pairs shaped the prompt.
  - Mitigation: running twice per item, and future undos feed the set automatically.
- Severity: low.
  - Concern: recall may drop on real corrections that say "it" without naming the item.
  - Mitigation: an abstain is always safe.
- Issues: follow-up to #3659 (closed). Verify-live remains #3693.

## PR link

(set after open)
