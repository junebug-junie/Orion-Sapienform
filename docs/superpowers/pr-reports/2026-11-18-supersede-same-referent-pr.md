## Summary

> #3709 merged at 06:12Z with v3.3 (the numbered-item rule checked after the votes). v3.4, which checks the rule before any vote and tightens the regex, ships in the follow-up PR #3735. Both versions have a recorded passing eval (v3.3 at 06:07Z, v3.4 at 09:31Z).

Juniper reviewed the two memory corrections that the v3.1 checker proposed in the dry run. She rejected both. In each case all 3 votes said "correction", and in each case the two memories only shared a project word:

- The first pair: the old memory measured one slice (overnight GPU overhead) and the new one gave whole-host totals.
- The second pair: the old memory said one fix was deployed, and the new one said nothing was deployed. That later note was about a different fix.

This PR makes the checker ask the missing question. It also adds a fixed rule for numbered items, because the model alone still got one of the two pairs wrong.

- **The checker now asks whether both memories are about the same specific thing.** Every vote must also answer `same_referent`: is B about the same PR, fix, pet, or measured quantity over the same scope as A? A shared project or host word does not count. A vote acts only when the answer is "yes". A missing or malformed answer abstains.
- **A fixed rule for numbered items.** Suppose the old memory names a PR or issue number, and the new one does not name that same number. Then the pair cannot act, and the model is not asked (no budget is spent). The model alone, at v3.2, still said "same referent: yes" 3 times out of 3 on the deploy pair in one of two repeats. The check version is now `supersede_classify.v3.4`.
- **The eval learns from undos.** Every pair someone undid after an apply (`supersede_undone`) becomes a hard negative, and the undoing actor is kept in the item's source. Juniper's two rejected pairs are in the labelled set this way.
- **The eval runs every item twice.** Sampled votes vary: one held-out negative passed at 00:44 and acted at 05:25. Scoring is pooled, so one false act in any repeat fails the run. `--record-db` requires `--repeats >= 2`.
- **The labelled set no longer depends on the live manifest.** The live manifest had lost its watch list, so the six watched row ids are now fixed in the script (ids only, already public in the issues). The set is deduped by pair, item ids are checked to be unique, and coverage floors count distinct pairs.
- **New made-up fixtures:** a measurement slice against a whole total, one fix against a different fix, and a true correction of the same fix.

## Outcome moved

The gate was open on a checker that was confidently wrong. It closed as soon as Juniper's two rejected pairs entered the eval. Every row below is a recorded run in `memory_supersede_eval_run`, scored on the same labelled set with Juniper's 2 rejected pairs included:

| Check version | Recorded | Precision | Recall | False acts | Errors | Result |
|---|---|---|---|---|---|---|
| v3.1 (before her rejections were in the set) | 04:18Z | 1.00 | 0.60 | 0 | 0 | passed: the gate was open |
| v3.1 (her 2 pairs added) | 05:25Z | **0.77** | 0.67 | **3** | 0 | failed: the gate closed |
| v3.2 (same-referent question) | 05:44Z | 0.95 | 0.63 | 1 (the deploy pair, in 1 of 2 repeats) | 0 | failed |
| v3.3 (+ numbered-item rule) | 06:07Z | **1.00** | 0.66 | **0** | 0 | passed |
| v3.4 (rule before votes, tighter regex) | 09:18Z | 1.00 | 0.69 | 0 | 1 gateway timeout | failed on the error |
| **v3.4 rerun (this PR)** | 09:31Z | **1.00** (22/22) | 0.69 | **0** in 120 runs | 0 | **passed**: the v3.4 gate is open |

The two pairs Juniper rejected, under v3.4:

| Pair (old -> new) | What it was | v3.1 | v3.4, repeat 1 | v3.4, repeat 2 |
|---|---|---|---|---|
| `65590f70` -> `4da33de0` | an overhead slice against whole-host totals | correction, 3/3 (acted) | compatible: the model said not the same referent | unrelated |
| `af9d823c` -> `06c7cf45` | one PR's deploy against "nothing deployed" for a different fix | correction, 3/3 (acted) | blocked by the rule (the old row names a PR the new one never names) | blocked by the rule |

The made-up true correction of the same PR still acts, 3/3 votes in both repeats.

## Current architecture

The v3.1 checklist asked for the same attribute but not the same specific item. The model said "same attribute: yes" when both memories talked about energy numbers or "deployed". The eval gate already reads the latest finished run for a check version, whatever its result. So a failing run after a pass closes the gate at once, and a stale pass cannot hold it open; the PG test `test_eval_gate_reads_the_latest_finished_run_for_this_check_version` pins this.

## Architecture touched

- `orion/memory/episode/supersede.py`: the prompt, `Vote.same_referent`, the `acting` rule, `numbered_items` / `same_numbered_item` (checked before any vote), and the version bump.
- `orion/memory/episode/supersede_eval.py`: coverage floors count distinct pairs.
- `scripts/memory_supersede_eval.py`: undone pairs, fixed watch ids, dedupe, `--repeats`, and the record guard.

## Files changed

- `orion/memory/episode/supersede.py`: the same-referent checklist question, the numbered-item rule, and v3.4.
- `orion/memory/episode/supersede_eval.py`: `base_id` and distinct-pair floors.
- `scripts/memory_supersede_eval.py`: as above.
- `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: 3 made-up pairs.
- `orion/memory/episode/tests/test_supersede.py`, `test_supersede_eval.py`, `test_supersede_pass_script.py`: tests.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: v3.2 note.

## Schema / bus / API changes

- Added: the vote field `same_referent` (in the model's JSON answer and the audit rows).
- Removed: none. Renamed: none.
- Behavior changed: `CHECK_VERSION` is now v3.4. The eval gate is per version, so write time needs a recorded v3.4 pass before it applies anything.
- Compatibility notes: no DB migration.

## Env/config changes

- None.

## Tests run

```text
PYTHONPATH=. pytest -q orion/memory/episode/tests   363 passed, 33 skipped (the Postgres-backed tests run in CI)
```

## Evals run

```text
scripts/memory_supersede_eval.py build                  23 real items (5 positives, 18 hard negatives incl. the 2 undone pairs)
scripts/memory_supersede_eval.py run --record-db        (default --repeats 2), one run per version; see the table above
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
- Second review (the numbered-item rule):
  - Finding: the rule ran after the votes, so pairs that could never act spent shared budget and could disable Juniper's path.
    - Fix: the rule is now checked before any call.
    - Evidence: the test asserts no prompt is asked.
  - Finding: the eval credited the rule's stops to the model.
    - Fix: these stops are now labelled `stopped_by=rule`.
  - Finding: the regex had gaps.
    - Fix: it now handles plurals, `PR-41`, and leading zeros, and no longer matches a Markdown `# 2026` or `#41st`.
    - Evidence: a parametrized regex test, plus edge cases (one of two numbers named; a renumbering that names the old number).
  - Accepted cost: "It was PR #41" against an old "#14" abstains. Abstaining is safe.
- Orion's thread questions: see the reply on this PR. Opened #3721 (pass apply needs Juniper's per-pair approval mark).

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Until this deploys, the running v3.1 code finds its gate closed (the 05:25Z run failed), so it proposes only. After deploy, v3.4 applies only if its latest recorded run passed.

## Risks / concerns

- Severity: medium.
  - Concern: the v3.2 numbers are partly in-sample, because Juniper's 2 pairs shaped the prompt.
  - Mitigation: running twice per item, and future undos feed the set automatically.
- Severity: low.
  - Concern: recall may drop on real corrections that say "it" without naming the item.
  - Mitigation: an abstain is always safe.
- Severity: low.
  - Concern: a gateway timeout during the eval fails the run (errors count). That is safe, and a rerun clears it.
- Issues: follow-up to #3659 (closed). Verify-live remains #3693. Opened #3721.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3709
