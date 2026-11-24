## Summary

- New read-only eval for Orion's "how I came to be" page (`page:origin`): `orion/memory/pages/evals/origin_history.py`.
- It scores every stored version: does the page still state the memory loss (Decision 3), which kinds of records it cites, how much the text changes night to night, and whether each sentence's cited records actually back it.
- Faithfulness is a cheap deterministic floor: shared content words between a sentence and its cited records, per sentence and per cite, plus every date and number in a sentence must be held by a cited record.
- It also compares Orion's self-sense answers to "what are you" in the 14 days before and after the first page:origin version.
- Privacy: the terminal output is counts only, built from allowlists. Sentences, record text and record ids go only to a local owner-only file outside any git checkout. This report holds aggregates only.

## Outcome moved

Before this, page:origin had a cite check that only proved a cited record exists. Now one command shows, per nightly version, whether the loss sentence survived, how stable the page is, and which sentences or cites the records do not back. The first live run found one real problem (numbers losing their backing as cites rotate, #3776).

## Current architecture

page:origin folds nightly in orion-durable-runs (`orion/memory/pages/origin.py`, shared code in `fold.py`) and is stored in `self_concept_history` with inline `[kind:id]` cites. `check_origin_page` enforces Decision 3 at fold time. The page evals folder held `juniper_facts.py` (LLM Q&A) and `lately_coverage.py` (counts-only coverage). There was no page:origin eval.

## Architecture touched

- New eval module and tests under `orion/memory/pages/`. It reuses `check_origin_page`, `memory_loss_refs`, `FOLD_SPEC.sources` and `fold._norm` instead of new rules.
- No service, schema, bus, env or Docker change. No writes to the DB (session is `read_only`, SELECTs only).

## Files changed

- `orion/memory/pages/evals/origin_history.py`: the eval (pure scorers plus a read-only `main`).
- `orion/memory/pages/tests/test_origin_history_eval.py`: 18 tests on made-up pages and records.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: one status bullet.
- `docs/superpowers/pr-reports/2026-11-24-page-origin-eval-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none at runtime
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest orion/memory/pages/tests -q   -> 99 passed (18 new)
pyflakes on the two new files                             -> clean
```

## Evals run

Live, read-only, 2026-11-24, against the 6 stored versions (v1 2026-11-18 .. v6 2026-11-23):

```text
.venv/bin/python -m orion.memory.pages.evals.origin_history --out ~/.orion/evals/page_origin_history_2026-11-24.json
```

1. Memory-loss sentence (Decision 3): 6 of 6 versions state the loss and cite a loss record. 1 loss sentence per version. `loss_dropped` = 0 of 6. Both loss records in today's pool are cited by every version. The live check passes on all 6.

2. Cite mix (stance / self-definition / episode_memory):

| version | words | sentences | stance | self-definition | episode_memory |
|---|---|---|---|---|---|
| 1 | 273 | 14 | 9 | 6 | 1 |
| 2 | 273 | 14 | 8 | 4 | 1 |
| 3 | 270 | 14 | 8 | 5 | 1 |
| 4 | 270 | 14 | 8 | 5 | 1 |
| 5 | 270 | 14 | 8 | 6 | 1 |
| 6 | 270 | 14 | 8 | 5 | 1 |

   Today's pool offers 44 stance, 443 episode_memory and 8 sampled self-definition records. Every version cites 1 of 443 episodes (Orion's question on #3734): the fold prompt's rule 8 keeps day-to-day records out by design. The pool is today's, not each night's, so older versions' self-definition cites read low against today's sample.

3. Stability: the text has barely moved. Word-level similarity night to night is 0.991 to 1.0, v1 to v6 is 0.991. 14 of 14 claims are kept at every step, 0 added, 0 dropped. The cites move a lot: ref Jaccard night to night is 0.56 to 0.80, and v1 to v6 is 0.50 (10 kept, 6 dropped, 4 added). The fold keeps the old sentences and re-points their cites as the self-definition sample shifts.

4. Faithfulness (deterministic floor):
   - Sentences: 84 across 6 versions. 84 supported, 0 weak, 0 unsupported, 0 citing a missing record, 0 uncited.
   - Cites flagged unrelated (fewer than 2 shared content words with the sentence): 1 per version, the same cite each time. Checked by hand: it is a loss record cited next to a second loss record on a sentence that paraphrases it. Topic-related, low word overlap. A false positive of the floor, not an unrelated cite.
   - Dates: 18 date claims, 0 unbacked.
   - Numbers: 8 number claims, 3 unbacked (2 in v2, 1 in v5). The sentence kept its number while its cite moved to a self-definition version that does not hold that number. Filed as #3776.

5. Self-sense `what_are_you` answers, 14 days each side of the first page:origin version (2026-11-18 10:15Z), answer source `http`:

| window | n | mean self-label (lower is better) | mean grounded-record | answers stating a memory loss |
|---|---|---|---|---|
| before | 94 | 0.149 | 2.681 | 0 |
| after | 41 | 0.073 | 3.195 | 3 |

   Confounded: page:juniper and page:lately went live the same week, and the date cortex-exec began rendering page:origin into turns is not recorded here (UNVERIFIED). The loss mentions (0 to 3) are the one change that points at page:origin specifically.

The full per-sentence output is a local owner-only file, never committed: `~/.orion/evals/page_origin_history_2026-11-24.json` on the athena host.

## Docker/build/smoke checks

```text
Not applicable: no service, image or compose change.
```

## Review findings fixed

A code-review subagent reviewed the eval and its tests.

- Finding: a version that dropped both the loss sentence and its cite would pass `check_ok`, because the live check only sees cited records.
  - Fix: new `loss_dropped` flag against today's pool; docstring says what `check_ok` covers.
  - Evidence: `test_a_page_that_drops_the_loss_and_its_cite_still_flags_against_the_pool`.
- Finding: ordinal dates ("August 13th") were invisible to both the date and number checks.
  - Fix: ordinal suffixes in both patterns.
  - Evidence: `test_an_ordinal_date_is_a_claim_too`.
- Finding: every record backed the day before its own date.
  - Fix: keep the full timestamp and back a date with the record's UTC day or Denver day only.
  - Evidence: `test_a_date_is_backed_by_the_record_utc_or_denver_day_or_its_text`.
- Finding: a sentence whose cited records were all missing read as "uncited".
  - Fix: new `no_record` verdict (plus `no_content_words`).
  - Evidence: `test_missing_records_are_counted_not_crashed_on`.
- Finding: the public output used exclusion rules and a colon split that a future field or error format could leak through.
  - Fix: allowlists for every public block, and error names are taken by regex, else "other".
  - Evidence: `test_aggregates_carry_no_text_or_ids` now runs a failing page and checks every record, id and sentence; `test_error_names_never_carry_page_text`.
- Finding: the `--out` guard missed sibling worktrees and ran after the DB pass, and the file was briefly readable under the umask.
  - Fix: the path is checked before connecting and refused inside any git checkout or under /mnt/scripts. The file is created 0600 and the folder 0700.
  - Evidence: `test_out_path_refuses_any_checkout`.
- Finding: the self-sense "before" side covered months of unrelated change.
  - Fix: equal windows each side (`--self-sense-days`, default 14).
  - Evidence: self-sense test with an out-of-window row.
- Finding: the pool is today's, so older versions read as ignoring records they were never shown.
  - Fix: `pool_is_today: true` in the output; noted in this report.
- Finding: refs were read from anywhere in the text, not only cite brackets.
  - Fix: `page_refs` reads cite brackets only.
  - Evidence: `test_refs_come_only_from_cite_brackets`.
- Not changed (nits, stated in the module docstring): stemming is crude (died/dies stay apart), and dates are checked on month and day, not year.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the faithfulness check is a word-overlap floor. A sentence that reuses a record's words in a wrong claim reads as supported, so "84 of 84 supported" is a floor, not a verdict. Mitigation: the date and number checks catch the most common invented specifics. An LLM judge on a local route is a possible next step.
- Severity: low. Concern: the issue's first idea, a Q&A eval (does a turn answer "how did you come to be" from the page, old self block vs. with page), is not built here. The self-sense before/after table is a confounded proxy. Mitigation: `juniper_facts.py` is the pattern if it is wanted.
- Issues opened: #3776 (page text frozen while cites rotate; numbers lose backing).
- Issues closed: Fixes #3733.

## PR link

(added after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
