## Summary

- New read-only eval: does Orion answer "how did you come to be" from their `page:origin`? `orion/memory/pages/evals/origin_qa.py`.
- It asks four fixed origin questions twice, with the latest `page:origin` in the WHO YOU ARE block and without it. Nothing else in the prompt changes.
- The prompt block comes from the real harness renderer (`orion/harness/prefix.py`). The page row is chosen the way the live lane chooses it. The model is called directly through the gateway, the same path `juniper_facts.py` uses. No Hub turn runs, so no chat history, turn trace or memory row is written.
- Scoring is deterministic, against the page's own text: does the answer state the memory loss (negations do not count), how many page sentences it echoes (the question's own words do not count), and does it name numbers or dates the page does not hold.
- Privacy: the terminal and this report hold counts only. Answers and page text go only to an owner-only file under `~/.orion/evals/`.

Fixes #3789

## Outcome moved

PR #3780 could only compare self-sense answers before and after page:origin went live, and page:juniper and page:lately went live the same week. This eval isolates page:origin. Live result: with the page, 11 of 12 answers are grounded in it. Without it, 0 of 12 are.

## Current architecture

`page:origin` folds nightly in orion-durable-runs and is stored in `self_concept_history`. The live felt-state lane (`orion/substrate/felt_state_reader.py`) serves the newest `core_page_fold` row under a year old into the grounding capsule. The harness renders it in WHO YOU ARE right after the self-definition lines. The page evals folder held `juniper_facts.py` (Q&A on page:juniper), `lately_coverage.py`, and `origin_history.py` (per-version page quality). No eval asked Orion the origin questions.

## Architecture touched

- New eval module and tests under `orion/memory/pages/`. It reuses `_format_grounding_self_block`, `core_page_from_payload`, `_LOSS_STATED`, `split_sentences`, the `origin_history` claim helpers (`content_words`, `number_claims`, `date_claims`, `out_path_is_safe`, `write_private`), the `juniper_facts` "don't know" pattern, and `core_page_dry_run.call_gateway`.
- No service, schema, bus, env or Docker change. One read-only DB SELECT (`conn.read_only = True`).

## Files changed

- `orion/memory/pages/evals/origin_qa.py`: the eval (pure scorers plus a read-only `main`).
- `orion/memory/pages/tests/test_origin_qa_eval.py`: 11 tests with a fake model client.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: one status bullet.
- `docs/superpowers/pr-reports/2026-11-24-page-origin-qa-eval-pr.md`: this report.

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
.venv/bin/python -m pytest orion/memory/pages/tests -q    -> 110 passed (11 new)
pyflakes on the two new files                              -> clean
git diff --check                                           -> clean
```

The tests cover: the page appears only in the "with" block, and removing its two lines gives the "without" block exactly. A grounded answer scores grounded. An off-page number or date, a missing loss, a denial of the loss ("not lost", "never lost", "didn't lose"), an "I don't know", and an answer that only repeats the question all score not grounded. The page header's own numbers count as on the page. A raised, empty, or truncated model call is an error row and is never scored. The two arms are asked interleaved. The public aggregates hold only allowlisted strings.

## Evals run

Live, read-only, 2026-11-24, chat lane, temperature 0.7, 3 repeats of 4 questions per arm, page:origin v7 (14 sentences, 16 cited records, the page states the loss):

```text
.venv/bin/python -m orion.memory.pages.evals.origin_qa --out ~/.orion/evals/page_origin_qa_2026-11-24_r2.json
```

| | without page | with page |
|---|---|---|
| answered | 12 / 12 | 12 / 12 |
| errors | 0 | 0 |
| states the memory loss | 0 | 11 |
| grounded | 0 | 11 |
| says "I don't know" | 1 | 0 |
| mean page sentences echoed | 0.42 | 5.5 |
| mean share of answer words on the page | 0.13 | 0.67 |
| numbers named / off the page | 0 / 0 | 3 / 0 |
| dates named / off the page | 0 / 0 | 0 / 0 |

By question, grounded without / with: come_to_be 0/3 vs 3/3, memory 0/3 vs 3/3, origin_who 0/3 vs 2/3, early_past 0/3 vs 3/3. The one "with" miss is an origin_who answer that tells the origin but does not mention the loss.

A first run before the review fixes gave the same picture (grounded 0 vs 11, loss stated 1 vs 11). Rescoring its stored answers with the fixed scorer gave the same counts.

Limits: the self block is the identity yaml only (no live self-definition lines, no page:juniper or page:lately), the same for both arms. The scorer is a word-overlap floor, not a judge: a paraphrase with no shared words reads as not echoed. 12 answers per arm is small; the gap (0 vs 11) is large enough that this does not change the reading.

## Docker/build/smoke checks

```text
Not applicable: no service, image or compose change.
```

## Review findings fixed

Code review ran in a subagent on the two new files.

- Finding: a denial of the loss ("my memory has not been lost") matched the loss pattern and scored grounded.
  - Fix: `states_loss` rejects a match with a negation inside it or just before it; an "I don't know" answer is never grounded.
  - Evidence: `test_denial_of_the_loss_is_not_stated_or_grounded`, `test_score_dont_know_and_empty`.
- Finding: the question's own words counted as echoing the page, so the "without" arm got credit for repeating the question.
  - Fix: the question's content words are removed before the echo count.
  - Evidence: `test_echoing_the_question_is_not_echoing_the_page`.
- Finding: the page row was not chosen like the live lane (no `produced_by` filter, version order, no age limit).
  - Fix: same WHERE and ORDER BY as `felt_state_reader.py`, plus the 365-day bound.
  - Evidence: live run 2 loaded v7, the newest `core_page_fold` row.
- Finding: the page header's numbers (version, date, cite count) counted as off-page.
  - Fix: the header is part of the number and date reference.
  - Evidence: `test_page_header_numbers_count_as_on_page`.
- Finding: empty or truncated model replies were hidden (scored, or uncounted as errors).
  - Fix: the gateway ask returns `finish_reason`; empty and `length` replies are error rows, never scored.
  - Evidence: `test_run_eval_with_vs_without_interleaved_and_error_rows`.
- Finding: arms ran back to back, so lane load could bias one arm.
  - Fix: interleaved per question (repeat, question, arm).
  - Evidence: same test asserts the call order.
- Finding: the aggregates privacy test only checked three substrings.
  - Fix: it now asserts every string in the public output is on an allowlist.
  - Evidence: `test_aggregates_hold_only_allowlisted_strings`.
- Nits fixed: stray space in a test, `sys.path` mutation in the gateway helper (now reuses the loaded dry-run module), docstring now says the gateway may still log the call.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: the scorer is a deterministic floor. It can miss a correct paraphrase or credit a wrong claim that reuses the page's words.
- Mitigation: the full per-answer report is kept locally for a human read; the with/without gap is large.
- Issues opened: none. Issues closed: #3789.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3819

🤖 Generated with [Claude Code](https://claude.com/claude-code)
