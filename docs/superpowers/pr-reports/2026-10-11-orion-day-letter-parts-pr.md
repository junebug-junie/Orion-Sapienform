## Summary

- Design for rereading Orion's Day letters grounded in the day's records, including the decisions Juniper made on 2026-10-11:
  - corrections persist and drive prior revision, through curiosity runs only
  - no privacy gate

  File: `docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md`.
- Patch 1 of that design: the letter can now be addressed by part. The email shows `¶N` before each note paragraph and `carry N` on each carry-forward item, plus a one-line hint ("ask Orion about '2026-10-09 ¶3'").
- New `orion/orion_day/letter_parts.py` does four things:
  - splits a letter into numbered parts, using the same splitter as the email
  - resolves carry-forward citations against the stored material
  - checks a note paragraph's concrete claims (numbers, timestamps, PRs, prior ids, code spans, quotes) against the day's raw records
- `budget.material_ref_records` is now the single map from each reference to its stored record. `material_refs` derives from it, with an identical key set.

## Outcome moved

Before, a conversation about a letter had nothing to point at, and Orion's prose note cited no records (0 citations across the last 5 letters).

After, on the real 2026-10-09 letter:
- 24 numbered paragraphs and 10 carry items
- all 21 carry citations resolve to stored records
- the claim check pulls 41 checkable tokens from the note; 40 are found in that day's records, each with the record that holds it ("260.6 hours" → `curiosity:e7d03d2ecdbb`)
- 1 is not found: a quote in ¶18 attributed to the reveries that appears nowhere in the day's records verbatim

## Current architecture

- The email rendered `note_md` and `carry_forward_md` as single markdown blobs.
- The letter writer cites records with `[kind:id]` refs defined in `orion/orion_day/budget.py`.
- `material` stores every record in full. Nothing outside the writer's own grounding count resolved those refs.

## Architecture touched

- `orion/orion_day` (pure helpers)
- Hub email rendering and its eval

No bus, schema, env or DB changes.

## Files changed

- `orion/orion_day/letter_parts.py`: new. `split_note`, `split_carry`, `find_part`, `resolve_citations`, `extract_claim_tokens`, `claim_check`.
- `orion/orion_day/budget.py`: `material_ref_records`. `material_refs` derives from it.
- `services/orion-hub/scripts/orion_day_email.py`: `numbered_html` and `numbered_text`, used by the HTML and plain-text parts.
- `services/orion-hub/templates/orion_day_letter.html.j2`: the reference hint line.
- `services/orion-hub/evals/run_orion_day_email_eval.py`: the full-text check runs per part. It also asserts the splitter lost or reordered nothing.
- Tests: `orion/orion_day/tests/test_letter_parts.py` (new, 15) and `services/orion-hub/tests/test_orion_day_letter.py` (+1).
- `docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md`: the design and decisions.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the email shows part numbers. Numbers are computed at render time and never stored.
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
pytest orion/orion_day                                   64 passed
services/orion-hub: tests/test_orion_day_letter.py + evals/test_orion_day_email_eval.py   52 passed
services/orion-durable-runs: test_orion_day_graph.py + test_orion_day_postgres.py         17 passed, 3 skipped
```

## Evals run

```text
services/orion-hub/evals/run_orion_day_email_eval.py (fixture letter)   0 failures
Same check on the real 2026-10-09 letter: 1 failure, "html missing curiosity:3afa48b9669b:journal_body".
  Pre-existing: identical on main. Checker false alarm: the write-up's inline "1." renders as
  <ol> numbering, so the normalised text differs. No content is missing.
```

## Docker/build/smoke checks

```text
No container change. The real 2026-10-09 letter was rendered through build_notification:
24 ¶ marks, 10 carry marks. Claim check across all paragraphs took 0.58s on 705KB of material.
```

## Review findings fixed

- Finding: a `* * *` / `- - -` rule inside the carry-forward took a carry number.
  - Fix: a thematic break ends the list and is unnumbered, as in markdown.
  - Evidence: `test_a_rule_inside_the_carry_forward_takes_no_number`
- Finding: a bare date leaked its year as an integer claim, which reads as "found" everywhere.
  - Fix: dates are claimed first and skipped.
  - Evidence: `test_a_bare_date_is_not_an_integer_claim`
- Finding: code spans and quotes containing `"` or `\` were never found, because the search ran on `json.dumps` output.
  - Fix: search the records' leaf values instead.
  - Evidence: `test_quotes_with_json_escaped_characters_are_found`
- Finding: the eval took its "what must be present" list from the splitter under test.
  - Fix: the eval also asserts the parts rejoin to the original text, in order.
  - Evidence: eval green on the fixture.
- Finding (nit): a `~~~` line closed a ``` fence.
  - Fix: only the opening marker closes.
  - Evidence: `test_only_the_opening_fence_marker_closes_a_fence`
- Finding (nit): a decimal was "found" inside a version string (`0.27` in `v0.27.1`).
  - Fix: the needle refuses a following `.digit`.
  - Evidence: `test_a_decimal_inside_a_version_string_is_not_support`
- Not fixed (nits, rare in real letters, which use `- ` bullets and plain prose):
  - reference-style link definitions
  - indented code blocks with blank lines
  - setext headings get a number
  - ordered carry lists show both "5." and "carry 1"
  - indented top-level bullets

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: "found" is weak support for common short decimals (e.g. `0.10`).
  - Mitigation: each hit names its record, so the reader can see where the number came from. Verdicts stay with Orion and Juniper.
- Severity: low
  - Concern: the existing email checker false alarm (inline "1." lists) remains.
  - Mitigation: noted for follow-up. It is unrelated to this patch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2606

🤖 Generated with [Claude Code](https://claude.com/claude-code)
