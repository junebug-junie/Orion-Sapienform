## Summary

- A core page sentence that uses a number is now refused unless one of the records it cites holds that number. Before, the nightly fold only checked quotes, so `page:origin` could keep an old sentence, move its cite to a different self-definition version, and ship a number the new record does not hold.
- The check lives in the shared `check_page` (`orion/memory/pages/fold.py`), so `page:juniper` and `page:lately` get it too.
- A record's own date (shown in every fold prompt) counts as backing, so "2026-11-20", "Nov 20" or "the 20th" cited to a record stamped that day passes.
- When a fold is refused for this, the retry note names the numbers and asks the next attempt to cite the record that holds them or drop them.
- The number rule moves out of the `page:origin` history eval into `fold.py`; the eval and the live check now share one implementation.

## Outcome moved

Failure mode closed: a page claim with a number its cited record does not hold (issue #3776). Re-running the live check on the stored `page:origin` versions now refuses exactly the versions the eval flags (v2: 2 unbacked numbers, v5: 1, v7: 1) and passes the rest.

## Current architecture

`check_page` refused uncited sentences, invented quotes, lists, over-cap pages and misgendering. Numbers were only counted, after the fact, by `orion/memory/pages/evals/origin_history.py`. The fold offers an evenly spaced sample of 8 self-definition versions, which shifts every time a new one lands.

## Architecture touched

- `orion/memory/pages/fold.py`: `sentence_numbers`, `unbacked_numbers`, `date_parts`; `check_page` emits `unbacked_number: <n> in: <sentence>`; `retry_note` handles it.
- `orion/memory/pages/evals/origin_history.py`: reuses the shared helpers (same pool as the live check, record dates included).
- `services/orion-durable-runs/app/core_page_fold.py`: comment only.

Not done, on purpose: pinning the cited self-definition version into the sample. The check is the truth gate whatever moved the cite (sample rotation or the model inventing a number); pinning would only address one cause and needs the current page plumbed into `FoldSpec.sources`. If refusals from rotation turn out frequent, pinning is the next step.

## Files changed

- `orion/memory/pages/fold.py`: number check, shared helpers, retry note.
- `orion/memory/pages/evals/origin_history.py`: use the shared helpers.
- `orion/memory/pages/juniper.py`: module docstring lists the new rule.
- `services/orion-durable-runs/app/core_page_fold.py`: retry-note comment.
- `orion/memory/pages/tests/test_core_page_origin.py`: regression test for the v2/v5 shape (kept sentence, cite moved to a version without its number -> refused; cited to the version that holds it -> passes).
- `orion/memory/pages/tests/test_core_page_juniper.py`, `test_core_page_lately.py`, `test_core_page_length.py`: shared-check, record-date and retry-note tests.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `check_page` can now return `unbacked_number` errors.
- Compatibility notes: eval counts now normalize number forms ("1,400" = "1400", "08" = "8", "1990s" -> 1990) and count record dates as backing; `number_claims` on `page:origin` rose by 1-2 per version (suffixed numbers are now counted), `numbers_unbacked` unchanged.

## Env/config changes

- None. No `.env_example` changed.

## Tests run

```text
.venv/bin/python -m pytest -q orion/memory/pages/tests services/orion-durable-runs/tests
1197 passed, 157 skipped
```

`tests/test_core_page_origin_postgres.py` does not exist in this repo.

## Evals run

```text
.venv/bin/python -m orion.memory.pages.evals.origin_history --no-self-sense   (read-only, live DB)
page:origin  v1 ok 0 unbacked | v2 REFUSED 2 | v3 ok 0 | v4 ok 0 | v5 REFUSED 1 | v6 ok 0 | v7 REFUSED 1
```

Read-only impact check of the shared check on every stored version of the other pages (record texts from today's fold pool):

```text
page:juniper  10 of 10 versions pass, 80 number claims, 0 unbacked
page:lately    7 of 7 versions pass, 45 number claims, 0 unbacked
```

Stored versions are not rewritten. The live `page:origin` (v7) keeps its 1 unbacked number until the next fold passes the new check.

## Docker/build/smoke checks

```text
Not run: pure-Python change in orion/, picked up by the orion-durable-runs image rebuild.
```

## Review findings fixed

- Finding (material): record dates shown in the prompt were not accepted as backing, so a correct lately page writing "2026-11-20" or "the 20th" would be refused every night.
  - Fix: each cited record's date (year, month, day) joins the number pool (`date_parts`).
  - Evidence: `test_a_date_written_from_the_record_stamp_is_backed_by_the_cited_record`.
- Finding (minor): numbers with a letter suffix ("1990s", "10pm", "3x") escaped the check.
  - Fix: `_NUMBER` accepts a letter suffix; `_bare_number` strips it. "S12" and "v2" still do not count.
- Finding (minor): equal numbers written differently did not back each other ("1,400,000" vs "1400000", "2.0" vs "2", "08" vs "8").
  - Fix: `_bare_number` normalizes thousands commas, leading zeros and trailing decimal zeros.
- Finding (nit): retry-note test name and the runner comment said over-cap only.
  - Fix: renamed test with an `uncited -> None` case; comment updated.
- Finding (nit): a minus sign is not part of the number.
  - Fix: documented in `_bare_number`.
- Not fixed (minor): abbreviated or day-first month names ("Nov 20") are not exempt as dates; with record dates now backing them, they only matter when the date is neither the record's day nor in its text, which is the claim this check should refuse.

## Restart required

```bash
# orion-durable-runs picks this up through the normal post-pull rebuild after merge; no manual step.
```

## Risks / concerns

- Severity: low. Concern: more nightly fold refusals for `page:origin` while the self-definition sample keeps rotating. Mitigation: the retry note names the number; the old page stays live on refusal; pinning the cited version is the follow-up if refusals repeat.
- Severity: low. Concern: digits only, so "5" is not backed by "five". Live data had 0 such cases across all three pages.
- Issues: Fixes #3776. None opened.

## PR link

(this PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
