Fixes the live-proof gap on #3463: page:lately's first two live folds were thrown away for length.

## Summary

- page:lately's first two live folds (`dry_run=false`) were rejected at 353 and 354 words, against the 350-word cap. Nothing was written.
- Every core page (juniper, origin, lately) now ends its prompt rules with the same length rule: about 280 words, at most 12 sentences, and a page over 350 is thrown away. The cap is unchanged.
- After a page is rejected for length, the next attempt that day says how long the last page was and asks for a shorter one. The sentence budget is scaled from the last page. This works in the nightly fold and in the dry-run script.
- The note only lasts while it is true. A later reject for another reason clears it, a successful fold clears it, it is ignored on another day, and a dry run never sets it.

## Outcome moved

**Live dry runs of page:lately** (read-only, nothing written, `memory_distill`, 243 sources):

| Prompt | Runs | Words per attempt (cap 350) |
|---|---|---|
| "about 280 words", no sentence budget (first commit) | 3 | run 1: 389, 435, 380, all rejected · run 2: 389 rejected, then 349 passed · run 3: 322 passed |
| "about 280 words, at most 12 sentences" (final) | 3 | 291 (12 sentences) · 303 (12) · 331 (13), each passed on attempt 1 |

- A word target alone does not hold the model to the cap: its sentences ran 21 to 27 words, and a higher target made pages longer.
- A sentence budget does hold it: all 3 final runs landed under 350 on the first try.
- The retry note was exercised with the first-commit prompt (run 2: 389 words, then 349 after the note). With the final prompt, no retry was needed.

The page text stays host-side (`~/.orion/core_pages/`, mode 600). The repo is public.

## Current architecture

- Each page wrote its own length line: juniper and origin said "At most 300 words", and lately said "about 220 words in 10 to 14 sentences".
- A rejected fold retried with the identical prompt on the next tick, up to 3 attempts a night.

## Architecture touched

- `orion/memory/pages/fold.py`: `TARGET_WORDS`, `TARGET_SENTENCES`, `length_rule()`, `retry_note()`, `with_retry_note()`.
- `services/orion-durable-runs/app/core_page_fold.py`: a per-folder, per-Denver-day retry note.
- `scripts/core_page_dry_run.py`: the same retry note between attempts.

## Files changed

- `orion/memory/pages/fold.py`: the shared length rule and the retry note.
- `orion/memory/pages/juniper.py`, `origin.py`, `lately.py`: use `length_rule()`.
- `services/orion-durable-runs/app/core_page_fold.py`: carry the note between attempts.
- `scripts/core_page_dry_run.py`: same, in the attempt loop.
- Tests:
  - `orion/memory/pages/tests/test_core_page_length.py` (new)
  - `services/orion-durable-runs/tests/test_core_page_fold.py` (2 tests)
  - `orion/memory/pages/tests/test_core_page_origin.py`

## Schema / bus / API changes

- None. Prompt text and fold retry behavior only.

## Env/config changes

- None.

## Tests run

```text
orion/memory/pages/tests                        81 passed
orion/memory orion/harness/tests                3500 passed, 41 skipped
services/orion-durable-runs tests/test_core_page_fold.py   24 passed
mutation: retry note never applied -> 1 failed; stale note kept after a thin reject -> 1 failed;
          dry run sets the note -> 1 failed (all restored)
git diff --check clean
```

## Evals run

```text
6 live dry runs of page:lately (scripts/core_page_dry_run.py --page lately --attempts 3 --no-page); table above.
page:juniper and page:origin were not re-run live: their rule changed from "at most 300 words" to the shared one.
```

## Docker/build/smoke checks

```text
No image build; Python only. No deploy (not requested).
```

## Review findings fixed

The code review ran in a subagent on the first commit.

- Finding: the note outlived the reject it described. After an over-cap reject and then a thin reject, attempt 3 still asked for a shorter page. It also survived a fold.
  - Fix: a later reject replaces or clears the note, and a fold clears it. The dry-run script uses only the last attempt's note.
  - Evidence: `test_the_retry_note_is_for_today_only_real_attempts_and_the_last_reject`, mutation-checked.
- Finding: a manual dry run (the endpoint's default) could set the note for the next real nightly attempt.
  - Fix: dry runs never set it.
  - Evidence: same test, mutation-checked.
- Finding: "about 280" asked page:lately for a longer page than the old "about 220", and juniper/origin lost their hard "at most 300" ceiling.
  - Fix: confirmed live (the table above). Added the sentence budget, which brought all 3 runs under the cap on attempt 1.
- Finding: the test's day key used the UTC date.
  - Fix: it now uses the Denver date.

## Restart required

After merge, from the primary checkout on main:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then `curl -X POST 'http://localhost:8124/core-pages/lately/fold?dry_run=false'`, or wait for the 01:00-03:00 Denver window.

## Risks / concerns

- Severity: low. juniper and origin now get "about 280 words, at most 12 sentences" instead of "at most 300 words". The live page:origin fold was 294 words in 16 sentences, so it may come out a little shorter. Not re-run live here.
- Severity: low. Three runs are a small sample. 331 words is 19 under the cap. The retry note remains the second line of defence.
- **UNVERIFIED live:** no `page:lately` row has been written yet. #3463 keeps `verify-live` until a live fold passes and a turn shows the page.
- Issues opened/closed: none.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
