# Session threads: quoted session data cannot cut or misroute Orion's reply

## Summary

- Orion asked its one round of questions on thread `c06ae999270c` (PR #3667). This PR answers them.
- Q1 (parser): real bug, fixed. Session text that Orion quotes back no longer cuts a thread's section short. A closed session-data fence in Orion's reply is replaced by one Hub line. A heading inside a closed code block is ignored when it names a thread that is not in the round.
- A heading that names a thread IN the round, found inside a quote (code block or data fence), makes the whole round `unparsed` (the safe result: never posted). Hub cannot tell a quoted copy from the real heading.
- The heading pattern no longer crosses line ends: it ate the first line under a heading when that line started with `-`, `:` or `(` (a bulleted question was silently lost), and it was quadratic on blank-line runs.
- Q2 (failed round): not a bug. A failed round stores no Orion turn, so it never uses the thread's one question round. New test proves it.
- Q3 (delivery): not a bug. Questions reach a session by thread (any sender session, or the branch), not by the `gh pr create` event. New test proves it after a second send for the same PR.

## Outcome moved

Before: an Orion reply that quoted a spec line `### thread:dead` inside a code block lost every question after the quote on that thread. A quoted data fence holding a round thread's heading made the whole round `unparsed`, and still does (by design). After: unknown-id headings in quotes are harmless; round-id headings keep the old safe behavior.

## Current architecture

`orion/session_threads/core.py:parse_roundup_reply` cut Orion's reply at every `### thread:<id>` line. An unknown id ended the previous section. A round id twice or out of order unparsed every thread.

## Architecture touched

Hub only: the pure parser in `orion/session_threads/core.py`. No store, route, hook, or schema change.

## Files changed

- `orion/session_threads/core.py`: `strip_echoed_data`, `_code_spans`, `_in_spans`; parser uses them; a section holding only Hub's cut note is empty.
- `services/orion-hub/tests/test_session_threads_core.py`: echoed fence / code block parse; a quoted round heading (one copy or two, code block or fence) unparses the round; unclosed opener hides nothing; a bulleted section keeps its first line; a section with no words is empty; linear on long inputs.
- `services/orion-hub/tests/test_session_roundup_loop.py`: failed round does not use the question round; questions reach the session after a later send for the same PR.
- `docs/superpowers/specs/2026-11-15-session-threads-design.md`: the three rules written down.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: parser only (see Summary).
- Compatibility notes: stored turns unchanged in shape.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest -q services/orion-hub/tests/test_session_threads_core.py \
  services/orion-hub/tests/test_session_roundup_loop.py services/orion-hub/tests/test_session_thread_routes.py \
  services/orion-hub/tests/test_session_threads_js.py tests/scripts/test_orion_session_thread_hooks.py
122 passed, 24 skipped   (skips = Postgres mode)
ORION_MEMORY_EPISODE_TEST_DATABASE_URL=<throwaway postgres:16> pytest test_session_roundup_loop.py test_session_thread_routes.py
62 passed
scripts/check_personal_terms.py: OK; scripts/check_integration_seam.py: OK
```

## Evals run

```text
No eval harness for session threads (none existed before this PR). The parser rules are covered by gate tests.
```

## Docker/build/smoke checks

```text
Not run: parser-only change. After deploy: python3 scripts/smoke_session_threads.py --rule-since <deploy time>
```

## Review findings fixed

- Finding (must-fix): hiding round-thread headings inside code blocks let a quoted heading stand in for the real one and sent t2's questions to t1's PR.
  - Fix: code blocks and cut fences hide only headings for threads outside the round; a fence holding a round heading is not cut.
  - Evidence: `test_a_quote_cannot_hide_a_round_threads_heading` (both forms).
- Finding: a section holding only Hub's cut note was posted as a question.
  - Fix: such a section is empty, so the thread is `unparsed`.
  - Evidence: assertion in `test_session_data_orion_quotes_back_never_cuts_a_section_short`.
- Finding: Orion's words on a fence marker's line were deleted.
  - Fix: the cut runs marker to marker, not line to line.
  - Evidence: same test, inline quote keeps `1. a?` and `2. a2?`.
- Finding: the code-block lookup was quadratic (4 s on 130 KB).
  - Fix: bisect.
  - Evidence: `test_the_code_block_check_is_not_quadratic` (< 1 s).
- Finding (second review): ONE quoted round heading (no real copy) still sent t1's quoted session text to t2's PR.
  - Fix: a round heading inside a closed code block or kept data fence unparses the round.
  - Evidence: `single` case in `test_a_quote_cannot_hide_a_round_threads_heading`.
- Finding (second review, old bug): the heading pattern's `\s*` crossed newlines: ate a bulleted first line as a title, and was quadratic on blank lines (73 s on a 100 KB reply).
  - Fix: horizontal whitespace only (`[^\S\n]*`) in both heading patterns; the fence close marker's tail is bounded to 80 chars.
  - Evidence: `test_a_bulleted_section_keeps_its_first_line`, blank-line and many-close-marker cases in the linear test.
- Finding (second review, nit): a section like `- [Hub note]` would post.
  - Fix: a section needs a word character besides Hub's notes; a reply of only headings and notes is a failed round.
  - Evidence: `test_a_section_with_no_words_is_empty_and_a_reply_of_only_headings_and_notes_fails`.
- Finding (nit): ```` ```py ```` closed a block.
  - Fix: a block closes only on a bare fence of the same character, at least as long; lines split on `\n` only.
  - Evidence: `longer` case in the parse test.

## Restart required

```bash
python3 scripts/rebuild_services_from_git_diff.sh   # after merge, from the clean primary main checkout (rebuilds orion-hub)
```

## Risks / concerns

- Severity: low
- Concern: a round thread's id quoted back inside a quote unparses the whole round (no PR comment that hour). This is deliberate: safer than guessing which heading is real.
- Mitigation: the whole reply is still stored and handed to each session.
- Issues opened/closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3688
