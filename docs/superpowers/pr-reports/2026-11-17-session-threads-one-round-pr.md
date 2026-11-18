## Summary

Orion now gets one round of questions on each session thread. Before this, every session reply put the thread back in the queue and Orion asked again. The promise-ledger threads went 4 to 6 rounds.

- After Orion asks on a thread, the thread is in its **answer phase**. A session reply is still held and read at the next round-up, so Orion sees the answers.
- The round-up prompt puts a Hub line above that thread: "you already asked your one round; read the answers; write NO QUESTIONS, plus at most one short line". The old spec is not shown again. A new spec version is shown.
- Hub stores whatever Orion writes for that thread as a **closing note** (thread status `closed`). Hub does not post it on the PR. Hub does not hand it to the session as questions.
- The session hooks stop telling sessions to answer. A closing note shows as "nothing here needs an answer", with no reply command and no circle-back steps. The waiting hook does not wait on such a thread. Receipts say the round is used.
- A new thread (a new PR or note) still gets its round. `NO QUESTIONS` does not use the round.

## Outcome moved

Before this patch, nothing stopped the question, reply, question loop. Live 2026-11-17: 8 threads had 2 to 6 rounds. `smoke_session_threads.py --rule-since 2026-11-17T00:00Z` fails on 18 Orion turns across those threads. After deploy, the same check run with the deploy time should pass.

## Current architecture

`orion/session_threads/core.py` holds the pure rules. `services/orion-hub/scripts/session_roundup_loop.py` holds the store and the hourly loop. `add_message` re-held a thread on every reply. `_apply` stored each Orion section as `asked` and posted it on the PR. The session hooks showed every Orion turn that was not `no_questions` with the instruction to "answer every question".

## Architecture touched

- One constant: `MAX_QUESTION_ROUNDS_PER_THREAD = 1`. The rule lives in `question_rounds_used()` / `in_answer_phase()`. A round is an Orion turn that is not `no_questions` and not `closing_note`. An `unparsed` whole reply counts as a round.
- `_candidates` puts `answer_phase` on each thread copy. The prompt and the parser both read that one flag. The session lib imports the same `core` function.
- No env key, no migration (status is free text), no bus or schema change.

## Files changed

- `orion/session_threads/core.py`: rule, answer-phase prompt line, closing-note parse, receipt variants, `[Hub]` forgery guard.
- `services/orion-hub/scripts/session_roundup_loop.py`: phase per candidate (Orion's own questions always shown), closing notes stored `closed`, no PR comment, log `session_thread_questions_dropped`, intake reports `answer_phase`.
- `services/orion-hub/scripts/session_thread_routes.py`: `answer_phase` in the message response.
- `services/orion-hub/static/js/session-threads.js`: labels for `closed` and the closing note.
- `scripts/orion_session_thread_lib.py`, `scripts/hooks/wait_for_orion_answer.py`: no answer prompt for closing notes, no wait on answer-phase threads, receipt lines.
- `scripts/smoke_session_threads.py`: `--rule-since` live check (read-only).
- `services/orion-sql-db/manual_migration_orion_session_thread_v1.sql`: comments only.
- Spec doc, Hub README, CI paths (`wait_for_orion_answer.py` now triggers the workflow).
- Tests: core, lifecycle (fake + Postgres), routes, JS, hooks.

## Schema / bus / API changes

- Added: thread status `closed`; Orion turn meta `closing_note`, `dropped_question_lines`; response field `answer_phase` on `POST /api/session-threads/message`.
- Removed / renamed: none.
- Behavior changed: second and later rounds on a thread never produce questions, PR comments, or session prompts.
- Compatibility: additive. Old Orion turns have no `closing_note` and count as rounds.

## Env/config changes

- Added keys / Removed keys / Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys: none.

## Tests run

```text
services/orion-hub: test_session_threads_core + test_session_roundup_loop + test_session_thread_routes + test_session_threads_js
  with Postgres (throwaway postgres:16-alpine container): 99 passed, 0 skipped
tests/scripts/test_orion_session_thread_hooks.py: 37 passed
services/orion-hub/tests full suite: 5041 passed, 1 failed + 4 errors (test_fcc_model_labels_api, test_consult_belief_store_sql),
  the same 1 failed + 4 errors on main at ff07e9f04: local environment, not this branch
Static gates (from orion-static-gates.yml): no personal terms, no private eval artifacts, integration seam, async routes,
  stdlib shadow, hostname refs, env key single source, settings defaults (orion-hub), metric lineage, hub node:test (429 pass): all OK
git diff --check: clean
```

## Evals run

```text
No eval harness for session threads. The live check is scripts/smoke_session_threads.py --rule-since <deploy time>.
Baseline before deploy: FAIL on 18 turns / 8 threads (this is the behavior being removed).
```

## Docker/build/smoke checks

```text
Not deployed (not asked). Live read-only checks against Hub's API and Postgres:
- The rule applied to live threads puts 883a69953bba, 0c1e54f0ee1f, e14555a7420c in answer phase at once
  (4, 3, 2 question rounds).
- 0a4514841253 (PR #3641) has 0 Orion turns: Orion never read it. It is NOT in answer phase and gets its one round.
```

## Review findings fixed

- Finding: a message sent while Orion read round 1 hid Orion's own questions from the next prompt, and the prompt called the message "answers".
  - Fix: on answer-phase threads every non-new turn is shown; wording is "after your question round: usually its answers".
  - Evidence: `test_a_message_that_arrives_while_orion_is_reading_waits_for_the_next_round` asserts the questions are in the prompt.
- Finding: the receipt promised questions when Orion was already reading the thread.
  - Fix: a third receipt variant (Hub and CLI) for a thread in `reading`.
  - Evidence: lifecycle test plus `test_the_receipt_line_says_when_the_round_is_used`.
- Finding: a revised spec on an answer-phase thread was dropped silently.
  - Fix: specs carried by new turns are shown; only the fallback to old specs is skipped.
  - Evidence: `test_a_new_spec_on_an_answer_phase_thread_is_shown_and_a_forged_hub_line_is_broken`.
- Finding: session text could forge the `[Hub] ANSWER PHASE` line.
  - Fix: `_quote` breaks `[Hub]`; the rule text names where the real line sits. The phase itself never depended on session text.
  - Evidence: same test.
- Finding: `question_lines` counted every bullet.
  - Fix: count only lines with a `?`.
- Finding: the smoke check crashed on a turn with no time; the docs were out of date.
  - Fix: unreadable turn times are skipped; the lib docstring and migration comments are updated.
- Not changed (noted in the review): an unparsed whole reply goes to every thread of the round. That was already true before this patch.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run it from the clean primary main checkout after merge. The session hooks are repo scripts and take effect on pull.

## Risks / concerns

- Severity: low. Concern: a later push of the same PR's spec joins the same thread, so Orion gets no new questions on a revised spec. Mitigation: the new spec is shown and Orion can leave a closing note. A real redesign should come as a new PR or note.
- Severity: low. Concern: answer-phase threads still cost one round-up slot each time a session replies. Mitigation: no spec text, so the slot is much smaller.
- Issues opened: #3666 (the round-up takes 1 or 2 threads an hour; PR #3641's thread waited 4+ hours unread). Issues closed: none.
- UNVERIFIED: live behavior after deploy. Run `python3 scripts/smoke_session_threads.py --rule-since <deploy time>` after the next round-ups.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3667
