## Summary

Orion asked three questions on session thread b3053f6df376 (PR #3688). This PR answers them in code.

- Hub sometimes cannot cut Orion's round-up reply into one section per thread (the reply is `unparsed`). That no longer spends the thread's one round of questions. Hub keeps the whole reply on the thread and sends it to nobody: no PR, no session, and the answer-wait hook cannot claim it. Hub then reads the thread again next hour. It counts as one failure, so a thread that is unparsed 3 times in a row ends `failed`.
- Some rounds fail because the body never gave Orion a turn: the GPU pool kept the run waiting past its deadline (heat, a busy lane), or the run was still waiting 30 minutes after it. Those are no longer strikes. The thread is held again with its count unchanged, and Hub writes one line per streak, not one per hour.
- These are still strikes: a deadline hit after Orion's turn was attempted, a run that is overdue while running or paused, and a request the pool can never place. A 14-day backstop also counts a body refusal once the thread's oldest unread message is that old.
- Question 2 is not built. Hub does not strip heading lines copied from session data, because a session can write the real heading byte for byte. Two tests pin this (see Risks).

## Outcome moved

- **Before:** a parse accident ended a thread's question round with Orion's questions sent nowhere. A hot week could leave every held thread `failed`, and only a new session message brought it back.
- **After:** the parse accident costs one hour, and a hot week costs only waiting.

## Current architecture

`orion/session_threads/core.py` holds the pure rules. `services/orion-hub/scripts/session_roundup_loop.py` holds the store and the hourly loop.

- An unparsed answer was stored as status `asked` and counted by `question_rounds_used`.
- `fail_round` added one to `roundup_failures` for every failure reason.
- `add_message` already re-holds a `failed` thread and resets its count. That path is unchanged and still tested.

## Architecture touched

- Hub's session-thread store and loop.
- The pure rules in core.
- The circle-back hook and its library.
- The Hub chat JS labels.

No bus, schema, migration, or env change.

## Files changed

- `orion/session_threads/core.py`:
  - `question_rounds_used` skips `meta.requeued`.
  - New `is_body_refusal`, `WAITING_RUN_STATUSES`, `BODY_REFUSAL_MAX_AGE_SEC`, `unparsed_text`, `body_refusal_text`.
- `services/orion-hub/scripts/session_roundup_loop.py`:
  - New `requeue_unparsed`.
  - `fail_round` takes a non-strike mode, writes one line per streak, and applies the age backstop.
  - `pending_for` and `mark_delivered` refuse requeued turns.
  - `_candidates` hides requeued turns.
  - `_poll` marks `workflow_deadline_after_turn`.
- `scripts/hooks/wait_for_orion_answer.py` and `scripts/orion_session_thread_lib.py`: a requeued turn is not an answer.
- `services/orion-hub/static/js/session-threads.js`: labels for the requeued turn and the Hub line.
- Tests:
  - `services/orion-hub/tests/session_thread_fakes.py` mirrors the SQL.
  - `test_session_roundup_loop.py`, `test_session_threads_core.py`, `test_session_threads_js.py` and `tests/scripts/test_orion_session_thread_hooks.py` cover the new behaviour.
- Docs: `docs/superpowers/specs/2026-11-15-session-threads-design.md` and `services/orion-hub/README.md`.

## Schema / bus / API changes

- Added: turn meta keys `requeued` (Orion turn), `reply_unparsed` (Hub line), and `body_refusal` (Hub line). These are JSONB keys in an existing column.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - The tick result of a failed round carries `strike`.
  - `GET pending` and `POST delivered` skip requeued turns.
- Compatibility notes: none live. There are 0 unparsed turns in the live store today.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub: test_session_threads_core.py test_session_roundup_loop.py test_session_thread_routes.py test_session_threads_js.py
  with ORION_MEMORY_EPISODE_TEST_DATABASE_URL on a throwaway postgres:16 container (not the live DB)
  -> 144 passed, 0 skipped (each lifecycle test runs on the fake store AND the real SQL)
tests/scripts/test_orion_session_thread_hooks.py -> 38 passed
tests/test_no_personal_terms.py -> 7 passed
scripts/check_integration_seam.py -> OK; check_settings_defaults.py orion-hub -> OK; git diff --check -> clean
```

## Evals run

```text
No eval harness for session threads. The lifecycle tests drive the live code paths (intake, tick, pending/delivered, the hook as a subprocess).
```

## Docker/build/smoke checks

```text
Not run: Python and JS only; no dependency, compose or env change.
Live state checked read-only: Hub was rebuilt at df45f4f71 (#3688) at 01:45Z. Thread b3053f6df376 is `asked`, with failures 0.
No session-roundup run has ever failed on body grounds (28 runs, all completed). The 4 failed rounds so far were all `orion_turn_cut_short`.
```

## Review findings fixed

A code review ran in a subagent. Each finding below lists its fix and the evidence.

- Finding (must): the answer-wait hook claimed the requeued uncut reply and woke the session with it.
  - Fix: the hook skips requeued turns, and `has_questions` returns False for them. `mark_delivered` refuses them in SQL and in the fake.
  - Evidence: `test_the_waiter_keeps_waiting_past_a_requeued_uncut_reply_and_wakes_on_the_real_questions`, plus a `mark_delivered == 0` check.
- Finding (should): `workflow_deadline` hit mid-turn, and a hung overdue run, could retry forever.
  - Fix:
    - A deadline after a turn becomes `workflow_deadline_after_turn`, which is a strike.
    - An overdue run is a body refusal only while it waits for the pool.
    - A 14-day backstop counts every body refusal after that age.
  - Evidence: the hot-week test, the never-ends test, and `test_a_body_refusal_on_a_very_old_message_counts_so_no_thread_retries_for_ever`.
- Finding (should): the requeued reply could reappear in a later answer-phase prompt as history.
  - Fix: `_candidates` drops requeued turns.
  - Evidence: the round-3 answer-phase assert in the unparsed lifecycle test.
- Finding (should): `gpu_pool_unavailable:shed:` had no producer, so its test passed for the wrong reason.
  - Fix: removed it. The test now asserts False.
- Finding (nit): a NULL meta in the `@>` filter.
  - Fix: `COALESCE`.
- Finding (nit): no test for a second apply or for a message sent while Orion is reading.
  - Fix: `test_requeue_is_once_per_round_and_a_message_sent_while_reading_is_read_next_round`.
- Finding (nit): a quiet body refusal pushes no chat frame.
  - Not fixed: cosmetic. The status stays `held`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build   # from clean primary main after merge (post-pull rebuild does this)
```

## Risks / concerns

- Severity: low.
  - Concern: body refusals now retry for up to 14 days, at about one background durable run per 3-4 hours while the body is busy.
  - Mitigation: a deadline after a turn is a strike, and the backstop counts every body refusal after 14 days.
- Severity: low.
  - Concern: question 2 is not done. Hub keeps "a round thread's heading in a quote means the whole round is unparsed".
  - Reason: a session knows its own thread id, so its text can hold Orion's real heading byte for byte. Stripping it could post one thread's questions on another thread's PR. Since this PR, the safe rule costs one hour and no question round.
  - Note: the spec's own examples use placeholders (`### thread:<id>`), which never match. Quoting them back is safe.
- Issues opened/closed: none (all three questions fixed or answered here).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3695

🤖 Generated with [Claude Code](https://claude.com/claude-code)
