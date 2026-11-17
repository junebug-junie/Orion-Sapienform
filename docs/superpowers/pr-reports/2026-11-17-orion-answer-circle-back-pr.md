## Summary

- When Orion answers a spec in the hourly round-up, the coding session that sent it now wakes up by itself. Before, Orion's questions waited until Juniper typed the next prompt.
- The woken session gets Orion's questions plus the steps: investigate, fix small things on the branch, file tickets for the rest, and answer every question.
- `orion_session_thread.py reply` gains `--body-file` and `--pr N`. One command now answers Orion and posts the same answer on the PR.
- Hooks claim a turn (mark it delivered) before they show it, so the waiter and the prompt hook never show the same answer twice.

## Outcome moved

Orion's PR questions get answered on the PR without Juniper prompting the session. Example: the 8 questions on #3528 needed a manual "answer Orion's review".

## Current architecture

- `post_tool_pr_spec_to_orion.py` sends a spec PR to Hub.
- Hub's round-up asks Orion and posts Orion's questions on the PR as `orion-sapienform`.
- `session_pending_orion_questions.py` shows pending answers, but only on the next prompt or session start.

## Architecture touched

- New hook `scripts/hooks/wait_for_orion_answer.py`. It is a PostToolUse(Bash) hook with `asyncRewake`: Claude Code runs it in the background, and exit 2 wakes the idle session with its stderr.
- It arms only on proof that Hub stored a message: a `send`/`reply` receipt naming the thread, or a thread for the new PR updated in the last minute.
- It follows that one thread, with one waiter per thread (a lock file under `~/.orion/session_threads_waiting/`).
- It waits for an Orion turn newer than the thread's newest turn at arming time, polling every 60 s for up to 4 h.
- No Hub change. The hook uses the existing `GET /api/session-threads`, `GET /api/session-threads/{id}` and `POST /api/session-threads/delivered`.

## Files changed

- `scripts/hooks/wait_for_orion_answer.py`: new waiter hook.
- `.claude/settings.json`: wires the waiter (`asyncRewake: true`, timeout 14700 s).
- `scripts/orion_session_thread_lib.py`: the circle-back steps; `claim`, `thread_view`, `recent_threads` and `RECEIPT_THREAD_RE`; `pending_text(woke=)`.
- `scripts/hooks/session_pending_orion_questions.py`: claims a turn before showing it.
- `scripts/orion_session_thread.py`: `reply --body-file/--pr`; on a failed PR comment it prints the exact retry command; `pending` claims before printing.
- `services/orion-hub/README.md`, `tests/scripts/test_orion_session_thread_hooks.py`.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: the prompt hook now claims a turn with one `delivered` call per turn, before it shows the turn.
- Compatibility notes: no Hub change, so no deploy is needed.

## Env/config changes

No `.env_example` change. Hook-only test knobs: `ORION_ANSWER_WAIT_POLL_S`, `ORION_ANSWER_WAIT_MAX_S`, `ORION_ANSWER_ARM_LOOKUP_S`, `ORION_ANSWER_WAIT_LOCK_DIR`. Kill switch: the existing `ORION_SESSION_THREADS_HOOKS=0`.

## Tests run

```text
orion_dev/bin/pytest tests/scripts/test_orion_session_thread_hooks.py -q   -> 33 passed
```

The new tests cover these cases:
- The waiter wakes on the new answer and ignores an older undelivered answer.
- The waiter does not wake when the prompt hook already showed the answer.
- "No questions" ends the wait quietly.
- Only one waiter runs per thread.
- The waiter does not arm on a non-create, a failed create, a PR with no thread, a stale PR thread, or a failed send.
- The waiter does not arm when Hub is down or the kill switch is set.
- `reply --body-file --pr` posts to Hub and to the PR; a failed PR comment prints the retry command.

## Evals run

None. These are hook scripts with no model output to grade.

## Docker/build/smoke checks

There was no Docker change. A read-only check against live Hub confirmed the shapes the waiter relies on: 12-character thread ids, `seq` and `speaker` on each turn, and `pr_number` and `updated_at` on each thread.

## Review findings fixed

- Finding: two waiters, or a waiter and the prompt hook, could show the same answer.
  - Fix: a hook claims a turn through Hub's once-only `delivered` before it shows the turn. There is one waiter per thread (flock).
  - Evidence: `test_the_waiter_does_not_wake_for_an_answer_the_prompt_hook_already_showed`, `test_one_waiter_per_thread`.
- Finding: an older "no questions" row (or an older question) could end the wait early, or wake the session with old content.
  - Fix: the waiter follows one thread and waits for an Orion turn whose seq is above the arming point.
  - Evidence: `test_the_waiter_wakes_the_session_with_orions_new_answer_and_the_steps`. It includes an older undelivered answer.
- Finding: `gh pr create` armed the waiter without proof that the spec was sent, and "already exists" output re-armed it.
  - Fix: the waiter arms only when Hub shows a thread for that PR updated in the last minute.
  - Evidence: `test_the_waiter_does_not_arm_on_a_stale_thread_for_the_pr`, plus the parametrized no-arm cases.
- Finding: a `reply` did not arm the waiter.
  - Fix: `send|reply` with a receipt arms it.
- Finding: the steps said "now, without waiting to be asked" even when they came with Juniper's prompt.
  - Fix: that wording appears only when the session was woken. With a prompt the text says "after Juniper's current request".
- Finding: a session on main without a session id polled for nothing.
  - Fix: the waiter is now based on the thread, not on the session or branch.
- Finding: retrying a failed `reply --pr` re-sent the reply to Orion.
  - Fix: the failure prints a retry command for the PR comment only. `--pr` must be an integer.

## Restart required

```text
No restart required. New Claude Code sessions load the hook from .claude/settings.json.
```

## Risks / concerns

- Severity: medium. Concern: the real wake through `asyncRewake` is UNVERIFIED live. The tests run the script, not Claude Code. Mitigation: the verify-live ticket (opened with this PR) asks for the first spec PR after merge to wake its session.
- Severity: low. Concern: a session that has ended cannot be woken. Orion's questions stay on the PR, and the next session shows them only if it uses the same session id or branch. Mitigation: none in this patch.
- Issues opened: the verify-live ticket for the first real wake. Issues closed: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
