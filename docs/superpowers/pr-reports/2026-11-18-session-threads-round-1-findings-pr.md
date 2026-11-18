## Summary

Orion asked six questions on session thread 455a41af5b44 (its one round, on PR #3695). Four of them found real bugs. This PR fixes all four and adds tests.

- The round-up prompt now tells Orion when a thread's pull request is closed or merged. Hub only comments on open PRs, so on those threads Orion's questions reach only the coding session and Hub chat.
- The 14-day backstop now measures from the NEWEST unread session message, not the oldest. A fresh message restarts that clock. A new message already reset the strike count.
- When a round holds more than one thread and Hub cannot find one thread's section, Hub no longer stores Orion's whole reply on that thread. The whole reply held the other threads' questions and any session text Orion quoted. Hub stores only its own line and keeps the reply on the round's durable run. This also covers answer-phase threads, which used to store the whole reply as a closing note. Later rounds then showed that note to Orion as the thread's history.
- Orion's questions are delivered once per session, not once per thread. A session that pulls the questions and exits without replying no longer loses them for every later session on the branch. Delivery stops once the thread moves on: a session writes on it, or Orion writes a closing note.
- A prompt the GPU pool refuses as too big for the lane (`min_ctx_exceeds_class`) is Hub's sizing mistake, not the thread's. It no longer counts as a strike. Hub cuts its prompt budget, using the lane's window when the refusal names it, and restores the budget after 24 h. A refusal at the 8,000-character floor still counts.

## Outcome moved

- Session text from one thread can no longer end up stored on another thread, and later prompts can no longer show it to Orion.
- A coding session that is thrown away no longer silently eats Orion's questions.
- A thread that a session still writes on is no longer ended by three hot hours.
- An oversized round no longer counts as a strike against its threads.
- Orion knows where its questions go when the PR has already merged.

## Current architecture

`services/orion-hub/scripts/session_roundup_loop.py` runs the hourly round-up over held threads, using the pure rules in `orion/session_threads/core.py`. Sessions read Orion's turns through `GET /pending` and claim them through `POST /delivered`, called by the hooks in `scripts/hooks/` and the CLI in `scripts/orion_session_thread*.py`. Before this patch:
- A claim set `delivered_at` once per turn, for the whole thread.
- An unparsed thread stored the whole reply on itself.
- The backstop measured from `min(created_at)`.
- `min_ctx_exceeds_class` counted as a strike.

## Architecture touched

- Hub session threads: the loop, the store SQL, and the routes.
- The pure rules in `orion/session_threads/core.py`.
- The session-side lib, the two hooks, and the CLI.
- One additive API field: an optional `session_id` on `POST /api/session-threads/delivered`.

## Files changed

- `orion/session_threads/core.py`:
  - `[Hub]` line for a closed or merged PR.
  - `ThreadAnswer.shared`.
  - `is_sizing_refusal`, `shrunk_budget`, `sizing_refusal_text`.
  - `unparsed_text(kept=...)`.
- `services/orion-hub/scripts/session_roundup_loop.py`:
  - `GitHubCommenter.pr_state` and `state_of`.
  - `_with_pr_states`.
  - The backstop now uses `max(created_at)`.
  - `requeue_unparsed(body=None)`.
  - Per-session `pending_for` and `mark_delivered`, with the redelivery rule.
  - The sizing budget: cut, then restore.
- `services/orion-hub/scripts/session_thread_routes.py`: `session_id` on `/delivered`. `meta.delivered_to` is hidden from API output.
- `scripts/orion_session_thread_lib.py`, `scripts/hooks/session_pending_orion_questions.py`, `scripts/hooks/wait_for_orion_answer.py`, `scripts/orion_session_thread.py`: claims carry the session id. A re-delivered question says that an earlier session saw it.
- Tests: `services/orion-hub/tests/test_session_roundup_loop.py`, `test_session_threads_core.py`, `test_session_thread_routes.py`, `session_thread_fakes.py`, `tests/scripts/test_orion_session_thread_hooks.py`.
- Docs: `docs/superpowers/specs/2026-11-15-session-threads-design.md`, `services/orion-hub/README.md`.

## Schema / bus / API changes

- Added:
  - An optional `session_id` on `POST /api/session-threads/delivered`.
  - `t.delivered_at` in `/pending` rows.
  - `meta.delivered_to` on turns. It is stored and stripped from API output.
  - `meta.reply_not_stored` on Hub's unparsed lines.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Delivery is once per session, until the thread moves on.
  - An unparsed thread in a multi-thread round stores no Orion turn.
  - The backstop clock runs from the newest message.
  - A sizing refusal is not a strike.
- Compatibility notes:
  - A client without `session_id` keeps the old claim-once behavior.
  - Turns claimed before deploy (no `delivered_to`) are never handed out again.
  - No migration. `meta` is already jsonb.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub (ORION_MEMORY_EPISODE_TEST_DATABASE_URL = throwaway postgres:16 container on 127.0.0.1:55499, never the live DB):
  pytest tests/test_session_threads_core.py tests/test_session_roundup_loop.py tests/test_session_thread_routes.py tests/test_session_threads_js.py
  -> 183 passed (fake and Postgres halves, 0 skipped)
pytest tests/scripts/test_orion_session_thread_hooks.py -> 39 passed
Static gates (orion-static-gates.yml): async_routes, body_economy, chat_route_poachers, definition_drift,
  env_key_single_source, integration_seam, metric_lineage, scripts_dir_no_stdlib_shadow,
  sentience_instruments --static-only, service_hostname_refs, system_health_producers,
  check_settings_defaults orion-hub --example-drift -> all OK
```

## Evals run

```text
No eval harness for session threads. The lifecycle suite runs on both a fake store and Postgres, and it is the behavioral check.
Live (Q1): python3 scripts/smoke_session_threads.py --rule-since 2026-11-18T05:10:06Z
  -> 18 threads, 3 Orion turns, 1 closing note; PASS no second round
python3 scripts/smoke_session_threads.py --rule-since 2026-11-17T23:46:11Z
  -> 16 Orion turns, 10 closing notes; PASS
```

## Docker/build/smoke checks

```text
Not built here (the change is Python only, with no new dependencies). Live evidence for Q2: Hub log 2026-11-18 05:26:55
`session_thread_pr_comment_skipped pr=3695 why=pull_request_closed` (and pr=3698). Live store: 0 unparsed or requeued turns and 0 sizing
refusals so far. Q4 and Q6 are latent, not yet seen live.
```

## Review findings fixed

- Finding: questions were still handed to new sessions after a closing note, when the reply had arrived while Orion was reading (that reply sits before the question turn).
  - Fix: redelivery also stops at any later non-requeued Orion turn.
  - Evidence: `test_questions_stop_going_to_new_sessions_once_orion_closed_the_thread`, on fake and Postgres.
- Finding: deploying would re-deliver every question delivered before this change, with no age limit, oldest first.
  - Fix: only turns with `delivered_to` can be redelivered, only for 14 days, and never-delivered turns are ordered first.
  - Evidence: `test_questions_delivered_before_per_session_delivery_are_not_handed_out_again`.
- Finding: the budget only shrank and ignored the lane window named in the refusal.
  - Fix: `core.shrunk_budget` caps the budget at 2 chars/token of `<ctx>`, and the budget restores 24 h after the last refusal.
  - Evidence: `test_the_budget_cut_follows_the_lanes_window_and_comes_back_a_day_later`.
- Finding: a stale comment and docstring, and `delivered_to` session ids were exposed through the API.
  - Fix: the comment and docstring are reworded, and `_public` strips `delivered_to`.
- Finding: no test claimed the same turn concurrently.
  - Fix: `test_two_hooks_of_one_session_claim_a_turn_once` (asyncio.gather x4, Postgres) gives a total of 1.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Severity: low.
  - Concern: the PR-state lookup adds one GitHub `GET` per PR thread per round (5 s timeout each, run concurrently, best effort). It runs only when PR comments are on.
  - Mitigation: if a lookup fails, the prompt is built without the line.
- Severity: low.
  - Concern: `_tick` selects threads without the `[Hub]` PR line, and `_submit` selects again with it. A thread at the budget edge can be released back to `held` at submit time.
  - Mitigation: the release path already handles this, and the thread is read next hour.
- Severity: low.
  - Concern: a redelivered question goes to at most 20 sessions in 14 days.
- Issues: none opened or closed. #3666 (round-up throughput) is related, but this PR does not change it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3718

🤖 Generated with [Claude Code](https://claude.com/claude-code)
