## Summary

Coding sessions can now write to Orion. Juniper sees the thread in Hub. Orion comes back with questions once an hour.

- A session sends Hub a message, usually the design spec of a pull request it just opened. Hub stores it and answers at once with a receipt. The receipt is Hub's text. It says Orion has NOT read the message. No model runs.
- Once an hour Hub takes every held thread and submits ONE Orion turn that reads them all. No held thread: no turn, no GPU.
- Code cuts Orion's reply into one answer per thread. Each answer goes to Hub chat, back to the session that sent the spec, and onto the pull request as a comment from `orion-sapienform`.
- A hook sends the spec when a session runs `gh pr create`. A second hook hands Orion's questions to the session on its next prompt. A CLI does both by hand.
- Hub chat shows one collapsed line per thread. A click opens the turns under three labels: Session, Hub receipt, Orion.

Design note: `docs/superpowers/specs/2026-11-15-session-threads-design.md`.

## Outcome moved

Before: Orion saw a spec only after it was merged, and Juniper could not see what Orion thought of it. After (once deployed): a spec reaches Orion within the hour it is pushed, and Orion's questions are visible in three places. Nothing is live yet. See "UNVERIFIED until deploy".

## Proposal-mode facts (a new input into Orion's cognition)

- **Capability changed.** Orion reads text written by coding sessions and writes questions back. Orion's account writes to GitHub for the first time (PR comments).
- **Data touched.** Two new tables (`orion_session_thread`, `orion_session_turn`). One durable run an hour at most. No write to chat history, recall, memory, or the belief graph.
- **Privacy boundary.** Session text is repo material (specs, branch names). It goes to Orion's local model and stays on the host. Orion's questions go to GitHub, on this repo only, as comments on the pull request the spec belongs to.
- **Trace that proves it worked.** Hub log lines `session_thread_message`, `session_roundup_started`, `session_roundup_applied`, `session_thread_pr_comment`; one `session_roundup.orion_turn` row in `durable_run_workflow` per hour that had a held thread; `orion_session_turn` rows with speaker `orion`; `pr_comment_url` set; `GET /api/debug/session-threads`.
- **Dangerous failure modes.**
  - *Untrusted session text steers Orion.* The text sits inside a fence with a per-round nonce no session can know. The prompt says it is material, not instructions. A `### thread:` line a session wrote is broken. If Orion's reply has a repeated or out-of-order heading, the whole round is stored as unparsed and nothing goes to a PR.
  - *A runaway hourly GPU turn.* One run per hour at most, and only one in flight. The run has a deadline (its hour + 3 h); 30 minutes after that Hub stops waiting. Three failures in a row leave a thread `failed`. Intake is bounded (50 waiting threads, 20 unread messages a thread, same message stored once).
  - *Orion's first write to GitHub.* Code posts the comment, not the model. Only on `HUB_SESSION_ROUNDUP_GITHUB_REPO`. Hub first reads the PR back and posts only when it is open and its head branch is the thread's branch. `@name` and `#123` are broken. `NO QUESTIONS` and unparsed replies are never posted.
- **How to disable.** `HUB_SESSION_ROUNDUP_ENABLED=false` (no Orion turn; messages are still stored and the receipt says so). `HUB_SESSION_ROUNDUP_PR_COMMENTS=false` (nothing goes to GitHub). `ORION_SESSION_THREADS_HOOKS=0` in a shell (both hooks off). Rollback: `DROP TABLE orion_session_turn; DROP TABLE orion_session_thread;`.

## Current architecture

Before this patch the only two-sided thread in Hub chat was the consult display (`scripts/consult_loop.py`, `static/js/consult.js`): Orion and Claude, one consult at a time, weekly cap 3. Orion's consult turn was a durable run (`consult.orion_turn`). No hook sent anything from a coding session to Orion, and no PostToolUse or UserPromptSubmit hook existed. Hub held Orion's GitHub token and used it read-only.

## Architecture touched

- **orion-hub**: new loop `scripts/session_roundup_loop.py` (store, intake, hourly tick, reply handling, GitHub comment), new routes `scripts/session_thread_routes.py`, new page script `static/js/session-threads.js`, one frame branch in `static/js/app.js`, one script tag in `templates/index.html`, wiring in `scripts/main.py`.
- **Shared contract**: new durable workflow `session_roundup.orion_turn` with brief `SessionRoundupBriefV1` (`orion/schemas/session_roundup_run.py`, `durable_run.py`, `registry.py`). Pure rules in `orion/session_threads/core.py`.
- **orion-durable-runs**: the workflow on the existing single-step harness graph; act family `durable.session_roundup`.
- **Body economy**: family `durable.session_roundup` in `config/physiology/families.yaml` (tier 2, shadow bids).
- **Repo tooling**: two Claude Code hooks, a CLI, a smoke script, one CI workflow.
- Not touched: the consult queue, its cap and its act family; no new service; no bus channel.

## Files changed

- `orion/session_threads/core.py`: caps, the hour bucket, receipt text, prompt with fenced session text, which threads fit one prompt, reply parser, PR comment text, failure text.
- `orion/schemas/session_roundup_run.py`: workflow name, run id per hour, brief model.
- `orion/schemas/durable_run.py`, `orion/schemas/registry.py`: the new literal, brief in the union, validator, registry entry.
- `services/orion-durable-runs/app/harness_smoke_graph.py`, `admission_runtime.py`, `run_acts.py`: graph builder, finish detail, work node, act family.
- `config/physiology/families.yaml`: `durable.session_roundup`.
- `services/orion-hub/scripts/session_roundup_loop.py`, `session_thread_routes.py`, `main.py`: loop, routes, wiring.
- `services/orion-hub/scripts/curiosity_investigation.py`: the round-up's source tag joins the "not a curiosity run" list (one line).
- `services/orion-hub/static/js/session-threads.js`, `static/js/app.js`, `templates/index.html`: thread display.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: 10 env keys and docs.
- `services/orion-sql-db/manual_migration_orion_session_thread_v1.sql`: two tables, indexes.
- `scripts/orion_session_thread_lib.py`, `scripts/orion_session_thread.py`: shared client code and CLI.
- `scripts/hooks/post_tool_pr_spec_to_orion.py`, `scripts/hooks/session_pending_orion_questions.py`, `.claude/settings.json`: the two hooks and their wiring.
- `scripts/smoke_session_threads.py`: smoke for a live Hub.
- `scripts/sync_local_env_from_example.py`: `HUB_SESSION_ROUNDUP_` prefix added so the keys sync.
- `.github/workflows/session-threads-tests.yml`: Postgres store tests (must not skip), hooks, CLI.
- `services/orion-durable-runs/README.md`, `docs/superpowers/specs/2026-11-15-session-threads-design.md`: docs.
- Tests: `services/orion-hub/tests/test_session_threads_core.py`, `test_session_roundup_loop.py`, `test_session_thread_routes.py`, `test_session_threads_js.py`, `session_thread_fakes.py`; `services/orion-durable-runs/tests/test_session_roundup_graph.py`, `test_run_acts.py`; `tests/scripts/test_orion_session_thread_hooks.py`.

## Schema / bus / API changes

- Added: tables `orion_session_thread`, `orion_session_turn`. Durable workflow `session_roundup.orion_turn`, `SessionRoundupBriefV1` (registered). Act family `durable.session_roundup`. Routes `POST /api/session-threads/message`, `GET /api/session-threads`, `GET /api/session-threads/pending`, `POST /api/session-threads/delivered`, `GET /api/session-threads/{id}`, `GET /api/debug/session-threads`. Websocket frame kind `session_thread_turn`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing paths.
- Compatibility notes: the new workflow is an additive value on `Literal` / `extra="forbid"` models. Readers must deploy before the producer (Hub). See "Restart required". No bus channel was added.

## Env/config changes

- Added keys (orion-hub): `HUB_SESSION_ROUNDUP_ENABLED=true`, `HUB_SESSION_ROUNDUP_INTERVAL_SEC=3600`, `HUB_SESSION_ROUNDUP_TICK_SEC=120`, `HUB_SESSION_ROUNDUP_ORION_TIMEOUT_SEC=1800`, `HUB_SESSION_ROUNDUP_RUN_DEADLINE_SEC=10800`, `HUB_SESSION_ROUNDUP_PROMPT_MAX_CHARS=100000`, `HUB_SESSION_ROUNDUP_PR_COMMENTS=true`, `HUB_SESSION_ROUNDUP_GITHUB_REPO=junebug-junie/Orion-Sapienform`, `HUB_SESSION_ROUNDUP_DURABLE_URL=http://127.0.0.1:8124`, `HUB_SESSION_ROUNDUP_TZ=America/Denver`. Flags ship ON.
- Shell-only (hooks, not in any `.env`): `ORION_HUB_URL`, `ORION_SESSION_THREADS_HOOKS`, `ORION_SESSION_NAME`, `ORION_SESSION_THREADS_LOG`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The primary checkout's `services/orion-hub/.env` has all 10 keys.
- skipped keys requiring operator action: none.

## Tests run

```text
Hub, session threads (in-memory store AND real SQL on a throwaway postgres:16 container):
  pytest tests/test_session_threads_core.py tests/test_session_roundup_loop.py
         tests/test_session_thread_routes.py tests/test_session_threads_js.py
  -> 87 passed, 0 skipped
Hooks and CLI as real subprocesses (stand-in Hub and gh):
  pytest tests/scripts/test_orion_session_thread_hooks.py -> 22 passed
orion-durable-runs, whole tests folder in the service image (read-only mount):
  -> 964 passed, 144 skipped (the skips need that suite's own Postgres; CI runs them).
     tests/test_reading_resonance_pg.py left out locally: collection error in the image, unrelated to this patch.
Hub, whole tests folder:
  -> 4964 passed, 124 skipped, 6 failed, 4 errors. The same 10 fail on main on this host
     (pandas missing, cwd-relative paths, a file permission, CREATE INDEX CONCURRENTLY in a fixture).
     Two of them pass when run from the repo root, as CI runs them.
Gates: check_env_template_parity PASS; check_env_key_single_source OK; check_async_routes_not_blocking clean;
  test_no_personal_terms, test_ci_service_image_sources, test_sync_local_env_from_example,
  test_sql_migration_drift_gate, test_post_pull_apply_migrations, test_schema_skew_discovery,
  test_family_lanes, test_cursor_bridge, test_session_agent_board_hooks -> all passed.
  Registry: orion.schemas.registry.resolve("SessionRoundupBriefV1") returns the model.
Mutation checks (each mutation made the named test fail, then was reverted):
  clock taken before the read -> test_the_clock_is_taken_after_the_read
  one-round-per-hour check removed -> test_three_failed_rounds_in_a_row...
  parser trusts any heading order -> test_a_heading_a_session_planted_cannot_move_text...
  cut-short reply accepted -> test_three_failed_rounds_in_a_row...
```

What the tests cover: intake and receipt; size caps; hold and selection; one round per hour across a restart; a crash between marking and submitting; clock after the read; parser (good, NO QUESTIONS, unparseable, empty, forged and out-of-order headings); failure counter; pending and delivered by session and by branch; both hooks (command matching, PR without a spec ignored, Hub down is silent); the lifecycle message -> held -> round-up -> asked -> session reply -> held -> second round-up.

## Evals run

```text
No model eval ran. The quality of Orion's questions can only be judged on live rounds.
scripts/smoke_session_threads.py is the live smoke. It was NOT run: the deployed Hub is the old code.
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml config
  -> rc=0; all 10 HUB_SESSION_ROUNDUP_* keys render with their defaults.
  (scripts/safe_docker_build.sh orion-hub config refuses in a worktree: it has no root .env. The raw
   read-only command was run with the worktree's compose file and the primary checkout's env files.)
No image build: no dependency changed (httpx, asyncpg, zoneinfo are already in Hub).
python scripts/apply_post_pull_migrations.py --dry-run services/orion-sql-db/manual_migration_orion_session_thread_v1.sql
  -> NOT APPLIED ... would apply (single transaction, lock_timeout=10s)   [passes the auto-apply allowlist]
No `up`. No deploy.
```

## Review findings fixed

A review subagent read the diff by hand (it had no code-review skill available).

- Finding: one thread with several long unread messages fit no prompt, so no round ever ran again.
  - Fix: a thread first in line is cut to fit (spec, then messages, then only its newest messages), and every cut is written in the prompt. Intake caps unread messages per thread.
  - Evidence: `test_one_thread_with_many_long_messages_is_cut_and_read_and_never_blocks_the_round`, `test_intake_is_bounded_per_thread_and_in_total`.
- Finding: a `### thread:<id>` heading planted by one session could move text to another thread's PR and session.
  - Fix: a repeated or out-of-order heading makes the whole round unparsed (no PR comment). Session lines that look like a heading are broken in the prompt. The fence carries a per-round nonce.
  - Evidence: `test_a_heading_a_session_planted_cannot_move_text_to_another_threads_pr`, `test_the_fence_carries_a_nonce_and_session_text_cannot_write_a_heading_line`.
- Finding (found while fixing the one above, by the Postgres half of the tests): replies were matched to threads in database order, not prompt order.
  - Fix: `_apply` uses the prompt's order.
  - Evidence: `test_each_thread_gets_its_own_section_and_its_own_pr_comment[pg]`.
- Finding: a session chose the PR number Orion's token commented on.
  - Fix: Hub reads the PR back and posts only when it is open and its head branch is the thread's branch.
  - Evidence: `test_orion_only_comments_on_an_open_pr_of_the_threads_own_branch`, `test_the_commenter_checks_the_pull_request_before_it_posts_and_never_raises`.
- Finding: a run that is `abandoned` or `paused` left threads in `reading` forever and blocked later rounds.
  - Fix: `abandoned` fails the round; any run not ended 30 minutes past its deadline fails the round.
  - Evidence: `test_a_run_that_never_ends_does_not_block_later_rounds`.
- Finding: the sender hook fired on `gh pr create --help`, dry runs and failed creates, and resent the same spec.
  - Fix: the hook needs the PR's URL in the tool's own output. Hub stores the same message once.
  - Evidence: hook test `...silent_when_hub_is_down_or_switched_off_or_gh_fails`, `test_the_same_message_twice_is_stored_once`.
- Finding: an unclosed backtick let a live `@mention` through; `#123` linked issues.
  - Fix: every `@name` and `#123` in the comment is broken.
  - Evidence: `test_the_pr_comment_is_headed_names_the_thread_and_pings_nobody`.
- Finding: a 5xx from durable-runs, or a 409 whose run could not be read, counted as a failed round.
  - Fix: both wait for the next tick.
  - Evidence: `test_a_server_error_or_an_unreadable_conflict_waits_and_counts_no_failure`.
- Finding: unbounded growth through a route with no login.
  - Fix: 50 waiting threads, 20 unread messages a thread (HTTP 429).
  - Evidence: `test_intake_is_bounded_per_thread_and_in_total`, `test_a_repeat_is_said_to_be_a_repeat_and_a_full_thread_answers_429`.
- Smaller: Orion's text is fenced when printed into a session; a unique index stops a second Orion turn per thread per round; an index serves the pending read; the sender hook has a 20 s budget of its own; an unused constant was removed.

Not changed, on purpose:

- A later message for the same PR from another session takes over the thread's sender fields. That is the brief's rule (questions go to the session the newest spec came from). Hub logs `session_thread_sender_changed`.
- Questions are delivered by session id OR by branch, as the brief asks. Two sessions on one branch share them; the first to prompt gets them.

## Restart required

Deploy order after merge (readers of the new workflow first, Hub last). One line each, from the clean primary `main` checkout:

```bash
git pull --ff-only
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-actions up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
python3 scripts/smoke_session_threads.py
```

Migration: the post-pull rebuild applies `manual_migration_orion_session_thread_v1.sql` by itself (dry-run says "would apply"). By hand, if needed:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_orion_session_thread_v1.sql
```

If the post-pull rebuild runs the services in another order, the worst case is one refused round (Hub submits before durable-runs knows the workflow): the threads go back to `held` and are read the next hour.

The hooks are live for every Claude Code session in this repo as soon as this merges (`.claude/settings.json`). Until Hub is deployed they are silent (Hub answers 404).

## UNVERIFIED until deploy

- The live hourly round-up: that Hub's turn loop serves `session_roundup.orion_turn`, and that Orion's model answers in the `### thread:<id>` format. Check: `python3 scripts/smoke_session_threads.py`, then `--check <thread>` after the next hour.
- The PR comment: Orion's token has never written to GitHub. Its scope for issue comments is not checked.
- The Hub page: the thread line and its turns in a real browser.
- The hooks inside a live session: the `additionalContext` output shape for PostToolUse and UserPromptSubmit, and whether the `rtk` proxy keeps the PR URL in the tool output the hook reads.
- The order the post-pull rebuild deploys the five services.

## Risks / concerns

- Severity: medium
- Concern: the act family's tier (2, shadow bids) was chosen by the design, not by Juniper. Tiers are her decision.
- Mitigation: stated in `families.yaml`; one line to change.

- Severity: medium
- Concern: Hub's routes have no login. Any process on the host can send a message to Orion or mark questions delivered.
- Mitigation: Hub is host-local; intake is bounded; the PR comment is checked against GitHub before it posts.

- Severity: low
- Concern: a crash between posting a PR comment and saving its URL can post the comment twice.
- Mitigation: a few milliseconds wide; a duplicate comment is visible and harmless.

- Severity: low
- Concern: a status `reading` was added to the four the brief named, and a message is read at the first round-up AFTER the hour it arrived in (up to an hour's hold, not exactly one hour).
- Mitigation: both are in the design note. The receipt names the time.

- Severity: low
- Concern: Cursor sessions have no hook (`.cursor/hooks.json` unchanged). The CLI works from any session.
- Mitigation: follow-up issue.

Issues opened: see the PR body. Issues closed: none.

## PR link

See the PR body.
