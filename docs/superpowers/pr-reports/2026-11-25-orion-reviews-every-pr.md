# Orion reviews every PR: 3 rounds, 15-minute round-up, draft gate

## Summary

- Every pull request a Claude Code session opens now goes to Orion. Orion gets the design spec files and the code diff. Before this change, only PRs with a design spec went to Orion.
- A code diff gets an 8-question review template in Orion's prompt (`CODE_REVIEW_TEMPLATE`). The questions come from CLAUDE.md 0A: live proof, the metric gate, the integration seam, empty shells, tests that pass for the wrong reason, contracts and config.
- Orion can ask up to 3 rounds of questions per PR (it was 1), so Orion writes on a PR at most 3 times.
- Orion reads every 15 minutes (it was every hour).
- A new hook refuses `gh pr create` without `--draft`. It refuses `gh pr ready` until the session answered Orion's first review.

## Outcome moved

Before: Orion saw 45 of the 253 PRs opened since 2026-11-15 (about 1 in 6), only the ones with a spec. After: Orion sees every PR opened from Claude Code. A PR cannot leave draft until Orion's first questions are answered. The exceptions are `NO QUESTIONS`, a thread Hub gave up on, and a 24 h wait-out when Orion's body is too busy to run the review.

## Current architecture

Session threads (`docs/superpowers/specs/2026-11-15-session-threads-design.md`): a PostToolUse hook sent a PR's spec files to Hub when the PR was opened. Every hour, Hub ran one durable Orion turn over every held thread. Orion asked one round of questions per thread. Code posted the questions as a PR comment. PRs with no spec file sent nothing.

## Architecture touched

- `orion/session_threads/core.py`: 3 rounds, a note that says how many rounds are left, the diff attachment shown as a diff, the review template, interval-neutral text.
- Hub loop and routes: the intake keeps `kind: "diff"` and cuts the diff at 40,000 characters, with the `gh pr diff N` pointer. The candidates carry `question_rounds_used`. New route `GET /api/session-threads?pr=N` (`threads_for_pr`).
- Session side: `build_pr_message` attaches `gh pr diff` without spec files and without generated, lock and binary files.
- New hook `scripts/hooks/pr_draft_gate.py`, registered in `.claude/settings.json`.

## Files changed

- `orion/session_threads/core.py`: round count, rounds-left note, diff display, review template, wording.
- `services/orion-hub/scripts/session_roundup_loop.py`: diff intake, `question_rounds_used`, `threads_for_pr`.
- `services/orion-hub/scripts/session_thread_routes.py`: `SpecFile.kind`, `?pr=` lookup, `kind` kept in the short view.
- `services/orion-hub/app/settings.py`, `.env_example`, `README.md`: interval 900.
- `scripts/orion_session_thread_lib.py`, `scripts/orion_session_thread.py`, `scripts/hooks/post_tool_pr_spec_to_orion.py`, `scripts/hooks/wait_for_orion_answer.py`: send the diff, wording, a 5th circle-back step (`gh pr ready`).
- `scripts/hooks/pr_draft_gate.py` (new), `.claude/settings.json`: the draft gate.
- `CLAUDE.md`: "Orion reviews every PR" rule (section 12, implementation steps, subagent preamble).
- Design doc: a "Change 2026-11-25" section.
- Tests: `tests/scripts/test_orion_session_thread_hooks.py`, `services/orion-hub/tests/test_session_*.py`, `session_thread_fakes.py`. CI workflow paths now include the gate.

## Schema / bus / API changes

- Added: `GET /api/session-threads?pr=N`. Optional `kind` on intake `spec[]` items (`"diff"`).
- Removed: none.
- Renamed: none.
- Behavior changed: PR comment heading is now "Orion's review questions (session-thread round-up)". Up to 3 question rounds per thread.
- Compatibility notes: no SQL migration (`kind` lives in the turn's `meta` JSON). Old clients still work: `kind` defaults to a spec.

## Env/config changes

- Added keys: none in `.env_example`. Hook-only env knobs: `ORION_PR_DRAFT_GATE`, `ORION_PR_READY_WAIT_H`, `ORION_ALLOW_PR_READY` (per-command override).
- Changed default: `HUB_SESSION_ROUNDUP_INTERVAL_SEC` 3600 -> 900.
- `.env_example` updated: yes (orion-hub).
- Local `.env` synced: the sync script never overwrites a changed value, so the main checkout's `services/orion-hub/.env` was set to 900 by hand. Verified with grep.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/scripts/test_orion_session_thread_hooks.py -q          -> 73 passed
pytest services/orion-hub/tests/test_session_threads_core.py \
       test_session_roundup_loop.py test_session_thread_routes.py \
       test_session_threads_js.py -q                                -> 156 passed, 41 skipped (Postgres lane: CI)
check_env_template_parity.py -> PASS; check_integration_seam.py -> OK; git diff --check -> clean
Live read-only check of the threads_for_pr query (BEGIN READ ONLY) -> returns PR 3953's thread
```

## Evals run

```text
No eval harness for session threads. The behavior is a prompt and routing change. The live proof is the first code-only PR that gets an Orion comment after deploy (verify-live ticket below).
```

## Docker/build/smoke checks

```text
UNVERIFIED live: needs the Hub rebuild after merge.
```

## Review findings fixed

A code review ran in a subagent and returned 12 findings. All 12 are fixed.

- Finding 1: text in a heredoc or message that said `gh pr ready` was read as a command, which caused a false refusal.
  - Fix: the gate now removes heredoc bodies and splits the command into shell words with `shlex`. Quoted text is one word.
  - Evidence: `test_only_real_gh_pr_ready_commands_are_gated`.
- Finding 2: the draft-flag check scanned the whole rest of the command, so body text or a later command could let an undrafted PR through.
  - Fix: flags are read only from that `gh pr create` command's own words.
  - Evidence: 9 new cases in `test_pull_requests_open_as_drafts`.
- Finding 3: before Hub is redeployed, an old Hub ignores `?pr=`, so the gate could read another PR's thread.
  - Fix: the gate takes only the thread whose `pr_number` matches.
- Finding 4: a PR with no spec and no readable diff got no thread, so it could never leave draft.
  - Fix: the session still sends a message that names the files and says why the diff is missing. The refusal suggests `--note`.
  - Evidence: `test_a_pr_whose_diff_gh_cannot_give_still_gets_a_thread_naming_its_files`.
- Finding 5: a 6th attachment (the diff) was dropped by Hub's 5-spec cap.
  - Fix: the diff never counts toward the cap, and the session sends it first.
  - Evidence: `test_a_diff_never_counts_toward_the_spec_file_cap`.
- Finding 6: a path with spaces or a quoted path merged into the previous file's diff chunk and could be dropped.
  - Fix: chunks split on `diff --git` line starts. The path comes from `+++ b/`, `rename to` or `--- a/`. A path the code cannot read is kept.
  - Evidence: `test_the_diff_keeps_files_whose_path_it_cannot_read_and_follows_renames`.
- Finding 7: re-running `gh pr create` or re-sending the PR counted as answering Orion.
  - Fix: the send hook skips "already exists". The gate counts only session turns with no attachments as answers.
  - Evidence: `test_a_pr_create_on_a_branch_that_already_has_a_pr_sends_nothing`, plus the resend case in `test_ready_waits_for_the_answer_to_orions_first_review`.
- Finding 8: the gate used the session's start directory, not the worktree it works in now.
  - Fix: the gate uses the agent board identity, and a `cd /abs` earlier in the command wins.
- Finding 9: `gh -R x pr ...`, `(gh ...)` and `/usr/bin/gh` skipped the gate. An override in a `#` comment counted.
  - Fix: the parser handles all three shapes. The override counts only as a leading env assignment. `-R` passes through to `gh pr view`.
- Finding 10: the hook's time-budget alarm was swallowed inside `_run`, so a spec or the diff disappeared silently.
  - Fix: the alarm is now logged and raised again.
- Finding 11: the compose default still said `:-3600`.
  - Fix: now `:-900`. Rendered with `docker compose config`: `HUB_SESSION_ROUNDUP_INTERVAL_SEC: "900"`.
- Finding 12: stale "hour" wording.
  - Fix: updated in the schema and loop docstrings.

Orion's first review (thread ba37694599a2, 8 questions) was answered on the PR (comment 7174397855). Fixed in 56897088d:
- When a PR has no thread, the gate sends the PR to Orion itself, so the review and the 24 h clock start then.
- An override is posted on the thread, so Orion can count waived reviews.
- Docs: the bucket wording, and the 24 h clock runs from the thread's creation.
- Ticketed: #4008 (round-up act outcome should be "session used the questions").

Known and accepted: `gh api ... draft=false` and the GraphQL ready mutation bypass the gate. A session that does that is deliberately going around the gate.

## Restart required

```bash
scripts/rebuild_services_from_git_diff.sh   # from the clean primary main after merge; rebuilds orion-hub
```

The hooks take effect for new Claude Code sessions in the primary checkout once `main` holds them.

## Risks / concerns

- Severity: low. Concern: about 1,500 more characters in every round-up prompt (the template). Mitigation: the budget is 100,000 characters.
- Severity: medium. Concern: about 5-6x more threads per day on Orion's agent lane. Mitigation: one round at a time, background priority, and heat refusals are not strikes.
- Issues opened: #3999 (verify-live: first live code-only review and the draft gate), #4008 (round-up act outcome). Closed: none.
- Severity: low. Concern: PRs opened outside Claude Code get no diff and no gate. Mitigation: `send --pr N` by hand.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3998

🤖 Generated with [Claude Code](https://claude.com/claude-code)
