## Summary

- Fix a second, previously-undiscovered agent-board bug found via live
  investigation while a user was querying a Stop-hook reminder: Claude
  Code's own `SessionStart`/`Stop` hooks (`scripts/hooks/
  session_start_agent_board.py`, `scripts/hooks/session_stop_agent_board.py`
  -- distinct from the git hooks shipped in the already-merged
  `chore/agent-board-heartbeat-coupling` PR) run with a process `cwd` fixed
  to wherever the session originally started, and do not track `cd` calls a
  Bash tool makes mid-session. Confirmed by inspecting the Stop hook's own
  stdin JSON payload: its `cwd` field reported the shared/primary checkout
  even while real git work had been happening in a linked worktree for many
  turns.
- `render_checkin_context()` was additionally *writing* a spurious presence
  row under that wrong path on every `SessionStart`, not just misreporting
  on read.
- Fixed with a new `resolve_current_identity()` in `agent_board_lib.py`:
  when a `session_id` is available, look up the most recently heartbeated
  presence row tagged with that session_id and use its `worktree_path`
  instead of the hook's own cwd. Works because the already-shipped
  `scripts/git_hooks/post-commit` and `scripts/safe_docker_build.sh` now
  tag their heartbeats with `$CLAUDE_CODE_SESSION_ID` (confirmed live to be
  present in the Bash tool's subprocess environment and to inherit
  correctly down to a git-hook subprocess). Falls back to the old
  cwd-based resolution when no matching session_id is on the board yet.
- Code review (dispatched subagent) confirmed the two direct regression
  tests actually fail against the pre-fix code (not placebo coverage) and
  found two minor nitpicks, both fixed here: a duplicated
  `_read_session_id()` helper consolidated into
  `read_session_id_from_stdin_hook_payload()`, and a TTY guard added so
  manual invocation for debugging doesn't block on stdin.

## Outcome moved

Before this fix, every concurrent Claude Code session working via linked
worktrees (this repo's own convention) had its `SessionStart`/`Stop`
board hooks silently attributing presence and open-item lookups to the
shared/primary checkout instead of its actual worktree -- weakening the
board's whole worktree-scoped collision-avoidance design for anything
routed through these two hooks. Confirmed live post-fix: a real commit's
heartbeat correctly carries this session's own `session_id`
(`4dd9ea4b-3afa-4362-8590-3bd65e3ac4ac`), which the fixed hooks can now
use to resolve the correct worktree.

## Current architecture

`scripts/agent_board_lib.py`'s `current_worktree_identity()` resolves the
board's notion of "current worktree" purely via `git rev-parse
--show-toplevel` from the calling process's own inherited cwd. Both
`session_start_agent_board.py` and `session_stop_agent_board.py` called
this directly with no way to override it. Separately,
`scripts/git_hooks/post-commit` and `scripts/safe_docker_build.sh`
(merged in `chore/agent-board-heartbeat-coupling`) already heartbeat the
board correctly, since they're invoked by `git`/`docker` with the correct
cwd, not by the Claude Code harness's own hook mechanism.

## Architecture touched

- `scripts/agent_board_lib.py`: new `resolve_current_identity()` and
  `read_session_id_from_stdin_hook_payload()`; `upsert_presence()` gained
  optional `worktree_path`/`branch` overrides; `render_checkin_context()`
  gained an optional `session_id` kwarg.
- `scripts/hooks/session_start_agent_board.py`,
  `scripts/hooks/session_stop_agent_board.py`: read `session_id` from
  their own stdin JSON payload, pass it through to the board lib.
- `scripts/git_hooks/post-commit`, `scripts/safe_docker_build.sh`: pass
  `--session-id "$CLAUDE_CODE_SESSION_ID"` to their heartbeat calls when
  that env var is set.
- `CLAUDE.md`: documents the bug and fix under "Agent workspace board".

## Files changed

- `scripts/agent_board_lib.py`: `resolve_current_identity()`,
  `read_session_id_from_stdin_hook_payload()` (new); `upsert_presence()`,
  `render_checkin_context()` (modified signatures, backward compatible --
  both new params are keyword-only with `None` defaults that short-circuit
  to the exact old behavior).
- `scripts/hooks/session_start_agent_board.py`,
  `scripts/hooks/session_stop_agent_board.py`: read and thread through
  `session_id`.
- `scripts/git_hooks/post-commit`, `scripts/safe_docker_build.sh`:
  `--session-id` passthrough when `$CLAUDE_CODE_SESSION_ID` is set.
- `tests/scripts/test_agent_board_lib.py`: 4 new tests for
  `resolve_current_identity` (session match, multi-worktree tie-break,
  no-match fallback, `session_id=None` fallback).
- `tests/scripts/test_session_agent_board_hooks.py`: 3 new tests --
  Stop-hook and SessionStart-hook resolution via session_id from a
  simulated wrong cwd, plus a no-match fallback case; `_run_hook` helper
  extended to pass an explicit stdin JSON payload.
- `tests/scripts/test_install_git_safety_hooks.py`,
  `tests/scripts/test_safe_docker_build_heartbeat.py`: new tests
  confirming `$CLAUDE_CODE_SESSION_ID` actually flows through into the
  heartbeat's `--session-id` argument.
- `CLAUDE.md`: new paragraph under "Agent workspace board".

## Schema / bus / API changes

None.

## Env/config changes

None. `$CLAUDE_CODE_SESSION_ID` is set by the Claude Code harness itself,
not something this repo configures.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest tests/scripts/ -q
199 passed in 47.3s
```

Plus live verification: a real commit on this branch tagged its heartbeat
with this session's actual `$CLAUDE_CODE_SESSION_ID`, confirmed via the
raw board JSONL.

## Evals run

No eval harness applies to shell/hook tooling in this repo.

## Docker/build/smoke checks

Not applicable -- no service touched.

## Review findings fixed

- Finding: `_read_session_id()` was duplicated verbatim across both hook
  scripts.
  - Fix: consolidated into `agent_board_lib.read_session_id_from_stdin_hook_payload()`.
  - Evidence: both hook files now import and call the single shared
    function; full suite still green (199 passed).
- Finding: `sys.stdin.read()` would block indefinitely if either hook is
  invoked manually from an interactive terminal for debugging (not a risk
  in the real Claude Code/Cursor invocation paths, where stdin is a pipe
  the harness writes-then-closes, but a new failure mode this diff
  introduced that didn't exist before it touched stdin at all).
  - Fix: `read_session_id_from_stdin_hook_payload()` checks
    `stream.isatty()` first and skips reading entirely on a live TTY.
  - Evidence: code inspection; the existing test suite's `input=""`-based
    invocations are piped, not TTY, so this path is exercised by every
    existing hook test without behavior change.
- Finding (verified, not a defect): tie-breaking on identical
  `heartbeat_at` timestamps within `resolve_current_identity` falls back
  to dict-iteration/insertion order rather than an explicit secondary key.
  - Not fixed: reviewer assessed a genuine same-second tie between two
    worktrees under one session_id as unlikely enough in practice not to
    matter; flagging as a known, accepted limitation rather than adding
    complexity for it.

## Restart required

```text
No restart required.
```

Local git-hook/hook-script tooling only. Anyone who wants the corrected
`SessionStart`/`Stop` resolution active needs no action beyond having this
merged code checked out -- these are plain Python scripts invoked fresh by
the harness each time, not something requiring a hook re-install (unlike
the git hooks, which do need `scripts/install_git_safety_hooks.sh` re-run
once per machine to pick up code changes, already covered by the earlier
PR).

## Risks / concerns

- Severity: Low
  - Concern: this fix's live verification was necessarily partial --
    confirming the git-hook side tags the real session_id into a real
    heartbeat is straightforward and done, but confirming the
    `SessionStart`/`Stop` hooks themselves correctly resolve via that
    session_id in a live Claude Code invocation (as opposed to the test
    suite's simulated stdin payloads) requires the fix to actually be
    merged and live in the shared checkout first, since `$CLAUDE_PROJECT_DIR`
    still points at pre-fix code until then.
  - Mitigation: the test suite's regression tests were confirmed to
    genuinely fail against the pre-fix code (not placebo coverage, per
    the dispatched review), and the underlying mechanism (session_id
    passed via stdin JSON, looked up against a real presence row) was
    exercised end-to-end in tests using the exact same code paths the
    real hooks call. Recommend a live spot-check after merge: make a
    commit from a linked worktree, then check whether the next Stop
    hook's reminder correctly reports items scoped to that worktree
    rather than the shared checkout.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/agent-board-session-cwd-lookup
