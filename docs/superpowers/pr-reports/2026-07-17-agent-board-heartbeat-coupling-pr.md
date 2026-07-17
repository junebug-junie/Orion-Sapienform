## Summary

- Couple the host-local agent workspace board (`scripts/agent_board.py`,
  `~/.orion/agent-board.jsonl`) to two habits that are already universal in
  this repo, instead of relying on agents remembering to run its CLI by
  hand. A live session found the board's only existing nudge -- a stop-hook
  banner reading "no open board items are recorded for this worktree" -- got
  read and ignored across roughly 20 consecutive turns, and the user
  confirmed real, ongoing collisions between concurrent agents working
  similar topics.
- New `scripts/git_hooks/post-commit`, installed by
  `scripts/install_git_safety_hooks.sh` alongside the existing `pre-commit`/
  `post-merge` hooks: heartbeats the board with the commit's own subject
  line after every commit, in any worktree.
- `scripts/safe_docker_build.sh` now heartbeats the board before every
  docker compose invocation it wraps, naming the service and tagging the
  task as `deploy:<service>` -- the wrapper's own documented incident (a
  concurrent agent silently reverting another's verified fix via a
  shared-checkout deploy) is exactly the collision class this covers.
- Both heartbeats are best-effort and never fail the operation they ride on.
- Mid-implementation, live testing found `post-commit` is a hook name real
  tooling in this repo already uses (`graphify hook install`'s
  knowledge-graph rebuild hook) -- a naive install silently disabled it. Two
  more real bugs surfaced the same way before this shipped (see below); all
  three are fixed and verified against live `git commit` runs, not just unit
  tests.

## Outcome moved

The agent board's collision-visibility tooling existed but had zero real
usage -- a passive reminder with no cost to ignoring it. It now gets a
presence heartbeat automatically on every commit and every docker deploy
action in this repo, across every worktree, without any agent needing to
remember to run the CLI.

## Current architecture

`scripts/agent_board.py` / `scripts/agent_board_lib.py` already existed with
a full CLI (`checkin`/`heartbeat`/`add`/`resolve`/`checkout`/`list`), a
JSONL event log at `~/.orion/agent-board.jsonl`, and test coverage
(`tests/scripts/test_agent_board_{cli,lib}.py`). `scripts/
install_git_safety_hooks.sh` already installed `pre-commit` (shared-checkout
refusal) and `post-merge` (worktree-hygiene nudge) via an overwrite-and-
backup strategy, both with their own test coverage. `scripts/
safe_docker_build.sh` already wrapped every `docker compose` invocation in
this repo with the shared-checkout guard and the dual `--env-file` pattern.
Nothing coupled the board's commands to either of those existing habits.

## Architecture touched

- `scripts/git_hooks/post-commit` (new): the heartbeat hook itself.
- `scripts/install_git_safety_hooks.sh`: adds `install_post_commit_hook()`,
  a dedicated install path for `post-commit` distinct from
  `install_one_hook` (used for `pre-commit`/`post-merge`), because
  `post-commit` is a hook name this repo's own `graphify hook install`
  already claims.
- `scripts/safe_docker_build.sh`: adds the pre-build heartbeat block.
- `CLAUDE.md`: documents both automatic heartbeats under the existing
  "Agent workspace board" section.
- `tests/scripts/test_install_git_safety_hooks.py`,
  `tests/scripts/test_safe_docker_build_heartbeat.py` (new): coverage for
  all of the above, including three regression tests for bugs found during
  development (below).

## Files changed

- `scripts/git_hooks/post-commit`: new hook, heartbeats with the commit
  subject; never calls a bare top-level `exit` itself, so it composes
  safely if something else ever concatenates around it.
- `scripts/install_git_safety_hooks.sh`: `install_post_commit_hook()` (new)
  + `_emit_post_commit_block()` (new helper) replace a plain
  `install_one_hook post-commit` call. Handles three cases: fresh install,
  refresh (our block already present), append (foreign hook present, no
  block of ours yet).
- `scripts/safe_docker_build.sh`: heartbeat block added after the
  shared-checkout refusal and service-argument validation, before the
  `exec docker compose` call.
- `CLAUDE.md`: new paragraph under "Agent workspace board".
- `tests/scripts/test_install_git_safety_hooks.py`: extended existing
  hook-install tests to cover all three hooks; added
  `test_post_commit_appends_to_existing_foreign_hook_without_backup`,
  `test_post_commit_heartbeat_survives_foreign_hook_early_exit`,
  `test_post_commit_refresh_preserves_content_appended_after_our_block`,
  `test_installed_post_commit_hook_heartbeats_the_agent_board`.
- `tests/scripts/test_safe_docker_build_heartbeat.py` (new): heartbeat
  fires before a fake `docker` binary runs; heartbeat failure (unwritable
  board path) never blocks the build.

## Schema / bus / API changes

None. This is tooling/hooks only, no bus/schema/API surface touched.

## Env/config changes

None. `agent_board.py` already reads `ORION_AGENT_BOARD_PATH` (defaults to
`~/.orion/agent-board.jsonl`); nothing new added.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest tests/scripts/ -q
190 passed in 44.6s
```

Plus manual live verification beyond the automated suite (see "Review
findings fixed" for what each one caught):
- Real `git commit` from a linked worktree in this repo's actual shared
  checkout, with the real `graphify hook install` hook in place: heartbeat
  landed with the correct commit subject on `~/.orion/agent-board.jsonl`.
- Reviewer's exact repro for the refresh/suffix-preservation bug (append
  content after our block, refresh 4 times): content survives every
  refresh, no blank-line growth on either side, file stays valid POSIX sh
  (`sh -n`) throughout.

## Evals run

No eval harness applies to shell tooling / git hooks in this repo.

## Docker/build/smoke checks

Not applicable -- no service `docker-compose.yml` touched. `scripts/
safe_docker_build.sh`'s own new heartbeat block is covered by
`tests/scripts/test_safe_docker_build_heartbeat.py` using a fake `docker`
binary on `PATH` instead of a real daemon.

## Review findings fixed

- Finding: `post-commit` is a hook name real tooling already uses here
  (`graphify hook install`'s rebuild hook) -- the first version of this
  branch used the same overwrite-and-backup strategy as `pre-commit`/
  `post-merge`, which silently disabled graphify's rebuild-on-commit,
  confirmed live against the real repo checkout.
  - Fix: `install_post_commit_hook()` appends to any existing foreign
    `post-commit` hook instead of overwriting it.
  - Evidence: live repro in `/tmp` restoring graphify's hook then
    reinstalling; confirmed both hooks' content present afterward.
- Finding: appending two shell scripts back-to-back is unsafe if either
  calls a bare `exit` -- graphify's hook legitimately calls `exit 0` for
  any commit from a linked worktree (its rebuild only runs from the
  primary checkout), which silently killed the appended heartbeat on every
  worktree commit -- this repo's actual common case (worktrees-only
  convention).
  - Fix: each fragment wrapped in its own `( ... ) || true` subshell.
  - Evidence: `test_post_commit_heartbeat_survives_foreign_hook_early_exit`
    (would fail against the pre-fix concatenation); also verified with a
    real `git commit` in `/tmp/dbgtest3` using a synthetic early-exiting
    foreign hook.
- Finding (dispatched review agent): the refresh path assumed our marker
  block was always the file's tail, silently discarding content another
  tool appended AFTER ours (confirmed reachable: re-running
  `graphify hook install` after ours was in place does exactly this,
  since graphify's own installer also appends to whatever exists) --
  `CLAUDE.md` and the installer's own log message both overclaimed this
  was already safe.
  - Fix: our fragment is now bounded by explicit BEGIN/END marker lines
    (not just one marker comment), so refresh locates both ends and
    preserves whatever sits on either side.
  - Evidence: `test_post_commit_refresh_preserves_content_appended_after_our_block`;
    manually reproduced the reviewer's exact scenario and refreshed 4
    times -- appended content survives every time, `sh -n` stays valid.
- Finding (dispatched review agent, minor): blank-line padding
  accumulated before the marker on repeated refreshes.
  - Fix: prefix trimmed of trailing blank lines via `awk` before
    reassembly; suffix symmetrically trimmed of leading blank lines for
    the same reason once the BEGIN/END fix made a suffix possible.
  - Evidence: manual 4x-refresh repro shows blank-line count stable at 1
    on both sides, not growing.
- Finding (dispatched review agent, minor): the append branch assumed a
  foreign hook's first line was always a shebang before skipping it.
  - Fix: only skips the first line if it actually starts with `#!`.
  - Evidence: code inspection; no foreign hook in this repo currently
    lacks a shebang, so no live repro was possible, but the fix is cheap
    and directly closes the gap the reviewer named.
- Finding (dispatched review agent, minor, not fixed): the board's
  underlying `flock` (in pre-existing `agent_board_lib.py`, not touched by
  this branch) has no timeout, and this branch is what wires that lock
  into two new high-frequency call sites (every commit, every docker
  wrapper call). A stuck concurrent writer could in principle stall a
  commit or docker build waiting on it.
  - Not fixed: touching the lock itself is out of scope for this branch
    (pre-existing code, not something this diff introduces a new failure
    mode in) and the exposure window is small (a single fast JSON-line
    append+flush, and OS `flock`s release automatically even on
    `SIGKILL`). Flagging as a known, low-severity limitation rather than
    fixing here.
- Finding (my own, self-caught via testing, not the dispatched review): a
  marker-duplication bug -- the appended block explicitly echoed the
  marker comment AND copied it a second time via `tail -n +2` on the
  source hook (whose own line 2 is that same marker comment).
  - Fix: `tail -n +3` skips both the source's shebang and its own marker
    line.
  - Evidence: marker count confirmed at exactly 2 (BEGIN + END) after
    install, not 3, across all test scenarios.

## Restart required

```text
No restart required.
```

This changes local git-hook and docker-wrapper tooling only, not a running
service. Anyone who wants the new `post-commit` heartbeat active in their
own checkout of this repo needs to (re-)run
`scripts/install_git_safety_hooks.sh .` once -- installed hooks live in the
shared/common git dir, so this covers every worktree of the repo, not just
the one it's run from.

## Risks / concerns

- Severity: Low
  - Concern: the underlying board lock has no timeout (see "Review
    findings fixed" above) -- a genuinely stuck concurrent board writer
    could stall a commit or docker build.
  - Mitigation: known limitation, flagged rather than fixed; the exposure
    window is small in practice. Worth a follow-up if it's ever actually
    observed to hang.
- Severity: Low
  - Concern: this branch's live testing required repeatedly mutating the
    real, shared `/mnt/scripts/Orion-Sapienform` checkout's installed git
    hooks (`.git/hooks/post-commit`) to verify behavior against the real
    `graphify hook install` hook already present there. That file is
    correctly restored to a clean, working state as of this report (graphify
    hook intact, our hook correctly appended, verified via a live commit),
    but it's worth double-checking after merge that no other concurrent
    session's hook state was disturbed in the process.
  - Mitigation: final live state verified via `grep`/`sh -n`/an actual test
    commit immediately before this report was written; no destructive git
    operations were used (`.git/hooks/post-commit` is not a tracked file).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/chore/agent-board-heartbeat-coupling
