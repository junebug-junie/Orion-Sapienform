# PR report: worktree hygiene tooling (status, prune, creation, hooks)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1046
Branch: `chore/worktree-hygiene-tooling`

## Summary

- `scripts/worktree_lib.py`: shared helpers (`list_worktrees`, `merged_branch_set`, `all_open_prs`, `dir_size_bytes`, `parallel_map`) used by every consumer below.
- `scripts/worktree_status.py`: reconciled table view (path, branch, merged status, open PR, disk size) across all worktree location conventions.
- `scripts/prune_merged_worktrees.py`: dry-run by default; `--yes` removes worktrees whose branch is merged into main. Never force-removes, never touches branches.
- `scripts/new_worktree.sh`: one front door for creating a sibling-directory worktree, with a cross-convention collision warning and a helpful error when the branch already exists.
- `scripts/git_hooks/post-merge` + `scripts/hooks/session_start_worktree_summary.py`: advisory nudges after every merge and at the start of every session.
- `CLAUDE.md`: documents all three worktree conventions this repo actually has, not just the one it previously described.

## Outcome moved

A live audit (not a guess) found **276 worktrees** under `/mnt/scripts/`, of which **248 are already merged into main and never cleaned up** (~25GB reclaimable), and only 26 represent active work. It also found **three simultaneously-used worktree location conventions** — sibling directories (65, the only one CLAUDE.md documented), `.worktrees/` (186, from other tooling), and `.claude/worktrees/agent-<id>` (15+, Claude Code's own `isolation: worktree` Agent-tool feature) — meaning the fragmentation problem this was built to address was less "agents ignore the policy" and more "worktree creation has three uncoordinated paths, and lifecycle (discovery, cleanup) had zero."

## Current architecture

Before this PR: no worktree cleanup tooling existed anywhere in `scripts/`. `graphify prs --worktrees` gave manual, on-demand visibility but nothing wired it into any automated workflow. CLAUDE.md documented one of three real worktree conventions.

## Architecture touched

`scripts/`, `scripts/hooks/`, `scripts/git_hooks/`, `tests/scripts/`, `Makefile`, `.claude/settings.json`, `CLAUDE.md`. Pure tooling — no service boundaries touched.

## Files changed

- `scripts/worktree_lib.py`: new, shared helpers.
- `scripts/worktree_status.py`, `scripts/prune_merged_worktrees.py`, `scripts/new_worktree.sh`: new.
- `scripts/git_hooks/post-merge`, `scripts/hooks/session_start_worktree_summary.py`: new hooks.
- `scripts/install_git_safety_hooks.sh`: refactored from a single hardcoded pre-commit-install block into a loop that installs both `pre-commit` and `post-merge`.
- `Makefile`: 4 new targets (`worktree-status`, `worktree-status-summary`, `worktree-status-stale`, `prune-merged-worktrees`), each accepting `BASE=` to override the default `origin/main` comparison.
- `.claude/settings.json`: new `SessionStart` hook entry.
- `CLAUDE.md`: documents all three worktree conventions and the new tooling.
- `tests/scripts/`: `conftest.py` (shared fixtures) + 7 test files, 125 tests total.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```
/mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest tests/scripts/ -q
125 passed in 19.47s
```

## Evals run

None — no eval harness applies to this class of tooling script; same rationale as the destructive-git-guard PR (#1041).

## Docker/build/smoke checks

Not applicable — no runtime/service surface touched.

## Live verification (real data, not synthetic-only)

All of the following were run against this repo's real, current worktree state (not just the synthetic-fixture test suite):

- `worktree_status.py --summary`: 0.152s for 278 worktrees.
- `worktree_status.py` (full table, batched PR lookup): **1.3s**, down from a measured **~2.5 minutes** with the original per-worktree `gh pr list` design (caught in review — see below).
- `prune_merged_worktrees.py` (dry-run, parallel `du`): **0.78s**, down from a measured **8.7s** sequential.
- `install_git_safety_hooks.sh`: installed both hooks fresh, then re-ran to confirm idempotent refresh of both, against a throwaway repo (not this one).
- `new_worktree.sh`: created and removed a real test worktree; confirmed the collision warning fires against this repo's real worktree list.

## Review findings fixed

Ran `code-review` at high effort (8 parallel finder angles). ~20 findings across correctness, reuse, simplification, efficiency, altitude, and CLAUDE.md conventions. Highest-severity fixed:

- **Finding (reuse, most severe)**: `open_pr_for_branch()` shelled out to `gh pr list --head <branch>` once *per worktree* — 276 separate subprocess+network round trips (measured ~2.5 min), and a transient failure on any single one silently reported that worktree as "no open PR," which `--stale-only` would then treat as a genuine prune candidate.
  - **Fix**: `all_open_prs()` — one `gh pr list --state open` call, mapped locally by branch. Returns `None` (not `{}`) on failure, so callers can distinguish "no open PRs" from "PR data unavailable." `--stale-only` now refuses to report (exit 1) rather than silently treating unknown-PR-status worktrees as stale.
  - **Evidence**: `test_all_open_prs_returns_none_when_gh_has_no_remote`, `test_stale_only_refuses_when_pr_data_unavailable`, live timing above.
- **Finding**: `prune_merged_worktrees.py`'s `du` calls were sequential (measured 8.8s at 248 worktrees) while `worktree_status.py`'s identical operation was already parallelized.
  - **Fix**: extracted `parallel_map()` into `worktree_lib.py`, used by both.
- **Finding**: `list_worktrees()`/`merged_branch_set()` had no error handling, unlike every other helper — a failure (bad ref, git unavailable) would either raise an uncaught traceback or (for a since-fixed version of `merged_branch_set`) silently return an empty set indistinguishable from "genuinely nothing merged."
  - **Fix**: both now raise `WorktreeLibError` on failure; callers catch it and print a clean `[tool-name] ERROR: ...` message instead of a traceback or a misleading "all clear."
  - **Evidence**: `test_list_worktrees_raises_outside_any_repo`, `test_merged_branch_set_raises_on_bad_base`, `test_bad_base_ref_reports_error_not_traceback`.
- **Finding**: `dir_size_bytes()` used `du -sb`, a GNU-only flag that fails silently (swallowed by a bare `except`) on BSD/macOS `du`.
  - **Fix**: switched to the portable `du -sk` (POSIX-ish, widely supported), converting KB to bytes.
- **Finding**: `install_git_safety_hooks.sh`'s source-hook-existence check moved (during the pre-commit/post-merge refactor) from before any mutation to inside the per-hook install function — a missing source hook now creates a `mkdir -p` side effect (an empty custom hooks dir) before erroring, when the original script failed clean.
  - **Fix**: both source hooks are validated to exist before `mkdir -p` runs at all.
- **Finding**: `new_worktree.sh` only implements 1 of 3 real worktree conventions and had no awareness of the other two, risking silently creating a duplicate attempt at work already in progress under a different convention.
  - **Fix**: added a `git worktree list`-based warning (not a hard block, since it's a heuristic) when an existing worktree's path or branch already mentions the requested name.
  - **Evidence**: `test_warns_on_name_collision_with_existing_worktree`.
- **Finding**: `session_start_worktree_summary.py` shelled out to a second `python3` process running `worktree_status.py --summary` purely to get a one-line string, adding a full duplicate interpreter startup (measured ~50ms) on every SessionStart firing, including every subagent spawn.
  - **Fix**: imports `worktree_lib` directly and builds the summary inline; also fixed the resulting timeout-headroom issue (inner subprocess timeout equaled the outer hook's own timeout, leaving zero margin).
- **Finding (test coverage)**: `worktree_status.py`'s own CLI logic (argparse, `--stale-only` filter, row assembly) and the new SessionStart hook had zero dedicated tests.
  - **Fix**: added `test_worktree_status.py` (9 tests) and `test_session_start_worktree_summary.py` (3 tests).
- **Finding (simplification)**: the `repo_with_worktrees` fixture and a `_git` helper were duplicated near-verbatim across 3 test files; one test duplicated another's coverage.
  - **Fix**: extracted `tests/scripts/conftest.py`; removed the redundant test.
- **Finding (CLAUDE.md conventions)**: shipping worktree tooling without correcting CLAUDE.md's incomplete (1-of-3-conventions) documentation in the same changeset violates this repo's own "runtime truth beats config truth" principle.
  - **Fix**: added a truthful section documenting all three conventions and the new tooling.

Minor, lower-severity findings not separately chased (documented, not silent): `git worktree remove` now anchors with `-C <toplevel>` (was relying on ambient cwd, latent not live); Makefile targets gained a `BASE=` override for consistency with `check-daily-schedule-collisions`'s existing pattern; a single pathological worktree's `du` call can still gate the full-table command's total runtime (bounded by the existing 30s timeout, not fixed further — diminishing returns for a manually-invoked command).

## Restart required

```text
No restart required. .claude/settings.json is read per-session; a new session picks up the SessionStart hook automatically once this branch is merged. Run `scripts/install_git_safety_hooks.sh .` once per existing checkout to pick up the new post-merge hook (same as any other hook update).
```

## Risks / concerns

- Severity: Low. `new_worktree.sh` only automates 1 of 3 real worktree conventions (by design — it targets the one AGENTS.md documents); review flagged that this doesn't reduce fragmentation on its own, only adds a consistent front door for the minority case. The collision-warning fix mitigates the sharpest risk (silent duplicate work) without attempting to unify all three mechanisms, which was out of scope for this PR.
- Severity: Low. A single pathological worktree (unusually large, e.g. from accumulated build artifacts) can still gate the full-table/`--stale-only` command's total runtime up to the 30s per-item timeout, since results are collected via a blocking `list(pool.map(...))` rather than streamed. Not fixed given the class of command this affects (manually-invoked, not session-start-blocking).
- Severity: Low. Comment/quote handling in `new_worktree.sh`'s collision-detection grep is a simple substring match, not exact — could produce false-positive warnings for coincidentally-similar names. Errs toward over-warning, not silent duplication.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1046
