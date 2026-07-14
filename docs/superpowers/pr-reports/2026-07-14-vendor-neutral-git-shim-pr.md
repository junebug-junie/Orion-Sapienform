# PR report: vendor-neutral git shim (system-wide PATH enforcement)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1051
Branch: `feat/vendor-neutral-git-shim`

## Summary

- `scripts/git_hooks/orion-git-shim`: a `git` replacement installed earlier in PATH than the real binary, blocking `git clean -f*`, `git reset --hard`, `git checkout`/`git switch --force` against the shared/primary checkout of any repo that has explicitly opted in.
- `scripts/install_orion_git_shim.sh`: installer, sets up the shim on PATH and opts a repo in via a marker file.
- Updates to `scripts/git_hooks/pre-commit`, `scripts/safe_docker_build.sh`, `scripts/hooks/destructive_git_guard.py`, and `CLAUDE.md` cross-referencing the new mechanism.

## Outcome moved

`scripts/hooks/destructive_git_guard.py`'s own docstring named this exact gap and deferred it: that hook only protects Claude Code sessions specifically — a plain terminal, a different AI tool, or a human typing the command directly sailed straight through unguarded. This closes that gap for any caller doing a normal PATH-resolved `git` lookup, which is the overwhelming majority of real-world git usage.

## Current architecture

Before this PR: three implementations of shared-checkout detection existed (`pre-commit`, `safe_docker_build.sh`, `destructive_git_guard.py`), all scoped to either `git commit` specifically or Claude Code specifically. No mechanism caught `git reset --hard`/`git clean -fd` from a plain terminal or another tool.

## Architecture touched

`scripts/git_hooks/`, `scripts/`, `scripts/hooks/`, `tests/scripts/`, `CLAUDE.md`. Pure tooling.

## Files changed

- `scripts/git_hooks/orion-git-shim`: new, the shim itself.
- `scripts/install_orion_git_shim.sh`: new, installer.
- `scripts/git_hooks/pre-commit`, `scripts/safe_docker_build.sh`: comment-only, cross-reference the new fourth detection implementation.
- `scripts/hooks/destructive_git_guard.py`: docstring updated — the "deferred wrapper" it named now exists.
- `CLAUDE.md`: documents the shim in section 2, clarifies section 13's actual (narrower) coverage.
- `tests/scripts/test_orion_git_shim.py`: 25 tests.

## Schema / bus / API changes

None.

## Env/config changes

None. `ORION_GIT_SHIM_DIR` is an optional installer override, not a new persistent env var.

## Tests run

```
/mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest tests/scripts/test_orion_git_shim.py -q
25 passed in 1.85s
```

## Evals run

None — no eval harness applies to this class of script.

## Docker/build/smoke checks

Not applicable.

## Live verification

Extensive manual testing against real (throwaway) repos with a fake PATH, never touching the real `~/.local/bin`: passthrough for non-destructive commands, block/allow for every guarded subcommand, escape hatch, worktree-vs-shared-checkout distinction, `-C` targeting, installer fresh-install/idempotency/symlink-refusal/unrelated-file-refusal, PATH-ordering warning. All of this is also now codified as the 25 automated tests.

**Not yet installed on the real machine.** This PR ships the code; actually placing the shim at the real `~/.local/bin/git` and opting this repo in is a separate, explicit step — deliberately not bundled into merging this PR, given the blast radius (affects `git` resolution for the whole shell environment, not just this repo).

## Review findings fixed

Ran a 4-dispatch high-effort review (line-by-line, reuse/CLAUDE.md-conventions, altitude/cross-file, security/removed-behavior). The first draft had four real, **empirically reproduced** bypasses — not theoretical:

- **Finding (alias bypass, most severe)**: `git config alias.hr "reset --hard"; git hr` totally bypassed the guard — the literal token "hr" matched none of clean/reset/checkout/switch, so the whole invocation took the immediate passthrough path with zero destructive-shape inspection. Reproduced live against the real shim.
  - **Fix**: resolves simple (non-`!`) aliases via `git config --get alias.<token>` and re-checks the expansion; `!`-prefixed shell aliases are an explicit, disclosed gap (not argv-analyzable).
  - **Evidence**: `test_simple_alias_to_reset_hard_blocked`.
- **Finding (--git-dir bypass, total, using ordinary syntax)**: `git --git-dir <path> --work-tree <path> reset --hard` had the path argument after `--git-dir` misidentified as the subcommand (the generic `-*)` catch-all skipped the flag but didn't consume its value), so the real "reset"/"--hard" tokens were never inspected. Reproduced live: exit 0, "HEAD is now at...", against an opted-in repo.
  - **Fix**: `--git-dir`/`--work-tree`/`-c`/`--namespace`/`--exec-path` now correctly consume their following argument; `--work-tree`'s value is used as the target-dir override (takes priority over `-C`, matching git's own precedence).
  - **Evidence**: `test_work_tree_flag_targeting_shared_checkout_blocked`, `test_git_dir_alone_does_not_misattribute_subcommand`.
- **Finding**: stacked `-C` flags (`git -C /a -C b`) overwrote the target dir instead of chaining relative to the previous one the way real git does.
  - **Fix**: each subsequent `-C` now resolves relative to the accumulated target dir.
  - **Evidence**: `test_stacked_dash_c_chains_correctly`.
- **Finding**: two independently-installed shim copies on PATH (e.g. installer re-run with a different `ORION_GIT_SHIM_DIR`, old copy never removed) recognized "self" only by exact path identity — each treated the other as "the real git" and exec'd into it, an unbounded mutual-recursion hang. Reproduced live: `timeout 5 git --version` killed (exit 124) with two shim copies on PATH.
  - **Fix**: self-detection now checks file *content* for the shim's own marker string, not path identity — any shim copy, however it got there, is correctly recognized and skipped.
  - **Evidence**: `test_two_shim_copies_on_path_do_not_infinite_loop`.
- **Finding**: `clean.requireForce=false` (a real, legitimate git config some repos carry) makes `git clean -d` destructive with **no** `-f`/`--force` anywhere on the command line — the shim's argv-only scan missed this entirely. Reproduced live: real untracked file deleted, guard never engaged.
  - **Fix**: checks `clean.requireForce` when no explicit force flag is present.
  - **Evidence**: `test_clean_requireforce_false_blocked_without_explicit_force_flag`.
- **Finding (efficiency)**: the opt-in check originally ran *after* destructive-shape detection and shelled out to `git config --get` — a full subprocess for every destructive-shaped command on every repo on the machine, opted-in or not.
  - **Fix**: switched opt-in from a git-config flag to a marker file, checked first, with a single `[ -f ... ]` — no subprocess for the common (not-opted-in) case at all.
- **Finding (CLAUDE.md conventions, corroborated by two independent review angles)**: shipping a system-wide mechanism without updating CLAUDE.md in the same changeset violates this repo's own "runtime truth beats config truth" principle; `destructive_git_guard.py`'s own docstring was left stale, claiming the wrapper was "deferred" after it had actually been built.
  - **Fix**: CLAUDE.md section 2 documents the shim; section 13 clarifies neither mechanism implements its blanket approval rule; `destructive_git_guard.py`'s docstring updated to reflect the shim now exists.
- **Finding**: `pre-commit`/`safe_docker_build.sh`'s own "if you fix a bug here, fix it there too" cross-reference comments didn't mention the new fourth detection implementation.
  - **Fix**: both updated.

Lower-severity findings disclosed rather than fixed (documented in the shim's own header comment): `sudo git ...` bypasses the shim (sudo's secure_path excludes user-local bin dirs); a caller that caches an absolute git path once (common in some libraries) never re-resolves through PATH and so never reaches the shim; TOCTOU between the shim's checks and its final exec, the same trust assumption inherent to any PATH-based tool; no central "which repos are protected" registry beyond grepping for the marker file.

## Restart required

```text
No restart required for the code itself. Actually activating protection requires running scripts/install_orion_git_shim.sh, a separate, explicit step from merging this PR (see "Live verification" above).
```

## Risks / concerns

- Severity: Medium. This shim, like any PATH-based tool, cannot catch `sudo git ...` or a caller using a cached absolute git path. Documented in the shim's own header, not silently overclaimed.
- Severity: Low. No central registry of which repos have opted in beyond the marker file itself — auditing coverage across many repos requires a filesystem sweep, not a single command. Not built in this PR (scope discipline); a `find / -name orion-safety-guard-enabled` sweep works today if needed.
- Severity: Low. Alias resolution handles the common (non-`!`) case; `!`-prefixed shell aliases are an explicit, disclosed gap given they're arbitrary shell commands, not argv-analyzable.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1051
