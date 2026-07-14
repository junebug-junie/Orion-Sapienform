# PR report: block destructive git commands in the shared checkout via PreToolUse hook

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1041
Branch: `feat/destructive-git-op-guard`

## Summary

- Adds `scripts/hooks/destructive_git_guard.py`, a Claude Code `PreToolUse` hook that blocks `git clean -f*`, `git reset --hard`, and `git checkout`/`git switch --force` from running against the shared/primary checkout of this repo.
- Wires it into `.claude/settings.json` alongside the existing `graphify hook-guard search` entry (same "Bash" matcher).
- Reuses the same git-dir/git-common-dir shared-checkout detection as `scripts/git_hooks/pre-commit` and `scripts/safe_docker_build.sh`; both got a comment update pointing at this third implementation.
- 57 tests in `tests/scripts/test_destructive_git_guard.py`, including a regression test for every bug found in review.

## Outcome moved

`git clean -fd`/`git reset --hard` destroy uncommitted files *before* any commit happens, so the existing `pre-commit` hook (which only gates `git commit`) can't help — the damage is already done. This closes that specific gap: it's the same incident class that already happened once this session (a concurrent session's presumed `git clean -fd` wiped `graphify-out/` extras and a spec doc from the shared checkout mid-session, discovered via reflog/timestamp investigation, not caused by this work).

## Current architecture

Before this PR: `scripts/git_hooks/pre-commit` blocked commits from the shared checkout; `scripts/safe_docker_build.sh` was a convention-based wrapper for `docker compose`. Neither covers destructive git operations that never reach a commit.

## Architecture touched

- `.claude/settings.json` — new PreToolUse hook entry.
- `scripts/hooks/destructive_git_guard.py` — new.
- `scripts/git_hooks/pre-commit`, `scripts/safe_docker_build.sh` — comment-only, point at the new third implementation of the shared-detection logic.
- `tests/scripts/test_destructive_git_guard.py` — new.

## Files changed

- `scripts/hooks/destructive_git_guard.py`: new PreToolUse hook script.
- `.claude/settings.json`: wires the hook into the existing "Bash" matcher.
- `tests/scripts/test_destructive_git_guard.py`: 57 tests, including a named regression test per confirmed review finding.
- `scripts/git_hooks/pre-commit`, `scripts/safe_docker_build.sh`: comment updates only (cross-reference the third detection implementation).

## Schema / bus / API changes

None. Pure tooling, no bus/schema/env surface touched.

## Env/config changes

None. No `.env_example` changes; nothing to sync.

## Tests run

```
/mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest tests/scripts/test_destructive_git_guard.py -q
57 passed in 1.53s
```

Note: `pyproject.toml`'s `norecursedirs = ["scripts", ...]` matches directory *basename*, so it also silently excludes `tests/scripts/` from bare `pytest`/`pytest tests -q` discovery (confirmed via `--collect-only`: 0 matches under default testpaths, full collection with the explicit path). This is a **pre-existing, repo-wide gap** — two sibling files (`test_orion_proposal_cli.py`, `test_sync_local_env_from_example.py`) already live there and are always invoked by explicit path in prior PR reports, never bare `pytest`. Flagging as a follow-up, not fixing in this PR since it would affect unrelated test suites.

## Evals run

None. This is a Claude Code hook script, not a `services/<name>` component with its own eval harness — no existing eval harness applies. The 57 tests include full end-to-end coverage (real subprocess over stdin/stdout, real git repos with actual worktrees, no mocks).

## Docker/build/smoke checks

Not applicable — no runtime/service/Docker surface touched.

## Live verification (not just tests)

Two live end-to-end tests were run with explicit Juniper approval via AskUserQuestion (the command itself, `git reset --hard HEAD`, is a genuine no-op against an unmoved HEAD in a clean tree, but AGENTS.md requires explicit approval for it regardless):

1. Temporarily wired the hook into the live shared checkout's `.claude/settings.json`, dispatched a fresh subagent to run `git reset --hard HEAD` there. **Blocked** — the subagent received the exact `permissionDecisionReason` the hook emits, the command never executed, confirmed via `git status` before/after. Wiring reverted after the test.
2. Confirmed `$CLAUDE_PROJECT_DIR` (used in the committed hook command so it works from any clone/machine) resolves correctly at hook-invocation time, via a non-destructive diagnostic hook writing to a log file — `/mnt/scripts/Orion-Sapienform` in all three firings, including from a dispatched subagent.

A third test (compound command tripping both the graphify hook and this one simultaneously, to confirm the deny still wins) was attempted but the dispatched subagent correctly declined to run a `git reset --hard` on a relayed "the user already approved this" claim from the orchestrating session, since agent-relayed approval isn't verifiable consent. That refusal is correct behavior, not a bug — this specific residual question (does Claude Code's harness honor a deny from the second of two hooks under one matcher when both produce output) is therefore **UNVERIFIED**, left as a documented low-severity gap rather than pushed further through relay.

## Review findings fixed

Ran the `code-review` skill at high effort (8 parallel finder angles, live-reproduced where possible). 10 findings reported, all fixed via a full rewrite of the core detection logic (the bugs shared one root cause: parsing the whole command string instead of walking statements in order while tracking directory state):

- **Finding**: Escape hatch searched the entire command string unscoped/unanchored — matched inside a shell comment (`# ORION_ALLOW_SHARED_CHECKOUT_WRITE=1`, never actually sets the env var) or an unrelated later statement, silently authorizing a real destructive command.
  - **Fix**: `_escape_hatch_set` now only matches a genuine leading env-assignment prefix on the *same* statement as the destructive command.
  - **Evidence**: `test_evaluate_escape_hatch_scoped_to_its_own_statement`, `test_escape_hatch_trailing_is_not_a_bypass`.
- **Finding**: `_resolve_target_dir` only handled a single leading `cd`; `echo x && cd ../shared && git reset --hard` (an ordinary two-hop chain) bypassed directory resolution entirely.
  - **Fix**: `_evaluate` now walks statements left-to-right tracking a running current directory across every `cd` statement.
  - **Evidence**: `test_evaluate_multi_hop_cd_bug`, `test_evaluate_sequential_cd_statements`.
- **Finding**: `git reset --hard` regex used `.*` without `re.DOTALL`, so a bash line-continuation (`git reset \` + newline + `--hard`) evaded detection.
  - **Fix**: added `re.DOTALL` to `_GIT_RESET_HARD`.
  - **Evidence**: `test_evaluate_line_continuation_reset_hard`.
- **Finding**: `-C <dir>`/`cd` extraction scanned the whole raw string unanchored, so decoy text in an earlier statement (e.g. inside an echoed string) could hijack directory resolution.
  - **Fix**: extraction now scoped per-statement, using quote-stripped text for pattern matching and the original text (aligned by offset) for real path values.
  - **Evidence**: `test_evaluate_decoy_dash_c_in_earlier_statement_no_longer_hijacks`.
- **Finding**: Detection was quoting-unaware — a commit message merely mentioning "git reset --hard" as text was misidentified as the command itself.
  - **Fix**: added `_strip_quoted`, applied before statement-splitting and pattern matching.
  - **Evidence**: `test_evaluate_quoted_mention_is_not_a_false_positive`.
- **Finding**: `_is_shared_checkout` used `os.path.realpath` (resolves symlinks) while the two existing shell hooks use logical `cd && pwd` (does not) — the "mirrored" implementations weren't actually equivalent for symlinked paths.
  - **Fix**: switched to `os.path.normpath`, matching logical-pwd semantics.
- **Finding**: `git checkout -f`/`git switch -f` (equally destroy uncommitted tracked-file changes) weren't detected, and unlike `push --force`/`branch -D` (explicitly commented as out-of-scope), this wasn't documented as intentional.
  - **Fix**: added detection for both; documented `switch -C`/`checkout -B` (force-create-branch variants) as a remaining, disclosed gap.
  - **Evidence**: `test_evaluate_checkout_force_blocked`, `test_evaluate_switch_force_blocked`.
- **Finding**: `_is_shared_checkout`'s fail-open behavior on git-subprocess error/timeout wasn't listed in the module's own "Known gaps" docstring.
  - **Fix**: documented explicitly.
- **Finding**: `tests/scripts/` silently excluded from bare `pytest` via `norecursedirs` basename matching.
  - **Fix**: not fixed in this PR (pre-existing, repo-wide, affects unrelated suites) — documented above under Tests run, PR report uses the explicit path.
- **Finding** (architecture-contradiction, most serious): a same-day prior PR report (`docs/superpowers/pr-reports/2026-07-14-agent-git-safety-hooks-pr.md`) explicitly evaluated and *rejected* "PreToolUse hook as enforcement," quoting Juniper directly ("I don't want to be beholden to developing with a paid subscription"), scoping PreToolUse hooks to disclosure-only and calling a vendor-neutral git-level wrapper the real, not-yet-built enforcement layer. This PR ships `permissionDecision: deny` — real enforcement — reversing that without initially reconciling it.
  - **Resolution**: surfaced directly to Juniper before proceeding (not fixed silently). Decision: keep this as real enforcement now; the vendor-neutral wrapper is deferred, not abandoned — next up once agent worktree-discipline noncompliance (why agents end up in the shared checkout at all) is addressed directly. Documented in the module docstring's "Deliberately scoped to Claude Code" section and in this PR report so the tradeoff is visible, not silent.

Two additional review angles (efficiency, simplification) suggested minor perf/style nits (precompile regexes, merge two `git rev-parse` calls into one) — the rewrite incorporates the regex-precompilation and single merged `git rev-parse --git-dir --git-common-dir` call naturally; the rest were not separately chased given diminishing returns relative to the correctness fixes above.

## Restart required

```text
No restart required. .claude/settings.json is read per-session; a new Claude Code session (or subagent) picks up the new hook automatically once this branch is merged to main.
```

## Risks / concerns

- **Severity: Medium.** This hook only protects Claude Code sessions that load this repo's `.claude/settings.json` — a different AI tool, a plain terminal, or a human typing the command directly is unprotected. Documented explicitly in the module docstring and this report; not fixed here by design (see architecture-contradiction finding above). Next step: build the vendor-neutral git-level wrapper, deferred until after agent worktree-discipline noncompliance is addressed.
- **Severity: Low.** Comments (`# ...`) aren't stripped before statement-splitting; a `&&`/`;`/`|` after a `#` on the same line can cause an incorrect statement split. Errs toward over-blocking (a harmless command misread as two statements), not under-blocking.
- **Severity: Low.** Variable expansion (`$VAR`, `$(...)`) isn't performed; a `cd "$VAR"` resolves to the literal string, which fails the "is this a real directory" check and so fails open (allowed).
- **Severity: Low.** The "two hooks fire simultaneously on a compound command" interaction is unverified end-to-end (see Live verification above) — believed safe (each hook's deny should be evaluated independently) but not proven live in this session.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1041
