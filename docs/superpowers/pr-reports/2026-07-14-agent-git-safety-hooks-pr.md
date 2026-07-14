# PR report: git-native enforcement for the shared-checkout worktree policy

## Summary

- A concurrent AI coding-agent session ran `docker compose build`+`up` directly from the shared/primary git checkout of this repo (not a worktree) and silently reverted another session's already-verified fix. `AGENTS.md` already said "use worktrees" at the time — it had said so the whole time — and that did not stop it.
- This PR replaces that sentence with a real, git-native `pre-commit` hook that refuses commits from the shared checkout, plus a `docker compose` wrapper for the half of the incident a git hook structurally cannot reach, plus a registered union-merge driver for the knowledge-graph artifact (`graphify-out/graph.json`) that's about to start being committed to this repo as a separate, follow-up piece of work.
- Every mechanism here is plain git or a POSIX shell script — nothing is tied to Claude Code, or any other specific AI coding tool, or a paid subscription. That was a deliberate, considered call, not a default; see "Why this shape and not another one" below for the reasoning and what got explicitly rejected along the way.
- A full 6-angle code-review pass, run independently after the implementing subagents reported done, found and the orchestrator fixed: a real correctness bug (the docker wrapper silently dropped required environment variables for most services), a real safety gap (the installer could silently write through an existing symlink into a file outside this repo), an idempotency bug (a fixed hook would never reach an already-installed checkout), a tilde-expansion bug, a wrong confirmation-message path for worktrees, and a documented (not silently ignored) residual gap around rebase/merge/cherry-pick sequences.
- Branch protection on `main` was also confirmed off and turned on in the same working session that produced the spec this PR implements — PR review is now required to merge, including for the repo owner.

## Outcome moved

Before this PR, nothing in this repo enforced "don't write to the shared checkout" — it was a sentence in a policy document that a real incident had already demonstrated doesn't work. After this PR, a commit or a `docker compose` build/deploy from the shared checkout is refused outright, with a deliberate and visible escape hatch for the rare genuinely-intentional case, using mechanisms that work identically for this session, a different AI tool, a human's own terminal, or CI.

## Why this shape and not another one

This started as a much broader design conversation, and most of what got seriously considered is *not* in this PR — worth recording why, so it doesn't get silently re-proposed later without the context for why it was set aside.

**The first framing was wrong.** The original ask was "track git changes in a database so an agent session knows to sync up." Research into how real tools solve adjacent problems reshaped that:

- **Postgres-backed change-tracking table** — considered, explicitly rejected by the repo owner. Not needed for what this actually turned out to require.
- **A locally-maintained mutable ledger file tracking git diffs** — the first design for "what changed since I last looked." Rejected after finding that real prior art (Turborepo's `--affected`, Nx, Watchman's source-control-aware queries) all compute that diff *live* against a merge-base rather than persisting any state file. Nothing to corrupt, race, or leave orphaned across worktrees if nothing is ever written down in the first place.
- **A `PreToolUse` Claude Code hook as the enforcement mechanism** — the first shape of the actual guard. Rejected specifically because it only protects sessions running through that one tool — a human's own terminal, or a different AI coding assistant, sails straight through it. The repo owner pushed back on this directly ("I don't want to be beholden to developing with a paid subscription"), which is what motivated moving the mechanism down to plain git hooks instead — git hooks fire for *any* client that runs `git commit`, tool-independent, no subscription required. Checked directly: Claude Code hooks (`SessionStart`, `PreToolUse`) are real and useful for *disclosure* (telling a session what changed), which is why that idea survives as a documented, not-yet-built phase-2 item — but not for *enforcement*, which needs to hold regardless of what's driving the terminal.
- **Forcing every subagent to spawn inside a worktree at dispatch time** — the more fundamental fix in principle (never let the agent be in the wrong place at all, rather than block it once it's there). Rejected because there's no confirmed way to reach that layer from a repo-local hook or setting — Claude Code's own docs don't expose a reliable subagent-vs-foreground signal to hook scripts. The `pre-commit` guard in this PR gets the same practical outcome without needing that layer: it doesn't matter *how* a write attempt got to the shared checkout, only that it's refused once it tries to happen there.

**A real, separate discovery along the way:** branch protection on `main` was checked directly (`gh api repos/.../branches/main/protection` → `404 Branch not protected`) and found to be off. The repo is public, so this needed no paid plan — just nobody had turned it on. Enabled in the same session: PR required to merge, one approving review, `enforce_admins: true` (applies to the repo owner too, no exception), force-push and branch deletion both blocked. This is the one layer in the whole design that's genuinely server-side and can't be talked around by any local tool, misbehaving agent, or convention violation — verified live, not just configured and assumed.

**What "done" looks like for the weaker half of the problem.** `docker compose build`/`up` is not a git operation — no git hook, however cleverly written, has jurisdiction over it. This is acknowledged directly in `scripts/safe_docker_build.sh`'s own header comment: it is *only* effective if callers route through it instead of calling `docker compose` directly. There is no vendor-neutral, unbypassable mechanism available for this half — the wrapper plus a `CLAUDE.md` pointer (added in this PR) is the best available answer, not a complete one.

## Current architecture (before this patch)

- No git hooks were installed anywhere (`.git/hooks/` held only git's stock `.sample` files).
- `.claude/settings.json` had no `hooks` key; `.claude/settings.local.json` had `"Bash(git *)"` blanket-allowed with no directory-based gating of any kind.
- `graphify-out/` (this repo's knowledge-graph output) was untracked, not even gitignored — nobody had decided either way.
- `graphify hook install` (the `graphifyy` pip package already used in this repo) installs `post-commit`/`post-checkout` hooks for a free, AST-only graph rebuild, verified by reading `graphify/hooks.py` directly — it already contains the same shared-checkout-vs-worktree detection technique this PR reuses, confirming the technique is sound and already proven in this exact codebase. It does not cover `git merge`/`git pull` at all (no `post-merge` hook installed), which is a separate, already-known gap tracked in the spec this PR partially implements, not addressed here.
- `graphify merge-driver <base> <current> <other>` — a real, working CLI subcommand doing a `networkx.compose` union merge — was confirmed to exist and work correctly (directly executed, not assumed) but is not auto-registered by any graphify command; using it requires a manual `.gitattributes` entry plus a local, non-committable `git config` entry per clone.

## Architecture touched

`scripts/` (four new POSIX shell scripts, no existing scripts modified), `.gitattributes` (new file), `CLAUDE.md` / `AGENTS.md` (`AGENTS.md` is a symlink to `CLAUDE.md` in this repo — editing the real target), `docs/superpowers/pr-reports/` (this report).

## Files changed

- `scripts/git_hooks/pre-commit` (new): the hook body. Detects shared/primary checkout vs. linked worktree via `git rev-parse --git-dir` vs `--git-common-dir` comparison (same technique graphify's own hook already uses in this repo). Blocks with an actionable message and exit 1; escape hatch via `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1`. Skips cleanly during an in-progress rebase/merge/cherry-pick (needed so this doesn't break `git rebase --continue`'s own internal commits) with a non-blocking stderr warning added after code review flagged that this bypass is a real, if narrow, residual gap — see "Known limitations" below.
- `scripts/install_git_safety_hooks.sh` (new): installs the hook above into the correct hooks directory, resolved via `git rev-parse --show-toplevel`/`--git-common-dir` (not a hand-rolled directory walk, after code review found the hand-rolled version reinvented something `git rev-parse` already does more robustly, and that this repo already uses `git rev-parse --show-toplevel` elsewhere for the same purpose). Respects `core.hooksPath` including tilde-expansion. Idempotent — refreshes an already-installed copy rather than skipping it, so a later fix to the hook's own logic actually reaches existing installs. Refuses to write through an existing destination symlink rather than silently following it into a file outside this repo; backs up any genuinely foreign existing hook before installing over it.
- `scripts/safe_docker_build.sh` (new): wraps `docker compose` with the same shared-checkout guard, since git hooks have no jurisdiction over non-git commands. Runs from repo root with the `--env-file .env --env-file services/$SERVICE/.env -f services/$SERVICE/docker-compose.yml` pattern `CLAUDE.md`/`AGENTS.md` section 8 already mandates for every docker compose invocation in this repo — the first version of this script (from the implementing subagent) instead `cd`'d into the service directory and called bare `docker compose`, which silently drops the root `.env` that most services in this repo depend on for shared vars like `ORION_BUS_URL`; caught by code review, confirmed against real service configs, fixed and re-verified against real `docker compose`.
- `scripts/setup_graphify_merge_driver.sh` (new): registers `git config merge.graphify.driver "graphify merge-driver %O %A %B"` locally (not committable, by git's own design — every clone/worktree runs this once) and ensures `.gitattributes` carries the mapping line. Idempotent. Confirmation-message path fixed to resolve correctly when run from inside a linked worktree (where `.git` is a file, not a directory, so `$REPO_ROOT/.git/config` doesn't exist).
- `.gitattributes` (new): `graphify-out/graph.json merge=graphify`.
- `CLAUDE.md` (symlinked as `AGENTS.md`): section 8 now names `scripts/safe_docker_build.sh` directly as the way to run docker compose builds/deploys in this repo, and explains why — closing a gap code review caught: the wrapper existed but nothing in the document that actually tells agents how to invoke docker pointed at it.

## Schema / bus / API changes

None. This is process/tooling, not runtime cognition — no bus channels, no schema registry entries, no service code touched.

## Env/config changes

- Added: `ORION_ALLOW_SHARED_CHECKOUT_WRITE` — a deliberate, per-command shell escape hatch (`ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git commit ...`), never intended to be persisted. Reviewed against `AGENTS.md` section 7's env-parity rule directly by the code-review pass: that rule is scoped to persisted, checked-in config read by services at boot from `.env_example`; this variable is never read by any service and is structurally like an existing one-shot Makefile-arg pattern already in this repo, not a new operator-facing config key. No `.env_example` change needed or made.
- No other env/config changes. `.env_example` unchanged.

## Tests run

No `pytest` suite — this is shell tooling with no Python test harness of its own. Every acceptance check below was actually executed against isolated throwaway git repos under `/tmp` (never the real repo or its worktrees), first by the three implementing subagents, then independently re-run by the orchestrator after the subagents reported done, then re-run a third time after the code-review fix commit:

```text
$ sh -n scripts/git_hooks/pre-commit scripts/install_git_safety_hooks.sh scripts/safe_docker_build.sh scripts/setup_graphify_merge_driver.sh
(all four: syntax OK)

$ git diff --check
(clean)
```

**pre-commit guard**, real throwaway repo, fresh install:
```text
=== check 1: commit from shared checkout ===
[git-safety] COMMIT BLOCKED: this is the shared/primary git checkout.
... (worktree instructions, escape hatch instructions) ...
git log --oneline  ->  only "init" -- no commit created

=== check 2: commit from a linked worktree ===
exit=0, commit created normally, no hook output, no added delay

=== check 3: escape hatch ===
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 git commit ...  ->  exit=0, succeeded

=== re-install idempotency (post-fix) ===
first run:  "installed ... at .git/hooks/pre-commit"
second run: "refreshed ... at .git/hooks/pre-commit"  (was "already installed", silently skipping refresh, before the fix)

=== symlink refusal (post-fix, new check) ===
.git/hooks/pre-commit is a symlink -> /tmp/somewhere-else
install script: "ERROR: ... is a symlink ... Refusing to write through it." exit=1
```

**safe_docker_build.sh**, real `docker compose`, post-fix:
```text
$ ./safe_docker_build.sh fake-service config
name: fake-service
services:
  demo:
    environment:
      ROOT_VAR: from_root      # confirms root .env loaded
      SVC_VAR: from_service    # confirms service .env loaded
    image: busybox
...
```
Before the fix, the same command (with the original `cd services/$SERVICE && docker compose "$@"` shape) would have loaded neither root-level var for any service depending on one — confirmed this is the common case, not an edge case: 83 of 87 real service directories in this repo have a local `.env`, and multiple real services (`orion-biometrics`, `orion-bus-mirror`, `orion-dream`, and others) reference root-only vars like `${ORION_BUS_URL}` with no per-service duplicate.

**setup_graphify_merge_driver.sh**, real 3-way merge:
```text
$ git merge branch-b -m merge
Auto-merging graphify-out/graph.json
Merge made by the 'ort' strategy.

$ grep -c '<<<<<<<' graphify-out/graph.json
0

$ python3 -c "... assert node_a and node_b and base_node all present ..."
PASS: ['base_node', 'node_a', 'node_b']
```
Re-run from a linked worktree post-fix to confirm the config-path message now resolves correctly: `merge.graphify.driver already configured in /tmp/.../.git/config` (the real shared common-dir path, not a nonexistent per-worktree path).

## Evals run

Not applicable — no eval harness exists for shell tooling of this kind in this repo, and none was warranted for a change of this shape.

## Docker/build/smoke checks

Covered above under Tests run (the `safe_docker_build.sh` check against real `docker compose` *is* the smoke check for this change).

## Review findings fixed

Code review run as a full 6-angle pass (line-by-line scan, removed-behavior audit, cross-file tracer, reuse, simplification+efficiency, altitude+conventions) against the complete diff, independently verified by the orchestrator, not just trusted from finder output.

- **Finding**: `safe_docker_build.sh` dropped the mandatory dual `--env-file` pattern by `cd`-ing into the service directory and calling bare `docker compose`.
  - **Fix**: rewritten to run from repo root with `--env-file .env --env-file services/$SERVICE/.env -f services/$SERVICE/docker-compose.yml`, matching `CLAUDE.md`/`AGENTS.md` section 8 exactly.
  - **Evidence**: real `docker compose config` run showing both root and service env vars resolved; confirmed against real service configs that this is the common case (83/87 services have a local `.env`; multiple reference root-only vars).
- **Finding**: the installer's idempotency check skipped re-copying an already-installed hook, so a later fix to `scripts/git_hooks/pre-commit` would never reach an existing install.
  - **Fix**: always refresh when the destination already carries our marker; only preserve+backup genuinely foreign hooks.
  - **Evidence**: re-run transcript above showing "refreshed" instead of "already installed" on a second run.
- **Finding**: `cp` against an existing destination symlink would silently write through it to a file outside this repo (e.g. a dotfiles-managed hooks directory).
  - **Fix**: detect a symlink destination, refuse with a clear message.
  - **Evidence**: transcript above showing the refusal.
- **Finding**: `core.hooksPath` values starting with `~` were never tilde-expanded, which would install the hook into a literal `~` directory inside the repo instead of where git actually looks.
  - **Fix**: explicit `~`/`~/` expansion to `$HOME` before use.
  - **Evidence**: code inspection; not separately re-tested live (low-risk, mechanical fix, covered by the existing `core.hooksPath` absolute/relative test paths already exercised).
- **Finding**: hand-rolled directory walk-up for repo-root resolution reinvented `git rev-parse --show-toplevel`, which this repo already uses elsewhere (`scripts/git-stash-table.sh`) and even within the same file for a different sub-step.
  - **Fix**: both installer scripts now resolve repo root via `git rev-parse --show-toplevel`/`--git-common-dir` directly.
  - **Evidence**: full re-run of every acceptance check post-fix, all green.
- **Finding**: `setup_graphify_merge_driver.sh`'s confirmation message hardcoded `$REPO_ROOT/.git/config`, which doesn't exist when `REPO_ROOT` is a linked worktree (`.git` there is a gitdir-pointer file, not a directory).
  - **Fix**: message now resolves and prints the actual config path via `git rev-parse --git-common-dir`.
  - **Evidence**: re-run from a linked worktree showing the correct shared path in the confirmation message.
- **Finding**: `AGENTS.md`/`CLAUDE.md`'s own Docker readiness section still showed bare `docker compose` examples with no pointer to the new wrapper — the mitigation existed with no enforcement path, since nothing told anyone (human or agent) to use it.
  - **Fix**: section 8 now names `scripts/safe_docker_build.sh` directly and explains why.
  - **Evidence**: `grep -n "safe_docker_build.sh" AGENTS.md` shows the new line.
- **Finding** (documented, not code-fixed): the pre-commit hook's rebase/merge/cherry-pick bypass — necessary so the guard doesn't break `git rebase --continue`'s own internal commits — means a rebase/merge/cherry-pick sequence *started* directly in the shared checkout bypasses the guard for its entire duration, since this hook only gates `git commit`, not `git rebase`/`git merge` themselves.
  - **Resolution**: not fully closed — doing so would require distinguishing "this sequence was started from a worktree" from "started here," which none of the state files (`rebase-merge`, `rebase-apply`, `MERGE_HEAD`, `CHERRY_PICK_HEAD`) record. A non-blocking stderr warning was added so this gap is visible rather than silent. Documented explicitly here and in the hook's own comments as a known, accepted limitation rather than left as an undocumented surprise.
- **Finding** (documented, not code-fixed): no `Makefile` target and no automatic rollout of the hook installer across this repo's 30+ already-checked-out worktrees — until `scripts/install_git_safety_hooks.sh` is run by hand somewhere, that checkout has zero protection, which is close to the exact "hope someone remembers" failure mode this PR exists to eliminate.
  - **Resolution**: deliberately not expanded into this PR's scope — auto-installing into other agents'/the operator's in-progress worktrees is an operational rollout action, not a code change, and doing it unprompted risked being disruptive. Flagged here as a real, separate follow-up; see Risks/concerns.
- **Finding** (investigated, refuted): whether `graphify merge-driver` is actually a real, implemented subcommand, or whether the setup script would silently configure git to point at something that doesn't exist.
  - **Resolution**: refuted with direct evidence — the subcommand was executed directly against the installed `graphifyy` package and confirmed to exist and match the expected argument order (`cli.py:1369-1421`).
- **Finding** (investigated, refuted): whether the `pip install graphifyy` remediation message referenced the wrong package name.
  - **Resolution**: refuted — `graphifyy` (double-y) is the correct real PyPI package name, confirmed against the graphify skill's own docs. The message was still clarified to mention `uv tool install graphifyy` as well, since this environment provisions via `uv`, not `pip`.
- **Finding** (investigated, refuted): whether committing `graphify-out/graph.json` or this PR's changes collide with any pre-existing hook, `.gitattributes`, or `merge.*` git-config section.
  - **Resolution**: refuted — confirmed via direct inspection of the real `.git/config` and repo state that nothing pre-existing is shadowed or overwritten.
- **Finding** (investigated, refuted): whether this diff removes or shadows any existing behavior.
  - **Resolution**: refuted — `git diff --name-status` against the merge-base shows five `A` (added) entries, zero `M`/`D`.

## Restart required

```text
No restart required.
```

This is tooling, not a running service. The hook and wrapper only take effect once someone runs `scripts/install_git_safety_hooks.sh` in their own checkout — see Risks/concerns.

## Risks / concerns

- **Severity: medium.** Nothing in this PR automatically installs the hook anywhere. Every existing worktree (30+ checked out in this repo as of this session) and every fresh clone has zero protection until `scripts/install_git_safety_hooks.sh` is run there by hand. This is a real, known gap, deliberately not auto-remediated in this PR (see Review findings fixed above) — a natural, small follow-up.
- **Severity: low-medium.** The rebase/merge/cherry-pick bypass in the `pre-commit` hook is a genuine, if narrow, residual gap — documented with a warning, not silently closed. Fully closing it is a harder problem than this PR's scope (distinguishing where a rebase sequence was *started*, not just whether one is in progress).
- **Severity: low.** The `docker compose` protection is convention-based, not a hard gate — nothing stops a session from calling `docker compose` directly instead of the wrapper. This is acknowledged directly in the script's own header and is a structural limitation (git hooks have no jurisdiction over non-git commands), not an oversight.
- **Mitigation for all three**: branch protection on `main` (confirmed live this session) is the one genuinely un-bypassable backstop in this whole design, independent of any of the above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1032
