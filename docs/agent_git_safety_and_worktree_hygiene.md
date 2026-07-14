# Agent Git Safety and Worktree Hygiene

What this repo builds toward, why, and the mechanisms currently live. This is a reference doc, not a PR report — for the incident history, evidence, and review findings behind any piece of this, follow the links to `docs/superpowers/pr-reports/`.

## Philosophy

This repo runs many concurrent AI coding agent sessions against the same working tree simultaneously. That is not a hypothetical stress case — it has actually destroyed real work more than once: a concurrent session ran `docker compose build`+`up` from the shared checkout and silently reverted another session's already-verified fix; a concurrent session's presumed `git clean -fd` wiped tracked extras and a spec doc mid-session, discovered only via reflog, not direct observation.

The response to that is not "tell agents to be careful" repeated louder. Per this repo's own operating contract (`AGENTS.md`/`CLAUDE.md` §0A): *if you have to repeat a rule twice, turn it into a script, test, check, hook, or make target.* Every mechanism below exists because an instruction-only version of it was tried first, or would obviously have been insufficient, and didn't hold.

Two other principles shape the specific shape these mechanisms take:

- **Runtime truth beats config truth.** A hook existing is not proof it fires. A doc describing one convention is not proof only one convention is in use. Every mechanism here was checked against live behavior, not just read — and in more than one case, checking live behavior instead of trusting the code as written is exactly what surfaced a real bug before it shipped (see "What review caught," below).
- **Defense in depth, not one clever layer.** No single mechanism here is airtight, and each one's own header comment says so plainly. The Claude-Code-scoped hook doesn't protect a plain terminal. The vendor-neutral shim doesn't survive `sudo` or a cached absolute git path. Neither replaces the other; they're deliberately redundant, so a gap in one is caught by another, and the honest limitations of each are stated where they live, not glossed over.

## The mechanism stack

Four independent layers, in the order a destructive `git clean -f*` / `git reset --hard` / `git checkout|switch --force` attempt would actually meet them, from earliest to last-resort:

| Layer | Scope | Catches | Doesn't catch |
|---|---|---|---|
| `scripts/hooks/destructive_git_guard.py` | Claude Code sessions using this repo's `.claude/settings.json` | Any Claude Code Bash call matching the destructive shape | Any other tool, a plain terminal, a human typing the command |
| `scripts/git_hooks/orion-git-shim` | Any repo with the opt-in marker file, for any caller doing a normal PATH-resolved `git` lookup | Plain terminals, other AI tools, scripts — anything that resolves `git` via PATH | `sudo git ...` (secure_path excludes user-local bin dirs), a caller that cached an absolute git path once, `!`-prefixed shell aliases |
| `scripts/git_hooks/pre-commit` | Any repo with the hook installed | `git commit` specifically, from the shared/primary checkout | Anything that doesn't reach a commit (this is why the two above exist — clean/reset/checkout destroy files *before* a commit would happen) |
| GitHub branch protection on `main` | This repo, server-side | Direct pushes to `main`, force-pushes, branch deletion | Nothing local — this is the one layer that can't be talked around by any local tool or misbehaving agent |

All three git-level mechanisms share one detection primitive — comparing `git rev-parse --git-dir` against `--git-common-dir` to tell the shared/primary checkout apart from a linked worktree — reimplemented independently in each (shell twice, Python once, POSIX sh once again in the shim) because two of them are hook files git itself copies verbatim with no import path back into this repo, and the third is a standalone installed binary with the same constraint. Each implementation's own comments cross-reference the others: fixing a bug in the detection logic itself means checking all four.

Escape hatch, identical across every layer: `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1`, set consciously per command, never as a habit.

## Installing the mechanisms

```bash
scripts/install_git_safety_hooks.sh .      # pre-commit + post-merge hooks (this repo, any worktree)
scripts/install_orion_git_shim.sh .        # system-wide git shim + opt-in marker for this repo
```

Both are idempotent — safe to re-run, refreshes to the current version, doesn't duplicate anything. Running only one leaves the other class of command (`git commit` vs. `clean`/`reset --hard`/`checkout --force`) unguarded by that layer specifically — `destructive_git_guard.py` still covers Claude Code sessions either way, since it's wired through `.claude/settings.json`, not either installer.

The shim's opt-in is a marker file (`.git/orion-safety-guard-enabled` in the shared checkout), not a blanket policy — installing the shim on a machine does not mean every git repo on that machine is now guarded, only ones the installer has explicitly touched.

## Worktree hygiene

A live audit (not an assumption) found this repo actually has **three simultaneously-used worktree location conventions**, not the one originally documented:

- `../Orion-Sapienform-<name>` (sibling directory) — the documented convention. `scripts/new_worktree.sh <type> <name>` creates one, and warns if a worktree mentioning the same name already exists under either convention below.
- `.worktrees/<name>` (nested inside the main checkout, gitignored) — driven by other tooling in this environment.
- `.claude/worktrees/agent-<id>` — Claude Code's own `isolation: "worktree"` Agent-tool feature.

Regardless of which convention created a given worktree, `git worktree list` sees all of them, and so does this repo's cleanup tooling, since it reads from that same source of truth rather than assuming a path shape:

```bash
make worktree-status              # full reconciled table: path, branch, merged?, open PR?, disk size
make worktree-status-summary      # one-line counts, no PR/disk lookups (this is what fires on every SessionStart and post-merge)
make worktree-status-stale        # merged worktrees with no open PR -- the actual prune candidates
make prune-merged-worktrees        # dry-run by default; YES=1 to actually remove
```

`prune-merged-worktrees` never passes `--force` to `git worktree remove` (a worktree with real uncommitted changes is skipped, reported, not destroyed) and never touches the branch itself, only the worktree directory.

The first real run of this reclaimed ~7.4GB and removed 202 of 276 worktrees on this machine — the scale that made building this worth it in the first place, not a hypothetical.

## What review caught

Worth naming explicitly, since it's the difference between "the mechanism exists" and "the mechanism actually works": every piece here went through multi-angle review before merging, and in two cases the first draft had real, live-reproducible bugs that would have quietly defeated the mechanism's own purpose.

The clearest example: the git shim's first draft passed its own hand-written tests and still had four real bypasses, found and *empirically reproduced* against the live script, not just reasoned about — a git alias (`git config alias.hr "reset --hard"; git hr`) totally skipped detection; `git --git-dir <path> reset --hard` (ordinary syntax, not a trick) had the path argument misidentified as the subcommand, silently passing the real destructive command straight through; two independently-installed shim copies on PATH could recognize each other as "the real git" and exec into one another forever, hanging every git invocation on the machine. All four are fixed, each with a regression test that reproduces the exact failing invocation from before the fix — see `docs/superpowers/pr-reports/2026-07-14-vendor-neutral-git-shim-pr.md` for the full account.

The general lesson, stated once here rather than repeated at every mechanism: a mechanism that looks correct and hasn't been tried against real, adversarial-shaped input is a claim, not a fact. Treat it as unverified until it's been pushed on.

## Further reading

- `docs/superpowers/pr-reports/2026-07-14-agent-git-safety-hooks-pr.md` — the original `pre-commit` hook and the incident that prompted it.
- `docs/superpowers/pr-reports/2026-07-14-destructive-git-guard-pr.md` — the Claude Code PreToolUse hook.
- `docs/superpowers/pr-reports/2026-07-14-worktree-hygiene-tooling-pr.md` — the three-convention audit and the status/prune tooling.
- `docs/superpowers/pr-reports/2026-07-14-vendor-neutral-git-shim-pr.md` — the system-wide shim and the four bypasses found in review.
- `docs/superpowers/pr-reports/2026-07-14-orion-repo-agent-pr.md` — the custom subagent definition carrying this context into dispatched subagents automatically.
- `CLAUDE.md` §2 ("Clean git and worktree rules") and §13 ("Safety rules") — the operating contract these mechanisms implement.
