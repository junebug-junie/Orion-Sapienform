## Summary

- New `Stop` hook (`scripts/hooks/stop_worktree_wip_snapshot.py`) snapshots every dirty worktree — including the shared/primary checkout — into a hidden `refs/orion-wip/<name>-<pathhash>/latest` git ref via `write-tree`/`commit-tree` plumbing.
- Requested directly, after a real near-miss (2026-08-18): ~1,000 lines of real, uncommitted work sat in a worktree whose branch was already merged, right before a main pull/deploy.
- The first design considered (periodic auto-commit) was rejected on real risk ("the risk is too big" — mid-edit commits, WIP branch-history noise). This sidesteps both structurally: never touches the branch (ref, not `refs/heads/`), never touches the real index/working tree (scratch `GIT_INDEX_FILE`), and can only ever fire at a genuine turn boundary (Stop), never mid-edit.
- Throttled to one scan per 5 minutes across all sessions; the hook itself hands the real scan to a detached background process and returns in <100ms regardless of fleet size (a synchronous scan across this repo's real ~258-worktree fleet measured ~20s).

## Outcome moved

Uncommitted worktree work — including in the shared/primary checkout — is no longer purely dependent on someone remembering to commit it. A real safety net now exists, verified live.

## Real bugs found and fixed (2 rounds)

**While building (before first commit):**
- A scratch `GIT_INDEX_FILE` pointed at a pre-created (0-byte) temp file reads as a *corrupt* index in git ("index file smaller than expected"), not a fresh one — fixed by reserving a path without creating the file.
- An early doc draft claimed `git stash apply <hash>` recovers a snapshot — verified live it doesn't (this snapshot's commit shape is plain single-parent, not stash-shaped, fails with "not a stash-like commit"). `git checkout <hash> -- .` is the form that works, confirmed live.

**Code review (3 finder agents), 5 findings, all fixed:**
1. Check-then-act throttle race (two concurrent Stop events could each launch a worker) — fixed with a non-blocking `fcntl.flock`, reusing this repo's own established `_StateLock` pattern (`scripts/bus_core_health_watchdog.py`).
2. `is_main` filtering silently excluded the shared/primary checkout (the highest-risk spot per this repo's own CLAUDE.md) while the docs claimed "every worktree" — fixed; verified live with a real snapshot of the main checkout.
3. Marker touched even on `Popen` failure, blocking the next real scan attempt for 5 minutes with zero signal — fixed to only touch on successful launch.
4. `git add -A` captures whatever a real commit would, automatically, with no review chance — disclosed prominently (not silently fixable), plus a real mitigation: added time-based pruning (48h) alongside count-based.
5. Ref-name collision for branches differing only by space-vs-underscore — fixed with a path-hash suffix that can't collide by definition.

## Files changed

- `scripts/hooks/stop_worktree_wip_snapshot.py` (new): the hook.
- `tests/scripts/test_stop_worktree_wip_snapshot.py` (new): 24 real-git-repo tests.
- `.claude/settings.json`: registered as a `Stop` hook (5s timeout — the hook itself returns near-instantly).
- `CLAUDE.md`: discoverability + verified recovery instructions.

## Tests run

```
tests/scripts/test_stop_worktree_wip_snapshot.py: 24/24 pass (real git repos, not mocked)
```

## Docker/build/smoke checks

Live end-to-end smoke, this repo's real ~258-worktree fleet: hook returns in ~0.08s, background worker completes in ~20s, real snapshots created and verified (content-correct, `.gitignore`-respecting, real working tree/index untouched, main checkout included).

## Review findings fixed

See "Real bugs found and fixed" above and the PR's own review-findings comment for the full evidence trail per finding.

## Recovery

```bash
git -C <worktree> show refs/orion-wip/<branch>-<hash>/latest:<path>      # one file
git -C <worktree> checkout refs/orion-wip/<branch>-<hash>/latest -- .    # whole snapshot
```

## Restart required

No restart required — a hook, not a running service.

## Risks / concerns

- Severity: low
- Concern: `git add -A` automatically captures any unignored sensitive content sitting in a dirty worktree, with no review chance before it lands in a git object.
- Mitigation: disclosed prominently in the module docstring; time-based (48h) + count-based (3) pruning bounds how long a ref points at it. Normal secret hygiene (`.gitignore` real secrets, never leave credentials in an untracked scratch file) remains exactly as necessary as it already was.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1720
