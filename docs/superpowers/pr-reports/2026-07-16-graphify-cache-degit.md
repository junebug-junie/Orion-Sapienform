# Revert: stop committing graphify's local extraction cache

## Summary

- PR #1076 committed `graphify-out/cache/semantic/` (360 hash-named JSON blobs, 2.0M) to main to solve a real problem (a fresh worktree has no way to skip re-extracting already-covered files) — but this directly violates `CLAUDE.md` section 2's explicit rule: "Never commit: ... local cache files." Flagged by Juniper.
- The actual fix (`scripts/graphify_seed_semantic_cache.py`, also from #1076) is sufficient on its own: it reconstructs the cache from `graph.json` deterministically, with no LLM call, in well under a second. There was never a real need to also commit the cache blobs — the script is cheap enough to just run every time.
- `git rm -r --cached graphify-out/cache/`, added `graphify-out/cache/` to `.gitignore`, and corrected the `CLAUDE.md` bullet that had claimed committing the cache was intentional.

## Outcome moved

Main no longer carries 360 opaque, content-hash-named cache blobs. The seed script remains the single source of truth for "skip already-extracted files" — cheap enough to run unconditionally, so there's no actual loss of the original PR's benefit.

## Current architecture

`graphify-out/cache/{ast,semantic}/` is graphify's own local, per-file, content-hash-keyed extraction cache (`graphify/cache.py`). It is derived state, fully reconstructable from `graph.json` at any time.

## Architecture touched

`.gitignore`, `CLAUDE.md`, git history of `graphify-out/cache/`. No runtime code.

## Files changed

- `.gitignore`: added `graphify-out/cache/`.
- `graphify-out/cache/semantic/*.json` (360 files): removed from git tracking via `git rm --cached` (left in place on disk locally — this worktree's cache still works, it's just no longer committed).
- `CLAUDE.md`: corrected the `## graphify` bullet added in #1076 — it previously said "the cache is checked into git specifically so this works with zero setup"; now says the opposite explicitly, with the mistake named so it doesn't get repeated.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

Not applicable — no application code. Verified the revert is correct:

```text
$ ls graphify-out/cache/semantic/ | wc -l
360   # unchanged on disk — git rm --cached does not touch the working tree
$ git check-ignore -v graphify-out/cache/semantic/<any-file>.json
.gitignore:151:graphify-out/cache/   <path>   # now correctly ignored
```

## Evals run

Not applicable.

## Docker/build/smoke checks

Not applicable.

## Review findings fixed

- Finding: PR #1076 committed a local cache directory in direct violation of `CLAUDE.md` section 2 ("Never commit: ... local cache files"), caught by Juniper reviewing main.
  - Fix: `git rm --cached` the directory, gitignore it, correct the `CLAUDE.md` claim that had justified committing it.
  - Evidence: `git ls-tree origin/main graphify-out/` (pre-fix) showed `cache/` as a tracked tree; post-fix it is absent from the index and matched by `.gitignore`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: every fresh worktree must now run `scripts/graphify_seed_semantic_cache.py` once before a semantic pass, rather than getting a pre-populated cache for free from git.
- Mitigation: the script costs nothing (no LLM, sub-second, pure local computation from `graph.json`) and `CLAUDE.md` documents it as a required, unconditional pre-step — the convenience lost by not committing the cache is negligible next to the git-hygiene cost of committing it.

## PR link

<link>
