# graph.json outgrew GitHub's 100MB blob cap — moved it to git-LFS, forward-only

## Summary

- `graphify-out/graph.json` (the knowledge-graph artifact) hit GitHub's hard
  100MB-per-blob push limit on a routine refresh (2026-09-08) and got
  rejected outright. Moved it to git-LFS starting from this branch —
  existing history is **not** rewritten (142 live worktrees/branches made
  that too risky).
- Because git hands a merge driver the *raw* LFS pointer stub text, not the
  real file, the existing union-merge driver (`graphify merge-driver`, wired
  up by `scripts/setup_graphify_merge_driver.sh`) would have started silently
  union-merging three ~130-byte pointer stubs instead of real graph JSON.
  Added `scripts/graphify_lfs_merge_driver.sh`, a wrapper that resolves real
  content via `git lfs smudge` before calling graphify's real merge, then
  re-cleans the result back into a pointer before handing it to git (a
  second, non-obvious step — merge drivers bypass git's normal clean-filter
  pipeline entirely, confirmed by direct repro).
- `scripts/check_graph_node_loss.py` (the pre-commit node-loss gate) reads
  `graph.json` via `git show`, which would also return a raw pointer stub
  post-migration. Fixed to detect and smudge it before comparing node counts.
- Both new subprocess-to-`git lfs` call sites got a timeout and an explicit
  `GIT_LFS_SKIP_SMUDGE` override, added after code review flagged a real hang
  risk (an unreachable LFS remote would otherwise block indefinitely).
- **Open risk, accepted by Juniper, not solved here:** this is a public repo
  on GitHub's free LFS tier (1GB storage + 1GB/month bandwidth, account-wide).
  See "Risks / concerns" below — this is why the status is
  `DONE_WITH_CONCERNS`, not `DONE`.

## Outcome moved

Before: a routine graph refresh silently could not be pushed once
`graph.json` crossed 100MB — GitHub rejects the push outright, no partial
state, no warning ahead of time. This already happened once (2026-09-08,
100.54MB).

After: `graph.json` is a ~130-byte pointer in the git history/pack; the real
content lives in LFS storage. **Proven, not just argued**: this branch's own
push uploaded the ~105MB real object through LFS transfer (`Uploading LFS
objects: 100% (1/1), 105 MB | 18 MB/s, done.`) and succeeded — the exact
scenario that would have been hard-rejected as a plain git blob before this
patch.

## Current architecture

- `graphify-out/graph.json`: committed as a plain ~100MB JSON blob on every
  refresh. `.gitattributes` mapped it to a custom union-merge driver
  (`merge=graphify`) via `scripts/setup_graphify_merge_driver.sh`, which
  configured `merge.graphify.driver = 'graphify merge-driver %O %A %B'`
  directly.
- `scripts/check_graph_node_loss.py`: the pre-commit gate compares HEAD's and
  the staged version's node counts via `git show <ref>:<path>` — a raw git
  blob read.
- No CI workflow reads `graph.json`'s content at all (confirmed by grep
  across `.github/workflows/*.yml` — zero references).

## Architecture touched

- `.gitattributes`: `graphify-out/graph.json` gains
  `filter=lfs diff=lfs -text` alongside the existing `merge=graphify`, plus
  an inline comment documenting the migration and the accepted LFS-quota
  risk.
- New script: `scripts/graphify_lfs_merge_driver.sh` — the actual
  `merge.graphify.driver` command now, instead of calling
  `graphify merge-driver` directly.
- `scripts/setup_graphify_merge_driver.sh`: points the driver at the new
  wrapper; also now checks the wrapper exists/is executable and that
  `git-lfs` itself is on PATH before configuring anything.
- `scripts/check_graph_node_loss.py`: detects and smudges LFS pointer-stub
  content (from both `HEAD:` and the staged blob) before parsing/comparing
  node counts.
- `CLAUDE.md`'s graphify section: one new bullet documenting the migration,
  the `git lfs install` requirement for a fresh clone/worktree, and the open
  LFS-quota risk.
- `tests/scripts/test_check_graph_node_loss.py`: new unit tests for pointer
  detection, plus two end-to-end tests using real throwaway git+git-lfs
  repos (successful smudge path, and an unresolvable-pointer fail-closed
  path).
- No runtime service touched. No bus/schema/env changes.

## Files changed

- `.gitattributes`: adds `filter=lfs diff=lfs -text` for `graphify-out/graph.json`; documents the migration and the accepted LFS free-tier risk inline.
- `scripts/graphify_lfs_merge_driver.sh` (new): LFS-aware wrapper around `graphify merge-driver` — smudges `%O`/`%A`/`%B` before the real merge, re-cleans the result before writing it back, timeouts + `GIT_LFS_SKIP_SMUDGE` guards on both `git lfs` calls, preserves real (un-cleaned) content on a merge-driver failure so a human can resolve it by hand.
- `scripts/setup_graphify_merge_driver.sh`: points `merge.graphify.driver` at the new wrapper (absolute path); checks the wrapper and `git-lfs` itself exist before configuring anything; updates the fallback `.gitattributes` line it appends if missing.
- `scripts/check_graph_node_loss.py`: detects an LFS pointer stub in `git show` output and smudges it (with timeout + `GIT_LFS_SKIP_SMUDGE` guard) before parsing, for both the HEAD and staged comparison sides.
- `tests/scripts/test_check_graph_node_loss.py`: unit tests for `_is_lfs_pointer`; an end-to-end test proving the gate reads real node counts through real LFS pointers (not raw stubs); an end-to-end test proving an unresolvable pointer fails closed (exit 2) instead of hanging or crashing.
- `CLAUDE.md`: one new bullet in the graphify section documenting the LFS migration, the `git lfs install` requirement, and the open quota risk.
- `graphify-out/graph.json`: migrated to LFS (`git add --renormalize`) — committed blob went from ~100MB to a 134-byte pointer; working-tree content unchanged (verified: same 75,802 nodes/164,544 links/104 hyperedges before and after).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `graphify-out/graph.json` is now git-LFS-tracked going forward. Merging a branch that touches it now runs through `scripts/graphify_lfs_merge_driver.sh` instead of `graphify merge-driver` directly.
- Compatibility notes: existing history is untouched — old commits still carry `graph.json` as a plain blob. Any clone/worktree that never runs `git lfs install` will check out a ~130-byte pointer stub instead of real content for any commit made from this branch onward (documented in `.gitattributes` and `CLAUDE.md`).

## Env/config changes

- Added keys: none (this is a git-config/git-attributes change, not an app env var). Two new **optional** override env vars for the merge driver's and the node-loss gate's LFS-call timeouts: `GRAPHIFY_LFS_MERGE_TIMEOUT` (default 60s) and `GRAPHIFY_LFS_SMUDGE_TIMEOUT` (default 60s). Neither needs `.env`/`.env_example` entries — they're git-hook/merge-driver-scoped, not service config.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable — no service `.env_example` touched.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable, no `.env_example` changed.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  tests/scripts/test_check_graph_node_loss.py \
  tests/scripts/test_safe_graphify_update.py \
  tests/scripts/test_graphify_hook_guard_gate.py \
  tests/scripts/test_bare_graphify_update_guard.py \
  tests/scripts/test_check_graph_worktree_integrity.py -q
-> 87 passed (22 in test_check_graph_node_loss.py, including 3 new LFS-specific
   tests: pointer detection, successful-smudge end-to-end, and
   unresolvable-pointer fail-closed end-to-end)

python3 scripts/check_graph_node_loss.py --json
-> live run against the real repo: {"before": {"nodes": 75802, ...},
   "after": {"nodes": 75802, ...}, "node_loss_pct": 0.0, "blocked": false}
   (this is the fixed gate correctly smudging the real committed pointer)

python3 scripts/check_graphify_scan_scope.py
-> "graphify sees 4517 code files; git tracks 6248; floor 3124" (OK, no
   change needed here -- confirmed it never touches graph.json)

sh -n scripts/graphify_lfs_merge_driver.sh && sh -n scripts/setup_graphify_merge_driver.sh
-> both syntax-clean

git diff --check
-> clean, no whitespace errors
```

## Evals run

No dedicated eval harness exists for `scripts/` git-tooling changes (this
isn't a service with an `evals/` directory). The merge-driver end-to-end
verification below is the functional equivalent of an eval for this patch:
real git repos, real LFS objects, real merge results inspected byte-for-byte.

## Docker/build/smoke checks

Not applicable — no Docker service touched, no runtime container behavior
changed. This is a git/CI-tooling-layer change only.

## Merge-driver verification (real evidence, not just exit codes)

All three scenarios below were run against real throwaway git+git-lfs repos
in `/tmp` scratch space (not the Orion-Sapienform history), using the exact
`scripts/graphify_lfs_merge_driver.sh` this PR ships (referenced by absolute
path from the scratch repo's git config, so it's the identical file).

**1. Success path — two branches, each adding a distinct node:**

```text
$ git merge branch-b --no-edit
Auto-merging graph.json
Merge made by the 'ort' strategy.
 graph.json | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
EXIT: 0

$ git ls-files -s graph.json
100644 027e69fdd9b23d7b4d9d8a0ea7ec88a878103710 0 graph.json
$ git cat-file -s 027e69fdd9b23d7b4d9d8a0ea7ec88a878103710
128        <- committed blob is a pointer, not the real ~350-byte JSON

$ cat graph.json   # actual working-tree file
{
  "directed": false, "multigraph": false, "graph": {},
  "nodes": [
    {"id": "BASE_1"}, {"id": "BASE_2"},
    {"id": "NODE_A"},   <- from branch A
    {"id": "NODE_B"}    <- from branch B
  ],
  "links": [{"source": "BASE_1", "target": "BASE_2"}]
}
```

Both nodes present, zero conflict markers, real content on disk, a proper
128-byte LFS pointer in the git object store. `git lfs status` reported
clean with no "should have been a pointer, but wasn't" warning (that warning
*was* present before the re-clean fix landed — see commit `71551dbfb`, the
mid-task bug this review process itself caught).

**2. Failure path — a merge `graphify merge-driver` can't auto-resolve
(corrupt JSON on one side):**

```text
$ git merge branch-c --no-edit
[graphify merge-driver] error loading graphs: Expecting value: line 1 column 1 (char 0)
Auto-merging graph.json
CONFLICT (content): Merge conflict in graph.json
Automatic merge failed; fix conflicts and then commit the result.
EXIT: 1

$ cat graph.json   # what a human resolving this by hand actually sees
{
  "directed": false, "multigraph": false, "graph": {},
  "nodes": [{"id": "BASE_1"}, {"id": "BASE_2"}, {"id": "NODE_A"}, {"id": "NODE_B"}],
  ...
}
```

Real, readable JSON is left at the conflict path (branch-a's real content,
since graphify's own error path exits before writing anything) — not a
stale pre-merge pointer stub, and not an unreadable re-cleaned pointer. This
is the should-fix finding from code review, fixed and verified live.

**3. Proof the migration actually fixes the original 100MB rejection:**
This branch's own `git push` uploaded the real, current 104,841,865-byte
`graphify-out/graph.json` and succeeded:

```text
Uploading LFS objects: 100% (1/1), 105 MB | 18 MB/s, done.
To github.com:junebug-junie/Orion-Sapienform.git
 * [new branch]          fix/graph-json-lfs -> fix/graph-json-lfs
```

That 105MB object went through LFS transfer, not the git pack — the exact
mechanism that avoids GitHub's hard 100MB-per-blob cap on regular pushes.

## Review findings fixed

- Finding: **must-fix** — `git lfs smudge`/`git lfs clean` calls in both `scripts/graphify_lfs_merge_driver.sh` and `scripts/check_graph_node_loss.py`'s `_lfs_smudge` had no timeout; an unreachable LFS remote or credential prompt would hang a merge (blocking) or every pre-commit hook run, indefinitely.
  - Fix: wrapped both call sites in a bounded timeout (`timeout` in the shell wrapper, `subprocess.run(..., timeout=...)` in Python), each overridable via `GRAPHIFY_LFS_MERGE_TIMEOUT` / `GRAPHIFY_LFS_SMUDGE_TIMEOUT` (default 60s).
  - Evidence: `sh -n` syntax-clean; full test suite still green (87 passed); commit `27cde6b49`.
- Finding: **should-fix** — on a merge-driver failure (real conflict), the wrapper left the pre-merge pointer stub at `%A` instead of the real content a human needs to resolve the conflict by hand.
  - Fix: on failure, copy back the real (un-cleaned) content `graphify merge-driver` produced instead of leaving `%A` untouched.
  - Evidence: live repro above (scenario 2) — real readable JSON at the conflict path, not a pointer.
- Finding: **should-fix** — neither `git lfs` call site explicitly unset `GIT_LFS_SKIP_SMUDGE`; if set in the invoking environment, smudge silently passes pointer text through unchanged instead of fetching real content.
  - Fix: both call sites now run with `GIT_LFS_SKIP_SMUDGE` explicitly unset.
  - Evidence: commit `27cde6b49`.
- Finding: **should-fix** — no test proved a smudge *failure* actually fails closed (exit 2, clear message) instead of hanging or crashing.
  - Fix: added `test_git_mode_fails_closed_on_unresolvable_lfs_pointer` — a real git+LFS repo with a syntactically valid but unresolvable pointer (no remote, unfetchable oid), asserting exit code 2 and non-empty stderr.
  - Evidence: `pytest tests/scripts/test_check_graph_node_loss.py -q` → 22 passed.
- Finding: **nice-to-have** — `setup_graphify_merge_driver.sh` never checked `git-lfs` itself was on PATH before configuring a driver that depends on it.
  - Fix: added a PATH check with a clear install pointer, run before any config is written.
  - Evidence: commit `27cde6b49`.
- Finding: **nice-to-have** — the wrapper's "POSIX sh only -- no bashisms" comment was misleading (`head -c`, `mktemp -d` aren't strict POSIX).
  - Fix: reworded to name the actual non-POSIX-but-near-universal utilities relied on, matching sibling scripts' existing convention in this repo.
  - Evidence: commit `27cde6b49`.
- Finding: **nice-to-have** — unused `monkeypatch` fixture parameter in the new e2e test.
  - Fix: removed.
  - Evidence: commit `27cde6b49`.
- Not fixed, accepted as documented follow-ups (all low severity, out of this patch's practical scope): the `.gitattributes` fallback-append in `setup_graphify_merge_driver.sh` could leave a duplicate attribute line on a machine that ran the pre-LFS version of the script before pulling this branch (git resolves it correctly via last-line-wins, just untidy); an absolute repo path containing a space would break the driver's git-config invocation (git's own driver command-line splitting has no quoting mechanism for this at all — not fixable from this script); other disk-reading consumers of `graph.json` (the bare `graphify` CLI, `scripts/safe_graphify_update.sh`, `scripts/check_graphify_scan_scope.py`) get a generic JSON-parse crash rather than an LFS-specific error message on a machine that never ran `git lfs install` — these are working-tree reads (LFS smudges transparently on checkout, so this only bites an unconfigured machine), and `CLAUDE.md`'s new note already tells operators to run `git lfs install`.

## Restart required

```text
No restart required.
```

This is a git/CI-tooling-layer change. No service process, container, or
runtime config was touched.

## Risks / concerns

- Severity: **material, accepted by Juniper, not solved by this patch**
- Concern: This repo is public on GitHub's free git-LFS tier — 1GB storage
  + 1GB/month download bandwidth, **account-wide across every repo on the
  account**, not just this one. `graphify-out/graph.json` alone is
  ~100MB/version and changed 6 times on `main` in the last 30 days before
  this migration (plus more across branches pre-merge, plus every CI
  checkout that would pull real LFS content). That budget can be exhausted
  by storage alone (10 versions) or by roughly 10 pulls in a month. Once
  exhausted, further LFS push/pull is blocked account-wide until a paid data
  pack is purchased.
- Mitigation: **Explicitly not solved here** — Juniper made this call
  knowingly. Documented prominently in two places so it isn't lost: an
  inline comment in `.gitattributes` right above the tracked-file line, and
  a new bullet in `CLAUDE.md`'s graphify section. Both point at `git lfs env`
  (local view) and GitHub's repo → Settings → Billing → "Git LFS Data" page
  (the real account-wide meter) as where to check actual consumption before
  assuming headroom.
- Severity: low
- Concern: a fresh clone/worktree that never runs `git lfs install` will
  check out `graph.json` as a ~130-byte pointer stub for any commit from
  this branch onward, and every disk-reading consumer of the file (the
  `graphify` CLI itself, `scripts/safe_graphify_update.sh`,
  `scripts/check_graphify_scan_scope.py`) will hit a generic JSON-parse
  crash rather than an LFS-specific error pointing at the real cause.
- Mitigation: documented in `CLAUDE.md`'s graphify section and inline in
  `.gitattributes`. Confirmed this specific dev machine already has
  `git lfs install` run globally (`git config --global --get
  filter.lfs.smudge` returns a value), so this doesn't block work here —
  it's a new-machine/CI-runner concern only, and no CI workflow currently
  reads `graph.json`'s content (confirmed by grep, zero references across
  `.github/workflows/*.yml`).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/graph-json-lfs
(filled in below once opened via `gh pr create`)
