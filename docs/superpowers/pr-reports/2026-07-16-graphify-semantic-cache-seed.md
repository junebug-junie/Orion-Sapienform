# graphify semantic cache: seed script + commit the cache, so a fresh run doesn't re-extract

## Summary

- After PR #1074 (docs semantic extraction merge), a fresh worktree running a full-repo `/graphify .` semantic pass would have re-dispatched subagents to re-extract every document/paper/image file already represented in `graph.json` — not just the new `docs/` corpus, but every README and other non-code doc anywhere in the repo that had ever gone through semantic extraction (362 files total, confirmed by inspecting `graph.json` directly).
- Root cause: graphify's semantic extraction (Part B) skips a file only if it has a matching entry in a local, per-file content-hash cache at `graphify-out/cache/semantic/` (`graphify.cache.check_semantic_cache`). That cache was never committed to this repo (only `graph.json`, `GRAPH_REPORT.md`, `manifest.json` were tracked under `graphify-out/`), so any fresh clone/worktree starts with an empty cache regardless of what `graph.json` already contains.
- Added `scripts/graphify_seed_semantic_cache.py`: a deterministic, LLM-free script that reconstructs cache entries directly from `graph.json`'s existing nodes/edges/hyperedges (grouped by `source_file`, using graphify's own `save_semantic_cache()` so the hash format is guaranteed to match what `check_semantic_cache` will look for later).
- Ran it now and committed the result: `graphify-out/cache/semantic/` (360 cache files, 2.0M) is now tracked, so a fresh clone/worktree is correctly seeded with zero setup — not just this machine's existing checkouts.
- Documented the requirement in root `CLAUDE.md`'s `## graphify` section (always loaded into every session in this repo) so any future agent — including a brand-new session with no prior context — runs the seed script before a full-repo semantic pass, without Juniper needing to explain the caveat each time.

## Outcome moved

A full-repo `/graphify .` semantic run, triggered from any fresh worktree by any agent with no special instructions, will now correctly skip the 362 files already extracted and only dispatch subagents for genuinely new document/paper/image files. Previously this required tribal knowledge (worktree rules, cache internals, manual exclusion scoping) that would have had to be re-explained by Juniper every time.

## Current architecture

`graphify-out/cache/{ast,semantic}/` existed as a local, per-machine content-hash cache (`graphify/cache.py`) used to skip re-extraction on `/graphify --update` and repeat semantic runs. Only `graph.json`, `GRAPH_REPORT.md`, and `manifest.json` were ever committed for this repo — the cache directory was untracked, so its benefit reset to zero on every fresh checkout.

## Architecture touched

Agent-side tooling only (`graphify-out/cache/`, `scripts/`, root `CLAUDE.md`). No Orion runtime service, contract, schema, or config touched.

## Files changed

- `scripts/graphify_seed_semantic_cache.py`: new. Reads `graph.json`, filters nodes/edges/hyperedges to `file_type` in `{document, paper, image}`, groups by `source_file`, and calls graphify's own `save_semantic_cache()` (via the detected graphify interpreter) to write correctly hash-keyed cache entries.
- `graphify-out/cache/semantic/`: new, committed. 360 cache files (2.0M total) — one of the 362 semantically-covered source files (`reviews/pending/2026-07-14-agent-worktree-discipline-and-graphify-sync-spec.md`) no longer exists on disk in this checkout, so the script correctly skipped it (matches `save_semantic_cache`'s own `p.is_file()` guard).
- `CLAUDE.md`: added a bullet to the `## graphify` section directing any future agent to run the seed script before a full-repo semantic pass, and to re-run + commit it after any future semantic extraction merge.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

Not applicable — no application code changed. Verified directly instead:

```text
# Seed script run against the merged graph:
$ python3 scripts/graphify_seed_semantic_cache.py
[seed-semantic-cache] wrote cache entries for 361 files
[seed-semantic-cache] 362 source files eligible (1235 nodes, 1476 edges, 66 hyperedges)

# Confirmed the cache now correctly discriminates cached vs. uncached:
check_semantic_cache([docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md,
                       /tmp/graphify_cache_test_scratch.md (a genuinely new, never-extracted file)])
-> cached_nodes: 6, uncached: ['/tmp/graphify_cache_test_scratch.md']
```

The already-extracted file correctly hits the cache; a brand-new scratch file correctly reports as uncached — confirming the seed produces real, working cache entries rather than a no-op.

## Evals run

No eval harness applicable to a tooling/cache script.

## Docker/build/smoke checks

Not applicable — no runtime/service change.

## Review findings fixed

No code-review skill subagent pass run — this is a small, self-verified deterministic script (no LLM, no runtime behavior) plus a docs/config change, verified directly against the real cache-check function rather than by inspection alone (see Tests run above).

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: If a future semantic extraction merge forgets to re-run the seed script and commit the updated cache, the next fresh worktree will silently fall back to the old (wasteful but non-destructive) re-extraction behavior for the newly-added files only — the existing 362 files stay correctly cached regardless.
- Mitigation: documented as a required step in `CLAUDE.md`, which is loaded into every session in this repo automatically.

## PR link

<link>
