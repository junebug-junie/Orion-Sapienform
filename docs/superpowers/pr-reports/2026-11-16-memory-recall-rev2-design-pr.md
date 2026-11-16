## Summary

- Adds a "Recall rev 2 (Juniper, 2026-11-16)" section to the core memory pages spec. Juniper approved it on 2026-11-16.
- Records why chat recall misses stored memories: no chat reader for `episode_memory`, the vector backend is off, and the read-only role has no grants on the memory tables.
- Records the new recall design: tiered memory, hybrid retrieval with a local reranker, a pre-turn query rewrite, link-following over `episode_memory_referent`, and corrections that supersede at write time. Each part names the research it follows.
- Changes phase 2: pages are born from read demand, `memory_page_mention` is dropped, and `orion.dig know` becomes a recall client.
- Adds a LongMemEval-style scoreboard with pass bars, and the build order with issues.
- Marks the replaced parts of the old text as superseded. The old text stays.

## Outcome moved

The spec now matches the approved design. Builders of #3533-#3537 have one written source for the plan and the pass bars.

## Current architecture

The spec planned pages from a mention ledger and facts from Graphiti. It said retrieval would not be rebuilt.

## Architecture touched

Docs only. No code, config, schema, or bus change.

## Files changed

- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: new Recall rev 2 section, superseded markers.
- `docs/superpowers/pr-reports/2026-11-16-memory-recall-rev2-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (design only; `memory_page_mention` is dropped from the plan before it was built).
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
git diff --check  -> clean
CI on this PR     -> see checks
```

## Evals run

```text
None. Docs only. The scoreboard is #3536.
```

## Docker/build/smoke checks

```text
None. Docs only.
```

## Review findings fixed

- Finding: none. Self-review for privacy (public repo): no family names, health details, or chat text added.
  - Fix: n/a
  - Evidence: diff of the spec.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: #3356 still states the wrong cause (`memory_tier`). The spec now says the missing piece is a reader.
- Mitigation: a comment on #3356 points to this spec.
- Issues opened: none. Issues closed: none. Related: #3533, #3534, #3535, #3536, #3537, #3409, #3463, #3462, #3356.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
