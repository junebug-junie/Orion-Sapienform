## Summary

- Adds a "Recall rev 2 (Juniper, 2026-11-16)" section to the core memory pages spec. Juniper approved it on 2026-11-16.
- Records why chat recall misses stored memories: no chat reader for `episode_memory`, the vector backend is off, and the read-only role has no grants on the memory tables.
- Records the new recall design: tiered memory, hybrid retrieval with a local reranker, a pre-turn query rewrite, link-following over `episode_memory_referent`, and corrections that supersede at write time. Each part names the research it follows.
- Changes phase 2: pages are born from read demand, `memory_page_mention` is dropped, and `orion.dig know` becomes a recall client.
- Adds a LongMemEval-style scoreboard with pass bars, and the build order with issues.
- Marks the replaced parts of the old text as superseded. The old text stays.
- After Orion's review of the amendment, the spec now also records:
  - the Graphiti trial numbers for both trials, and which checks failed;
  - corrections that keep a `supersedes_memory_id` pointer, a one-time supersede pass over existing rows (in #3533), and a deterministic test that a superseded row is never served or folded;
  - phase 1a (`page:juniper`, PR #3418, and anchor lines, PR #3478) marked DONE and live, with the fold's provenance;
  - a flattening guard (#3553), vetoes stored as negative facts, `matters` gate numbers and a later re-grade, a per-turn recall trace for counting false denials, and recall that degrades but never skips under heat shedding;
  - a recommendation to keep core pages and memory lines out of Claude consult briefs unless Juniper opts in (#3541), and the paraphrase gap in the mesh-edge check (#3554).

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

- Finding: Orion's review raised nine gaps (trial numbers; existing contradictions; no test for "a correction outranks"; phase 1a status; flattening; vetoes re-entering; `matters` "came up more than once" unknowable at distill time and no gate numbers; no trace for false denials and no heat behavior; consult briefs and paraphrases at the mesh edge).
  - Fix: each is now a subsection of Recall rev 2, with markers on the older text it changes. Deferred work is filed as #3553 and #3554.
  - Evidence: spec diff in this PR.
- Finding: privacy self-review (public repo).
  - Fix: no family names, health details, or chat text added. The trip and pet contradictions are described by kind only.
  - Evidence: spec diff in this PR.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: #3356 still states the wrong cause (`memory_tier`). The spec now says the missing piece is a reader.
- Mitigation: a comment on #3356 points to this spec.
- Severity: low
- Concern: phase 1a is marked live on the coordinator's report and the merged PRs. This PR did not query the live page rows from this host. UNVERIFIED here.
- Mitigation: check with `SELECT concept_id, max(version) FROM self_concept_history WHERE concept_id='page:juniper' GROUP BY 1`.
- Issues opened: #3553 (flattening guard), #3554 (paraphrase gap). Issues closed: none. Commented: #3533, #3541. Related: #3533, #3534, #3535, #3536, #3537, #3409, #3463, #3462, #3356.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3538

🤖 Generated with [Claude Code](https://claude.com/claude-code)
