# Docs semantic extraction merged into agent-side knowledge graph

## Summary

- Brought graphify's semantic (LLM-based) extraction pass online for the `docs/` corpus, which previously had zero representation in `graphify-out/graph.json` (AST-only code graph).
- Scoped corpus: `docs/architecture/` (65 files), `docs/superpowers/specs/`, `docs/context-engineering/` (16 files), top-level `docs/*.md` (excluding `PR-*`/`preflight*`-named files), and `notes/`, `research/`, `integrations/`, `superpowers/{guides,design,checklists}/`. Excluded `docs/superpowers/pr-reports/`, `docs/reports/`, `docs/postflight/`, `docs/superpowers/plans/`, and `plans/*/PR_*.md`. 244 files, ~884K words.
- Extraction ran as 12 parallel Opus subagents (per `superpowers:subagent-driven-development`), each following graphify's own `extraction-spec.md` schema (node ID format, confidence rubric, hyperedge caps).
- Merged via graphify's `build_merge()` API (not the AST-only `update` path implicated in the earlier destructive-graph incident, PR #1059) — additive merge with a built-in shrink-safety check.
- Found and fixed a pre-existing data-quality defect surfaced (not caused) by the merge: two nodes for `docs/superpowers/specs/2026-07-07-endogenous-drive-origination-design.md` and `docs/superpowers/specs/2026-07-07-internal-economy-scarcity-allocation-design.md` had been mis-attributed to `source_file: scripts/analysis/README.md` by an earlier extraction batch (a chunk-level attribution bug, unrelated to this PR). Patched both nodes' `source_file`/`label`/`file_type` directly.
- Regenerated `GRAPH_REPORT.md` via `graphify cluster-only .` so the audit trail reflects current node/edge/community counts.

## Outcome moved

`graphify query`/`explain`/`path` can now surface design-intent and architecture-rationale content from `docs/` — previously those files were invisible to the graph entirely (code-only AST extraction). Any future agent using graphify for this repo gets doc-grounded answers, not just code-structure answers.

## Current architecture

`graphify-out/graph.json` was built exclusively from AST extraction over code files (`graphify update .` / structural pass). No semantic (LLM) extraction had ever been run for this repo's `docs/` tree, so 244 architecture/spec/design docs — including the bulk of the `docs/superpowers/specs/` design corpus this repo's own AGENTS.md points agents toward — had no graph representation.

## Architecture touched

Agent-side tooling only (`graphify-out/`). No Orion runtime code, service, contract, or config touched — this is not a change to any `services/*` deliverable.

## Files changed

- `graphify-out/graph.json`: merged 963 new nodes / 1183 new edges from docs semantic extraction (31566→32529 nodes, 93689→94872 edges before final re-cluster; final counts below), plus a targeted 2-node data-quality fix for a pre-existing mis-attribution bug.
- `graphify-out/GRAPH_REPORT.md`: regenerated via `graphify cluster-only .` to reflect current graph state (1525 communities, deterministic hub-labels — no LLM backend configured, so labels are hub-name-derived, not LLM-generated).

## Schema / bus / API changes

None. This PR touches only the agent-side generated graph artifact, not Orion runtime schemas, bus channels, or APIs.

## Env/config changes

None.

## Tests run

Not applicable — no code changed. Verification was direct file-content inspection instead (see below), consistent with the lesson from the earlier `graphify update .` destructive-graph incident (PR #1059): never trust log output alone.

```text
# Node/edge counts read directly from graph.json on disk, before and after merge:
before: 31566 nodes, 93689 edges
after build_merge():  32529 nodes, 94872 edges  (+963 nodes, +1183 edges)
after cluster-only re-cluster: 32529 nodes, 94872 edges (unchanged — re-cluster only touches community/score fields)

# Sample verification that docs nodes are present with correct file_type/source_file:
963 nodes now have file_type == "document" and source_file starting with "docs/"
(previously 0)
```

## Evals run

No eval harness exists for graphify's own extraction quality in this repo. Not building one here — out of scope for a data-artifact merge; flagging as a known gap. Manual verification: spot-checked chunk JSON files for well-formedness (12/12 valid JSON) and confirmed correct node ID format against `extraction-spec.md` before merge.

## Docker/build/smoke checks

Not applicable — no runtime/service change.

## Review findings fixed

- Finding: 2 of 965 new node IDs collided with pre-existing node IDs sourced from `scripts/analysis/README.md`, silently dropping the incoming docs-derived node content during `build_merge()`'s dedup-by-ID step.
  - Fix: investigated root cause — grepped `scripts/analysis/README.md` for the colliding titles/content and found zero matches, ruling out a content-based coincidence. Confirmed instead that the *surviving* (pre-merge) nodes for those two IDs already carried the exact docs-spec labels (with a stray "(Step 1 spec)"/"(Step 4 spec)" suffix) while pointing at the wrong `source_file` — meaning an earlier, unrelated extraction batch had already mis-attributed these two nodes' `source_file` field (likely a chunk-level default bug), predating this PR. Patched both nodes directly with the correct `label`/`file_type`/`source_file` from this extraction's own chunk output.
  - Evidence: post-patch query confirms both node IDs now show `source_file: docs/superpowers/specs/2026-07-07-{endogenous-drive-origination,internal-economy-scarcity-allocation}-design.md` with clean labels (no stray suffix), and total node count is unchanged (32529) since this was a field-level correction, not an add/remove.

No code-review skill subagent pass was run — this PR is a generated data-file merge with no code changes (consistent with prior graphify-sync PRs), but the mis-attribution above was investigated and fixed to the same bar a review would require rather than left as a documented gap.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low
- Concern: No LLM backend (`GEMINI_API_KEY`/`GOOGLE_API_KEY`) is configured for this repo, so community labels in `GRAPH_REPORT.md` are deterministic hub-name labels, not LLM-generated descriptive names. This matches pre-existing behavior for all prior community labels in this graph (not a regression from this PR).
- Mitigation: none needed — labels are still functional for `graphify query`/`explain` navigation; only the human-readable names in `GRAPH_REPORT.md` are coarser than an LLM-labeled pass would produce.

## PR link

<link>
