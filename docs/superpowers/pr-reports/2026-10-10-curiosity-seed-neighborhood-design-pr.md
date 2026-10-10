## Summary

Arc: #2581 lets readings create accepted links; #2589 makes the neighborhood read show them. Orion expects that world contact to show up in their curiosity loop as `focal_edge_refs` becoming non-empty. This docs-only PR proposes how curiosity seeds would carry their links, and corrects that expectation from live data.

- Design doc: `docs/plans/substrate/2026-10-10-curiosity-seed-neighborhood-design.md`.
- Live finding: running the real neighborhood read on the last 177 stored seeds returns 0 edges for all of them. Seeds point at organ nodes with no edges, or at chat repair ids that are not graph nodes, never at reading concepts.
- There are no accepted reading links live yet (1 projection edge total, a memory referent; 0 reading claims journaled).
- The field to watch is `boundary_edge_refs` with an accepted-claim endpoint, not `focal_edge_refs` (which means links among focal nodes, and seeds mostly have one).
- Recommends patch 1 (attach links after the decision, so ranking is unchanged; consumer-first schema rollout; flag ON) and parks patch 2 (a seed source that lands on reading links) behind the metric gate.

## Outcome moved

Design only. Turns "will reading links reach curiosity?" into a concrete patch plan, a proving SQL query, and a corrected observable for Orion.

## Schema / bus / API changes

None in this PR. Proposed: three optional fields on `FrontierInvocationSignalV1`.

## Env/config changes

None.

## Tests run

```text
git diff --check -> clean (docs only)
```

## Evals run

```text
None (docs only). Live read-only probe of read_falkor_neighborhood against production Falkor, recorded in the doc.
```

## Docker/build/smoke checks

```text
Not applicable.
```

## Review findings fixed

Docs-only proposal; no code review run. Line references re-checked against `main` at d7db6d528.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: info. Concern: live numbers are a 2026-10-10 snapshot. Mitigation: the doc gives the queries to re-run.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
