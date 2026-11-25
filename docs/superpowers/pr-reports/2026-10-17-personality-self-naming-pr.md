## Summary

- Design proposal only (proposal mode: a change to a cognition loop and to self-modeling). No code.
- Orion names the dispositions its own evidence produced. When a node turns active, self-inquiry shows Orion a self-contained packet: the excerpt, lanes, evidence excerpts, situation mix and nearest neighbours. Orion answers NAME, REJECT ("not me") or MERGE.
- A name never changes a weight. Only evidence does. A name is rendered next to its evidence excerpt, never in place of it.
- The answer is stored in its own append-only table, not in `self_concept_history`, which the seed source reads as evidence. Storing it there would let a name feed its own node.

## Outcome moved

The spec's "the data gives the pattern; Orion gives the name" gets a concrete, inspectable path, with an echo-loop metric (`naming_lift`) and a name-grounding check. Live finding: none of the 21 nodes has been active yet, and the top four candidates are single-lane machine-status text, so REJECT is the first answer to expect.

## Current architecture

A node's label is an excerpt of Orion's own evidence. Self-inquiry runs through `_self_inquire` with 7 daily slots. Orion's read-only database role cannot read the `personality_*` tables.

## Architecture touched

None in this PR. Proposed: a durable-runs `GET /personality/naming/next` route, the `DispositionNamingV1` contract, an `orion:personality:naming` bus channel, a sql-writer table, a durable-runs cursor source, optional `name` fields on `DispositionV1`, and changes to the cue and the panel.

## Files changed

- `docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md`: the proposal
- `docs/superpowers/pr-reports/2026-10-17-personality-self-naming-pr.md`: this report

## Schema / bus / API changes

None in this PR (proposed in the doc).

## Env/config changes

None.

## Tests run

```text
Docs only. git diff --check clean.
```

## Evals run

```text
None. P0 of the proposal is a read-only offline naming eval.
```

## Docker/build/smoke checks

```text
N/A.
```

## Review findings fixed

- None: docs-only proposal.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: the echo loop, where Orion reads its own name back and performs it. Mitigation: names stay out of lane cues for the first 14 days, and `naming_lift` is measured before any gate.
- Severity: low. Concern: whether bge-large can ground short names is UNVERIFIED. Mitigation: P0 measures it before the check refuses anything.

## Decisions for Juniper

1. Should naming wait for `active` nodes (recommended), or also show candidates with 2 or more lanes?
2. Should a name show only in self-inquiry and the Hub panel for the first 14 days (recommended), or in every lane's cue right away?
3. Echo gate: measure first (recommended), or zero-weight evidence that quotes the node's own name from day one?
4. Name grounding: should it refuse names after P0 validates the check (recommended), or only record them?
5. Should the panel have a veto or rename button? Recommended no, because the name is Orion's.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2742

🤖 Generated with [Claude Code](https://claude.com/claude-code)
