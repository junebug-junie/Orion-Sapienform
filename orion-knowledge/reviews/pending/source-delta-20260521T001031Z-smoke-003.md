# Source Delta Review

## Source
- id: `source:smoke-003`
- kind: `design_doc`
- path: `raw/sources/smoke-003.md`

## Source summary
**Hub Source Delta Test**

- Hub should let operators ingest design docs.
- Source ingest should create pending reviews, not accepted claims.
- The result should show proposed claims and warnings.

## Proposed claims
- [ ] Hub should let operators ingest design docs. _(L5 · Requirements § Hub should let operators ingest design docs.)_
- [ ] Source ingest should create pending reviews, not accepted claims. _(L6 · Requirements § Source ingest should create pending reviews, not accepted claims.)_
- [ ] The result should show proposed claims and warnings. _(L7 · Requirements § The result should show proposed claims and warnings.)_
- [ ] No auto-accept. _(L11 · Non-goals § No auto-accept.)_
- [ ] No vector search. _(L12 · Non-goals § No vector search.)_
- [ ] No autonomous rewrite. _(L13 · Non-goals § No autonomous rewrite.)_
- [ ] Dry run writes no files. _(L17 · Acceptance checks § Dry run writes no files.)_
- [ ] Write mode creates a pending review. _(L18 · Acceptance checks § Write mode creates a pending review.)_

## Possibly affected specs
- `spec:knowledge-forge-ideation-review-v1`

## Suggested context packs
- compile context pack for `spec:knowledge-forge-ideation-review-v1` after claims are accepted

## Human action needed
- accept/reject proposed claims
- update affected specs if needed
