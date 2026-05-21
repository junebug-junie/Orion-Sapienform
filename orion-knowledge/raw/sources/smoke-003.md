# Hub Source Delta Test

## Requirements

- Hub should let operators ingest design docs.
- Source ingest should create pending reviews, not accepted claims.
- The result should show proposed claims and warnings.

## Non-goals

- No auto-accept.
- No vector search.
- No autonomous rewrite.

## Acceptance checks

- Dry run writes no files.
- Write mode creates a pending review.
