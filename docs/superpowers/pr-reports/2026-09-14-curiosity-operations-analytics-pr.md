# Curiosity operations analytics

## Status

`READY_FOR_REVIEW`

## Summary

Adds a privacy-reduced Curiosity operations slice to Orion Analytics and a
deployed Lightdash dashboard for answering:

- which Curiosity pipeline types have run;
- which durable lifecycle states and resume/failure events were recorded;
- how much wall time, harness time, and harness work runs actually used;
- how much approved material was available and offered;
- what graph element types each lane produced; and
- which runs explicitly produced no graph artifact.

The patch does not label a stale non-terminal row as a running process and does
not invent historical configured-budget values that were never persisted.
Lifecycle measures are explicitly limited to the currently retained source
window (90 days by default).

## Data contracts and privacy

- Adds four `security_barrier` projections in `analytics_source` for bounded
  structural lifecycle and deterministic journal-footer metadata.
- Adds four grain-separated dbt facts: runs, transitions, material-pool rows,
  and graph-write rows.
- Keeps prompts, findings, journal prose, error text, correlation IDs, raw JSON,
  graph claims, and checkpoints outside analytics.
- Grants the transformer only the reduced projections. The Lightdash reader can
  read marts but cannot read `analytics_source` or the underlying operational
  tables.
- Makes no writes to FalkorDB and does not run or refresh Graphify.

## Dashboard

`Curiosity Operations` contains 13 tiles: eight query-backed charts and five
scope/section tiles. It covers the run ledger, pipeline types, lifecycle event
status, actual elapsed time, offered material, approved pool composition,
graph-write types, and graph-write coverage.

## Live evidence

The 2026-09-14 reconciliation found:

- 96 observed run IDs, including 50 with durable lifecycle evidence;
- 640 persisted lifecycle transitions;
- 50 completed durable runs, 186 failed transition events, and 245 resume
  events;
- 212 run/graph-element rows and 234 run/material-kind rows;
- 68 runs with recorded graph elements, 26 explicitly recording no graph
  write, and two legacy runs without graph evidence; and
- zero source/fact count deltas or duplicate-key deltas.

All eight uploaded Lightdash charts executed successfully against the live
warehouse. Lightdash metadata contains the deployed dashboard, eight saved
charts, eight chart tiles, and five markdown tiles.

## Checks

- analytics role bootstrap and fail-closed privilege audit: pass
- dbt deps / parse / compile / run: pass (`16` views)
- dbt tests: pass (`210`, zero warnings/errors)
- read-only live reconciliation: pass (all deltas zero)
- analytics-reader and transformer denial checks: pass
- static analytics contract tests: pass (`16`)
- Lightdash semantic compile: pass (`9` explores, zero errors)
- Lightdash lint: pass
- Lightdash upload: pass
- live Lightdash chart execution: pass (`8/8`)
- `git diff --check`: pass

## Review findings fixed

- Unanchored regexes could parse model-authored prose before the producer
  footer. Parsing now isolates and validates only the final footer line;
  structural fragment labels are bounded, and an adversarial test proves fake
  earlier phrases cannot affect the result.
- Raw `grounding_status` could carry an exception or refusal string. The safe
  view now exposes only whitelisted status codes and maps everything else to
  `other`. Live verification found and redacted 29 such diagnostic values;
  zero unbounded values remain exposed.
- Lifecycle descriptions implied journal-only meant pre-durable even though
  the state table has default 90-day retention. Model, dashboard, source, and
  operator docs now consistently say retained-window evidence, with a static
  contract test protecting that disclosure.
- Lightdash upload validation found one unused graph-family dimension. It was
  removed, then validation and all eight live chart queries passed.

The review found no other material grain, permission, Lightdash, scope, or
Graphify issues. A two-thread dbt view refresh briefly deadlocked in an
unrelated visual-Reverie view; the complete refresh passed serially before the
test run.

## Known limitation and next patch

Daily caps, cooldowns, outer deadlines, governor deadlines, and stream-stall
deadlines are configuration rather than persisted run facts. A follow-up should
emit typed, non-narrative budget telemetry at admission/kickoff and clear the
metric-quality gate on live data before Lightdash exposes budget headroom or
remaining-budget measures.
