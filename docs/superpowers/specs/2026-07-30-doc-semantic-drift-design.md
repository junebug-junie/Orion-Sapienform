# Doc/Narrative Semantic Drift — cheap-prefilter concept layer (design)

Status: design/proposal mode. Third domain under the `orion/concepts/` registry proposed in
`2026-07-30-codebase-mass-signal-design.md`'s Phase 2 — same registry, not a competing one.

## Arsonist summary

graphify already has a real semantic (LLM) extraction pass (`/graphify` Part B, on
docs/papers/images) with an already-incremental, content-hash-keyed cache
(`graphify_seed_semantic_cache.py`) — but per Juniper it has effectively never been run against
docs/READMEs because a full-corpus pass is a real frontier-LLM cost. This spec doesn't build new
extraction; it scopes the existing mechanism to only the doc files touched in a given commit/PR
diff (naturally small), and adds a cheap, non-frontier pre-filter in front of it so LLM calls are
spent only where a real semantic shift is likely.

## Current architecture

- graphify Part B: real, already exists (root `CLAUDE.md`'s own graphify section). Semantic
  cache keyed by content hash under `graphify-out/cache/semantic/`, gitignored, local.
  `graphify_seed_semantic_cache.py` reconstructs cache entries directly from `graph.json`'s
  existing nodes/edges/hyperedges with no LLM call — sub-second, per CLAUDE.md's own
  description.
- Conventional-commit prefixes (`feat/fix/chore/docs/test`) are already used in the large
  majority of this repo's real commit history (confirmed by sampling git log this session) — a
  free, already-present impact/intent classification, zero new extraction needed to use it.
- `orion/concepts/` registry (sibling spec, Phase 2): `chat` and `architecture` domains
  proposed. This spec adds a third: `narrative`/`doc`.
- Existing-mechanism check: no embedding-diff or lexical-drift mechanism for docs/READMEs found
  in this repo — clean.

## Missing questions

- What non-frontier embedding model is acceptable to run inline — routed through
  `orion-llm-gateway`'s existing model routing, or a new lightweight local dependency? Needs a
  real answer, not an assumption, before this is buildable.
- What threshold on embedding-diff should gate escalation to a real graphify Part B call? Needs
  calibration against real historical doc diffs (there's plenty of history to calibrate
  against), not a guessed constant.
- Should the free commit-prefix signal be treated as a strong classifier or only a weak feature
  alongside the embedding-diff score? `docs:`-prefixed commits are self-declared, not verified —
  a `docs:` commit could still carry a real code-adjacent concept change, or a `feat:` commit
  could touch docs incidentally.

## Proposed schema / API changes

**Split, 2026-07-30, matching the rest of this design arc's shared-library-vs-service
pattern:** the I/O-heavy observation work (embedding calls, graphify Part B escalation) is a
*producer*, not a concept-domain classifier — it moves to
`services/orion-cocreation-signals/app/producers/doc_semantic_drift.py` (see
`2026-07-30-codebase-mass-signal-design.md`'s "Dedicated service" section), alongside
`structural_mass`, `dev_economics`, and the affective-state producers. `orion/concepts/`'s
`narrative.py` domain (sibling spec's Phase 2) stays a pure classification/registry library that
consumes this producer's output — it does no I/O of its own, same relationship
`orion/substrate/prediction_error.py` has to the `structural_mass` producers.

1. **Cheap tier (always runs, in the new service):** embedding-diff between before/after text
   for any docs/README files touched in a commit/PR — non-frontier, cheap, local or
   gateway-routed.
2. **Escalation tier (gated, in the new service):** only files whose embedding-diff crosses a
   calibrated threshold get a real graphify Part B semantic extraction call, scoped to just
   those files. Reuses the existing content-hash cache, so anything already covered by a prior
   graphify run is never re-billed.
3. **Free tier (always attached):** conventional-commit-prefix classification as a weak feature,
   regardless of the above two tiers' outcome.
4. **Classification (in `orion/concepts/domains/narrative.py`):** the producer's output —
   embedding-diff score, any escalated concept extraction, the commit-prefix feature — becomes a
   candidate concept instance in the shared registry, same shape the `architecture` domain
   consumes from `structural_mass`.

**Self-narrative coherence signal — the concrete answer to "how do we operationalize this
beyond a vague sense of self-growth":** compare `structural_mass`'s code-change magnitude for a
window against this domain's doc-semantic-delta magnitude for the same window. A large mismatch
— substantial code delta, ~zero doc delta — is a real, traceable "documentation lag" finding,
not narrative flavor. It's a measurable gap, the same shape as every other real signal in this
program, and a strong candidate to feed the SSP Objective-3 AST/HOT reducer's self-model
artifact or the journal surface as an actual finding.

## Files likely to touch

- `services/orion-cocreation-signals/app/producers/doc_semantic_drift.py` (new): the I/O-heavy
  producer — embedding-diff tier, gated graphify Part B escalation tier.
- `orion/concepts/` (new, shared with the sibling spec's Phase 2): registry + `narrative.py`
  domain — pure classification, no I/O.
- Reuses `scripts/graphify_seed_semantic_cache.py`, `graphify-out/cache/semantic/` as-is — no
  changes needed there.
- `scripts/analysis/measure_doc_semantic_drift.py` (new): read-only replay script against real
  historical doc diffs.

## Non-goals

- Not running graphify Part B repo-wide on any cadence — scoped to per-commit/PR diffs only,
  always, by design (this is the whole point of the pre-filter).
- Not implementing the embedding-diff pre-filter as a frontier-LLM call — it must be genuinely
  cheap/non-frontier, or it doesn't solve the cost problem it exists to solve.
- Not wiring the self-narrative-coherence comparison to any consumer until both
  `structural_mass` and this domain independently have real replayed data — "measure before
  minting," same as everywhere else in this program.

## Acceptance checks

- Embedding-diff pre-filter tested against a handful of real historical doc commits (some
  trivial typo/formatting diffs, some real rewrites) and shown to separate them at a real,
  calibrated threshold — not an arbitrary guess.
- Confirmed the escalation tier only re-extracts files not already present in the semantic
  cache — i.e., doesn't silently re-bill something graphify already covered.
- Self-narrative-coherence replay produces at least one real historical example of a large
  code/doc mismatch — the drives arc (PR #879 → #1486, now merged) is a strong first candidate
  given how much code changed relative to how much narrative documentation had to be hand-updated
  to track it.

## Recommended next patch

The embedding-diff pre-filter alone, run read-only against a sample of this repo's real
historical doc commits, calibrating a real threshold — before touching graphify Part B
integration or the `orion/concepts/` registry shape at all.
