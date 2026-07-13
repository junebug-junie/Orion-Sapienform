# PR report: concept-relation decision log + belief-revision digest

Implements item 1 of `docs/superpowers/specs/2026-07-13-recall-followups-loop-retirement-saturation-gate-spec.md`.

## Summary

- `maybe_resolve_concept_relation()`'s LLM decisions were only observable via a log line that fired *after* the confidence-floor/relation filter — every `unrelated` decision and every sub-floor `contradicts`/`refines` decision vanished silently. Now every real LLM decision writes a row to a new `memory_concept_relation_decisions` table, regardless of outcome.
- New `scripts/concept_relation_digest.py` reads undigested rows and produces two things in one pass: a threshold-tuning report (call volume, relation distribution, near-miss counts below `CONCEPT_RELATION_CONFIDENCE_FLOOR`), and — for real belief-revision decisions (`same`/`refines`/`contradicts` that cleared the floor) — a new `reflection`-kind crystallization: a deterministic, non-LLM trace of Orion revising its own beliefs over time.
- `reflection` is a new `CrystallizationKind` with a real `salience.py::KIND_BASE` entry (0.4) so it participates in real ranking, not an inert schema-valid label — satisfies this repo's "no keyword cathedral" requirement (producer + consumer in the same patch).
- Self-caught review finding, already fixed: the decision-log write was initially unguarded — wrapped in try/except so a transient DB error on this purely observational write can no longer take down an entire consolidation window.
- Live-verified end to end, twice (once by the implementing agent, once independently by the orchestrator): seed a real decision, run the digest, confirm a real reflection crystallization lands in Postgres with real salience/confidence, confirm a second run produces zero duplicates.

## Outcome moved

The concept-relation resolver (proven working in PR #1000) goes from "we can only see its decisive outputs" to "every decision is inspectable, including the near-misses that would tell us if the 0.6 floor needs retuning" — and its real belief revisions now leave a structured trace instead of disappearing into a link table nobody reads back.

## Current architecture (before this patch)

`maybe_resolve_concept_relation()` called a real LLM, computed a relation + confidence, and either attached a link/reinforced a match (decisive outcomes) or returned `None` silently (everything else). No table, no digest, no `reflection` kind existed.

## Architecture touched

`orion/memory/crystallization/` (schema, salience, concept_relation, repository), one new top-level script, one new SQL table (via the existing idempotent-DDL-file mechanism, not a new migration-file system — this repo has none).

## Files changed

- `orion/core/storage/sql/memory_crystallizations.sql`: new `memory_concept_relation_decisions` table (`CREATE TABLE IF NOT EXISTS`, matching how `memory_crystallization_retrieval_events` already lives in the same file) + index on `(digested, decided_at)`.
- `orion/memory/crystallization/repository.py`: `insert_concept_relation_decision()`, mirroring `insert_retrieval_event()`'s exact shape.
- `orion/memory/crystallization/concept_relation.py`: `maybe_resolve_concept_relation()` now inserts a decision row (try/except-guarded) after `resolve_concept_relation()` returns, before the existing floor/relation filter — so nothing vanishes silently anymore.
- `orion/memory/crystallization/schemas.py`: `reflection` added to `CrystallizationKind`; `concept_relation_decision` added to `CrystallizationSourceKind` (required for the reflection crystallization's evidence ref — schema is `extra="forbid"`).
- `orion/memory/crystallization/salience.py`: `reflection` added to `KIND_BASE` at 0.4 (below `episode`'s 0.45 — the lowest tier, since these are meta-observations about the decision loop itself, not first-hand evidence).
- `scripts/concept_relation_digest.py` (new): standalone script matching `check_activation_saturation.py`'s exact conventions (argparse, `POSTGRES_URI`/`--postgres-uri`, `--json`, sys.path guard). Runs the whole read/create-reflections/mark-digested sequence inside one real Postgres transaction via a `_SingleConnPool` shim, so a mid-run crash rolls back cleanly instead of double-creating reflections on retry.
- Tests: `tests/test_memory_crystallization_concept_relation.py` (decision-log round-trip, unrelated/sub-floor decisions now logged, write-failure-doesn't-break-formation regression), `tests/test_concept_relation_digest.py` (report counts, reflection creation gated correctly on `floor_cleared`, second-run-no-duplicates, clean-empty-run), `tests/test_memory_crystallization.py` (`reflection` kind scores real nonzero salience).

## Design decisions worth flagging

**Status `"active"`, not `"proposed"`.** Reflection crystallizations are system-derived observations about the decision loop's own history, not claims requiring governor review — same trust tier as other automated provenance already in this codebase. `governance.approved_by`/`approval_mode` set to `"system:concept_relation_digest"`/`"auto_policy"` to make that origin explicit and auditable, not hidden behind a human-looking approval.

**One transaction, not a naive read-then-write.** The `_SingleConnPool` shim lets `insert_crystallization()` (which expects a real `asyncpg.Pool`) run on the same already-open connection/transaction as the digest's own reads and digested-flag update. This is the correct fix for the obvious failure mode (crash mid-digest leaving some rows digested with no reflection, or vice versa) rather than leaving it as a known gap.

**No new migration-file system invented.** This repo's schema for `orion/memory/crystallization/` lives in one idempotent `CREATE TABLE IF NOT EXISTS`-per-table SQL file, re-applied wholesale via `apply_memory_crystallizations_schema()`. The new table was added there, not as a new standalone migration mechanism.

## Schema / bus / API changes

- Added: `memory_concept_relation_decisions` table (new, additive). `reflection` `CrystallizationKind` value (new, additive — existing kind-based logic elsewhere either explicitly lists kinds it cares about, unaffected by an unlisted new one, or falls back safely, per the implementing agent's review pass confirming `KIND_TO_BUCKET.get()` has a safe default and `formation_policy.GATED_KINDS` doesn't need this kind since the digest bypasses the governor path entirely).
- Removed: none.
- Behavior changed: `maybe_resolve_concept_relation()` now performs one additional DB write per real LLM decision (previously zero for non-decisive outcomes). Guarded to never raise.
- Compatibility notes: no existing caller's contract changes; the new write is purely additive and independently guarded.

## Env/config changes

None. No new env keys — `CONCEPT_RELATION_CONFIDENCE_FLOOR` (pre-existing) is read, not changed.

## Tests run

```text
$ source venv/bin/activate && python -m pytest tests/test_memory_crystallization_concept_relation.py \
    tests/test_memory_crystallization.py tests/test_concept_relation_digest.py \
    tests/test_encode_reinforce_not_duplicate.py -v
78 passed, 1 failed
```
The 1 failure (`TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap`) is the same pre-existing, unrelated failure confirmed multiple times earlier this session (registry-gap issue, zero relation to this patch's files). Independently re-run by the orchestrator, not taken from the implementing agent's report alone.

## Evals run

No eval harness applies — this is deterministic aggregation logic, fully covered by the unit tests above. The live verification below is the standard this session holds "does it actually work" claims to.

## Docker/build/smoke checks

```text
# Schema applied live, confirmed via \d:
$ psql -c '\d memory_concept_relation_decisions'
(all 8 columns present as designed, including the digested/decided_at index)

# Orchestrator's own independent live end-to-end run (separate from the implementing
# agent's own verification, seeded fresh, not reusing their test data):
$ python verify_concept_relation_digest.py
seeded target: e4dde759-dbad-4508-a3fd-c01e9b2850dd
seeded decision: 0cf96e0d-24d5-4e21-8000-91f457e7f9ca

$ POSTGRES_URI=postgresql://postgres:postgres@127.0.0.1:55432/conjourney python scripts/concept_relation_digest.py
concept_relation_digest: 1 decision(s) since last run
  relation distribution: contradicts: 1 (rest 0)
  near-misses: contradicts: 0, refines: 0
  reflection crystallizations created: 1
    040407e5-4f06-4638-883f-6e84c2133901

$ (second run) -> "0 decision(s) since last run ... nothing new to report." (no duplicate)

# Confirmed the reflection crystallization actually landed correctly, by direct query:
$ psql -c "select crystallization_id, kind, status, subject, salience, confidence
           from memory_crystallizations where crystallization_id='040407e5-...'"
040407e5-... | reflection | active | Concept relation decision: contradicts | 0.491 | possible
```
Real, non-zero, non-default salience (0.491) and a real computed confidence (`possible`, via the already-merged `infer_confidence()` from PR #1002) — not an empty-shell stub. Test rows deprecated/deleted after verification, no permanent pollution.

## Review findings fixed

- Finding: `insert_concept_relation_decision()`'s call site in `maybe_resolve_concept_relation()` was unguarded — unlike `resolve_concept_relation()` (documented to never raise), a transient DB error on this purely observational write would propagate up through `intake_pipeline.py` and fail an entire consolidation window over the loss of one diagnostic row.
  - Fix: wrapped in `try/except Exception` with a `logger.warning`, matching the file's existing never-raise convention.
  - Evidence: commit `4429bf74`; new regression test `test_decision_log_write_failure_does_not_break_formation` asserts `maybe_resolve_concept_relation()` still returns the correct outcome even when the decision-log insert raises.

Orchestrator independently re-read every diff in full, re-ran the full targeted test suite, and ran an independent live end-to-end verification (fresh seed data, not reused from the implementing agent's own run) rather than trusting the report alone.

## Restart required

```text
No restart required for the code itself (no env/schema-deploy-time changes beyond the
idempotent SQL file, which orion-memory-consolidation and orion-hub already re-apply on
their normal startup path via apply_memory_crystallizations_schema()).
```

## Risks / concerns

- Severity: Low — `scripts/concept_relation_digest.py` is on-demand/cron-category, not a live service loop (matches the spec's explicit non-goal). Nothing runs it automatically yet; someone needs to invoke it (manually or via a future scheduled job) for the loop to actually close in production. This mirrors `check_activation_saturation.py`'s exact same status.
- Severity: Low — `reflection` crystallizations are created with `requires_manual_review=False` and never surface in the governor review queue by design (system-derived, not requiring human approval). If this trust level is ever reconsidered, the change is isolated to `_build_reflection_crystallization()`'s governance fields.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/concept-relation-decision-loop?expand=1`
