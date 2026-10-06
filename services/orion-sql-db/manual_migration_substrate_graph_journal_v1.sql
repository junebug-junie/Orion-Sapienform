-- Shared substrate graph journal (memory Stage 2 PR A, 2026-10-06): proposal -> decision ->
-- materialization receipts for claims projected into the substrate graph (Falkor).
-- Spec: docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md section 1.4;
-- contract: orion/core/schemas/substrate_graph_journal.py.
--
-- Append-only: rows are inserted, never updated or deleted (SubstrateGraphJournal has no
-- update/delete path). `payload` holds the full validated event; the columns beside it are the
-- ones queries filter or enforce uniqueness on.
-- Writers: AssertionProjector (materializations) and claim producers (proposals, decisions;
-- memory referents in PR B). Reader: AssertionProjector via SubstrateGraphJournal.
-- Apply BEFORE deploying any service that runs AssertionProjector. Additive only.
-- Rollback: manual_migration_substrate_graph_journal_v1_rollback.sql (drops the journal; the
-- Falkor projections stay and can be retracted by producer).

CREATE TABLE IF NOT EXISTS substrate_graph_journal (
    event_id TEXT PRIMARY KEY,
    event_kind TEXT NOT NULL,              -- proposal | decision | materialization
    proposal_kind TEXT NOT NULL,           -- relationship_assertion (validated in Python)
    proposal_id TEXT NOT NULL,
    decision_id TEXT,                      -- decision + materialization rows
    target_id TEXT NOT NULL,               -- the Assertion node id
    revision INTEGER,                      -- decision: resulting revision; materialization: applied revision
    outcome TEXT,                          -- materialization: applied | failed
    actor TEXT NOT NULL,
    payload JSONB NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT substrate_graph_journal_kind_chk CHECK (event_kind IN ('proposal', 'decision', 'materialization')),
    CONSTRAINT substrate_graph_journal_outcome_chk CHECK (
        (event_kind = 'materialization' AND outcome IN ('applied', 'failed')) OR
        (event_kind <> 'materialization' AND outcome IS NULL))
);
-- Expected-revision check: two decisions cannot both claim the same next revision of a target.
CREATE UNIQUE INDEX IF NOT EXISTS uq_substrate_graph_journal_decision_revision
    ON substrate_graph_journal (target_id, revision) WHERE event_kind = 'decision';
-- At most one applied materialization per decision; failed attempts may repeat.
CREATE UNIQUE INDEX IF NOT EXISTS uq_substrate_graph_journal_applied
    ON substrate_graph_journal (decision_id) WHERE event_kind = 'materialization' AND outcome = 'applied';
CREATE INDEX IF NOT EXISTS idx_substrate_graph_journal_target ON substrate_graph_journal (target_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_substrate_graph_journal_proposal ON substrate_graph_journal (proposal_id);
CREATE INDEX IF NOT EXISTS idx_substrate_graph_journal_decision ON substrate_graph_journal (decision_id)
    WHERE decision_id IS NOT NULL;
