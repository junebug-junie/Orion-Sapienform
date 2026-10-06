-- Rollback of manual_migration_substrate_graph_journal_v1.sql. Stop every AssertionProjector first.
DROP TABLE IF EXISTS substrate_graph_journal;
DROP FUNCTION IF EXISTS substrate_graph_journal_append_only();
