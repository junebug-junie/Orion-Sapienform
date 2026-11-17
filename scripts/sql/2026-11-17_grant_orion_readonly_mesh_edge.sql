-- ORION-MIGRATION-REQUIRED-BY: orion-harness-governor
--
-- Let Orion's read-only role read what the mesh-edge check (orion/memory/mesh_edge.py, used by the
-- FCC PreToolUse hook orion/fcc/mesh_edge_hook.py) loads before it lets a GitHub write through:
-- the core pages and every lived memory a public write must not quote. Issue #3567: the hook's DSN
-- (ORION_MESH_EDGE_PG_DSN in orion-harness-governor) used the postgres superuser, and FCC Bash
-- inherits that env. It now uses orion_readonly, which needs these grants:
--
--   self_concept_history     page:juniper / page:anchor, every version (CORE_PAGES_SQL)
--   episode_memory           lived memories (GUARD_MEMORY_SQL)
--   memory_crystallizations  stance crystallizations (GUARD_MEMORY_SQL)
--   person_fact, person      person facts and display names (GUARD_MEMORY_SQL)
--   episode_memory_referent  who/what each lived memory is about (granted with the set, so a guard
--                            that joins referents later does not silently fail closed)
--
-- Without them the guard cannot load and the hook refuses EVERY GitHub write from an Orion turn
-- (fail closed, not open). So apply this BEFORE the governor runs with the orion_readonly DSN.
--
-- Read-only grants; no write. Not under services/orion-sql-db/: GRANT is not on the post-pull
-- auto-apply allowlist, and the deploy gate cannot see a grant (a grant-only file has no schema
-- effect to check; scripts/check_migration_required_by.py rejects REQUIRED-BY on such a file).
-- The REQUIRED-BY line above is the record of who needs it, not an enforced gate.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < scripts/sql/2026-11-17_grant_orion_readonly_mesh_edge.sql
-- or the one-line form:
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON episode_memory, memory_crystallizations, person_fact, person, self_concept_history, episode_memory_referent TO orion_readonly;"

BEGIN;

GRANT SELECT ON public.episode_memory, public.memory_crystallizations, public.person_fact, public.person,
    public.self_concept_history, public.episode_memory_referent TO orion_readonly;

COMMIT;

-- Undo (every Orion GitHub write is then refused until the DSN changes again):
--   REVOKE SELECT ON public.episode_memory, public.memory_crystallizations, public.person_fact,
--     public.person, public.self_concept_history, public.episode_memory_referent FROM orion_readonly;
--
-- Verify (expect t for every SELECT and f for every INSERT):
--   SELECT t, has_table_privilege('orion_readonly', t, 'SELECT') AS can_select,
--          has_table_privilege('orion_readonly', t, 'INSERT') AS can_insert
--   FROM unnest(ARRAY['episode_memory', 'memory_crystallizations', 'person_fact', 'person',
--                     'self_concept_history', 'episode_memory_referent']) AS t;
