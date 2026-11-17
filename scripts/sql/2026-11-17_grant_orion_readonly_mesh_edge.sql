-- ORION-MIGRATION-REQUIRED-BY: orion-harness-governor
--
-- Let Orion's read-only role read what the mesh-edge check (orion/memory/mesh_edge.py, used by the
-- FCC PreToolUse hook orion/fcc/mesh_edge_hook.py) loads before it lets a GitHub write through:
-- the core pages and every lived memory a public write must not quote. Issue #3567: the hook's DSN
-- (ORION_MESH_EDGE_PG_DSN in orion-harness-governor) used the postgres superuser, and FCC Bash
-- inherits that env. It now uses orion_readonly, which needs these grants:
--
--   self_concept_history     page:juniper / page:anchor, every version (CORE_PAGES_SQL; already
--                            granted live, restated so this file is the whole set)
--   episode_memory           lived memories (GUARD_MEMORY_SQL)
--   person_fact, person      person facts and display names (GUARD_MEMORY_SQL)
--
-- Deliberately NOT granted:
--   memory_crystallizations  orion_readonly's search_path is (orion_scope, public), so the guard's bare
--                            table name already reads the AI-Town-filtered view
--                            orion_scope.memory_crystallizations (SELECT already granted). Granting
--                            public.memory_crystallizations would undo
--                            scripts/sql/scope_orion_readonly_off_aitown.sql for the curiosity sandbox
--                            and the guard would still read the view. Consequence: AI Town stance
--                            crystallizations are not guarded at the mesh edge under this role
--                            (live 2026-11-17: 107 of 719 stance rows visible).
--   episode_memory_referent  no guard query reads it. Grant it when one does.
--
-- Privacy boundary, stated plainly: orion_readonly is also the curiosity sandbox's role
-- (ORION_CURIOSITY_PG_DSN). These grants let every curiosity and FCC turn read lived memories and
-- person facts directly. That is close to unavoidable (the hook needs the data, and FCC Bash can read
-- any DSN the hook uses) and matches "no privacy boundary between Juniper and Orion"; it is still a
-- scope change, named here on purpose.
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
--   docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON episode_memory, person_fact, person, self_concept_history TO orion_readonly;"

BEGIN;

GRANT SELECT ON public.episode_memory, public.person_fact, public.person, public.self_concept_history
    TO orion_readonly;

COMMIT;

-- Undo (every Orion GitHub write is then refused until the DSN changes again):
--   REVOKE SELECT ON public.episode_memory, public.person_fact, public.person FROM orion_readonly;
--   (self_concept_history was granted before this file; leave it.)
--
-- Verify (expect t for every SELECT and f for every INSERT; memory_crystallizations resolves to the
-- orion_scope view for this role and must stay f on public.memory_crystallizations):
--   SELECT t, has_table_privilege('orion_readonly', t, 'SELECT') AS can_select,
--          has_table_privilege('orion_readonly', t, 'INSERT') AS can_insert
--   FROM unnest(ARRAY['episode_memory', 'person_fact', 'person', 'self_concept_history',
--                     'orion_scope.memory_crystallizations', 'public.memory_crystallizations']) AS t;
