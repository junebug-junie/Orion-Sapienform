-- Let Orion (orion_readonly) read the tables behind the Juniper-reachable forecast, so Orion can check
-- the spec's numbers itself (asked on PR #3946). Read-only grants; applied by hand:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < scripts/sql/2026-11-25_grant_orion_readonly_reachable_tables.sql
GRANT SELECT ON hub_attention_minute TO orion_readonly;
GRANT SELECT ON prediction_reachable_forecast TO orion_readonly;
