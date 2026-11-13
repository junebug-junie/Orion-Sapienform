-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- "Same person" (2026-10-29): Juniper says two names in the person record are one
-- person (a first name and a full name, or a nickname). The merged row stays, so
-- its id keeps resolving; merged_into names the person kept. Set from the Hub
-- People panel (services/orion-hub/scripts/people_routes.py); read by the link
-- planner (orion/person_model/record.py), which moves the merged name's camera
-- links to the person kept.
-- No foreign key: an FK on a new column is not on the post-pull auto-apply allowlist.
-- The Hub checks the target exists; the planner ignores a dangling id.
ALTER TABLE person ADD COLUMN IF NOT EXISTS merged_into TEXT;
