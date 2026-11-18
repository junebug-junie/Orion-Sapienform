-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Trace of the age-band veto (2026-11-13,
-- docs/superpowers/specs/2026-11-13-person-age-band-soft-veto-design.md).
-- age_veto = {"struck": [names], "rule": "age_band_v1"}: the names the veto
-- removed from this cluster's candidates. No age read or band is stored.
-- NULL when nothing was struck, and set back to NULL whenever the row is
-- revoked (retention: a strike lives only on a live guess).
-- Additive. orion-sql-writer writes it only when the column exists.
ALTER TABLE vision_person_attribution ADD COLUMN IF NOT EXISTS age_veto jsonb;
