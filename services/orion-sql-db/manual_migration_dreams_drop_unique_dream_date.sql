-- Allow multiple dream rows per day.
-- Previous schema enforced unique(dream_date), which prevented multiple dream artifacts
-- in the same calendar day from being persisted as separate rows.

ALTER TABLE IF EXISTS dreams
DROP CONSTRAINT IF EXISTS dreams_dream_date_key;

DROP INDEX IF EXISTS dreams_dream_date_key;
DROP INDEX IF EXISTS ix_dreams_dream_date;
