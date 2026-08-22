-- Attestation must survive to the scored row (2026-08-21).
--
-- manual_migration_artifact_rating.sql made an artifact rating require a
-- user_id, on the reasoning that an unattributed rating cannot be told apart
-- from Orion rating itself. That check happened at the schema boundary and
-- then the information was DROPPED: substrate_action_ratings recorded the
-- verdict, the categories and the free text, and not who gave it.
--
-- So the belief could not distinguish a human verdict from a deploy smoke
-- test, and the whole point of the artifact-rating path is that its grader is
-- not Orion. Found immediately: the first row through the resolver was a
-- smoke rating tagged `deploy-smoke`, and nothing downstream could have known.

ALTER TABLE substrate_action_ratings
    ADD COLUMN IF NOT EXISTS rated_by TEXT;

ALTER TABLE substrate_action_ratings
    ADD COLUMN IF NOT EXISTS rating_source TEXT;

-- The audit access pattern: "whose opinions is this belief made of".
CREATE INDEX IF NOT EXISTS substrate_action_ratings_rated_by_idx
    ON substrate_action_ratings (rated_by, rated_at DESC);
