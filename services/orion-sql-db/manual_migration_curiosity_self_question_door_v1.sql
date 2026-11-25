-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- The self-question door (2026-11-25, docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md).
-- Every question Orion writes for its own self-inquiry (a sitting's :SelfQuestionMint, or the
-- self-monitor's "Why?" question) is written door = 'held' and is never picked while held. Hub's
-- self_question_door loop judges it with the finding-register judge (prompt v2, PASSED_JUDGE_MODELS):
--
--   door                 NULL  = a Juniper row, or an Orion row written before the door
--                        'held'      waiting for the judge (also: a judge error, retried)
--                        'admitted'  philosophical: enters the self-inquiry pool as before
--                        'routed'    operational: never asked as self-inquiry; offered to the
--                                    investigate line as a self focus request when the slot is free
--                        'demoted'   was admitted, but the sitting that answered it wrote an
--                                    operational finding (door_demoted_by_run names that run)
--                        'unjudged'  no label after door_attempts failed judgements; never admitted
--                                    or routed by default
--   door_reason          the judge's one-sentence reason, or the last error
--   door_judged_at       when the door last judged (or tried to)
--   door_judge_model     the served model that answered
--   door_prompt_sha256   the judge prompt's sha (register.JUDGE_PROMPT_SHA256)
--   door_attempts        failed judgements so far (gateway refusals are not counted)
--   door_demoted_by_run  the curiosity run whose operational finding demoted the question
--
-- No CHECK on door: a CHECK added to this live table would need NOT VALID plus a later VALIDATE, and the
-- writer (Hub, orion/curiosity/self_question_pool.py DOOR_VALUES) is the only one. The status CHECK is not
-- touched (changing it is not additive).
-- Writer: orion-hub only. Readers: Hub; orion_readonly already has SELECT on the table.
-- Safe for the post-pull auto-apply: ADD COLUMN only; nullable, or one constant default.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql
-- Rollback (additive, by hand, reviewed): ALTER TABLE curiosity_self_questions DROP COLUMN door, DROP COLUMN door_reason,
--   DROP COLUMN door_judged_at, DROP COLUMN door_judge_model, DROP COLUMN door_prompt_sha256, DROP COLUMN door_attempts,
--   DROP COLUMN door_demoted_by_run;

ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door text;
ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door_reason text;
ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door_judged_at timestamptz;
ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door_judge_model text;
ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door_prompt_sha256 text;
ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door_attempts integer NOT NULL DEFAULT 0;
ALTER TABLE curiosity_self_questions ADD COLUMN IF NOT EXISTS door_demoted_by_run text;
