-- 2026-07-22, SelfStateV1 burn (docs/superpowers/specs/2026-07-22-self-state-
-- phi-endo-origination-burn-spec.md). orion-feedback-runtime no longer reads
-- or writes SelfStateV1 -- it builds FeedbackFrameV1 directly from
-- FieldStateV1 snapshots. source_self_state_id was already nullable, so no
-- constraint change is needed here; this just adds the replacement column.
--
-- Historical rows and the column itself are left in place -- this is a stop-
-- writing change, not a data-loss one. Per CLAUDE.md sec 13, an actual DROP
-- COLUMN/DROP TABLE is a separate, explicitly-gated action, not bundled here.

alter table substrate_feedback_frames
    add column if not exists source_field_tick_id text;
