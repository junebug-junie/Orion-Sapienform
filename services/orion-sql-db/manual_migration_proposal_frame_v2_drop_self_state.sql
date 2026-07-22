-- 2026-07-22, SelfStateV1 burn (docs/superpowers/specs/2026-07-22-self-state-
-- phi-endo-origination-burn-spec.md). orion-proposal-runtime no longer reads
-- SelfStateV1 -- it builds ProposalFrameV1 directly from FieldStateV1 +
-- FieldAttentionFrameV1. New rows never populate source_self_state_id, so its
-- NOT NULL constraint must be relaxed or every future insert would fail.
--
-- Historical rows and the column itself are left in place -- this is a stop-
-- writing change, not a data-loss one. Per CLAUDE.md sec 13, an actual DROP
-- COLUMN/DROP TABLE is a separate, explicitly-gated action, not bundled here.

alter table substrate_proposal_frames
    alter column source_self_state_id drop not null;

alter table substrate_proposal_frames
    add column if not exists source_field_generated_at timestamptz;

-- No longer useful: nothing populates source_self_state_id going forward, so
-- an index on it stops earning its write-cost. Left creatable via the index
-- name still existing in the original migration's IF NOT EXISTS guard if
-- ever needed again; this only drops the currently-live one.
drop index if exists idx_substrate_proposal_frames_source_self_state;
