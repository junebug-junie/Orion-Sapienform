-- 2026-07-22, SelfStateV1 burn (docs/superpowers/specs/2026-07-22-self-state-
-- phi-endo-origination-burn-spec.md). orion-policy-runtime no longer reads or
-- writes SelfStateV1 -- it builds PolicyDecisionFrameV1 directly from
-- ProposalFrameV1 (which already carries source_field_tick_id). New rows
-- never populate source_self_state_id, so its NOT NULL constraint must be
-- relaxed or every future insert would fail.
--
-- Historical rows and the column itself are left in place -- this is a stop-
-- writing change, not a data-loss one. Per CLAUDE.md sec 13, an actual DROP
-- COLUMN/DROP TABLE is a separate, explicitly-gated action, not bundled here.

alter table substrate_policy_decision_frames
    alter column source_self_state_id drop not null;

alter table substrate_policy_decision_frames
    add column if not exists source_field_tick_id text;

-- No longer useful: nothing populates source_self_state_id going forward, so
-- an index on it stops earning its write-cost.
drop index if exists idx_substrate_policy_decision_frames_source_self_state;

create index if not exists idx_substrate_policy_decision_frames_source_field_tick
    on substrate_policy_decision_frames (source_field_tick_id);
