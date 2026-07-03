-- Hub social-room ops v1: align live Postgres with orion-social-memory models.
-- Run against the conjourney DSN used by orion-social-memory (see service README).

BEGIN;

-- social_participant_continuity (hygiene + artifact dialogue columns)
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS shared_artifact_proposal JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS shared_artifact_revision JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS shared_artifact_confirmation JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS calibration_signals JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS peer_calibration JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS trust_boundary JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS memory_freshness JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS decay_signals JSONB;
ALTER TABLE social_participant_continuity ADD COLUMN IF NOT EXISTS regrounding_decisions JSONB;

-- social_room_continuity (thread/deliberation/gif/hygiene columns)
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS shared_artifact_proposal JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS shared_artifact_revision JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS shared_artifact_confirmation JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS active_threads JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS current_thread_key TEXT;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS current_thread_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS handoff_signal JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS active_claims JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS recent_claim_revisions JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS claim_attributions JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS claim_consensus_states JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS claim_divergence_signals JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS bridge_summary JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS clarifying_question JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS deliberation_decision JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS turn_handoff JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS closure_signal JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS floor_decision JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS gif_usage_state JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS active_commitments JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS calibration_signals JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS peer_calibrations JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS trust_boundaries JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS memory_freshness JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS decay_signals JSONB;
ALTER TABLE social_room_continuity ADD COLUMN IF NOT EXISTS regrounding_decisions JSONB;

COMMIT;
