-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Simulation isolation: review/apply explicitly before deploying the writer.
BEGIN;
CREATE TABLE IF NOT EXISTS town_sim_scene_resolved (
 id text PRIMARY KEY, world_id text NOT NULL, scene_id text NOT NULL,
 participants jsonb NOT NULL, purpose text NOT NULL, legal_results jsonb NOT NULL,
 chosen_results jsonb NOT NULL, effects_applied jsonb NOT NULL, seed bigint NOT NULL,
 status text NOT NULL, transcript jsonb NOT NULL, created_at timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS town_sim_prediction_score (
 id text PRIMARY KEY, world_id text NOT NULL, target_id text NOT NULL, purpose text NOT NULL,
 probabilities jsonb NOT NULL, outcome jsonb NOT NULL, brier double precision NOT NULL,
 brier_resample double precision, brier_base_rate double precision NOT NULL,
 baseline_count integer NOT NULL, resample_count integer NOT NULL,
 status text NOT NULL, signal_status text NOT NULL CHECK (signal_status='unverified'),
 created_at timestamptz DEFAULT now()
);
REVOKE ALL ON town_sim_scene_resolved, town_sim_prediction_score FROM PUBLIC;
DO $$ BEGIN
 IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='orion_readonly') THEN
   REVOKE ALL ON town_sim_scene_resolved, town_sim_prediction_score FROM orion_readonly;
 END IF;
END $$;
COMMIT;
