-- Durable self-inquiry question pool (orion/curiosity/self_question_pool.py).
-- Hub merges YAML seed with these rows and records each ask before kickoff.
--
-- Apply (from the host):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < scripts/sql/2026-09-18_curiosity_self_questions.sql

CREATE TABLE IF NOT EXISTS curiosity_self_questions (
  question_id text PRIMARY KEY,
  text text NOT NULL,
  family text NOT NULL CHECK (family IN ('lived', 'anatomy')),
  pinned boolean NOT NULL DEFAULT false,
  minted_by text NOT NULL CHECK (minted_by IN ('juniper', 'orion')),
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'answered', 'parked')),
  ask_count integer NOT NULL DEFAULT 0,
  last_asked_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
