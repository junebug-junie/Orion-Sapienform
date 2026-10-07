-- Orion's learned shed action (attend-to-act loop A1): one row per orion_self_shed request.
-- Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md ("Amendment 2026-09-29").
-- orion-gpu-pool also creates this lazily (CREATE TABLE IF NOT EXISTS, short lock_timeout); applying it
-- here first is the operator path. Additive; safe to re-run.
CREATE TABLE IF NOT EXISTS public.gpu_pool_orion_shed (
    shed_id text PRIMARY KEY,
    dispatch_id text NOT NULL,
    reason text NOT NULL DEFAULT 'orion_self_shed',
    state text NOT NULL,
    refusal text,
    ttl_sec double precision,
    requested_at timestamptz NOT NULL,
    started_at timestamptz,
    valid_until timestamptz,
    ended_at timestamptz,
    drained_at timestamptz,
    grants_withheld integer NOT NULL DEFAULT 0,
    delayed_grant_sec double precision NOT NULL DEFAULT 0,
    background_live_at_start integer NOT NULL DEFAULT 0,
    correlation jsonb NOT NULL DEFAULT '{}'::jsonb,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS gpu_pool_orion_shed_dispatch_idx ON public.gpu_pool_orion_shed (dispatch_id);
CREATE INDEX IF NOT EXISTS gpu_pool_orion_shed_requested_idx ON public.gpu_pool_orion_shed (requested_at DESC);
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orion_readonly') THEN
    GRANT SELECT ON public.gpu_pool_orion_shed TO orion_readonly;
  END IF;
END $$;
