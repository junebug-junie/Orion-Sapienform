-- One-time backfill for the reverie theme_key format fix (fix/reverie-verdict-
-- aware-narration). chain.theme_key_for() used to prefix a loop-selected
-- theme_key as "loop:<id>" -- a format only that function ever wrote or read,
-- so it never matched attention_loops_store.suppress_loop()'s (orion-hub)
-- bare-id convention, silently breaking refractory suppression on a human's
-- Resolve/Dismiss action. The code now emits the bare id everywhere; this
-- backfill re-keys rows written before that deploy so in-flight refractory
-- suppression windows and resonance recurrence history survive the rollout
-- instead of being orphaned under the old prefixed key.
--
-- Idempotent: safe to run more than once (WHERE theme_key LIKE 'loop:%' matches
-- nothing on a second run). Apply once, any time at or after deploying
-- fix/reverie-verdict-aware-narration.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_reverie_theme_key_bare_id_backfill.sql

-- substrate_reverie_refractory: theme_key is the primary key, so re-key via
-- upsert-then-delete rather than a plain UPDATE, in case a post-deploy chain
-- already wrote a bare-id row for the same loop (keep whichever suppression
-- window extends furthest into the future -- the more protective one).
insert into substrate_reverie_refractory (theme_key, suppressed_until, updated_at)
select substring(theme_key from 6), suppressed_until, updated_at
from substrate_reverie_refractory
where theme_key like 'loop:%'
on conflict (theme_key) do update
    set suppressed_until = greatest(
            substrate_reverie_refractory.suppressed_until,
            excluded.suppressed_until
        ),
        updated_at = now();

delete from substrate_reverie_refractory where theme_key like 'loop:%';

-- substrate_reverie_chain: theme_key is a plain column (chain_id is the primary
-- key), so a direct rename is safe -- no uniqueness constraint to collide with.
update substrate_reverie_chain
set theme_key = substring(theme_key from 6)
where theme_key like 'loop:%';
