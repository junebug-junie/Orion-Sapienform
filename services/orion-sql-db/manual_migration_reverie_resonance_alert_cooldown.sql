-- Resonance health-monitor alert cooldown (Phase H+ follow-up). Real dedupe
-- for ResonanceHealthMonitor's "worsening" pages: NotificationRequest.dedupe_key
-- / dedupe_window_seconds are accepted by orion-notify and stored, but nothing
-- there ever reads them back to suppress a repeat (confirmed by search; see
-- BusFallbackAlertState's docstring for the same finding on its own caller).
-- One row per check_key (e.g. "reverie_resonance_worsening:<theme_key>") holds
-- the last time a page actually went out; store.py's resonance_alert_cooldown_*
-- functions gate on it before calling attention_request again.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_reverie_resonance_alert_cooldown.sql

create table if not exists substrate_reverie_resonance_alert_cooldown (
    check_key text primary key,
    last_alerted_at timestamptz not null,
    updated_at timestamptz not null default now()
);

create index if not exists idx_substrate_reverie_resonance_alert_cooldown_last_alerted
    on substrate_reverie_resonance_alert_cooldown (last_alerted_at);
