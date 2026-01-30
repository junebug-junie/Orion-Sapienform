-- Turn-effect time series + alert overlay (Postgres)
-- Notes:
-- - Expects chat_history_log.spark_meta JSONB column.
-- - Optional alert overlay uses collapse_mirror tags (metacog.alert.*), if available.
-- - If your alert events are not stored in SQL, omit the overlay CTE.

WITH base AS (
    SELECT
        created_at,
        correlation_id,
        session_id,
        spark_meta
    FROM chat_history_log
    WHERE created_at >= NOW() - INTERVAL '7 days'
),
alerts AS (
    SELECT
        correlation_id,
        MAX(
            CASE
                WHEN tags ILIKE '%metacog.alert.sev.error%' THEN 'error'
                WHEN tags ILIKE '%metacog.alert.sev.warn%' THEN 'warn'
                WHEN tags ILIKE '%metacog.alert.sev.info%' THEN 'info'
                ELSE NULL
            END
        ) AS alert_severity_max,
        STRING_AGG(DISTINCT tag, ',') AS alert_rules
    FROM (
        SELECT
            correlation_id,
            tags
        FROM collapse_mirror
        WHERE tags IS NOT NULL
    ) cm
    CROSS JOIN LATERAL (
        SELECT unnest(cm.tags) AS tag
    ) t
    WHERE t.tag LIKE 'metacog.alert.%'
    GROUP BY correlation_id
)
SELECT
    b.created_at,
    b.correlation_id,
    b.session_id,
    b.spark_meta->'metadata'->'spark_meta_rich'->'phi_before' AS phi_before,
    b.spark_meta->'metadata'->'spark_meta_rich'->'phi_after' AS phi_after,
    b.spark_meta->'metadata'->'spark_meta_rich'->'phi_post_before' AS phi_post_before,
    b.spark_meta->'metadata'->'spark_meta_rich'->'phi_post_after' AS phi_post_after,
    b.spark_meta->>'turn_effect_summary' AS turn_effect_summary,
    (b.spark_meta->'turn_effect'->'user'->>'coherence')::float AS delta_user_coherence,
    (b.spark_meta->'turn_effect'->'user'->>'valence')::float AS delta_user_valence,
    (b.spark_meta->'turn_effect'->'user'->>'energy')::float AS delta_user_energy,
    (b.spark_meta->'turn_effect'->'user'->>'novelty')::float AS delta_user_novelty,
    (b.spark_meta->'turn_effect'->'assistant'->>'coherence')::float AS delta_assistant_coherence,
    (b.spark_meta->'turn_effect'->'assistant'->>'valence')::float AS delta_assistant_valence,
    (b.spark_meta->'turn_effect'->'assistant'->>'energy')::float AS delta_assistant_energy,
    (b.spark_meta->'turn_effect'->'assistant'->>'novelty')::float AS delta_assistant_novelty,
    (b.spark_meta->'turn_effect'->'turn'->>'coherence')::float AS delta_turn_coherence,
    (b.spark_meta->'turn_effect'->'turn'->>'valence')::float AS delta_turn_valence,
    (b.spark_meta->'turn_effect'->'turn'->>'energy')::float AS delta_turn_energy,
    (b.spark_meta->'turn_effect'->'turn'->>'novelty')::float AS delta_turn_novelty,
    alerts.alert_severity_max,
    alerts.alert_rules
FROM base b
LEFT JOIN alerts
    ON alerts.correlation_id = b.correlation_id
ORDER BY b.created_at DESC;
