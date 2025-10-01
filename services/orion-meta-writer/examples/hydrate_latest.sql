-- Hydrate collapse_mirror with the most recent enrichment
CREATE OR REPLACE VIEW collapse_with_latest_enrichment AS
SELECT cm.*,
       ce.id          AS enrichment_id,
       ce.service_name,
       ce.service_version,
       ce.enrichment_type,
       ce.tags,
       ce.entities,
       ce.salience,
       ce.ts           AS enrichment_ts
FROM collapse_mirror cm
LEFT JOIN LATERAL (
    SELECT *
    FROM collapse_enrichment ce
    WHERE ce.collapse_id = cm.id
    ORDER BY ce.ts DESC
    LIMIT 1
) ce ON TRUE;
