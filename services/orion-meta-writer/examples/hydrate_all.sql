-- Hydrate collapse_mirror with all enrichments
CREATE OR REPLACE VIEW collapse_with_all_enrichments AS
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
LEFT JOIN collapse_enrichment ce
       ON cm.id = ce.collapse_id;
