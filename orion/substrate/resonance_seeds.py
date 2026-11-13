"""One unscored concept-expansion seed per accepted reading resonance, bounded by retention."""
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1, UNSCORED_EVENT_NOTE
from orion.schemas.reading_resonance import ReadingResonanceResultV1

UNSEEDED_RESONANCE_SQL = """
SELECT r.result_json FROM reading_resonance r
WHERE r.completed_at >= now() - (:lookback_hours * interval '1 hour')
  AND r.result_json->'check'->>'accepted'='true'
  AND r.result_json->'check'->>'decision' IN ('share','note')
  AND coalesce(r.result_json->'judgment'->>'explore_next','') <> ''
  AND (jsonb_array_length(coalesce(r.result_json->'packet'->'landed_concept_ids','[]'::jsonb)) > 0
       OR EXISTS (SELECT 1 FROM jsonb_array_elements(r.result_json->'judgment'->'citations') citation
                  WHERE citation->>'ref_kind'='atlas_concept'))
  AND NOT EXISTS (
      SELECT 1 FROM substrate_endogenous_curiosity_candidates c
      WHERE c.generated_at >= r.completed_at - interval '1 hour'
        AND c.candidates_json @> CAST(:marker AS jsonb)
        AND EXISTS (SELECT 1 FROM jsonb_array_elements(c.candidates_json) s
          WHERE s->'notes' ? ('resonance:' || (r.result_json->>'resonance_id'))))
ORDER BY r.completed_at LIMIT 12
"""


def resonance_seeds(rows, *, cap=4):
    out, seen = [], set()
    for row in rows:
        result = ReadingResonanceResultV1.model_validate(row["result_json"])
        j = result.judgment
        if not result.check.accepted or not j or j.decision == "none" or not (j.explore_next or "").strip():
            continue
        anchors = sorted(set(result.packet.landed_concept_ids) | {c.ref_id for c in j.citations if c.ref_kind == "atlas_concept"})
        if not anchors or result.resonance_id in seen:
            continue
        seen.add(result.resonance_id)
        out.append(FrontierInvocationSignalV1(signal_type="curiosity_candidate", anchor_scope="orion",
            subject_ref="entity:orion", target_zone="concept_graph", task_type_candidate="concept_expand",
            focal_node_refs=anchors, signal_strength=0.0, confidence=0.0, evidence_summary=j.explore_next,
            notes=["endogenous_seed", "source:reading_resonance", UNSCORED_EVENT_NOTE, "resonance:" + result.resonance_id]))
        if len(out) >= min(4, max(0, cap)):
            break
    return out if cap > 0 else []
