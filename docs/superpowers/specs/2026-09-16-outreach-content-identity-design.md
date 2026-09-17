# Outreach content identity — ledger before kinds

Status: implementing. Thin slice only. No motive taxonomy.

## Arsonist summary

Do not ship `outreach_reason.v1` with six kinds. That recreates the drives
failure: categories invented in chat before history shows what clusters exist.
This patch only makes the existing decision log record **which** prior /
curiosity candidate reached the prompt (IDs, not prose), so a later cluster
report can be derived from real sends. Cap-coupled "wanting" stays parked.

## Current architecture

- `grounding_summary()` already persists lane booleans/counts into
  `endogenous_outreach_decisions.result_json.grounding`.
- Provenance capsule (`outreach_provenance.v1`) stores the full prompt for
  "why I spoke" — still no durable ID list for novelty / cluster measurement.
- Talkable-content fire uses existence of live priors / curiosity / daydream,
  not novelty against prior sends.

## Proposed schema / API changes

Additive keys on existing `result_json.grounding` (no new table, channel, or
registry event):

- `prior_ids: string[]` — worldview `:Prior.prior_id` values that reached the
  prompt this tick (parallel to `priors_count`).
- `curiosity_content_ids: string[]` — content-stable keys for curiosity
  candidates whose summaries reached the prompt (source note + sorted focal
  refs, or a short hash of the evidence summary). **Not** reminted
  `FrontierInvocationSignalV1.signal_id` UUIDs.

No free-text claims, no `kind`, no `why`.

## Non-goals

- Six-kind (or any) outreach_reason taxonomy
- Novelty gate / `has_talkable_content` rewrite (second slice, after data)
- Daily-cap coupling to internal pressure
- Wanting / longing meters

## Acceptance checks

Owned by `scripts/analysis/measure_outreach_reason_clusters.py` (read-only):

1. Target monoculture: no `target_id` > 70% of sends / 7d
2. Cap pin: daily send count must not equal `DAILY_CAP` every day / 14d
3. Content gate falsifiability: `tension_without_content` > 0 over 14d (may
   still fail until novelty gate ships — report honestly)
4. Identity coverage: among sends with `priors_count > 0` or
   `curiosity_summaries > 0`, the matching ID lists must be present and
   length-aligned after this patch deploys
5. Repeat content: same prior_id or curiosity_content_id in two sends within
   7d (informational until novelty gate)

## Recommended next patch (after this)

Novelty gate: talkable content means content Orion has not already used,
using these IDs. Then re-run the measure script; only then consider deriving
kinds from clusters.
