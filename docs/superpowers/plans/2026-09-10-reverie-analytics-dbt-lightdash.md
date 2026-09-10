# Reverie analytics with dbt Core and Lightdash OSS

## Goal

Give Juniper a self-hosted, human-controlled way to explore completed reverie-chain activity without inventing SQL joins, grains, or denominators.

## Follow-up scope correction

The original recommendation to model expectation verdicts next omitted the
parallel diffusion/visual Reverie system. The corrected sequence is:

1. Text Reverie activity.
2. Visual/diffusion Reverie, as separate chain-grain and artifact-grain facts.
3. Reverie expectation scoring.
4. Cross-Reverie relationships only where a real key exists.
5. Curiosity analytics.
6. Governed agent access.

Text and visual facts may share the UTC date and terminal-reason dimensions and
appear on one dashboard, but they must not be unioned or joined into a
mixed-grain fact.

## Current architecture

- PostgreSQL 15 runs from `services/orion-sql-db/docker-compose.yml`; the operational database is `conjourney` and has no warehouse schema.
- `services/orion-thought/app/chain.py` creates one `ReverieChainV1` record after a chain loop terminates and `services/orion-thought/app/store.py::persist_reverie_chain` inserts it into `public.substrate_reverie_chain`.
- The live source had 25,060 rows and 25,060 distinct `chain_id` values at the final 2026-09-10 audit, spanning 2026-07-24 through 2026-09-10.
- Live terminal reasons were `no_coalition` (23,009), `max_steps` (1,484), and `pressure_discharged` (567). Declared but not yet observed reasons are `refractory` and `low_salience`.
- No dbt project, dimensional warehouse, Lightdash service, or other general semantic layer exists. Grafana is present only in a separate signal-gateway telemetry stack.
- The existing `orion_readonly` database role is intentionally scoped for Orion's self-inquiry and must not be broadened for analytics.

## Thin vertical slice

1. Declare `public.substrate_reverie_chain` as one dbt source and stage only non-narrative fields.
2. Build ordinary views:
   - `fct_reverie_chains`: one row per source `chain_id`.
   - `dim_reverie_outcomes`: one row per declared terminal reason.
   - `dim_reverie_dates`: one row per UTC date from the first observed chain through the current date.
3. Expose one governed Lightdash measure, distinct reverie-chain count, with explicit many-to-one joins from the fact to both dimensions and `event_at` as its default time dimension.
4. Ship content-as-code charts and a starter dashboard for daily volume, terminal reasons, and missing days.
5. Run Lightdash with its own PostgreSQL metadata container. Keep warehouse credentials separate and require a dedicated read-only Lightdash role.

## Metric quality gate

- **Provenance:** the measure counts `chain_id` produced in `run_reverie_chain`, persisted by `persist_reverie_chain`, and declared as the source table primary key by its migration.
- **Independence:** there is only one measure in this slice. Terminal reason and UTC date are dimensions of the same row, not additional signals.
- **Theory anchor:** this is an event-ledger count, not a latent cognition score. One unique persisted chain ID is one persisted chain record.
- **Live sanity:** the key is non-null and unique in all 25,060 live rows; counts vary by date and terminal reason. The measure has a genuine zero state on dates with no rows, represented by the date spine.
- **Existing mechanism:** repository search found local debug queries and a Hub reverie panel, but no governed metric or reusable dimensional model.
- **Reversibility:** all models are views in a dedicated schema; the service and schema can be removed without altering source rows.

`ema_salience`, proposal commitment, theme keys, thought IDs, and JSON summaries are not exposed. The first is a separate cognitive signal requiring its own quality review; the others are degenerate, identity-adjacent, or narrative-bearing in the live data.

## Files likely to touch

- New service: `services/orion-analytics/` for dbt, Lightdash Compose, tests, evals, role bootstrap, and operating docs.
- New PR report: `docs/superpowers/pr-reports/2026-09-10-reverie-analytics-dbt-lightdash-pr.md`.
- No source schemas, migrations, writers, bus contracts, or existing services change.

## Non-goals

- No production data rewrite, backfill, materialized view, incremental model, agent query service, or model of reverie meaning.
- No exposure of `theme_key`, summaries, thought content, session identity, or other narrative context.
- No generalized enterprise warehouse and no second analytical subsystem.

## Acceptance checks

- dbt dependency resolution, parse, compile, run, and test pass against an isolated PostgreSQL fixture.
- Source row count equals fact row count and distinct fact key count.
- Both fact-to-dimension joins preserve the fact row count.
- A date-spine query visibly returns zero-count days when gaps exist.
- Docker Compose renders, images are version-pinned, and both Lightdash metadata and warehouse-reader boundaries are explicit.
- Lightdash semantic metadata and content YAML pass static contract tests; live Lightdash compilation is either verified after first-user/project setup or reported `UNVERIFIED` with exact setup commands.

## Recommended next patch

Add the visual/diffusion slice before expectation scoring. Model
`reverie_visual_chain` and `reverie_visual_artifact` as separate facts, preserve
their one-to-many relationship with deterministic tests, and expose only
content-free operational measures. After that slice has been used, add
reverie-thought expectation outcomes only if their live verdict coverage is
non-degenerate and the privacy boundary can remain content-free.
