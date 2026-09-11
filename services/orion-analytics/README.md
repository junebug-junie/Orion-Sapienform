# Orion analytics

This service adds a narrow, self-hosted analytics path over Orion's existing
PostgreSQL data:

```text
public.substrate_reverie_chain
  -> text staging/fact views ──────────────┐
public.reverie_visual_chain                ├─> shared outcome/date dimensions
  -> visual-chain staging/fact views       ├─> Lightdash OSS
public.reverie_visual_artifact             │
  -> visual-artifact staging/fact views ───┘
```

It does not add a second warehouse, copy operational data, or expose reverie
themes, prompts, captions, continuity text, storage paths, or narrative
payloads. The artifact content hash remains only as the fact primary key and is
hidden from Lightdash users. That `hidden` setting is a presentation boundary,
not removal from the analytics database; database readers can still query the
key. Text and visual Reverie remain separate facts because their grains,
schedules, and meanings differ.

## What exists in PostgreSQL

The first subject is `public.substrate_reverie_chain`, created by
`services/orion-sql-db/manual_migration_substrate_reverie_chain.sql`.
`services/orion-thought/app/chain.py::run_reverie_chain` produces a
`ReverieChainV1`; `services/orion-thought/app/store.py::persist_reverie_chain`
inserts it directly.

Final live inspection on 2026-09-10 found 25,060 rows, all with distinct non-null
`chain_id` values, spanning 2026-07-24 through 2026-09-10. Recorded stop reasons
were `no_coalition` (23,009), `max_steps` (1,484), and
`pressure_discharged` (567). `refractory` and `low_salience` are declared by the
schema but were not present in the live rows checked.

Mind activity was considered first, but `mind_runs` was not a useful initial
star: all 793 inspected rows had `trigger=user_turn`, `ok=true`, no error code,
and `router_profile_id=default`.

The parallel visual system uses `public.reverie_visual_chain` for persisted
chain attempts and `public.reverie_visual_artifact` for persisted generated
images. `services/orion-thought/app/visual_chain.py::_run_visual_chain_body`
generates, stores, observes, and builds the records;
`services/orion-thought/app/store.py::persist_reverie_visual_chain` and
`persist_reverie_visual_artifact` insert them. PostgreSQL enforces the artifact
`chain_id` foreign key.

Live inspection on 2026-09-10 found 1,442 distinct visual chains and 1,433
distinct artifacts. Nine chains had no artifact, 1,398 artifacts had a nonblank
observation caption, and nine persisted-chain gaps exceeded the live 45-minute
watchdog threshold. Of 1,257 rows with the producer's per-run continuity
marker, 940 proved that continuity was used; 185 legacy rows without that marker
remain unknown and are excluded from the rate. Artifact counts were exactly
zero or one per chain in that sample, but the models preserve the declared
one-to-many relationship.

## Models and grains

- `stg_reverie_chains`: one row per persisted source `chain_id`; selects only
  the key, UTC record-creation time/date, stop reason, and insertion time. The
  record timestamp is assigned after the chain loop terminates, just before
  persistence; it is not the chain's start time.
- `fct_reverie_chains`: one row per persisted reverie chain. Primary key
  `reverie_chain_id`; foreign keys `outcome_key` and `event_date`.
- `dim_reverie_outcomes`: one row per stop reason declared in
  `ReverieChainV1`.
- `dim_reverie_dates`: one row per UTC date from the first persisted chain
  through today.
- `stg_visual_reverie_chains`: one row per persisted visual chain; derives only
  a continuity-used flag from the producer's numeric `continuity_streak`
  marker and discards the raw JSON. Missing legacy markers remain null rather
  than being mislabeled.
- `stg_visual_reverie_artifacts`: one row per persisted visual artifact;
  derives only a caption-present flag and discards caption text and storage
  path. The source content hash is retained only as the fact primary key.
- `fct_visual_reverie_chains`: one row per persisted visual chain, with a
  pre-aggregated artifact count so zero-artifact chains remain visible without
  changing the grain.
- `fct_visual_reverie_artifacts`: one row per generated image that reached
  artifact persistence. Its source foreign key is tested against the visual
  chain fact, but the facts are not joined in Lightdash: that would invite
  cross-grain measures.

All are ordinary PostgreSQL views in the configured `analytics` schema. A
missing fact row means only that no chain reached persistence; it is not proof
that the service was down. Stop reasons are never relabeled as successes or
failures.

## Metric-quality gate: visual Reverie

The visual metrics were checked before wiring them into Lightdash:

1. **Provenance.** Chain and artifact counts come from the two insert functions
   named above. Artifact bytes comes from
   `StoredVisualArtifact.bytes`; caption coverage comes from nonblank
   `ReverieVisualArtifactV1.description`; continuity use comes from the
   producer's per-run `chain_json.continuity_streak > 0` marker. The raw JSON
   is not selected into a model. Late intervals compare adjacent persisted
   chain `created_at` values, and recency matches
   `store.py::visual_chain_age_minutes`.
2. **Independence.** Images-per-chain, missing-artifact count, caption coverage,
   continuity rate, and late-gap count are explicitly derived diagnostics, not
   independent cognitive signals. They are retained because they expose
   distinct generation, observation, continuity, and scheduling failure modes;
   they must not be interpreted as evidence of cognition quality.
3. **Theory anchor.** Counts and coverage use relational primary-key/foreign-key
   conservation. Caption coverage measures whether the generate → observe path
   produced its persisted observation marker. Continuity rate measures only
   whether the current run's prompt used prior-description continuity. The
   45-minute late threshold is the operational watchdog contract, not a learned
   cutoff.
4. **Live sanity.** The 2026-09-10 sample was nondegenerate: 1,442 chains,
   1,433 artifacts, 1,398 captioned artifacts, 940 continuity-using chains among
   1,257 rows with a known marker, nine zero-artifact chains, and nine late
   gaps. The 185 legacy rows without a marker stay unknown. Artifact bytes
   ranged from 111,195 to 2,140,299 (mean 884,069.24). Median persisted-chain
   gap was 2.53 minutes and p95 was 10.86 minutes.
5. **Existing mechanism.** The late threshold and latest-chain-age calculation
   reuse the visual-chain watchdog contract; no second liveness definition is
   introduced.
6. **Reversibility.** Every metric is a view/YAML definition over unchanged
   operational tables. Removing it requires no source migration or data rewrite.

The proposed “generation-to-artifact persistence delay” fails the provenance
gate and is intentionally absent. The producer stores neither generation-start
time nor artifact-persistence time: chain `created_at` is assigned after image
generation and observation, while artifact `created_at` is assigned before its
insert. Subtracting them would mislabel object-construction timing as generation
or persistence latency.

Likewise, the dashboard reports observed gaps longer than the watchdog
threshold, not an invented count of “missing runs.” The worker sleeps after a
run completes, so elapsed wall time cannot be divided by its 600-second sleep
interval to recover how many executions should have occurred.

## Lightdash surface

The `fct_reverie_chains` Explore exposes:

- dimensions: chain record time/date and terminal reason, plus labels and date
  attributes through explicit joins;
- measure: `Reverie chain count`, a distinct count of the source primary key;
- joins: fact to outcome and date are both declared `many-to-one`;
- default time dimension: `event_at` at day grain.

The content-as-code starter dashboard at `lightdash/dashboards/reverie-activity.yml`
contains daily activity, terminal-reason volume, and a date-spine query for
zero-row days. Uploading it requires a Lightdash user or service account token;
the token is not committed.

The unified `Reverie Overview` dashboard keeps text and visual sections
separate. Visual charts show persisted chains, persisted generated images,
caption coverage, 45-minute late intervals, and recorded terminal reasons.
The visual-chain Explore additionally exposes images per chain, chains without
an artifact, continuity-used rate, average artifact bytes, captioned image
count, and minutes since the latest selected persisted chain.

## Credentials and read boundaries

Do not reuse the PostgreSQL superuser or Orion's `orion_readonly` self-inquiry
role. The idempotent bootstrap script creates two separate principals:

- `orion_analytics_transformer`: `SELECT` on exactly
  `public.substrate_reverie_chain`, `public.reverie_visual_chain`, and
  `public.reverie_visual_artifact`, plus ownership of the `analytics` schema so
  dbt can create views there;
- `orion_analytics_reader`: `SELECT` on analytics views only, a read-only
  transaction default, and a 30-second statement timeout. This is the
  Lightdash warehouse credential.

Role creation changes database metadata, so review and run it as the operator:

```bash
python3 services/orion-analytics/scripts/init_local_env.py
set -a
. services/orion-analytics/.env
set +a

docker exec -i orion-athena-sql-db \
  psql -U postgres -d "$ORION_ANALYTICS_DATABASE" \
  -v analytics_transformer_password="$ORION_ANALYTICS_DBT_PASSWORD" \
  -v analytics_reader_password="$ORION_ANALYTICS_READER_PASSWORD" \
  < services/orion-analytics/scripts/bootstrap_analytics_roles.sql
```

The initializer creates the ignored `.env` from `.env_example` when needed and
fills all five required password/secret blanks with independent random values.
It does not replace existing nonblank values, does not print generated values,
is safe to run again, and sets the file mode to `0600`. The two generated
analytics role passwords still need to be applied with the reviewed role
bootstrap above before dbt or Lightdash connects to Orion's database.

The metadata password and S3 secret belong only to Lightdash's private
PostgreSQL and MinIO containers. Lightdash 2.184.6 requires S3-compatible
storage; the compose stack pins MinIO, creates the bucket idempotently, and
exposes its API on localhost port 8266 for browser-signed downloads. Lightdash
and dbt anonymous analytics are disabled. Do not expose either backing service
to a public interface.

`ORION_HOST_REPO_ROOT` anchors Compose bind mounts to the durable primary
checkout. For disposable local worktree validation only, point it at that
worktree in the ignored `.env`; do not leave production containers mounted to
a worktree that may be removed.

## Build and test the models

From the repository root, using the repo's Docker safety wrapper:

```bash
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt deps
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt parse
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt compile
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt run
scripts/safe_docker_build.sh orion-analytics --profile tools run --rm analytics-dbt test
```

Run the reconciliation after `dbt run` with the transformer role, because it
is the only analytics principal allowed to see both the selected source and
the mart. The SQL file explicitly opens a read-only transaction:

```bash
PGPASSWORD="$ORION_ANALYTICS_DBT_PASSWORD" psql \
  -h 127.0.0.1 -p 55432 -U orion_analytics_transformer -d conjourney \
  -f services/orion-analytics/evals/reconcile_reverie_analytics.sql

PGPASSWORD="$ORION_ANALYTICS_DBT_PASSWORD" psql \
  -h 127.0.0.1 -p 55432 -U orion_analytics_transformer -d conjourney \
  -f services/orion-analytics/evals/reconcile_visual_reverie_analytics.sql
```

Text success is `source_fact_delta=0`, `duplicate_fact_delta=0`, and
`join_fanout_delta=0`. Visual success requires every `*_delta` column to be
zero. Remaining columns report safe live coverage, recency, and late-gap
observations.

## Start Lightdash

```bash
scripts/safe_docker_build.sh orion-analytics --profile analytics up -d
scripts/safe_docker_build.sh orion-analytics --profile analytics ps
```

Open <http://localhost:8265>. On first launch:

1. Create the first admin account and organization.
2. Create a PostgreSQL project named `Orion analytics`.
3. Use host `orion-sql-db`, port `5432`, database `conjourney`, schema
   `analytics`, and the `orion_analytics_reader` credentials.
4. Set the dbt project path/subdirectory to `services/orion-analytics` when
   using a Git connection, or deploy the local project once with the pinned
   Lightdash CLI.
5. Confirm the `Reverie chains`, `Visual reverie chains`, and `Visual reverie
   artifacts` Explores contain the dimensions and measures above. Lightdash's
   query panel exposes the generated SQL.
6. Authenticate the CLI, deploy the semantic project, and upload the starter
   content:

```bash
npx --yes @lightdash/cli@2.184.6 login http://localhost:8265
cd services/orion-analytics
npx --yes @lightdash/cli@2.184.6 deploy --target prod --profiles-dir .
npx --yes @lightdash/cli@2.184.6 upload --force
```

The pinned CLI requires Node.js 24 or newer. The deploy command also needs
`dbt` 1.9.0 available in the same shell and the analytics connection/password
variables exported. The fixed transformer and reader role names are part of
the security policy and are not configurable. The Lightdash project keeps
the read-only warehouse credential for interactive Explore and SQL Runner
queries. Users with developer access can write and save SQL, but PostgreSQL
still refuses writes from that connection.

The compose default binds Lightdash and MinIO only to `127.0.0.1`. Keep that
default for first-admin setup. A remote deployment may set
`LIGHTDASH_BIND_ADDRESS` only after adding a trusted reverse proxy, HTTPS,
secure-cookie settings, and a matching externally reachable S3 public
endpoint.

## Agent boundary

A future agent-facing tool should accept a Lightdash metric identifier plus
dimensions, filters, and a bounded time range, then call Lightdash's compiled
metric-query API. It may also query the compiled `analytics` views with the
reader role for fixed workflows. It must not accept arbitrary cross-schema SQL
or receive transformer credentials. That extension needs authentication,
query-cost limits, result-size limits, metric allowlists, audit events, and an
evaluation proving the generated query uses the declared grain and join path.
No agent service or Model Context Protocol server is added here.

## Limitations and next subject

- Lightdash's first admin, project UUID, and API token are runtime state; the
  dashboard YAML cannot be uploaded until that human-owned setup exists.
- The live database role/bootstrap script is supplied but is not applied by
  deployment automatically.
- dbt views are current at query time; no volume evidence justified tables,
  incremental models, or materialized views.
- `ema_salience` is excluded until its producing signal passes a separate
  metric-quality review. `committed_proposal_id` was null in every inspected
  row. Theme keys and JSON are excluded for privacy.
- The operational 45-minute visual-chain watchdog setting is mirrored as a dbt
  variable. Change and deploy them together if the runtime contract changes.
- Next smallest useful subject: reverie-thought expectation verdicts, but only
  after checking live coverage of `expectation_verdict` and confirming that no
  narrative text needs to enter the mart.
