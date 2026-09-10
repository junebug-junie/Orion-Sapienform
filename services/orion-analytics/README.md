# Orion analytics

This service adds a narrow, self-hosted analytics path over Orion's existing
PostgreSQL data:

```text
public.substrate_reverie_chain
  -> dbt staging view
  -> reverie fact + outcome/date dimensions
  -> Lightdash OSS
```

It does not add a second warehouse, copy operational data, or expose reverie
themes and narrative payloads.

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

All are ordinary PostgreSQL views in the configured `analytics` schema. A
missing fact row means only that no chain reached persistence; it is not proof
that the service was down. Stop reasons are never relabeled as successes or
failures.

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

## Credentials and read boundaries

Do not reuse the PostgreSQL superuser or Orion's `orion_readonly` self-inquiry
role. The idempotent bootstrap script creates two separate principals:

- `orion_analytics_transformer`: `SELECT` on exactly
  `public.substrate_reverie_chain`, plus ownership of the `analytics` schema so
  dbt can create views there;
- `orion_analytics_reader`: `SELECT` on analytics views only, a read-only
  transaction default, and a 30-second statement timeout. This is the
  Lightdash warehouse credential.

Role creation changes database metadata, so review and run it as the operator:

```bash
export ORION_ANALYTICS_DBT_PASSWORD='generate-a-long-random-password'
export ORION_ANALYTICS_READER_PASSWORD='generate-another-long-random-password'

docker exec -i orion-athena-sql-db \
  psql -U postgres -d "$ORION_ANALYTICS_DATABASE" \
  -v analytics_transformer_password="$ORION_ANALYTICS_DBT_PASSWORD" \
  -v analytics_reader_password="$ORION_ANALYTICS_READER_PASSWORD" \
  < services/orion-analytics/scripts/bootstrap_analytics_roles.sql
```

Copy `.env_example` to the ignored `.env`, set those two passwords, and set
fresh values for `LIGHTDASH_METADATA_PASSWORD`, `LIGHTDASH_SECRET`, and
`LIGHTDASH_S3_SECRET_KEY`. The metadata password and S3 secret belong only to
Lightdash's private PostgreSQL and MinIO containers. Lightdash 2.184.6 requires
S3-compatible storage; the compose stack pins MinIO, creates the bucket
idempotently, and exposes its API on localhost port 8266 for browser-signed
downloads. Lightdash and dbt anonymous analytics are disabled. Do not expose
either backing service to a public interface.

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
```

Success is `source_fact_delta=0`, `duplicate_fact_delta=0`, and
`join_fanout_delta=0`. The final column reports the number of explicit
zero-activity dates in the date spine.

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
5. Confirm the `Reverie chains` Explore contains the dimensions and measure
   above. Lightdash's query panel exposes the generated SQL.
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
- Next smallest useful subject: reverie-thought expectation verdicts, but only
  after checking live coverage of `expectation_verdict` and confirming that no
  narrative text needs to enter the mart.
