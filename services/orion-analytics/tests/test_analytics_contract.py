from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str):
    return yaml.safe_load((ROOT / relative).read_text())


def test_fact_is_one_safe_source_row_with_no_narrative_fields() -> None:
    staging = (ROOT / "models/staging/stg_reverie_chains.sql").read_text().lower()
    fact = (ROOT / "models/marts/fct_reverie_chains.sql").read_text().lower()
    assert "source('orion_operational', 'substrate_reverie_chain')" in staging
    for private_field in (
        "theme_key",
        "chain_json",
        "ema_summary",
        "thought_ids",
        "committed_proposal_id",
        "interpretation",
    ):
        assert private_field not in staging
        assert private_field not in fact


def test_lightdash_measure_and_join_cardinality_are_explicit() -> None:
    models = _yaml("models/marts/marts.yml")["models"]
    fact = next(model for model in models if model["name"] == "fct_reverie_chains")
    assert fact["meta"]["primary_key"] == "reverie_chain_id"
    assert fact["meta"]["default_time_dimension"] == {"field": "event_at", "interval": "DAY"}
    assert {join["relationship"] for join in fact["meta"]["joins"]} == {"many-to-one"}
    chain_id = next(column for column in fact["columns"] if column["name"] == "reverie_chain_id")
    metric = chain_id["meta"]["metrics"]["reverie_chain_count"]
    assert metric["type"] == "count_distinct"


def test_lightdash_is_pinned_and_metadata_is_isolated() -> None:
    compose = _yaml("docker-compose.yml")
    services = compose["services"]
    assert services["analytics-dbt"]["image"] == "ghcr.io/dbt-labs/dbt-postgres:1.9.0"
    assert services["lightdash"]["image"] == "lightdash/lightdash:2.184.6"
    assert services["lightdash-db"]["image"] == "postgres:15.14-alpine"
    assert services["lightdash-object-storage"]["image"] == (
        "minio/minio:RELEASE.2025-09-07T16-13-09Z"
    )
    assert services["lightdash-object-storage-init"]["image"] == (
        "minio/mc:RELEASE.2025-08-13T08-35-41Z"
    )
    assert services["lightdash-db"]["networks"] == ["analytics-internal"]
    assert services["lightdash-object-storage"]["networks"] == ["analytics-storage"]
    assert services["lightdash"]["environment"]["S3_FORCE_PATH_STYLE"] == "true"
    assert services["lightdash"]["environment"]["RUDDERSTACK_ANALYTICS_DISABLED"] == "true"
    assert services["lightdash"]["ports"] == [
        "${LIGHTDASH_BIND_ADDRESS:-127.0.0.1}:${LIGHTDASH_PORT:-8265}:8080"
    ]
    assert services["analytics-dbt"]["volumes"][0].startswith(
        "${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}/"
    )
    assert services["lightdash"]["volumes"][0].startswith(
        "${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}/"
    )
    assert compose["networks"]["analytics-internal"]["internal"] is True


def test_reader_is_read_only_and_cannot_read_operational_source() -> None:
    roles = (ROOT / "scripts/bootstrap_analytics_roles.sql").read_text().lower()
    assert "nosuperuser nocreatedb nocreaterole noinherit noreplication nobypassrls" in roles
    assert "from pg_auth_members" in roles
    assert "revoke all privileges on all sequences in schema" in roles
    assert "revoke all privileges on schema" in roles
    assert "alter role orion_analytics_reader set default_transaction_read_only = on" in roles
    assert "grant select on all tables in schema analytics to orion_analytics_reader" in roles
    assert "grant select on public.substrate_reverie_chain to orion_analytics_reader" not in roles
    assert "raise exception 'analytics transformer can select an undeclared source relation'" in roles
    assert "raise exception 'analytics reader can select a non-analytics relation'" in roles
    assert "raise exception 'analytics reader can modify or owns a user relation'" in roles
    assert "raise exception 'analytics reader can create in or owns a user schema'" in roles

    reconciliation = (ROOT / "evals/reconcile_reverie_analytics.sql").read_text().lower()
    assert "begin transaction read only" in reconciliation
    assert reconciliation.rstrip().endswith("rollback;")


def test_schema_and_role_names_are_fixed_security_policy() -> None:
    env_example = (ROOT / ".env_example").read_text()
    profile = (ROOT / "profiles.yml").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "ORION_ANALYTICS_SCHEMA" not in env_example + profile + compose
    assert "ORION_ANALYTICS_DBT_USER" not in env_example + profile + compose
    assert "ORION_ANALYTICS_READER_USER" not in env_example + profile + compose
    assert "user: orion_analytics_transformer" in profile
    assert "schema: analytics" in profile


def test_dashboard_references_every_starter_chart() -> None:
    dashboard = _yaml("lightdash/dashboards/reverie-activity.yml")
    referenced = {
        tile["properties"]["chartSlug"]
        for tile in dashboard["tiles"]
        if "chartSlug" in tile["properties"]
    }
    assert referenced == {
        "reverie-activity-by-day",
        "reverie-chains-by-terminal-reason",
        "reverie-missing-days",
    }

    missing_days = _yaml("lightdash/charts/reverie-missing-days.yml")
    assert missing_days["tableName"] == "dim_reverie_dates"
    assert missing_days["metricQuery"]["metrics"] == [
        "fct_reverie_chains_reverie_chain_count"
    ]
    assert "sql" not in missing_days

    dashboard_filter = dashboard["filters"]["dimensions"][0]
    assert dashboard_filter["tileTargets"]["reverie-missing-days"] == {
        "fieldId": "dim_reverie_dates_date_key_day",
        "tableName": "dim_reverie_dates",
    }
