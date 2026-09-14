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


def test_visual_facts_keep_private_content_out_and_preserve_grains() -> None:
    chain_staging = (ROOT / "models/staging/stg_visual_reverie_chains.sql").read_text().lower()
    artifact_staging = (
        ROOT / "models/staging/stg_visual_reverie_artifacts.sql"
    ).read_text().lower()
    chain_fact = (ROOT / "models/marts/fct_visual_reverie_chains.sql").read_text().lower()
    artifact_fact = (
        ROOT / "models/marts/fct_visual_reverie_artifacts.sql"
    ).read_text().lower()

    assert "source('orion_operational', 'reverie_visual_chain')" in chain_staging
    assert "source('orion_operational', 'reverie_visual_artifact')" in artifact_staging
    assert "chain_json -> 'continuity_streak'" in chain_staging
    assert "prior_description" not in chain_staging
    assert "description is not null" in artifact_staging
    assert "group by visual_chain_id" in chain_fact

    semantic_models = _yaml("models/marts/marts.yml")["models"]
    chain_model = next(
        model for model in semantic_models if model["name"] == "fct_visual_reverie_chains"
    )
    artifact_model = next(
        model
        for model in semantic_models
        if model["name"] == "fct_visual_reverie_artifacts"
    )
    exposed_columns = {
        column["name"] for model in (chain_model, artifact_model) for column in model["columns"]
    }
    assert exposed_columns.isdisjoint(
        {"path", "description", "prompt", "prior_description", "theme_key", "chain_json"}
    )
    assert chain_model["meta"]["primary_key"] == "visual_chain_id"
    assert artifact_model["meta"]["primary_key"] == "visual_artifact_id"
    assert all(
        join["relationship"] == "many-to-one"
        for model in (chain_model, artifact_model)
        for join in model["meta"]["joins"]
    )


def test_visual_lightdash_measures_are_explicit_and_grain_safe() -> None:
    models = _yaml("models/marts/marts.yml")["models"]
    chain = next(model for model in models if model["name"] == "fct_visual_reverie_chains")
    artifact = next(
        model for model in models if model["name"] == "fct_visual_reverie_artifacts"
    )

    chain_metrics = {
        metric_name: metric
        for column in chain["columns"]
        for metric_name, metric in column.get("meta", {}).get("metrics", {}).items()
    }
    artifact_metrics = {
        metric_name: metric
        for column in artifact["columns"]
        for metric_name, metric in column.get("meta", {}).get("metrics", {}).items()
    }
    assert set(chain_metrics) == {
        "visual_chain_count",
        "minutes_since_last_visual_chain",
        "continuity_used_rate",
        "images_per_chain",
        "chains_without_artifact",
        "late_interval_count",
    }
    assert chain_metrics["visual_chain_count"]["type"] == "count_distinct"
    assert chain_metrics["images_per_chain"]["type"] == "average"
    assert chain_metrics["continuity_used_rate"]["format"] == "percent"
    assert "legacy rows" in chain_metrics["continuity_used_rate"]["description"].lower()
    assert set(artifact_metrics) == {
        "visual_artifact_count",
        "average_artifact_bytes",
        "captioned_artifact_count",
        "caption_coverage_rate",
    }
    assert artifact_metrics["visual_artifact_count"]["type"] == "count_distinct"
    assert artifact_metrics["caption_coverage_rate"]["format"] == "percent"
    assert "generation" not in " ".join(chain_metrics | artifact_metrics)

    derived_flags_test = (ROOT / "tests/assert_visual_reverie_derived_flags.sql").read_text()
    assert "continuity_used_flag is distinct from" in derived_flags_test
    assert "jsonb_typeof(chain_json -> 'continuity_streak') = 'number'" in derived_flags_test


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
    assert "public.reverie_visual_chain" in roles
    assert "public.reverie_visual_artifact" in roles
    assert "grant select on public.reverie_visual_chain to orion_analytics_reader" not in roles
    assert "grant select on public.reverie_visual_artifact to orion_analytics_reader" not in roles
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
    assert "sslmode: disable" in profile


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


def test_reverie_overview_keeps_text_and_visual_queries_separate() -> None:
    dashboard = _yaml("lightdash/dashboards/reverie-overview.yml")
    referenced = {
        tile["properties"]["chartSlug"]
        for tile in dashboard["tiles"]
        if "chartSlug" in tile["properties"]
    }
    assert referenced == {
        "reverie-activity-by-day",
        "reverie-chains-by-terminal-reason",
        "reverie-missing-days",
        "visual-reverie-chains-by-day",
        "visual-reverie-artifacts-by-day",
        "visual-reverie-caption-coverage",
        "visual-reverie-late-intervals",
        "visual-reverie-chains-by-terminal-reason",
    }

    for chart_path in (ROOT / "lightdash/charts").glob("visual-reverie-*.yml"):
        chart = _yaml(str(chart_path.relative_to(ROOT)))
        query = chart["metricQuery"]
        assert query["exploreName"] in {
            "fct_visual_reverie_chains",
            "fct_visual_reverie_artifacts",
        }
        assert "fct_reverie_chains" not in str(query)

    target_tables = {
        target["tableName"]
        for target in dashboard["filters"]["dimensions"][0]["tileTargets"].values()
    }
    assert target_tables == {
        "dim_reverie_dates",
        "fct_visual_reverie_chains",
        "fct_visual_reverie_artifacts",
    }


def test_curiosity_models_only_read_privacy_reduced_sources() -> None:
    model_paths = list((ROOT / "models/staging").glob("stg_curiosity*.sql")) + list(
        (ROOT / "models/marts").glob("fct_curiosity*.sql")
    )
    model_text = "\n".join(path.read_text().lower() for path in model_paths)
    assert "source('curiosity_safe'" in model_text
    for forbidden in (
        "journal_entries",
        "substrate_durable_run_state",
        "correlation_id",
        "finding_text",
        "reach_out_why",
        "self_definition",
        "checkpoint",
        "body",
        "prompt",
        "error_text",
    ):
        assert forbidden not in model_text

    roles = (ROOT / "scripts/bootstrap_analytics_roles.sql").read_text().lower()
    assert "with (security_barrier = true)" in roles
    assert roles.count("reverse(split_part(reverse(rtrim(body") == 3
    assert "regexp_match(body" not in roles
    assert "when grounding[1] is not null then 'other'" in roles
    assert "^((-> )?[a-za-z][a-za-z0-9_]{0,62}) ([0-9]+)$" in roles
    assert "^([a-z][a-z0-9_]{0,62}) ([0-9]+)$" in roles
    assert "grant select on all tables in schema analytics_source" in roles
    assert "grant select on public.journal_entries" not in roles
    assert "grant select on public.substrate_durable_run_state" not in roles
    assert "transformer_raw_curiosity_sources_are_denied" in roles
    assert "reader_curiosity_source_schema_is_denied" in roles


def test_curiosity_metrics_keep_run_and_child_grains_separate() -> None:
    models = _yaml("models/marts/curiosity.yml")["models"]
    by_name = {model["name"]: model for model in models}
    assert set(by_name) == {
        "fct_curiosity_runs",
        "fct_curiosity_run_transitions",
        "fct_curiosity_graph_writes",
        "fct_curiosity_material_pool",
    }
    assert by_name["fct_curiosity_runs"]["meta"]["primary_key"] == "run_id"
    assert by_name["fct_curiosity_run_transitions"]["meta"]["primary_key"] == "transition_id"
    assert by_name["fct_curiosity_graph_writes"]["meta"]["primary_key"] == "graph_write_id"
    assert by_name["fct_curiosity_material_pool"]["meta"]["primary_key"] == "material_pool_id"
    assert all("joins" not in model["meta"] for model in models)

    run = by_name["fct_curiosity_runs"]
    metrics = {
        metric_name: metric
        for column in run["columns"]
        for metric_name, metric in column.get("meta", {}).get("metrics", {}).items()
    }
    assert metrics["observed_run_count"]["type"] == "count_distinct"
    assert metrics["failed_transition_event_count"]["type"] == "sum"
    assert "not failed runs" in metrics["failed_transition_event_count"]["description"].lower()
    assert metrics["graph_write_coverage_rate"]["format"] == "percent"
    assert "configured timeout" in metrics["average_whole_turn_seconds"]["description"].lower()


def test_curiosity_dashboard_references_every_curiosity_chart() -> None:
    dashboard = _yaml("lightdash/dashboards/curiosity-operations.yml")
    referenced = {
        tile["properties"]["chartSlug"]
        for tile in dashboard["tiles"]
        if "chartSlug" in tile["properties"]
    }
    chart_slugs = {
        _yaml(str(path.relative_to(ROOT)))["slug"]
        for path in (ROOT / "lightdash/charts").glob("curiosity-*.yml")
    }
    assert referenced == chart_slugs
    assert len(referenced) == 8

    explore_names = {
        _yaml(str(path.relative_to(ROOT)))["metricQuery"]["exploreName"]
        for path in (ROOT / "lightdash/charts").glob("curiosity-*.yml")
    }
    assert explore_names == {
        "fct_curiosity_runs",
        "fct_curiosity_run_transitions",
        "fct_curiosity_graph_writes",
        "fct_curiosity_material_pool",
    }

    scope = next(
        tile for tile in dashboard["tiles"] if tile["tileSlug"] == "curiosity-operations-scope"
    )["properties"]["content"].lower()
    scope = " ".join(scope.split())
    assert "not a live process claim" in scope
    assert "does not invent historical budget-remaining" in scope
    assert "retained for 90 days by default" in scope
    assert "journal-only does not mean pre-durable" in scope


def test_curiosity_lifecycle_contract_discloses_source_retention() -> None:
    readme = (ROOT / "README.md").read_text().lower()
    marts = (ROOT / "models/marts/curiosity.yml").read_text().lower()
    assert "substrate_durable_run_state_retention_days=0" in readme
    assert "currently retained evidence" in readme
    assert "journal-only run may predate durable execution or may simply have" in readme
    assert "lifecycle evidence is limited to the source's currently retained window" in marts
    assert "journal-only does not imply a pre-durable run" in marts
