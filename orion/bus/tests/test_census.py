from __future__ import annotations

from pathlib import Path

from orion.bus.census import compute_census, load_channel_catalog_names, normalize_channel_name


def test_compute_census_finds_silent_and_undeclared_literal_channels() -> None:
    catalog = {"orion:system:health", "orion:journal:write"}
    active = {"orion:system:health": 1.0, "orion:mystery:channel": 0.5}

    result = compute_census(catalog, active)

    assert result.declared_silent == ["orion:journal:write"]
    assert result.undeclared_active == ["orion:mystery:channel"]


def test_compute_census_wildcard_prefix_matches_dynamic_reply_channels() -> None:
    catalog = {"orion:exec:result:*"}
    active = {
        "orion:exec:result:LLMGatewayService:0cf10589-4bae-4cb7-bc13-40684d7d321e": 0.1,
        "orion:exec:result:RecallService:9b3c5e84-0fda-4f78-9873-777814191e1a": 0.1,
    }

    result = compute_census(catalog, active)

    assert result.declared_silent == []
    assert result.undeclared_active == []


def test_compute_census_bare_star_wildcard_without_colon_matches() -> None:
    # channels.yaml has both "name:*" and "name*" wildcard forms.
    catalog = {"orion:cortex:result*"}
    active = {"orion:cortex:result:184992b1-7922-4616-8b26-f2f40e5d0c77": 0.2}

    result = compute_census(catalog, active)

    assert result.declared_silent == []
    assert result.undeclared_active == []


def test_compute_census_wildcard_with_zero_matches_is_declared_silent() -> None:
    catalog = {"orion:exec:result:*"}
    active = {"orion:unrelated:channel": 1.0}

    result = compute_census(catalog, active)

    assert result.declared_silent == ["orion:exec:result:*"]
    assert result.undeclared_active == ["orion:unrelated:channel"]


def test_compute_census_empty_active_marks_everything_silent() -> None:
    catalog = {"orion:a", "orion:b:*"}

    result = compute_census(catalog, {})

    assert result.declared_silent == ["orion:a", "orion:b:*"]
    assert result.undeclared_active == []


def test_compute_census_empty_catalog_marks_everything_undeclared() -> None:
    active = {"orion:x": 1.0, "orion:y": 2.0}

    result = compute_census(set(), active)

    assert result.declared_silent == []
    assert result.undeclared_active == ["orion:x", "orion:y"]


def test_compute_census_overlapping_wildcard_family_matches_narrowest_and_broadest() -> None:
    # Mirrors the real orion/bus/channels.yaml shape: one broad wildcard plus
    # several narrower per-service wildcards all sharing the same prefix
    # family, exactly what the live 0-undeclared_active mesh check depends on.
    catalog = {
        "orion:exec:result:*",
        "orion:exec:result:LLMGatewayService:*",
        "orion:exec:result:PadRpc:*",
        "orion:exec:result:StateService:*",
        "orion:exec:result:RecallService:*",
        "orion:exec:result:ContextExecService:*",
    }
    active = {
        "orion:exec:result:LLMGatewayService:0cf10589": 0.1,
        "orion:exec:result:RecallService:9b3c5e84": 0.1,
    }

    result = compute_census(catalog, active)

    assert result.undeclared_active == []
    # The two service-specific wildcards with no live traffic (PadRpc,
    # StateService, ContextExecService) are correctly silent; the broad
    # umbrella and the two matched service wildcards are not.
    assert result.declared_silent == [
        "orion:exec:result:ContextExecService:*",
        "orion:exec:result:PadRpc:*",
        "orion:exec:result:StateService:*",
    ]


def test_normalize_channel_name_collapses_a_dynamic_reply_channel() -> None:
    catalog = {"orion:exec:result:*"}
    channel = "orion:exec:result:LLMGatewayService:0cf10589-4bae-4cb7-bc13-40684d7d321e"

    assert normalize_channel_name(channel, catalog) == "orion:exec:result:*"


def test_normalize_channel_name_leaves_exact_catalog_channels_unchanged() -> None:
    catalog = {"orion:system:health", "orion:exec:result:*"}
    assert normalize_channel_name("orion:system:health", catalog) == "orion:system:health"


def test_normalize_channel_name_exact_match_short_circuits_a_wildcard_sibling() -> None:
    # Regression test for a real bug caught in review: channels.yaml has
    # standalone literal entries that share a prefix with a broader wildcard
    # sibling (e.g. "orion:exec:result:LLMGatewayService" alongside the
    # "orion:exec:result:*" umbrella). Before this fix, the literal entry was
    # incorrectly collapsed into the broad bucket -- exactly backwards, since
    # it's already a real, bounded, declared catalog channel that needs no
    # collapsing at all. Live-verified against the real channels.yaml: 8
    # literal entries hit exactly this case.
    catalog = {"orion:exec:result:LLMGatewayService", "orion:exec:result:*"}
    assert normalize_channel_name("orion:exec:result:LLMGatewayService", catalog) == "orion:exec:result:LLMGatewayService"


def test_normalize_channel_name_leaves_uncataloged_channels_unchanged() -> None:
    catalog = {"orion:exec:result:*"}
    assert normalize_channel_name("orion:totally:unrelated:channel", catalog) == "orion:totally:unrelated:channel"


def test_normalize_channel_name_picks_the_narrowest_matching_wildcard() -> None:
    # Mirrors the real orion/bus/channels.yaml shape: a broad umbrella plus a
    # narrower per-service wildcard both match the same live channel -- the
    # narrower one is more useful (distinguishes which service, not just
    # "some" reply channel).
    catalog = {"orion:exec:result:*", "orion:exec:result:LLMGatewayService:*"}
    channel = "orion:exec:result:LLMGatewayService:0cf10589"

    assert normalize_channel_name(channel, catalog) == "orion:exec:result:LLMGatewayService:*"


def test_normalize_channel_name_handles_bare_star_wildcards() -> None:
    catalog = {"orion:cortex:result*"}
    channel = "orion:cortex:result:184992b1-7922-4616-8b26-f2f40e5d0c77"

    assert normalize_channel_name(channel, catalog) == "orion:cortex:result*"


def test_normalize_channel_name_empty_catalog_returns_channel_unchanged() -> None:
    assert normalize_channel_name("orion:x", set()) == "orion:x"


def test_load_channel_catalog_names_reads_the_real_channels_yaml() -> None:
    # No path given -- must resolve orion/bus/channels.yaml relative to this
    # module's own location, not the test process's CWD.
    names = load_channel_catalog_names()

    assert len(names) > 100  # real catalog has 264 entries as of this arc
    assert "orion:system:health" in names


def test_load_channel_catalog_names_missing_file_returns_empty_set() -> None:
    assert load_channel_catalog_names(Path("/nonexistent/channels.yaml")) == set()


def test_load_channel_catalog_names_explicit_path_overrides_default(tmp_path) -> None:
    catalog_file = tmp_path / "custom_channels.yaml"
    catalog_file.write_text(
        "channels:\n  - name: orion:custom:channel\n  - name: orion:other:*\n",
        encoding="utf-8",
    )

    names = load_channel_catalog_names(catalog_file)

    assert names == {"orion:custom:channel", "orion:other:*"}
