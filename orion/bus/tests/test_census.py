from __future__ import annotations

from orion.bus.census import compute_census


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
