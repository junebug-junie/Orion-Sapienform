from __future__ import annotations

import pytest

from app.substrate.routing import (
    ATOM_TYPE_OPERATOR_KIND,
    BOUNDARY_BULK_CUT,
    BOUNDARY_SITES,
    BULK_SITES,
    N_SITES,
    ORGAN_SITE_MAP,
    UnroutableAtomTypeError,
    UnroutableOrganError,
    route_atom,
)


def test_organ_site_map_matches_confirmed_live_producers() -> None:
    # Cross-referenced this session against orion/bus/channels.yaml's
    # orion:grammar:event producer list and
    # services/orion-substrate-runtime/app/worker.py's reducer cursor names.
    assert set(ORGAN_SITE_MAP.keys()) == {
        "orion-hub",
        "orion-biometrics",
        "orion-cortex-exec",
        "orion-bus",
        "orion-cortex-orch",
    }


def test_organ_sites_are_a_contiguous_boundary_block() -> None:
    assert sorted(ORGAN_SITE_MAP.values()) == list(BOUNDARY_SITES)
    assert set(BOUNDARY_SITES) | set(BULK_SITES) == set(range(N_SITES))
    assert set(BOUNDARY_SITES).isdisjoint(BULK_SITES)


def test_boundary_bulk_cut_is_the_seam_between_the_two_blocks() -> None:
    assert BOUNDARY_BULK_CUT == max(BOUNDARY_SITES) + 1
    assert BOUNDARY_BULK_CUT == min(BULK_SITES)


def test_every_grammar_atom_type_has_an_operator_kind() -> None:
    # orion/schemas/grammar.py's AtomType Literal, copied here as a change
    # detector -- if grammar.py gains a new atom type, this test should fail
    # loud rather than route_atom() silently defaulting somewhere.
    from orion.schemas.grammar import AtomType

    expected = set(AtomType.__args__)
    assert set(ATOM_TYPE_OPERATOR_KIND.keys()) == expected


def test_route_atom_happy_path() -> None:
    assignment = route_atom(
        source_service="orion-hub",
        atom_type="observation",
        confidence=0.9,
        salience=0.6,
        uncertainty=0.2,
    )
    assert assignment.site_index == ORGAN_SITE_MAP["orion-hub"]
    assert assignment.operator_kind == ATOM_TYPE_OPERATOR_KIND["observation"]
    assert assignment.confidence == pytest.approx(0.9)
    assert assignment.salience == pytest.approx(0.6)
    assert assignment.uncertainty == pytest.approx(0.2)


def test_route_atom_defaults_missing_fields() -> None:
    assignment = route_atom(
        source_service="orion-biometrics",
        atom_type="signal",
        confidence=None,
        salience=None,
        uncertainty=None,
    )
    assert assignment.confidence == 1.0
    assert assignment.salience == 0.5
    assert assignment.uncertainty == 0.5


def test_route_atom_rejects_unlisted_organ() -> None:
    with pytest.raises(UnroutableOrganError):
        route_atom(
            source_service="orion-vision-retina",
            atom_type="observation",
            confidence=1.0,
            salience=1.0,
            uncertainty=0.0,
        )


def test_route_atom_rejects_unknown_atom_type() -> None:
    with pytest.raises(UnroutableAtomTypeError):
        route_atom(
            source_service="orion-hub",
            atom_type="not_a_real_atom_type",
            confidence=1.0,
            salience=1.0,
            uncertainty=0.0,
        )
