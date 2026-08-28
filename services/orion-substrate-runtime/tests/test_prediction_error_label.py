"""The substrate prediction-error node's label is read by a person.

It used to be ``f"substrate:{node_id}"`` -- i.e.
``substrate:node:substrate.harness_closure``, a non-empty string that is not a
label, just the id said twice. Confirmed live 2026-08-28: 10 of the 24 nodes
in the Concept Atlas rendered like that.
See docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md.
"""
from __future__ import annotations

import pytest

from app.worker import _prediction_error_label


@pytest.mark.parametrize(
    "node_id,expected",
    [
        ("node:substrate.harness_closure", "Harness closure prediction error"),
        ("node:substrate.biometrics", "Biometrics prediction error"),
        ("node:substrate.bus_synaptic", "Bus synaptic prediction error"),
        ("substrate.route", "Route prediction error"),
        ("node:vision", "Vision prediction error"),
    ],
)
def test_derives_a_readable_domain_name(node_id: str, expected: str) -> None:
    assert _prediction_error_label(node_id) == expected


def test_never_returns_the_old_id_shaped_label() -> None:
    for node_id in ("node:substrate.chat", "node:substrate.execution"):
        label = _prediction_error_label(node_id)
        assert not label.startswith("substrate:")
        assert node_id not in label


@pytest.mark.parametrize("node_id", ["", None, "   "])
def test_never_returns_an_empty_label(node_id) -> None:
    """ConceptNodeV1 requires a label; returning "" would fail validation and
    take down the write."""
    assert _prediction_error_label(node_id).strip()


def test_unrecognized_shape_falls_back_without_inventing_a_domain() -> None:
    assert _prediction_error_label("weird-id") == "Weird-id prediction error"
