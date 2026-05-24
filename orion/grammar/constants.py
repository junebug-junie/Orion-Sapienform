"""Canonical layer and dimension enums for Substrate Atlas (spec §5.4–5.5)."""

from __future__ import annotations

GRAMMAR_LAYERS: tuple[str, ...] = (
    "raw_input",
    "sensor_raw",
    "sensor_semantic",
    "organ_signal",
    "semantic_interpretation",
    "memory",
    "stance",
    "reasoning",
    "action_candidate",
    "speak",
    "metacognitive",
)

GRAMMAR_DIMENSIONS: tuple[str, ...] = (
    "linguistic",
    "visual",
    "spatial",
    "temporal",
    "affective",
    "social",
    "epistemic",
    "intentional",
    "agency",
    "memory",
    "identity",
    "world",
    "action",
)
