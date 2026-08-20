"""Schema-level registry tests for the reverie visual chain (Patch 1,
docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md).

Pins two things:
- both schemas are registered in *both* registry dicts (CLAUDE.md §6:
  "Do not publish unregistered event shapes").
- `ReverieVisualChainV1.prior_description` really exists as a real field, not
  a name that only appears in prose -- the whole point of this schema is to
  not repeat the dead `SpontaneousThoughtV1.next_focus`/`drift` pattern
  (design doc §2).
"""

from __future__ import annotations

from orion.schemas.registry import SCHEMA_REGISTRY, _REGISTRY, resolve
from orion.schemas.reverie_visual import ReverieVisualArtifactV1, ReverieVisualChainV1

REVERIE_VISUAL_SCHEMAS = (
    "ReverieVisualChainV1",
    "ReverieVisualArtifactV1",
)


def test_reverie_visual_schemas_registered_in_both_registries() -> None:
    for schema_id in REVERIE_VISUAL_SCHEMAS:
        assert schema_id in _REGISTRY
        assert schema_id in SCHEMA_REGISTRY
        assert resolve(schema_id) is _REGISTRY[schema_id]
        assert SCHEMA_REGISTRY[schema_id].model is _REGISTRY[schema_id]


def test_reverie_visual_chain_kind_and_model_are_the_real_classes() -> None:
    assert resolve("ReverieVisualChainV1") is ReverieVisualChainV1
    assert SCHEMA_REGISTRY["ReverieVisualChainV1"].kind == "reverie.visual.chain.v1"

    assert resolve("ReverieVisualArtifactV1") is ReverieVisualArtifactV1
    assert SCHEMA_REGISTRY["ReverieVisualArtifactV1"].kind == "reverie.visual.artifact.v1"


def test_prior_description_is_a_real_field_not_prose() -> None:
    chain = ReverieVisualChainV1(
        chain_id="chain-1",
        terminal_reason="max_steps",
        prior_description="a quiet room, warm light",
    )
    assert chain.prior_description == "a quiet room, warm light"

    validated = ReverieVisualChainV1.model_validate(chain.model_dump(mode="json"))
    assert validated.prior_description == "a quiet room, warm light"


def test_reverie_visual_chain_round_trip() -> None:
    sample = ReverieVisualChainV1(
        chain_id="chain-42",
        theme_key="dusk",
        terminal_reason="pressure_discharged",
        ema_salience=0.6,
        prior_description="an empty hallway",
        chain_json={"steps": [1, 2, 3]},
    )
    validated = ReverieVisualChainV1.model_validate(sample.model_dump(mode="json"))
    assert validated.chain_id == "chain-42"
    assert validated.theme_key == "dusk"
    assert validated.terminal_reason == "pressure_discharged"
    assert validated.ema_salience == 0.6
    assert validated.prior_description == "an empty hallway"
    assert validated.chain_json == {"steps": [1, 2, 3]}


def test_reverie_visual_artifact_round_trip() -> None:
    sample = ReverieVisualArtifactV1(
        sha256="a" * 64,
        chain_id="chain-42",
        step_index=0,
        mime="image/png",
        bytes=12345,
        width=512,
        height=512,
        path="/mnt/storage-lukewarm/orion/reverie-visual/aaaa.png",
        description="a rendered hallway, dim light",
    )
    validated = ReverieVisualArtifactV1.model_validate(sample.model_dump(mode="json"))
    assert validated.sha256 == "a" * 64
    assert validated.chain_id == "chain-42"
    assert validated.step_index == 0
    assert validated.mime == "image/png"
    assert validated.bytes == 12345
    assert validated.width == 512
    assert validated.height == 512
    assert validated.description == "a rendered hallway, dim light"
