"""Relations — directed couplings between atoms or molecules.

Predicates are deliberately kept as free strings, but a small default vocabulary
is registered so cross-organ usage stays comparable.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_PREDICATES: Final[tuple[str, ...]] = (
    "supports",
    "contradicts",
    "depends_on",
    "elicits",
    "constrains",
    "amplifies",
    "decays",
    "references",
    "co_occurs_with",
    "transforms",
)


class ConceptRelationV1(BaseModel):
    """A directed edge between two loci.

    `source` and `target` are opaque keys — they may point at atom keys,
    molecule ids, or atom roles inside a molecule. The schema kernel does not
    resolve them; the substrate or caller does.

    `weight` is non-negative magnitude. `polarity` is in [-1, 1]; negative
    means inhibitory/dissonant, positive means supportive.
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    target: str = Field(min_length=1)
    weight: float = 1.0
    polarity: float = 0.0

    @field_validator("weight")
    @classmethod
    def _weight_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("weight must be non-negative")
        return value

    @field_validator("polarity")
    @classmethod
    def _polarity_bounded(cls, value: float) -> float:
        if value < -1.0 or value > 1.0:
            raise ValueError("polarity must be in [-1, 1]")
        return value
