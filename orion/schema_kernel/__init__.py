"""Schema kernel — universal grammar for the Orion substrate.

Atoms are reusable semantic invariants (not domain nouns).
Relations connect atoms with predicate + weight + polarity.
The registry pins the allowed vocabulary.

This module is intentionally small. No ontology trees, no inheritance, no
embeddings, no graph backends.
"""

from .atom import (
    ATOM_KINDS,
    DEFAULT_ATOMS,
    ConceptAtomV1,
)
from .relation import (
    DEFAULT_PREDICATES,
    ConceptRelationV1,
)
from .composite import (
    CompositeV1,
)
from .gradient import (
    DEFAULT_GRADIENT_KEYS,
    clamp_gradient,
    empty_gradient_vector,
)
from .registry import (
    SchemaKernelRegistry,
    default_registry,
)
from .validator import (
    SchemaValidationError,
    validate_atom,
    validate_composite,
    validate_relation,
)

__all__ = [
    "ATOM_KINDS",
    "DEFAULT_ATOMS",
    "ConceptAtomV1",
    "DEFAULT_PREDICATES",
    "ConceptRelationV1",
    "CompositeV1",
    "DEFAULT_GRADIENT_KEYS",
    "clamp_gradient",
    "empty_gradient_vector",
    "SchemaKernelRegistry",
    "default_registry",
    "SchemaValidationError",
    "validate_atom",
    "validate_composite",
    "validate_relation",
]
