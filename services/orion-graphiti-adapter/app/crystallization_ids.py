from __future__ import annotations

import re

_CRYSTALLIZATION_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_crystallization_id(crystallization_id: str) -> str:
    """Reject IDs that would break parameterized or legacy Cypher construction.

    Real crystallization_id values are raw UUIDs (`memory_crystallizations.crystallization_id`),
    not `crys_`-prefixed strings (that prefix is only used for derived ids: chroma doc_id,
    graphiti entity_id `gent_<id>`, episode_id `gep_<id>`). This only guards against characters
    that would be unsafe if ever interpolated outside a parameterized query.
    """
    cid = str(crystallization_id).strip()
    if not _CRYSTALLIZATION_ID.match(cid):
        raise ValueError(f"invalid_crystallization_id:{cid}")
    return cid
