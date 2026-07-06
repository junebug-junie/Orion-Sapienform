from __future__ import annotations

import re

_CRYSTALLIZATION_ID = re.compile(r"^crys_[a-zA-Z0-9_-]+$")


def validate_crystallization_id(crystallization_id: str) -> str:
    """Reject IDs that would break parameterized or legacy Cypher construction."""
    cid = str(crystallization_id).strip()
    if not _CRYSTALLIZATION_ID.match(cid):
        raise ValueError(f"invalid_crystallization_id:{cid}")
    return cid
