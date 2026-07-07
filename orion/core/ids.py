"""Dependency-free stable identifiers (hashlib only).

Use from thin services (orion-thought, orion.schemas) instead of importing
``orion.substrate.ids``, which loads the full substrate package graph stack.
"""

from __future__ import annotations

import hashlib


def stable_hash_id(prefix: str, parts: list[str]) -> str:
    """Return ``{prefix}_{sha256(preimage)[:24]}`` from ordered semantic parts."""
    normalized = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    preimage = "|".join(normalized)
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"
