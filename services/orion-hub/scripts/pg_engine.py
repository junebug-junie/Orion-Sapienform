"""One cached SQLAlchemy engine for `POSTGRES_URI`, shared within a tick.

Not a repo-wide migration: `services/orion-hub/scripts/*.py` has ~20 other
near-identical inline `create_engine(os.getenv("POSTGRES_URI"), ...)` call
sites (a real, pre-existing pattern flagged by code review, out of scope for
this patch to fix wholesale -- that would be an unrelated, invasive change
across files this patch has no other reason to touch).

This module exists narrowly for the two call sites this patch itself adds to
the SAME outreach tick: `tension_outreach_trigger.current_run` and
`endogenous_outreach._fetch_recent_turns`. Before this module, each built (or,
in `tension_outreach_trigger`'s case, cached separately) its own connection
pool against the identical database -- two independent pool lifecycles doing
the same job on every single tick. Sharing one cached engine here removes
that duplication for this patch's own two call sites, and gives the next
orion-hub module a real shared factory to reuse instead of writing a 23rd
copy.
"""

from __future__ import annotations

import os
from typing import Any, Optional

_engine: Any = None


def get_engine() -> Optional[Any]:
    """Cached SQLAlchemy engine for `POSTGRES_URI`, or `None` if unset.

    Never disposed by callers -- it is meant to outlive any single call and
    be reused by every consumer of this module for the life of the process.
    """
    global _engine
    if _engine is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            return None
        from sqlalchemy import create_engine

        _engine = create_engine(uri, pool_pre_ping=True)
    return _engine
