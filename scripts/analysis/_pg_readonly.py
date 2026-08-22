"""Shared read-only Postgres connection helper for scripts/analysis/'s measure_*.py probes.

Factored out 2026-08-11 (review fix on measure_goal_provenance_streak_distribution.py):
`open_readonly_connection` was already duplicated byte-for-byte across
measure_emergent_clustering_probe.py, measure_ast_hot_reducer.py, and
measure_capability_salience_coupling.py before this module existed -- adding a 4th copy
for the new streak-distribution probe would have made that four independent copies to keep
in sync by hand. This module stops the count growing further; it does not migrate the
three pre-existing copies (a separate, larger cleanup, out of scope for this patch).

Moved to `orion/db_readonly.py` 2026-08-19 (review finding on
`orion/metrics/liveness.py`, phase 5 of the metric semantic layer): that
module needed this exact contract but couldn't import from `scripts/`
without inverting this repo's layering, so it grew its own copy instead --
the same duplication this file exists to prevent, one layer down. This is
now a thin re-export so `measure_goal_provenance_streak_distribution.py`'s
existing `from _pg_readonly import open_readonly_connection` keeps working
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This module's own callers only put scripts/analysis/ on sys.path (see e.g.
# measure_goal_provenance_streak_distribution.py's own comment on why), never
# repo root -- add it here so the re-export below resolves regardless of how
# the caller was invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.db_readonly import open_readonly_connection  # noqa: E402, F401
