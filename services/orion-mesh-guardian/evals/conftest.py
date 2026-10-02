from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
MESH_GUARDIAN_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]

# Mesh-guardian service root must come first so `app.*` resolves here. Added alongside the
# heartbeat-chassis patch: previously this suite only collected when invoked with `cd
# services/orion-mesh-guardian && pytest tests` (or an explicit PYTHONPATH); this conftest
# brings it in line with every other service's `pytest services/<name>/tests -q` convention.
if str(MESH_GUARDIAN_ROOT) not in sys.path:
    sys.path.insert(0, str(MESH_GUARDIAN_ROOT))
# Repo root last so `orion.*` resolves from the repo (not overriding anything above).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
