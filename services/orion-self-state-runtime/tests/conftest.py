from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
SELF_STATE_RUNTIME_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]

# Self-state-runtime service root must come first so `app.*` resolves here.
if str(SELF_STATE_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SELF_STATE_RUNTIME_ROOT))
# Repo root last so `orion.*` resolves from the repo (not overriding anything above).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
