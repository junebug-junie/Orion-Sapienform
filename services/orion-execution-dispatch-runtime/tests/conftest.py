from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
EXECUTION_DISPATCH_RUNTIME_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]

# Execution-dispatch-runtime service root must come first so `app.*` resolves here.
if str(EXECUTION_DISPATCH_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(EXECUTION_DISPATCH_RUNTIME_ROOT))
# Repo root last so `orion.*` resolves from the repo (not overriding anything above).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# The settings default is cwd-relative (the container runs from /app); pin it so
# a worker built in tests loads the repo's policy from any working directory.
os.environ.setdefault(
    "EXECUTION_DISPATCH_POLICY_PATH",
    str(REPO_ROOT / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"),
)
