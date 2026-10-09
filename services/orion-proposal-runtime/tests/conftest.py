from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
SERVICE_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]

# Service root first so `app.*` resolves to proposal-runtime; repo root for `orion.*`.
for key in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
    del sys.modules[key]
if str(SERVICE_ROOT) in sys.path:
    sys.path.remove(str(SERVICE_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
# app.settings needs these at import; tests never touch a real database or bus.
os.environ.setdefault("POSTGRES_URI", "postgresql://unused@127.0.0.1:1/unused")
