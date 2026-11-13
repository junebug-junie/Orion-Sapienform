from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
SERVICE_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
