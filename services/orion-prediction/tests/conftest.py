from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ORION_BUS_URL", "redis://127.0.0.1:6379/0")
os.environ.setdefault("POSTGRES_URI", "postgresql://unused")

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

REGISTRY_PATH = REPO / "config" / "prediction" / "series.v1.yaml"
