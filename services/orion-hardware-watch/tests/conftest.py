from __future__ import annotations

import os

os.environ.setdefault("ORION_BUS_URL", "redis://127.0.0.1:6379/0")
os.environ.setdefault("POSTGRES_URI", "postgresql://unused")
