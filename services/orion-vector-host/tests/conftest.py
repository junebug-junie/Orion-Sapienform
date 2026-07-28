from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
VECTOR_HOST_ROOT = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]

# Service root must come first so `app.*` resolves here.
if str(VECTOR_HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(VECTOR_HOST_ROOT))
# Repo root last so `orion.*` resolves from the repo (not overriding anything above).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Settings() requires these; tests never hit a real bus or embedding backend.
os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("VECTOR_HOST_EMBEDDING_MODEL", "test-embedding-model")
# Keep OrionTissue's snapshot writes out of /mnt/graphdb (not guaranteed to
# exist in a test sandbox) -- tissue_feed.py's module-level TISSUE singleton
# reads this at import time.
os.environ.setdefault(
    "ORION_TISSUE_SNAPSHOT_PATH",
    str(Path(tempfile.gettempdir()) / "orion-vector-host-test-tissue" / "tissue-brain.npy"),
)
