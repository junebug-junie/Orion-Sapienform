import os
import sys
from pathlib import Path

os.environ.setdefault("ORION_BUS_URL", "redis://127.0.0.1:6379/0")
os.environ.setdefault("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/conjourney")

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
