import os
import sys
from pathlib import Path

# app.main reads settings at import; tests never touch a real bus or database.
os.environ.setdefault("ORION_BUS_URL", "redis://127.0.0.1:1/0")
os.environ.setdefault("POSTGRES_URI", "postgresql://unused@127.0.0.1:1/unused")
SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE.parents[1]), str(SERVICE)]
