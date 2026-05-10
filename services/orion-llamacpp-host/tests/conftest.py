import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("SERVICE_NAME", "orion-llamacpp-host")
os.environ.setdefault("SERVICE_VERSION", "0.1.0")
os.environ.setdefault("NODE_NAME", "llamacpp-node")
os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("ORION_BUS_ENFORCE_CATALOG", "false")
