import os
import sys

# The image runs orion-llamacpp-host's wrapper; test against that code.
SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
WRAPPER_DIR = os.path.join(REPO_ROOT, "services", "orion-llamacpp-host")
for path in (WRAPPER_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("SERVICE_NAME", "orion-llamacpp-bonsai-host")
os.environ.setdefault("SERVICE_VERSION", "0.1.0")
os.environ.setdefault("NODE_NAME", "llamacpp-node")
os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("ORION_BUS_ENFORCE_CATALOG", "false")
