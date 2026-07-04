import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

os.environ.setdefault("LLM_PROFILE_NAME", "test-profile")
