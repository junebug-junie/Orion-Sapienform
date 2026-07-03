"""Isolate spark-introspector imports for cross-service contract tests."""
from __future__ import annotations

import os
import sys

SPARK_SERVICE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "services", "orion-spark-introspector")
)
REPO_ROOT = os.path.abspath(os.path.join(SPARK_SERVICE, "..", ".."))


def _purge_app_modules_if_wrong_service(expected_subdir: str) -> None:
    mod = sys.modules.get("app")
    loc = (getattr(mod, "__file__", "") or "").replace("\\", "/")
    if mod is not None and expected_subdir not in loc:
        for key in list(sys.modules):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]


_purge_app_modules_if_wrong_service("/orion-spark-introspector/")

if SPARK_SERVICE not in sys.path:
    sys.path.insert(0, SPARK_SERVICE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("SERVICE_NAME", "spark-introspector")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.chdir(SPARK_SERVICE)
