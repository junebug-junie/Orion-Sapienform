import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _purge_app_modules_if_wrong_service(expected_subdir: str) -> None:
    mod = sys.modules.get("app")
    loc = (getattr(mod, "__file__", "") or "").replace("\\", "/")
    if mod is not None and expected_subdir not in loc:
        for key in list(sys.modules):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]


_purge_app_modules_if_wrong_service("/orion-spark-introspector/")

if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("SERVICE_NAME", "spark-introspector")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
