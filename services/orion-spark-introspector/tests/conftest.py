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
# app.worker constructs a module-level OrionTissue() singleton at import
# time. Without this override it defaults to loading/writing
# /mnt/graphdb/orion/spark/tissue-brain.npz -- the live production snapshot
# actively written by the running orion-spark-introspector service. Tests
# that exercise handle_semantic_upsert (which calls TISSUE.propagate() and
# TISSUE.snapshot()) would otherwise read and clobber real production state.
# Force-set, not setdefault: services/orion-spark-introspector/.env_example
# (and a sourced .env) already sets this key to the production path, so
# setdefault would silently no-op under the exact "source .env before
# running pytest" workflow this repo's own tooling encourages.
os.environ["ORION_TISSUE_SNAPSHOT_PATH"] = "/tmp/orion-spark-introspector-test-tissue.npy"

# main.py mounts StaticFiles relative to CWD (app/static)
os.chdir(SERVICE_DIR)
