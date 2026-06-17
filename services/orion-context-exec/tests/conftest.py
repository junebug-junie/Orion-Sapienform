import os
import sys

import pytest

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def _sync_runner_settings_module() -> None:
    """Keep app.* settings singletons aligned after partial app module reloads."""

    def _sync() -> None:
        try:
            from app.settings import settings
            import app.runner as runner_mod

            runner_mod.settings = settings
            for mod_name in (
                "app.rlm_engine",
                "app.alexzhang_rlm_engine",
                "app.organ_runtime",
                "app.llm_profile_resolver",
                "app.workspace",
                "app.storage",
            ):
                mod = sys.modules.get(mod_name)
                if mod is not None and hasattr(mod, "settings"):
                    mod.settings = settings
        except ImportError:
            pass

    _sync()
    yield
    _sync()
