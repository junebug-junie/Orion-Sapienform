import os
import sys
from types import ModuleType

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _purge_app_modules_if_wrong_service(expected_subdir: str) -> None:
    mod = sys.modules.get("app")
    loc = (getattr(mod, "__file__", "") or "").replace("\\", "/")
    if mod is not None and expected_subdir not in loc:
        for key in list(sys.modules):
            if key == "app" or key.startswith("app."):
                del sys.modules[key]


REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))


def _ensure_cortex_exec_paths() -> None:
    """Make the top-level ``app`` package resolve to orion-cortex-exec.

    In a multi-service pytest session another service's conftest (e.g.
    orion-hub's ``pytest_configure``) prepends its own root to ``sys.path`` and
    re-points ``app``. Re-assert this service so ``from app.settings import ...``
    below never imports the wrong service's Settings.
    """
    _purge_app_modules_if_wrong_service("/orion-cortex-exec/")
    # Insert SERVICE_DIR then REPO_ROOT so the final order is
    # [REPO_ROOT, SERVICE_DIR, ...] (matches historical priority) while both
    # sit ahead of any other service root a sibling conftest prepended.
    for path in (SERVICE_DIR, REPO_ROOT):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
        sys.path.insert(0, path)


_ensure_cortex_exec_paths()

os.environ.setdefault("SERVICE_NAME", "orion-cortex-exec")
os.environ.setdefault("SERVICE_VERSION", "0.2.0")
os.environ.setdefault("NODE_NAME", "athena")
os.environ.setdefault("ORION_BUS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("ORION_BUS_ENFORCE_CATALOG", "false")


def _stub_spacy_for_router_imports() -> None:
    """Avoid importing full spaCy when tests only need app.router (pulls autonomy -> concept_induction)."""
    if "spacy" in sys.modules:
        return

    def _load(_name: str) -> object:
        class _Nlp:
            pass

        return _Nlp()

    spacy_mod = ModuleType("spacy")
    spacy_mod.load = _load  # type: ignore[method-assign]
    sys.modules["spacy"] = spacy_mod

    lang_mod = ModuleType("spacy.language")

    class _Language:
        pass

    lang_mod.Language = _Language  # type: ignore[attr-defined]
    sys.modules["spacy.language"] = lang_mod


_stub_spacy_for_router_imports()


def pytest_sessionstart(session):
    """Service .env often sets large LLM_CHAT_* dev budgets; unit tests expect canonical caps.

    Runs after every collected conftest's ``pytest_configure``, so a sibling
    service may currently own the top-level ``app`` package. Re-assert this
    service and guard the import: in a shared session where cortex-exec tests are
    not actually collected, importing the wrong ``app.settings`` must not abort
    the run.
    """
    _ensure_cortex_exec_paths()
    try:
        from app.settings import settings
    except Exception:
        return
    # Confirm we actually imported cortex-exec's settings (not a sibling's that
    # still owns ``app.settings`` in sys.modules) before mutating it.
    settings_file = (getattr(sys.modules.get("app.settings"), "__file__", "") or "").replace("\\", "/")
    if "/orion-cortex-exec/" not in settings_file:
        return

    settings.llm_chat_quick_max_tokens = 384
    settings.llm_chat_general_max_tokens = 768
    settings.llm_chat_fallback_max_tokens = 512
    settings.chat_pcr_enabled = False
