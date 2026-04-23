import os
import sys
from types import ModuleType

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
