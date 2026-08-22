"""Exists because none of the other tests import app.main -- a syntax error
inside model_runtime.py's deferred-import block (a real one: `from X import
*` is invalid inside a function body, caught only by a manual py_compile
sweep, not by pytest, since nothing else in this suite imports app.main)
went undetected until that sweep. This closes that gap cheaply: importing
app.main pulls in model_runtime.py, settings.py, and face_extract.py without
needing a GPU, model weights, or a live bus (FastAPI app construction alone
doesn't connect to anything -- that happens in the lifespan/start()).
"""
from __future__ import annotations


def test_app_module_imports_cleanly():
    import app.main as main_module

    assert main_module.app is not None
    assert main_module.settings.SERVICE_NAME == "affectgpt-worker"


def test_model_runtime_module_imports_cleanly():
    import app.model_runtime as model_runtime_module

    assert model_runtime_module.AffectGptRuntime is not None
