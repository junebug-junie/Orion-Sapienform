import os
import sys

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest


@pytest.fixture(autouse=True)
def _reset_upstream_admission_between_tests():
    """`upstream_admission.py`'s admission gate is a process-global singleton,
    keyed by upstream URL. Until lane_contention.py (2026-09-24), only tests
    that deliberately exercised it ever touched it -- other test files freely
    reuse dummy/shared URLs across unrelated fixtures (harmless when nothing
    reads a global counter keyed by that string). `plan_llm_chat()` now reads
    that gauge for any metacog/quick request (see lane_contention.py), so
    without this, one test's inflight/semaphore state can leak into an
    unrelated test file that happens to reuse the same dummy URL, corrupting
    its counts or (worse) leaving a semaphore at a value a later test's
    `await ... acquire()` then blocks on for real. Reset before AND after
    every test regardless of which test caused the leak."""
    from app.upstream_admission import reset_upstream_admission_for_tests

    reset_upstream_admission_for_tests()
    try:
        from app.main import reset_executor_for_tests

        reset_executor_for_tests()
    except Exception:
        pass
    yield
    reset_upstream_admission_for_tests()
    try:
        from app.main import reset_executor_for_tests

        reset_executor_for_tests()
    except Exception:
        pass
