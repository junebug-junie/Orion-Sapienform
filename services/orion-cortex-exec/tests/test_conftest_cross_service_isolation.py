"""Regression: cortex-exec's ``pytest_sessionstart`` must not abort a shared
multi-service pytest session.

In a combined session orion-hub's ``pytest_configure`` prepends its own root to
``sys.path`` and re-points the top-level ``app`` package. If cortex-exec's
``pytest_sessionstart`` then does a bare ``from app.settings import settings`` it
imports orion-hub's ``Settings`` (which requires ``CHANNEL_VOICE_*`` keys absent
from the root ``.env``) and aborts the whole session with an INTERNALERROR before
any test runs. The conftest must re-assert its own service and guard the import.

Run in a clean subprocess so the in-process conftest state does not mask the bug.
Note: this asserts only the *no session-abort* property. Fully green combined
single-invocation runs are a separate, pre-existing cross-service ``scripts``/``app``
namespace concern that the plan sidesteps by running each service separately.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_combined_session_does_not_abort_on_sibling_settings() -> None:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "services/orion-hub/tests/test_attention_loops_api.py",
        "services/orion-cortex-exec/tests/test_attention_frame.py",
        "-q",
        "-p",
        "no:cacheprovider",
        "--co",  # collection only: enough to run pytest_sessionstart
    ]
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    tail = out[-3000:]
    assert "INTERNALERROR" not in out, tail
    assert "CHANNEL_VOICE_TRANSCRIPT" not in out, tail
