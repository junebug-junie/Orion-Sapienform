"""A backend failure must not be recorded as Orion's reasoning.

2026-08-29: a ~45 minute circe outage produced 663 gateway timeouts, each returning
`[Error: llamacpp timed out after waiting]` in the result dict's `text`. Because
`orion-cortex-exec` falls back to `res.final_text` when no real reasoning trace
exists, those landed in `orion_metacognitive_trace` as `trace_role="reasoning"`,
`model="unknown"` -- 936 such rows going back to 2026-08-16.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from orion.cognition.cortex_payload_extract import looks_like_error_text

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text",
    [
        # The exact strings behind the 936 persisted rows.
        "[Error: llamacpp timed out after waiting]",
        "[Error: llamacpp failed: Connection refused]",
        "[Error: ollama timed out after waiting]",
        # Variants a narrower prefix-only check would have missed. This patch
        # originally shipped its own `"[Error: "` constant; review caught that as a
        # third copy of an existing canonical detector, and each of these is a case
        # the private version would have let through into `trace_role="reasoning"`.
        "[error: llamacpp timed out]",
        "[Error:llamacpp timed out]",
        "Traceback (most recent call last):\n  File ...",
        "Internal Server Error",
    ],
)
def test_gateway_failure_text_is_detected(text: str) -> None:
    assert looks_like_error_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        # Real reflective prose that mentions errors must survive -- this gate sits
        # on content that would otherwise be stored as Orion's reasoning.
        "the codebase is throwing errors I can't map yet",
        "I read the deploy log and found nothing conclusive.",
    ],
)
def test_real_content_is_not_flagged(text: str) -> None:
    assert not looks_like_error_text(text)


def test_cortex_exec_gates_on_the_canonical_detector() -> None:
    """Pins WHICH detector the gate uses.

    The regression this patch fixes is only closed if cortex-exec consults the
    canonical `looks_like_error_text`. A future edit swapping in a private
    prefix check would silently reopen the lowercase / no-space / traceback holes,
    so assert the import and the call site directly.
    """
    main_py = (REPO_ROOT / "services" / "orion-cortex-exec" / "app" / "main.py").read_text()
    assert (
        "from orion.cognition.cortex_payload_extract import looks_like_error_text" in main_py
    )
    assert "if reasoning_trace is None and looks_like_error_text(" in main_py


def test_gpu_cluster_power_settings_expose_service_version() -> None:
    """api.py reads settings.service_version on every heartbeat; it was never
    defined, so the tick raised AttributeError once per beat and the service
    published no heartbeat at all.

    Loaded by explicit file path under a unique module name rather than via
    `sys.path` + `from app.settings import ...`. Roughly 20 root tests bind a
    top-level `app` package to a *different* service, and Python resolves `app`
    from `sys.modules` first -- so the path-based version returned whichever
    service was imported earlier in the collection. Review confirmed that made this
    test fail beside `test_receipt_pruner.py` and pass VACUOUSLY beside
    `test_attention_runtime_store.py`, whose own Settings happens to declare
    `service_version` too. It also left a poisoned `sys.modules['app']` behind for
    later tests.
    """
    settings_path = (
        REPO_ROOT / "services" / "orion-gpu-cluster-power" / "app" / "settings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "orion_gpu_cluster_power_settings_under_test", settings_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # deliberately NOT registered in sys.modules

    assert "service_version" in module.Settings.model_fields
    field = module.Settings.model_fields["service_version"]
    assert field.alias == "SERVICE_VERSION"
    assert field.default, "needs a non-empty default; compose passes SERVICE_VERSION anyway"
