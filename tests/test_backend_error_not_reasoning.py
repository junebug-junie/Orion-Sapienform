"""A backend failure must not be recorded as Orion's reasoning.

2026-08-29: a ~45 minute circe outage produced 663 gateway timeouts, each returning
`[Error: llamacpp timed out after waiting]` in the result dict's `text`. Because
`orion-cortex-exec` falls back to `res.final_text` when no real reasoning trace
exists, those landed in `orion_metacognitive_trace` as `trace_role="reasoning"`,
`model="unknown"` -- 936 such rows going back to 2026-08-16.
"""

from __future__ import annotations

import pytest

from orion.llm.backend_errors import BACKEND_ERROR_PREFIX, is_backend_error_text


@pytest.mark.parametrize(
    "text",
    [
        "[Error: llamacpp timed out after waiting]",
        "[Error: ollama timed out after waiting]",
        "[Error: llamacpp failed: Connection refused]",
        "  [Error: llamacpp timed out after waiting]",  # callers strip inconsistently
        "\n[Error: x]",
    ],
)
def test_gateway_failure_text_is_detected(text: str) -> None:
    assert is_backend_error_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "The deploy threw [Error: boom] and I retried.",  # sentinel is a PREFIX only
        "I could not find an error in the logs.",
        "Error: something",  # no bracket -- not the gateway's format
    ],
)
def test_real_content_is_not_flagged(text: str | None) -> None:
    assert not is_backend_error_text(text)


def test_prefix_matches_what_the_gateway_actually_emits() -> None:
    """Pins the shared constant against the gateway's own format string.

    The constant only helps if producer and detector cannot drift; this reads the
    real source rather than trusting that they still agree.
    """
    from pathlib import Path

    backend = (
        Path(__file__).resolve().parents[1]
        / "services"
        / "orion-llm-gateway"
        / "app"
        / "llm_backend.py"
    ).read_text()
    assert 'f"{BACKEND_ERROR_PREFIX}{backend_name} timed out after waiting]"' in backend
    assert 'f"{BACKEND_ERROR_PREFIX}{backend_name} failed: {str(e)}]"' in backend
    # And the constant still produces the exact historical string, so the 936
    # already-persisted rows remain matchable by the same predicate.
    assert f"{BACKEND_ERROR_PREFIX}llamacpp timed out after waiting]" == (
        "[Error: llamacpp timed out after waiting]"
    )


def test_gpu_cluster_power_settings_expose_service_version() -> None:
    """api.py reads settings.service_version on every heartbeat; it was never
    defined, so the tick raised AttributeError once per beat and the service
    published no heartbeat at all."""
    import sys
    from pathlib import Path

    svc = Path(__file__).resolve().parents[1] / "services" / "orion-gpu-cluster-power"
    sys.path.insert(0, str(svc))
    try:
        from app.settings import Settings  # type: ignore[import-not-found]

        assert "service_version" in Settings.model_fields
        assert Settings().service_version  # non-empty default
    finally:
        sys.path.remove(str(svc))
