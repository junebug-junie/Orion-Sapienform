"""looks_like_error_text -- promoted 2026-08-19 from
services/orion-hub/scripts/endogenous_outreach.py to this shared module, so
that orion/harness/finalize.py's extract_voice_finalize_text()/
extract_finalize_reflection_payload() (the SAME finalize chain every real
unified turn runs through) can use it too, not just outreach's own bare
cortex_client.chat() path. See this module's own comment for the two real
incidents this responds to.
"""
from __future__ import annotations

import pytest

from orion.cognition.cortex_payload_extract import looks_like_error_text


@pytest.mark.parametrize(
    "raw",
    [
        # The exact string that reached Juniper's chat thread on 2026-08-14.
        "[Error: llamacpp failed: Client error '400 Bad Request' for url "
        "'http://100.121.214.30:8013/v1/chat/completions']",
        # The exact string confirmed live, 2026-08-19, as orion_voice_
        # finalize's own final_text during a real circe-worker outage.
        "[Error: llamacpp timed out after waiting]",
        "Error: connection refused",
        "Traceback (most recent call last):\n  File ...",
        "Internal Server Error",
        "llamacpp failed: read timeout after 60s",
    ],
)
def test_error_shaped_text_is_recognised(raw) -> None:
    assert looks_like_error_text(raw) is True


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        # Orion's real first outreach -- must NOT be swallowed by the backstop.
        "The codebase is throwing errors I can't map yet. It's like trying to fix "
        "a machine without knowing which parts are broken.",
        "I keep hitting errors in the same place and I think that means something.",
        "There is something about the way a timeout feels from the inside.",
    ],
)
def test_real_prose_about_errors_is_not_swallowed(raw) -> None:
    assert looks_like_error_text(raw) is False
