"""Hub's COMPUTE lane picks the model, and the lane's real window is the budget.

Covers the two halves of the 2026-09-03 finding that Mode=Agent + Compute=Agent
never reached the 27B: the model was chosen on a different axis than the
dropdown wrote to, and the context ceiling was a process-wide constant that
could not be right for two lanes at once.
"""

from __future__ import annotations

import pytest

from orion.fcc.context_budget import (
    context_risk_level,
    is_provider_error_envelope,
    max_context_chars,
    max_context_tokens,
)
from orion.harness.fcc_motor import label_to_claude_model_id
from orion.hub.turn_orchestrator import (
    DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL,
    _resolve_fcc_model_label,
)
from orion.llm.routes import fcc_model_for_route

# The real ~/.fcc/.env shape, as read live 2026-09-03.
FCC_ENV = {
    "MODEL": "llamacpp/harness",
    "MODEL_OPUS": "llamacpp/harness",
    "MODEL_SONNET": "llamacpp/harness",
    "MODEL_HAIKU": "nvidia_nim/z-ai/glm-5.2",
}


# --- Juniper's hard constraint: Mode=Orion + Compute=Chat must not move -------

def test_orion_mode_label_is_untouched_by_the_compute_lane() -> None:
    """The regression guard for "do not break mode Orion, compute chat".

    Hub's app.js always stamps an explicit fcc_model_label for Mode=Orion, so
    the explicit branch wins and NO compute lane -- not even one that resolves
    to a real model -- can redirect Orion mode.
    """
    for lane in ("chat", "agent", "quick", "metacog", None, "nonsense"):
        payload = {"fcc_model_label": "MODEL_SONNET", "llm_route": lane}
        assert _resolve_fcc_model_label(payload, "orion") == "MODEL_SONNET"


def test_orion_mode_default_is_unchanged_when_hub_sends_no_label() -> None:
    assert (
        _resolve_fcc_model_label({"llm_route": "chat"}, "orion")
        == DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL
    )


def test_compute_default_quick_cannot_demote_orion_mode() -> None:
    """COMPUTE defaults to `quick` (an 8B). If the lane steered every mode, a
    fresh page load would silently drop Orion's primary lane onto it."""
    payload = {"fcc_model_label": "MODEL_SONNET", "llm_route": "quick"}
    assert _resolve_fcc_model_label(payload, "orion") == "MODEL_SONNET"


# --- the ask: Mode=Agent + Compute=Agent runs the 27B ------------------------

def test_agent_mode_with_agent_lane_selects_the_agent_route() -> None:
    label = _resolve_fcc_model_label({"llm_route": "agent"}, "agent")
    assert label == "llamacpp/agent"
    # ...and that label reaches `claude --model` as-is, rather than missing the
    # env lookup and silently falling back to MODEL (= the harness lane).
    assert label_to_claude_model_id(label, FCC_ENV) == "llamacpp/agent"


@pytest.mark.parametrize(
    "lane,expected",
    [("agent", "llamacpp/agent"), ("chat", "llamacpp/chat"), ("chat_quick", "llamacpp/quick")],
)
def test_agent_mode_follows_whichever_lane_is_selected(lane: str, expected: str) -> None:
    assert _resolve_fcc_model_label({"llm_route": lane}, "agent") == expected


@pytest.mark.parametrize("lane", ["harness", "nonsense", "", None])
def test_agent_mode_falls_back_rather_than_guessing(lane: object) -> None:
    """`harness` is SYSTEM_LLM_ROUTES -- never a human's Compute choice -- and an
    unrecognised name means "no override". Both keep today's default."""
    assert (
        _resolve_fcc_model_label({"llm_route": lane}, "agent")
        == DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL
    )
    assert fcc_model_for_route(lane) is None


# --- label resolution: the silent-wrong-model trap ---------------------------

def test_env_key_labels_still_resolve_exactly_as_before() -> None:
    assert label_to_claude_model_id("MODEL_SONNET", FCC_ENV) == "llamacpp/harness"
    assert label_to_claude_model_id("MODEL_HAIKU", FCC_ENV) == "nvidia_nim/z-ai/glm-5.2"


def test_a_route_spec_is_never_resolved_through_the_MODEL_fallback() -> None:
    """The bug this ordering exists to stop: `env.get("llamacpp/agent")` misses,
    and `or env.get("MODEL")` would then serve the harness lane's 35B while
    reporting success. Pinned by asserting the *distinguishing* value -- if the
    fallback fired, this returns "llamacpp/harness"."""
    assert label_to_claude_model_id("llamacpp/agent", FCC_ENV) != FCC_ENV["MODEL"]
    assert label_to_claude_model_id("llamacpp/agent", FCC_ENV) == "llamacpp/agent"


def test_unknown_bare_label_still_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unchanged behaviour for non-route labels, deliberately: narrowing that is
    a separate decision with its own blast radius."""
    assert label_to_claude_model_id("MODEL_NOT_SET", FCC_ENV) == "llamacpp/harness"


# --- the ceiling follows the lane -------------------------------------------

def test_live_window_beats_the_process_env_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", "131072")
    assert max_context_tokens() == 131072
    assert max_context_tokens(32768) == 32768, "the agent lane's real window must win"


@pytest.mark.parametrize("bad", [None, 0, -1, True, "32768"])
def test_unknown_window_falls_back_never_to_unlimited(
    monkeypatch: pytest.MonkeyPatch, bad: object
) -> None:
    """None/absent means "not known" (older gateway, worker down), and every
    non-positive or non-int shape is treated the same way."""
    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", "131072")
    assert max_context_tokens(bad) == 131072  # type: ignore[arg-type]


def test_warn_is_reachable_below_critical_on_a_small_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second-order bug: `context_risk_level` took the ceiling from its
    caller but the warn line from the env. On a 32768-token lane under a
    131072 env default, warn_at (~91750 tokens) sat ABOVE critical (32768), so
    a turn's first signal was `critical` -- after the overrun, not before it.
    """
    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", "131072")
    monkeypatch.setenv("HARNESS_FCC_CONTEXT_PRESSURE_PCT", "70")
    lane_chars = max_context_chars(32768)

    # 50% of the lane: quiet.
    assert context_risk_level(accumulated_chars=int(lane_chars * 0.5), max_chars=lane_chars) == "ok"
    # 80% of the lane: this is the nudge that has to fire while there is still
    # room to answer. Under the old code this was still "ok".
    assert context_risk_level(accumulated_chars=int(lane_chars * 0.8), max_chars=lane_chars) == "warn"
    # Over the lane: critical.
    assert context_risk_level(accumulated_chars=lane_chars + 1, max_chars=lane_chars) == "critical"


# --- a laundered provider error is a failure, not an answer ------------------

OVERFLOW_REPLY = (
    "Upstream provider LLAMACPP returned HTTP 400.\n"
    "Category: exceed_context_size_error\n"
    "Mapped message: Invalid request sent to provider.\n\n"
    "Upstream error:\n"
    '{"error":{"code":400,"message":"request (40056 tokens) exceeds the available '
    'context size (32768 tokens), try increasing it","type":"exceed_context_size_error"}}'
)


def test_the_real_captured_overflow_reply_is_recognised_as_an_error() -> None:
    """Verbatim from a live 2026-09-03 probe against circe:8015, which returned
    it with HTTP 200 and stop_reason "end_turn"."""
    assert is_provider_error_envelope(OVERFLOW_REPLY) is True


def test_prose_about_an_overflow_is_not_destroyed() -> None:
    """Orion introspects on its own infrastructure, so an answer *discussing* a
    context overflow is a real turn and must survive. This is why the check is
    on the envelope's framing, not on the presence of the error words."""
    assert (
        is_provider_error_envelope(
            "I hit exceed_context_size_error earlier; the agent lane only holds "
            "32768 tokens, so I trimmed the file reads before retrying."
        )
        is False
    )


@pytest.mark.parametrize("text", ["", "   ", None, "Here is the answer you asked for."])
def test_ordinary_replies_are_not_flagged(text: object) -> None:
    assert is_provider_error_envelope(text) is False  # type: ignore[arg-type]
