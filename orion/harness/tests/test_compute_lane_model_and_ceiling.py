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

    # Both guards are independently sufficient, so assert them SEPARATELY --
    # review caught that the cases above all return at the explicit-label branch
    # and would still pass with the `mode_tag == "agent"` gate deleted. This is
    # the one that pins the gate: no explicit label, a lane that DOES resolve,
    # and Orion mode must still not follow it.
    assert (
        _resolve_fcc_model_label({"llm_route": "agent"}, "orion")
        == DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL
    )
    assert fcc_model_for_route("agent") == "llamacpp/agent", "the lane really does resolve"


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


# --- a lane too small to host the turn says so, before spending anything -----

async def _drive_turn(monkeypatch: pytest.MonkeyPatch, *, n_ctx: int | None, prompt: str) -> list:
    """Run the motor far enough to reach (or pass) the lane-capacity guard.

    Every resource the motor would acquire AFTER the guard is booby-trapped, so
    a guard that fires too late fails loudly instead of silently leaking.
    """
    from orion.harness import fcc_motor

    monkeypatch.setattr(fcc_motor, "load_fcc_env", lambda *_a, **_k: dict(FCC_ENV))

    async def _probe(*_a, **_k):
        return "Qwen3.8-27B-UD-Q4_K_XL", n_ctx

    monkeypatch.setattr(fcc_motor, "probe_route_runtime", _probe)

    def _boom_preflight(*_a, **_k):
        raise AssertionError("guard must return before the FCC preflight")

    def _boom_mcp(*_a, **_k):
        raise AssertionError("guard must return before an MCP config is rendered")

    monkeypatch.setattr(fcc_motor, "_preflight_fcc_server", _boom_preflight)
    monkeypatch.setattr(fcc_motor, "_maybe_render_mcp_config", _boom_mcp)

    return [
        ev
        async for ev in fcc_motor.run_fcc_turn(
            prompt=prompt,
            correlation_id="corr-guard",
            fcc_model_label="llamacpp/agent",
            workspace="/tmp",
            fcc_server_url="http://fcc:8082",
            auth_token="t",
            claude_bin="claude",
            timeout_sec=5.0,
        )
    ]


@pytest.mark.asyncio
async def test_a_lane_too_small_fails_before_spending_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COMPUTE defaults to `quick`, which serves n_ctx=4096 live while real
    harness prompts run 6.3k tokens median. Mode=Agent on the already-selected
    default lands there routinely, so it must fail immediately and legibly --
    not as a provider error mid-stream, and not after burning a turn."""
    events = await _drive_turn(monkeypatch, n_ctx=4096, prompt="x" * 40_000)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["error_code"] == "fcc_lane_context_too_small"
    # The operator needs both numbers to act on it, and the lane's identity.
    assert "4096" in events[0]["error"]
    assert "40000" in events[0]["error"]
    assert events[0]["metadata"]["fcc_lane_n_ctx"] == 4096


@pytest.mark.asyncio
async def test_a_lane_that_fits_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must not become a general brake: a prompt well inside the 27B's
    32768-token window has to pass it and go on to spawn.

    The prompt length is chosen to sit BETWEEN the raw token count (32768) and
    the real character ceiling (32768 * chars-per-token). A guard that compared
    the prompt's length in characters against a window measured in TOKENS would
    refuse this turn, and would still look correct on any prompt outside that
    band -- so this is the case that pins the unit conversion.
    """
    fits_in_chars_not_in_tokens = 50_000
    assert 32768 < fits_in_chars_not_in_tokens < max_context_chars(32768)

    with pytest.raises(AssertionError, match="before the FCC preflight"):
        await _drive_turn(monkeypatch, n_ctx=32768, prompt="x" * fits_in_chars_not_in_tokens)


@pytest.mark.asyncio
async def test_an_unknown_window_does_not_refuse_the_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """None means "not known" (older gateway, worker down, non-llamacpp backend).
    Refusing on it would turn a missing fact into an outage.

    The prompt deliberately exceeds even the env-fallback ceiling, so that a
    guard which dropped the "is the window known" test would refuse here. A
    shorter prompt would pass this test either way and prove nothing.
    """
    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", "65536")
    over_the_env_fallback = max_context_chars() + 1_000

    with pytest.raises(AssertionError, match="before the FCC preflight"):
        await _drive_turn(monkeypatch, n_ctx=None, prompt="x" * over_the_env_fallback)


# --- the error envelope must not ride in the partial-draft slot -------------

def _provider_error_frame() -> dict:
    """The motor's own error frame for a laundered provider error."""
    return {
        "type": "error",
        "error": "Upstream provider LLAMACPP returned HTTP 400.",
        "error_code": "fcc_context_overflow",
        "metadata": {"exit_code": 0, "provider_error_text": OVERFLOW_REPLY},
    }


def test_the_error_frame_carries_no_partial_draft() -> None:
    """`llm_response` on an error frame is the PARTIAL-DRAFT contract, not a
    diagnostic slot -- `runner.py`'s error branch promotes any non-empty value
    into `draft_text` with verdict "partial", the governor only aborts on an
    EMPTY draft, and the finalize+voice chain would then speak the provider's
    error as Orion's answer and persist it. Caught in review: the first version
    of this branch put the envelope there and re-created the laundering one
    layer up."""
    frame = _provider_error_frame()

    assert not frame.get("llm_response")
    # ...but the text must still be recoverable for diagnosis.
    assert frame["metadata"]["provider_error_text"] == OVERFLOW_REPLY


@pytest.mark.asyncio
async def test_the_runner_marks_a_provider_error_turn_failed_not_partial() -> None:
    """End of the chain that actually matters: an error frame shaped like the
    motor's now yields a FAILED turn with an empty draft, so the governor aborts
    before finalize instead of voicing the envelope."""
    from unittest.mock import AsyncMock
    from typing import Any, AsyncIterator

    from orion.harness.runner import HarnessRunner
    from orion.harness.tests.fixtures import make_thought
    from orion.schemas.cognition.answer_contract import AnswerContract
    from orion.schemas.context_exec import ContextExecPermissionV1
    from orion.schemas.harness_finalize import HarnessRunRequestV1

    async def _provider_error_runner(**_: Any) -> AsyncIterator[dict[str, Any]]:
        yield _provider_error_frame()

    request = HarnessRunRequestV1(
        correlation_id="c-provider-error",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    result = await HarnessRunner(AsyncMock(), fcc_runner=_provider_error_runner).run(request)

    assert result.draft_text == "", "the envelope must never become a draft"
    assert result.compliance_verdict == "failed"
    assert OVERFLOW_REPLY not in (result.draft_text or "")


# --- one overflow hint, not two contradictory ones --------------------------

def test_a_second_hint_with_a_different_window_is_not_appended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The motor hints with the LANE's window; `runner.py` and the hub call the
    same helper with no n_ctx and would resolve the container default. The
    idempotence check used to compare the whole hint text, so two different
    numbers no longer matched and the operator got both -- "context window full
    (~32768 tokens)" immediately followed by "(~131072 tokens)"."""
    from orion.fcc.context_budget import apply_context_overflow_hint

    monkeypatch.setenv("HARNESS_FCC_MAX_CONTEXT_TOKENS", "131072")
    raw = "llama_decode: exceed_context_size_error"

    once = apply_context_overflow_hint(raw, n_ctx=32768)
    twice = apply_context_overflow_hint(once)  # downstream caller, no lane

    assert twice == once
    assert twice.count("context window full") == 1
    assert "32768" in twice and "131072" not in twice


# --- a yielding lane is not an interactive Compute choice -------------------

@pytest.mark.parametrize("lane", ["quick_background", "metacog_background"])
def test_background_lanes_are_refused(lane: str) -> None:
    """A human picking a yielding lane for a live turn "buys nothing but
    latency" (orion/llm/routes.py's own words). Reachable via a raw
    POST /api/chat body or a stale localStorage entry predating the Hub
    picker's priority filter, so the refusal belongs at this seam."""
    assert fcc_model_for_route(lane) is None
    assert _resolve_fcc_model_label({"llm_route": lane}, "agent") == (
        DEFAULT_UNIFIED_TURN_FCC_MODEL_LABEL
    )
