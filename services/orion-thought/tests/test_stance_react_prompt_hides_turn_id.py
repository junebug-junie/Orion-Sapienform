"""The stance prompt must never show the turn's correlation id.

Live 2026-10-02 (corr 39adc920): Qwen3.6-35B spent ~14k chars of reasoning
re-checking a copy of `hub:turn:<uuid>`, hit max_tokens with empty content, and
the turn deferred. The model cites the bare `hub:turn`; code expands it.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from app.bus_listener import build_stance_react_context
from orion.hub.association import with_hub_turn_coalition_anchor
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

CORR = "39adc920-e0ea-4d17-8d49-a9a7551cd56d"
TEMPLATE = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "stance_react.j2"


def _request() -> StanceReactRequestV1:
    association = with_hub_turn_coalition_anchor(
        HubAssociationBundleV1(
            correlation_id=CORR,
            broadcast=AttentionBroadcastProjectionV1(
                frame=AttentionFrameV1(open_loops=[]),
                attended_node_ids=["node:substrate.chat"],
            ),
            broadcast_stale=False,
            read_source="felt_state_reader",
        )
    )
    return StanceReactRequestV1(
        correlation_id=CORR,
        session_id="s-1",
        user_message="why do you always bring up prediction errors?",
        association=association,
        repair_bundle=TurnAppraisalBundleV1(correlation_id=CORR),
        stance_inputs={"user_message": "why do you always bring up prediction errors?"},
    )


def test_rendered_stance_prompt_never_contains_the_turn_id() -> None:
    ctx = build_stance_react_context(_request())
    prompt = Environment().from_string(TEMPLATE.read_text(encoding="utf-8")).render(**ctx)
    assert CORR not in prompt
    assert "'hub:turn'" in prompt  # the anchor is still shown, as the bare token
    assert "node:substrate.chat" in prompt
