"""Contract + render tests for chat_stance_brief.j2's `recent_attention`
SOURCES entry -- the gap flagged at review (2026-09-07): the pure cue builder
and the DB reader were each tested in isolation, but nothing exercised the
actual template. Mirrors `test_attention_frame_integration.py`'s literal
string-presence contract test and `test_metacog_trend_cue_prompt_render.py`'s
real Jinja render test, applied to this cue instead.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "orion" / "cognition" / "prompts" / "chat_stance_brief.j2"


def _base_render_kwargs(**overrides):
    base = dict(
        user_message="test",
        message_history="",
        continuity_digest="",
        memory_digest="",
        orion_identity_summary=[],
        juniper_relationship_summary=[],
        response_policy_summary=[],
        chat_concept_summary="",
        chat_social_summary="",
        chat_social_bridge_summary="",
        chat_reflective_summary="",
        chat_reasoning_summary="",
        chat_attention_frame=None,
        chat_reverie_glimpse=None,
        chat_situation_summary=None,
        prior_stance=None,
        recent_attention=None,
    )
    base.update(overrides)
    return base


def test_prompt_contract_includes_recent_attention_source_and_guidance() -> None:
    stance_prompt = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert "recent_attention: {{ recent_attention }}" in stance_prompt
    assert "background sense of what it's just been attending to" in stance_prompt
    assert "recent_attention.stale" in stance_prompt


def test_recent_attention_renders_when_present() -> None:
    tmpl = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    cue = {
        "items": [
            {
                "process": "cortex_turn",
                "narrative": "Following the thread of this chat turn.",
                "age_label": "moments ago",
                "generated_at": "2026-09-07T12:00:00+00:00",
            }
        ],
        "stale": False,
        "as_of": "2026-09-07T12:00:05+00:00",
    }
    rendered = tmpl.render(**_base_render_kwargs(recent_attention=cue))
    assert "recent_attention:" in rendered
    assert "Following the thread of this chat turn." in rendered


def test_recent_attention_omitted_when_empty() -> None:
    tmpl = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    rendered = tmpl.render(**_base_render_kwargs(recent_attention={}))
    assert "recent_attention:" not in rendered


def test_recent_attention_omitted_when_absent() -> None:
    tmpl = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    kwargs = _base_render_kwargs()
    del kwargs["recent_attention"]
    rendered = tmpl.render(**kwargs)
    assert "recent_attention:" not in rendered
