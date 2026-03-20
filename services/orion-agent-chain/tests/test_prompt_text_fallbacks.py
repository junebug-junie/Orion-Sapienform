from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "orion" / "cognition" / "prompts"


def test_prompt_templates_fallback_to_text():
    renderer = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    ctx = {"text": "hello world"}

    cases = [
        ("triage_prompt.j2", "INCOMING REQUEST"),
        ("plan_action_prompt.j2", "GOAL"),
        ("goal_formulate_prompt.j2", "RAW INTENTION"),
        ("summarize_context_prompt.j2", "INPUT CONTEXT"),
        ("tag_enrich_prompt.j2", "TARGET FRAGMENT"),
        ("pattern_detect_prompt.j2", "CANDIDATE FRAGMENTS"),
    ]

    for template_name, section_heading in cases:
        rendered = renderer.get_template(template_name).render(**ctx)
        assert section_heading in rendered
        assert "hello world" in rendered
