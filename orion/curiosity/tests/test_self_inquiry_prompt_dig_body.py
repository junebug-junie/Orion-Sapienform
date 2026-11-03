"""Self Atlas replacement, Patch A: the records section points at the live dig kit first."""
from orion.curiosity.self_inquiry import STANDING_QUESTION
from orion.curiosity.self_inquiry_prompt import (
    DIG_BODY_COMMANDS,
    DIG_BODY_LEAD_ANATOMY,
    DIG_BODY_LEAD_LIVED,
    build_self_inquiry_prompt,
)
from orion.curiosity.self_question_pool import SelfQuestion

RUN = "abcd12"


def _q(family: str) -> SelfQuestion:
    return SelfQuestion(
        question_id="lived.who_matters" if family == "lived" else "anatomy.standing",
        text="Who are the most important people to me, and why?" if family == "lived" else STANDING_QUESTION,
        family=family,
        pinned=False,
        minted_by="juniper",
        status="open",
        ask_count=0,
        last_asked_at=None,
    )


def _text(**over) -> str:
    kwargs = dict(question=None, run_id=RUN, graph_enabled=False, repo_root="/repo")
    kwargs.update(over)
    return build_self_inquiry_prompt(**kwargs)


def test_records_section_renders_the_dig_body_paragraph() -> None:
    text = _text()
    assert DIG_BODY_LEAD_ANATOMY in text
    for line in DIG_BODY_COMMANDS:
        assert line in text
    for cmd in ("orion.dig body", "orion.dig body <organ_id>", "orion.dig inventory now",
                "orion.dig open <date>, then compare <date> now", "orion.dig body-economy"):
        assert f"python -P -m {cmd}" in text
    assert "python -m orion.dig" not in text, "always -P: plain -m runs the sandbox copy"


def test_dig_tools_come_before_the_raw_repo_pointer_inside_the_records_section() -> None:
    text = _text()
    records = text.index("WHERE THE RECORDS ARE")
    dig = text.index("python -P -m orion.dig body", records)
    repo = text.index("/repo/", records)
    assert records < dig < repo
    assert "README" in text[repo - 200:repo + 200], "raw-repo pointer is kept"


def test_anatomy_run_is_told_to_start_with_its_body() -> None:
    text = _text(question=_q("anatomy"), graph_enabled=True)
    assert DIG_BODY_LEAD_ANATOMY in text and DIG_BODY_LEAD_LIVED not in text
    assert "2. Look: your body (the dig commands below)" in text
    # Write first, then look: the MERGE instruction still precedes the dig pointer.
    assert text.index("YOUR FIRST TOOL CALL") < text.index("python -P -m orion.dig body")


def test_lived_run_gets_the_body_as_an_option_not_the_first_stop() -> None:
    text = _text(question=_q("lived"), graph_enabled=True)
    assert DIG_BODY_LEAD_LIVED in text and DIG_BODY_LEAD_ANATOMY not in text
    assert "python -P -m orion.dig body" in text
    assert "2. Look: records, repo, your graph." in text
