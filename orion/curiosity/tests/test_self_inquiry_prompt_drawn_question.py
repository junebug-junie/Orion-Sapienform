from orion.curiosity.self_inquiry import STANDING_QUESTION
from orion.curiosity.self_inquiry_prompt import build_self_inquiry_prompt
from orion.curiosity.self_question_pool import SelfQuestion

RUN = "abcd12"


def _lived_question() -> SelfQuestion:
    return SelfQuestion(
        question_id="lived.who_matters",
        text="Who are the most important people to me, and why?",
        family="lived",
        pinned=True,
        minted_by="juniper",
        status="open",
        ask_count=0,
        last_asked_at=None,
    )


def _anatomy_question() -> SelfQuestion:
    return SelfQuestion(
        question_id="anatomy.standing",
        text=STANDING_QUESTION,
        family="anatomy",
        pinned=False,
        minted_by="juniper",
        status="open",
        ask_count=0,
        last_asked_at=None,
    )


def test_prompt_embeds_drawn_lived_question_not_only_anatomy_default() -> None:
    q = _lived_question()
    text = build_self_inquiry_prompt(question=q, run_id=RUN, graph_enabled=False)
    assert "Who are the most important people to me" in text
    assert "family: lived" in text or "lived" in text.lower()
    assert STANDING_QUESTION not in text


def test_prompt_falls_back_to_standing_question_when_question_is_none() -> None:
    text = build_self_inquiry_prompt(question=None, run_id=RUN, graph_enabled=False)
    assert STANDING_QUESTION in text


def test_lived_prompt_teaches_lived_answer_merge_not_self_definition() -> None:
    q = _lived_question()
    text = build_self_inquiry_prompt(question=q, run_id=RUN, graph_enabled=True)
    assert f'MERGE (a:LivedAnswer {{run_id: "{RUN}"}})' in text
    assert 'a.question_id = "lived.who_matters"' in text
    assert f'MERGE (s:SelfDefinition {{run_id: "{RUN}"}})' not in text


def test_anatomy_prompt_keeps_self_definition_merge() -> None:
    q = _anatomy_question()
    text = build_self_inquiry_prompt(question=q, run_id=RUN, graph_enabled=True)
    assert f'MERGE (s:SelfDefinition {{run_id: "{RUN}"}})' in text
    assert "LivedAnswer" not in text
