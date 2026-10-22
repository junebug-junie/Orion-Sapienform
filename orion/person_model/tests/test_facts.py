"""Person questions and facts (orion/person_model/facts.py, categories.yaml)."""

from __future__ import annotations

from orion.person_model import facts as pf
from orion.schemas.person_model import PersonRefV1


def test_first_question_about_someone_is_who_they_are_to_you() -> None:
    assert pf.next_question(name="Sam", role="other", done=()) == ("relationship", "Who is Sam to you?")


def test_questions_follow_the_yaml_order_and_skip_done_ones() -> None:
    asked = []
    done: set = set()
    while (nxt := pf.next_question(name="Sam", role="other", done=done)) is not None:
        asked.append(nxt[0])
        done.add(nxt[0])
    assert asked[:5] == ["relationship", "pronouns", "age", "gender", "sexuality"]
    assert "goals" not in asked          # about_other is null
    assert asked[-1] == "anything_else"


def test_the_interlocutor_gets_you_questions_and_self_gets_none() -> None:
    assert pf.next_question(name="Alex", role="interlocutor", done=()) == (
        "gender", "What is your gender, in your own words?")
    assert pf.next_question(name="Orion", role="self", done=()) is None


def test_no_question_guesses_a_pronoun_for_someone_else() -> None:
    for c in pf.load_categories():
        if c.about_other:
            text = c.about_other.format(name="Sam").lower().split()
            assert not {"she", "he", "her", "his", "him", "they", "them", "their"} & set(t.strip("?,.") for t in text), c.key


def test_sensitive_categories_are_marked() -> None:
    sensitive = {c.key for c in pf.load_categories() if c.sensitive}
    assert {"gender", "sexuality"} <= sensitive


def test_ask_ref_round_trips_and_rejects_unknown_categories() -> None:
    ref = pf.ask_ref("person-1", "gender")
    assert pf.parse_ask_ref(ref) == ("person-1", "gender")
    assert pf.parse_ask_ref("person-1|shoe_size") is None
    assert pf.parse_ask_ref("garbage") is None


def test_answers_are_kept_verbatim_only_whitespace_collapsed() -> None:
    assert pf.answer_value("  my   neighbor's kid ") == "my neighbor's kid"
    assert pf.answer_value("   ") is None


def test_stated_pronouns_become_facts_but_not_for_orion() -> None:
    people = [PersonRefV1(node_id="r-you", display_name="Alex", role="interlocutor", pronouns="she/her"),
              PersonRefV1(node_id="r-self", display_name="Orion", role="self", pronouns="they/them"),
              PersonRefV1(node_id="r-x", display_name="Sam")]
    facts = pf.stated_pronoun_facts(people)
    assert [(f["person_id"], f["value_text"], f["voice"]) for f in facts] == [("r-you", "she/her", "authored_profile")]
