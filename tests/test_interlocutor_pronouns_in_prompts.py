"""P0 regression: prompts must not misgender the interlocutor.

Source of truth for pronouns: config/person_model/people.yaml (stated-only; the
interlocutor said "My pronouns are she/her"). stance_react.j2:131 said "Juniper
telling Orion something about their own life" (misgendering, found 2026-10-14);
orion-mind's stance handoff and chat_quick.j2 had the same slip.
"""

import re
from pathlib import Path

from orion.person_model.person import interlocutor, load_people, pronoun_forms

ROOT = Path(__file__).resolve().parents[1]
PROMPT_SURFACES = [
    ROOT / "orion/cognition/prompts/stance_react.j2",
    ROOT / "orion/cognition/prompts/chat_quick.j2",
    ROOT / "services/orion-mind/app/stance_handoff.py",
]


def _me():
    me = interlocutor(load_people())
    assert me is not None
    return me, pronoun_forms(me)


def test_interlocutor_pronouns_are_stated_she_her():
    me, forms = _me()
    assert me.pronouns == "she/her" and forms is not None and forms.possessive == "her"
    assert any("she/her" in ev.quote for ev in me.pronouns_evidence)


def test_stance_react_uses_her_own_life():
    me, forms = _me()
    text = (ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    assert f"{me.display_name} telling Oríon something about {forms.possessive} own life" in text
    assert "their own life" not in text


def test_no_they_pronoun_bound_to_the_interlocutor_in_prompt_surfaces():
    me, forms = _me()
    name = re.escape(me.display_name)
    wrong = r"\b(their own life|their life|their week|meet them|who they are to you|they are heading)\b"
    for p in PROMPT_SURFACES:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(name, line) and re.search(wrong, line):
                raise AssertionError(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:160]}")
