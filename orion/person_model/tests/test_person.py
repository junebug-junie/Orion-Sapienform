from pathlib import Path

import pytest

from orion.person_model.person import interlocutor, load_people, possessive, pronoun_forms
from orion.schemas.person_model import PersonRefV1

ROOT = Path(__file__).resolve().parents[3]


def test_interlocutor_pronouns_come_from_config():
    people = load_people()
    me = interlocutor(people)
    assert me is not None and me.display_name == "Juniper"
    assert me.pronouns == "she/her"
    assert possessive(me) == "her"
    assert pronoun_forms(me).subject == "she"


def test_self_is_they_them():
    self_row = next(p for p in load_people() if p.role == "self")
    assert possessive(self_row) == "their"


def test_unstated_pronouns_render_the_name_not_a_guess():
    p = PersonRefV1(node_id="referent-x", display_name="Brenda")
    assert pronoun_forms(p) is None
    assert possessive(p) == "Brenda's"


def test_pronouns_without_evidence_are_refused(tmp_path):
    f = tmp_path / "people.yaml"
    f.write_text("people:\n  - node_id: n1\n    display_name: X\n    pronouns: he/him\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stated-only"):
        load_people(f)


def test_pronoun_evidence_quotes_exist_in_their_repo_sources():
    """Repo-resident sources are checked verbatim; chat sources are checked by the eval."""
    for p in load_people():
        for ev in p.pronouns_evidence:
            path, _, _ = ev.source_ref.partition("#")
            if ev.source_kind == "authored_profile":
                assert ev.quote in (ROOT / path).read_text(encoding="utf-8"), (p.node_id, ev.source_ref)
