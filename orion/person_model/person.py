"""Who a person is to the person model: a node id, a name, stated pronouns.

Every public function in ``orion.person_model`` takes a ``PersonRefV1`` (or a
node id). Nobody is special in code: the interlocutor is whoever
``config/person_model/people.yaml`` marks ``role: interlocutor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from orion.schemas.person_model import PersonRefV1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PEOPLE_PATH = REPO_ROOT / "config" / "person_model" / "people.yaml"


@dataclass(frozen=True)
class PronounForms:
    subject: str  # she / they / he
    object: str  # her / them / him
    possessive: str  # her / their / his
    reflexive: str  # herself / themselves / himself


# Grammar of the common stated sets. A stated set outside this table still
# renders by name (forms() returns None): no guessing at an unknown neopronoun.
_FORMS: Dict[str, PronounForms] = {
    "she/her": PronounForms("she", "her", "her", "herself"),
    "he/him": PronounForms("he", "him", "his", "himself"),
    "they/them": PronounForms("they", "them", "their", "themselves"),
}


def normalize_pronouns(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parts = [p.strip().lower() for p in str(value).replace(" ", "").split("/") if p.strip()]
    return "/".join(parts[:2]) if parts else None


def pronoun_forms(person: PersonRefV1) -> Optional[PronounForms]:
    """Forms for a person's stated pronouns, or None (caller uses the name)."""
    return _FORMS.get(normalize_pronouns(person.pronouns) or "")


def possessive(person: PersonRefV1) -> str:
    """'her' / 'their' / ... for stated pronouns; "<Name>'s" when none are stated."""
    forms = pronoun_forms(person)
    return forms.possessive if forms else f"{person.display_name}'s"


def load_people(path: Path | str = DEFAULT_PEOPLE_PATH) -> List[PersonRefV1]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rows = data.get("people") or []
    people = [PersonRefV1.model_validate(row) for row in rows]
    ids = [p.node_id for p in people]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate node_id in {path}")
    for p in people:
        if p.pronouns and not p.pronouns_evidence:
            raise ValueError(f"{p.node_id}: pronouns are stated-only and need pronouns_evidence")
    if sum(1 for p in people if p.role == "interlocutor") > 1:
        raise ValueError(f"{path}: more than one interlocutor")
    return people


def interlocutor(people: Iterable[PersonRefV1]) -> Optional[PersonRefV1]:
    for p in people:
        if p.role == "interlocutor":
            return p
    return None


def by_node_id(people: Iterable[PersonRefV1]) -> Dict[str, PersonRefV1]:
    return {p.node_id: p for p in people}
