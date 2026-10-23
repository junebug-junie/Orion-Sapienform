"""What Orion asks the interlocutor about a person, and what the answer becomes (2026-10-22).

Spec: docs/superpowers/specs/2026-10-22-people-one-record-design.md section 4.

Pure. The question list is ``categories.yaml``. A question is asked once per
(person, category): answered or dismissed, it is not asked again; expired, it
may come back after a cooldown (the caller's SQL decides that). The answer is
stored verbatim as a ``person_fact`` (voice ``ask_answer``): nothing is parsed,
nothing is inferred, so a sensitive category only ever holds the interlocutor's own words.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml

from orion.schemas.person_model import PersonRefV1

CATEGORIES_PATH = Path(__file__).with_name("categories.yaml")
SOURCE_KIND_PERSON_FACT = "person_fact"
REF_SEP = "|"
VOICE_ASK_ANSWER = "ask_answer"
VOICE_AUTHORED = "authored_profile"
VOICE_HUB_EDIT = "hub_edit"
ROLE_INTERLOCUTOR = "interlocutor"
ROLE_SELF = "self"


@dataclass(frozen=True)
class Category:
    key: str
    about_other: Optional[str]
    about_you: Optional[str]
    sensitive: bool


@lru_cache(maxsize=4)
def load_categories(path: str = str(CATEGORIES_PATH)) -> Tuple[Category, ...]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = []
    for row in data.get("categories") or []:
        out.append(Category(str(row["key"]), row.get("about_other"), row.get("about_you"),
                            bool(row.get("sensitive", False))))
    keys = [c.key for c in out]
    if len(set(keys)) != len(keys):
        raise ValueError("categories.yaml: duplicate key")
    return tuple(out)


def category_keys() -> List[str]:
    return [c.key for c in load_categories()]


def question_text(category: Category, *, name: str, role: str) -> Optional[str]:
    """The question about one person, or None when this category is not asked of them."""
    if role == ROLE_SELF:
        return None
    template = category.about_you if role == ROLE_INTERLOCUTOR else category.about_other
    if not template:
        return None
    return template.format(name=" ".join(str(name).split()))


def next_question(*, name: str, role: str, done: Iterable[str]) -> Optional[Tuple[str, str]]:
    """(category, question) for the first category not yet answered, dismissed or
    pending for this person, in categories.yaml order; None when there is nothing left."""
    skip = set(done)
    for c in load_categories():
        if c.key in skip:
            continue
        text = question_text(c, name=name, role=role)
        if text:
            return c.key, text
    return None


def ask_ref(person_id: str, category: str) -> str:
    return f"{person_id}{REF_SEP}{category}"


def parse_ask_ref(ref: str) -> Optional[Tuple[str, str]]:
    person_id, sep, category = str(ref or "").rpartition(REF_SEP)
    if not sep or not person_id or category not in category_keys():
        return None
    return person_id, category


def answer_value(answer: Optional[str]) -> Optional[str]:
    """Her answer, whitespace-collapsed, or None when there is nothing to keep."""
    text = " ".join(str(answer or "").split())
    return text or None


def fact_id(person_id: str, category: str, voice: str, source_ref: str) -> str:
    raw = f"{person_id}|{category}|{voice}|{source_ref}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def stated_pronoun_facts(people: Sequence[PersonRefV1]) -> List[dict]:
    """people.yaml pronouns as facts (voice authored_profile), so every reader
    gets pronouns from one table. Only stated pronouns; a row without them adds nothing."""
    out = []
    for p in people:
        if not p.pronouns or p.role == ROLE_SELF:
            continue
        ref = p.pronouns_evidence[0].source_ref if p.pronouns_evidence else "config/person_model/people.yaml"
        out.append({"fact_id": fact_id(p.node_id, "pronouns", VOICE_AUTHORED, ref), "person_id": p.node_id,
                    "category": "pronouns", "value_text": p.pronouns, "voice": VOICE_AUTHORED, "source_ref": ref})
    return out
