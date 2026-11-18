"""Known entities for the short-message gate of lived-memory recall (issue #3613).

A short chat message ("what's up with hecate?") has too few content words for the step 1 floor,
but it names something Orion has memories about. This index answers one question: does a text
name a known entity?

Known entities come from Orion's own stores, never from a hand-written list:

- **Referents** (``episode_memory_referent``): the referent key's slug ("project:hecate" ->
  "hecate") and every name-class alias of it (``referent_alias``), for entity types only
  (``ENTITY_TYPES``: people, pets, places, projects, services, hosts, past events). Concepts,
  files and PR numbers are topics, not entities, and are left out.
- **People** (``person``): active display names, except the two participants (role ``self``
  and ``interlocutor``). For people and pets the first name alone also counts.

Junk filters (the same rules link-following uses, ``lived_links.usable_name``): an alias needs 4+
characters and a letter, and is not a function word. A referent on more than ``HUB_MAX_DEGREE``
memories is a hub (the two participants): naming it says nothing about which memory fits, and
"thanks Orion" must not search.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Set, Tuple

from .lived_links import HUB_MAX_DEGREE, normalize, usable_name

# Referent key prefixes that name an entity (a person, pet, place, project, service, host) or a
# past event. Everything else in the store today is concept:, file: or pr:.
ENTITY_TYPES = frozenset({"person", "pet", "place", "project", "service", "host", "event"})
PARTICIPANT_ROLES = frozenset({"self", "interlocutor"})
FIRST_NAME_TYPES = frozenset({"person", "pet"})

REFERENT_DEGREE_SQL = """
SELECT referent_key, count(*)::int AS n
FROM episode_memory_referent
GROUP BY referent_key
"""

REFERENT_ALIASES_SQL = """
SELECT a.alias_norm AS alias, k.alias_norm AS referent_key
FROM referent_alias a
JOIN referent_alias k ON k.node_id = a.node_id AND k.alias_class = 'key'
WHERE a.alias_class = 'name'
"""

PERSON_NAMES_SQL = """
SELECT display_name, role
FROM person
WHERE status = 'active'
"""


def _slug_name(referent_key: str) -> str:
    return normalize(referent_key.split(":", 1)[1] if ":" in referent_key else referent_key)


def _entity_type(referent_key: str) -> str:
    return referent_key.split(":", 1)[0] if ":" in referent_key else ""


class EntityIndex:
    """Whole-word name lookup: ``mentions(text)`` -> the entity keys named in ``text``."""

    def __init__(self, referent_degrees: Iterable[Tuple[str, int]] = (),
                 aliases: Iterable[Tuple[str, str]] = (),
                 persons: Iterable[Tuple[str, str]] = (), *,
                 hub_max_degree: int = HUB_MAX_DEGREE) -> None:
        self.names: Dict[str, Set[str]] = defaultdict(set)
        aliases = list(aliases)
        persons = list(persons)
        keep: Set[str] = set()
        hubs: Set[str] = set()
        for key, n in referent_degrees:
            key = str(key or "")
            if int(n or 0) > hub_max_degree:
                hubs.add(key)
            elif _entity_type(key) in ENTITY_TYPES and int(n or 0) > 0:
                keep.add(key)
        # A participant's or hub's name never counts, even as another entity's first word
        # ("<participant> sister" must not make the participant's own name a trigger).
        self.blocked: Set[str] = {_slug_name(k) for k in hubs}
        self.blocked |= {normalize(a) for a, k in aliases if str(k or "") in hubs}
        self.blocked |= {normalize(d) for d, role in persons if str(role or "") in PARTICIPANT_ROLES}
        self.blocked |= {b.split(" ", 1)[0] for b in list(self.blocked) if b}
        for key in keep:
            self._add(_slug_name(key), key, first_name=_entity_type(key) in FIRST_NAME_TYPES)
        for alias, key in aliases:
            key = str(key or "")
            if key in keep:
                self._add(alias, key, first_name=_entity_type(key) in FIRST_NAME_TYPES)
        for display, role in persons:
            if str(role or "") in PARTICIPANT_ROLES:
                continue
            full = normalize(display)
            self._add(full, f"person_name:{full}", first_name=True)

    def _add(self, alias: str, key: str, *, first_name: bool = False) -> None:
        name = normalize(alias)
        if usable_name(name) and name not in self.blocked:
            self.names[name].add(key)
        # People are called by their first name: "Ada" for a person named "Ada Lovelace".
        first = name.split(" ", 1)[0] if first_name and name else ""
        if first and first != name and usable_name(first) and first not in self.blocked:
            self.names[first].add(key)

    def __len__(self) -> int:
        return len(self.names)

    def matched_names(self, text: str) -> Set[str]:
        """The known names that appear in ``text`` as whole words."""
        norm = f" {normalize(text)} "
        if not norm.strip():
            return set()
        return {alias for alias in self.names if f" {alias} " in norm}

    def mentions(self, text: str) -> Set[str]:
        """Entity keys whose name appears in ``text`` as whole words."""
        out: Set[str] = set()
        for alias in self.matched_names(text):
            out |= self.names[alias]
        return out


async def load_entities(pool: Any) -> EntityIndex:
    degrees = [(str(r["referent_key"]), int(r["n"])) for r in await pool.fetch(REFERENT_DEGREE_SQL)]
    aliases = [(str(r["alias"]), str(r["referent_key"])) for r in await pool.fetch(REFERENT_ALIASES_SQL)]
    persons = [(str(r["display_name"]), str(r["role"])) for r in await pool.fetch(PERSON_NAMES_SQL)]
    return EntityIndex(degrees, aliases, persons)


__all__ = ["ENTITY_TYPES", "EntityIndex", "PARTICIPANT_ROLES", "load_entities"]
