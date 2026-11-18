"""Known entities for the short-message gate of lived-memory recall (issue #3613).

A short chat message ("what's up with hecate?") has too few content words for the step 1 floor,
but it names something Orion has memories about. This index answers one question: does a text
name a known entity?

Known entities come from Orion's own stores, never from a hand-written list:

- **Referents** (``episode_memory_referent``): every name-class alias (``referent_alias``) of a
  referent of an entity type (``ENTITY_TYPES``: people, pets, places, projects, services, hosts,
  past events). Concepts, files and PR numbers are topics, not entities, and are left out. The
  key's slug is not a name on its own ("project:curiosity" would make "out of curiosity" an
  entity): only what the distiller wrote down as a name counts.
- **People** (``person``): active display names. A display name that is a proper name (two or
  more words, each capitalized) also counts by its first word; a role-like name ("Mail carrier")
  does not.

The two participants never count, so "thanks Orion" does not search: the ``person`` rows with
role ``self`` and ``interlocutor``, every referent that has one of their names as a key slug or
alias, and their first names as anyone's name. Only when no participant row can be read does a
degree backstop apply instead (a referent on more than ``HUB_MAX_DEGREE`` memories is left out).

Junk filters (the same rules link-following uses, ``lived_links.usable_name``): a name needs 4+
characters and a letter, and is not a function word.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

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


def _first(name: str) -> str:
    return name.split(" ", 1)[0] if name else ""


def is_proper_name(display: str) -> bool:
    """Two or more words, each starting with a capital letter ("Ada Lovelace", not "Mail carrier")."""
    words = str(display or "").split()
    return len(words) >= 2 and all(w[:1].isupper() for w in words)


class EntityIndex:
    """Whole-word name lookup: ``mentions(text)`` -> the entity keys named in ``text``."""

    def __init__(self, referent_degrees: Iterable[Tuple[str, int]] = (),
                 aliases: Iterable[Tuple[str, str]] = (),
                 persons: Iterable[Tuple[str, str]] = (), *,
                 hub_max_degree: int = HUB_MAX_DEGREE) -> None:
        self.names: Dict[str, Set[str]] = defaultdict(set)
        degrees = [(str(k or ""), int(n or 0)) for k, n in referent_degrees]
        aliases = [(normalize(a), str(k or "")) for a, k in aliases]
        persons = [(str(d or ""), str(r or "")) for d, r in persons]
        names_of: Dict[str, Set[str]] = defaultdict(set)
        for alias, key in aliases:
            names_of[key].add(alias)

        participants = {normalize(d) for d, role in persons if role in PARTICIPANT_ROLES}
        participants.discard("")
        # A participant's name, and its first word, is never anyone's name.
        self.blocked: Set[str] = participants | {_first(p) for p in participants}
        if participants:
            excluded = {k for k, _n in degrees
                        if _slug_name(k) in self.blocked or names_of.get(k, set()) & self.blocked}
        else:
            # No participant rows readable: fall back to "too many memories to say anything".
            excluded = {k for k, n in degrees if n > hub_max_degree}
            self.blocked |= {_slug_name(k) for k in excluded} | {a for k in excluded for a in names_of.get(k, ())}
            self.blocked |= {_first(b) for b in list(self.blocked)}

        keep = {k for k, n in degrees if n > 0 and _entity_type(k) in ENTITY_TYPES and k not in excluded}
        for alias, key in aliases:
            if key in keep:
                self._add(alias, key, first_name=_entity_type(key) in FIRST_NAME_TYPES)
        for display, role in persons:
            if role in PARTICIPANT_ROLES:
                continue
            full = normalize(display)
            self._add(full, f"person_name:{full}", first_name=is_proper_name(display))

    def _add(self, alias: str, key: str, *, first_name: bool = False) -> None:
        name = normalize(alias)
        if usable_name(name) and name not in self.blocked:
            self.names[name].add(key)
        # People are called by their first name: "Ada" for a person named "Ada Lovelace".
        first = _first(name) if first_name else ""
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
    degrees: List[Tuple[str, int]] = [(str(r["referent_key"]), int(r["n"]))
                                      for r in await pool.fetch(REFERENT_DEGREE_SQL)]
    aliases = [(str(r["alias"]), str(r["referent_key"])) for r in await pool.fetch(REFERENT_ALIASES_SQL)]
    persons = [(str(r["display_name"]), str(r["role"])) for r in await pool.fetch(PERSON_NAMES_SQL)]
    return EntityIndex(degrees, aliases, persons)


__all__ = ["ENTITY_TYPES", "EntityIndex", "PARTICIPANT_ROLES", "is_proper_name", "load_entities"]
