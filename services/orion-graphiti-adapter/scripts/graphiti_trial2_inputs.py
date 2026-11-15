"""Trial 2 inputs: raw chat exchanges, a person-fact seed episode, a small ontology, and a
RediSearch-safe fulltext sanitizer (spec 2026-11-13 core memory pages, Graphiti trial 2).

Trial 1 fed Graphiti Orion's distilled first-person notes with no ontology and thinking on, and
scored NO-GO. Trial 2 runs Graphiti the standard way, to tell "Graphiti" apart from "our setup":

- raw chat: one ``chat_history_log`` row (one Juniper prompt + Orion's reply) = one
  EpisodeType.message episode, speaker-labelled, reference_time = the turn time;
- a seed episode built from ``person_fact`` so the extractor knows each person's own pronouns and
  relation before the first chat;
- custom entity types (Person, Pet, Place, Event, OrionPart) with the generic ``Entity`` type
  excluded, so narrator junk ("I", "Speaker", "fact") has no type to land in.

Pure helpers (no graphiti import) so the gate tests run without graphiti-core.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

# A reply longer than this is cut to head + marker + tail. Trial 1's p90 episode was 390 s on
# ~130-char notes; chat replies average ~1,100 chars and reach 9,700, which would blow the 540 s
# episode timeout. Head keeps what Orion answered; tail keeps where the reply landed.
ORION_MAX_CHARS = 1500
ORION_HEAD, ORION_TAIL = 1100, 300
JUNIPER_MAX_CHARS = 2500
JUNIPER_HEAD, JUNIPER_TAIL = 2000, 400

CHAT_SQL = """
select id::text as id, prompt, response, created_at as at
from chat_history_log
where coalesce(prompt, '') <> '' and coalesce(response, '') <> ''
  and coalesce(source, '') <> 'collapse_mirror_reply'
"""
PERSON_FACT_SQL = """
select coalesce(p.display_name, f.person_id) as name, f.category, f.value_text
from person_fact f left join person p on p.person_id = f.person_id
where f.removed_at is null and f.category in ('pronouns', 'relationship')
order by 1, 2, f.created_at
"""


def trim(text: str, limit: int, head: int, tail: int) -> tuple[str, int]:
    """(text, chars cut). Over ``limit``: first ``head`` chars + a marker + last ``tail`` chars."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text, 0
    cut = len(text) - head - tail
    return f"{text[:head].rstrip()} [... {cut} chars trimmed ...] {text[-tail:].lstrip()}", cut


def exchange_body(prompt: str, response: str) -> tuple[str, int]:
    """Speaker-labelled message episode, Graphiti's documented "<speaker>: <text>" shape."""
    p, cut_p = trim(prompt, JUNIPER_MAX_CHARS, JUNIPER_HEAD, JUNIPER_TAIL)
    r, cut_r = trim(response, ORION_MAX_CHARS, ORION_HEAD, ORION_TAIL)
    return f"Juniper: {p}\nOrion: {r}", cut_p + cut_r


def seed_body(facts: list[dict[str, Any]]) -> str:
    """One line per person: name, stated pronouns, and how they relate to Juniper, in her words."""
    people: dict[str, dict[str, list[str]]] = {}
    for f in facts:
        name = str(f.get("name") or "").strip()
        val = str(f.get("value_text") or "").strip()
        if not name or not val or "don't know" in val.lower():
            continue
        slot = people.setdefault(name, {"pronouns": [], "relationship": []})
        bucket = slot.setdefault(str(f.get("category")), [])
        norm = re.sub(r"\s*/\s*", "/", val.lower()) if f.get("category") == "pronouns" else val
        if norm not in bucket:
            bucket.append(norm)
    lines = ["Juniper: These are the people in my life, with the pronouns each one uses.",
             "Juniper: I am Juniper. My pronouns are she/her. Orion's pronouns are they/them."]
    for name in sorted(people):
        if name.lower() == "juniper":
            continue
        slot = people[name]
        pron = ", ".join(slot.get("pronouns") or []) or "not stated"
        rel = " ".join(slot.get("relationship") or [])
        lines.append(f"Juniper: {name} uses {pron} pronouns." + (f" {rel}" if rel else ""))
    return "\n".join(lines)


def chat_rows_to_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        body, cut = exchange_body(str(r["prompt"]), str(r["response"]))
        at = r["at"]
        if isinstance(at, str):
            at = datetime.fromisoformat(at)
        if at.tzinfo is None:  # chat_history_log.created_at is naive UTC
            at = at.replace(tzinfo=timezone.utc)
        out.append({"id": str(r["id"]), "body": body, "at": at, "trimmed_chars": cut})
    return out


def seed_time(first_turn: datetime) -> datetime:
    """The seed must come before every exchange so the pronouns are known from the first one."""
    return first_turn - timedelta(hours=1)


# RediSearch (FalkorDB fulltext) treats more characters as syntax than Lucene does. graphiti 0.19's
# lucene_sanitize leaves e.g. '%' (fuzzy operator), '@', '$', ';', '.', ',', '<', '>', '=', "'" and
# '#' alone, and an entity named "38% outreach figure" crashed every search for it in trial 1
# ("RediSearch: Syntax error at offset 42 near figure"). Fulltext here is a fuzzy candidate search,
# so dropping punctuation to spaces loses nothing.
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def redisearch_safe(query: str) -> str:
    return re.sub(r"\s+", " ", _NON_WORD.sub(" ", query or "")).strip()


def install_redisearch_sanitizer() -> None:
    """Route graphiti's fulltext queries through ``redisearch_safe`` first (trial process only)."""
    from graphiti_core.search import search_utils  # noqa: PLC0415
    if getattr(search_utils.lucene_sanitize, "_orion_safe", False):
        return
    original = search_utils.lucene_sanitize

    def safe(query: str) -> str:
        return original(redisearch_safe(query))

    safe._orion_safe = True  # type: ignore[attr-defined]
    search_utils.lucene_sanitize = safe


def ontology() -> tuple[dict[str, Any], list[str]]:
    """(entity_types, excluded_entity_types) for add_episode. Small on purpose: only what the
    page checks need. The generic ``Entity`` type is excluded, so a thing that fits none of these
    is not extracted at all (that is where "I", "Speaker", "fact" landed in trial 1)."""
    from pydantic import BaseModel, Field  # noqa: PLC0415

    class Person(BaseModel):
        """A human being, named or referred to by role ("my wife", "the mail carrier")."""
        pronouns: str | None = Field(None, description="The person's own pronouns as stated, e.g. she/her")
        relation_to_juniper: str | None = Field(None, description="How this person relates to Juniper, e.g. wife, son")

    class Pet(BaseModel):
        """A companion animal with a name."""
        species: str | None = Field(None, description="dog, cat, ...")

    class Place(BaseModel):
        """A geographic place or building: a city, a home, a hotel, a hospital."""
        place_kind: str | None = Field(None, description="city, home, hotel, hospital, ...")

    class Event(BaseModel):
        """Something that happened over a bounded time: a trip, a surgery, a meeting, a visit."""
        event_kind: str | None = Field(None, description="trip, surgery, meeting, ...")

    class OrionPart(BaseModel):
        """A part of Orion's own body or mind: a service, machine, sensor, camera or model."""
        part_kind: str | None = Field(None, description="service, machine, sensor, model, ...")

    return ({"Person": Person, "Pet": Pet, "Place": Place, "Event": Event, "OrionPart": OrionPart},
            ["Entity"])
