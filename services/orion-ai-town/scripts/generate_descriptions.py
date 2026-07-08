#!/usr/bin/env python3
"""Deterministically compose AI Town `Descriptions` from the tracked card set.

Source of truth: ``services/orion-ai-town/cards/town_cards.yaml``.
Target: ``services/orion-ai-town/upstream/data/characters.ts`` (the ``Descriptions``
array only; imports + ``characters`` sprite table are left untouched).

NPC identities are the rich, prompt-injected self-descriptions read by
``convex/agent/conversation.ts`` at prompt time. Juniper (human) and Orion
(external join) are NOT emitted here — Juniper's blurb goes in ``convex/world.ts``
and Orion's card is applied by the embodiment bootstrap.

Run from repo root:  python services/orion-ai-town/scripts/generate_descriptions.py
Idempotent: re-running with unchanged cards produces an unchanged file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve()
_SERVICE = _HERE.parents[1]
CARDS = _SERVICE / "cards" / "town_cards.yaml"
CHARACTERS_TS = _SERVICE / "upstream" / "data" / "characters.ts"
WORLD_TS = _SERVICE / "upstream" / "convex" / "world.ts"
CONSTANTS_TS = _SERVICE / "upstream" / "convex" / "constants.ts"
GENERATED = _SERVICE / "cards" / "generated"

# Fixed NPC order (matches the seeded roster / sprite table).
NPC_ORDER = [
    "mara_vale",
    "nico_sable",
    "elian_cross",
    "juno_park",
    "tessa_quinn",
    "vale_moreno",
    "sofia_bell",
    "cam_lin",
]


def _collapse(text: str | None) -> str:
    return " ".join((text or "").split())


def compose_identity(card: dict) -> str:
    """Rich, third-person identity injected into the agent's prompts."""
    parts = [
        _collapse(card.get("public_description")),
        _collapse(card.get("deeper_bio")),
        _collapse(card.get("orion_dynamic")),
    ]
    pressure = _collapse(card.get("private_pressure"))
    if pressure:
        parts.append(f"Privately: {pressure}")
    signature = _collapse(card.get("signature_line"))
    if signature:
        parts.append(f"A line they often use: {signature}")
    return " ".join(p for p in parts if p)


def compose_presence_blurb(card: dict) -> str:
    """Rich blurb for players seeded outside `Descriptions` (Juniper human join,
    Orion external join). Uses public_description + deeper_bio + town_presence +
    signature so residents perceive a full character, not a one-liner."""
    parts = [
        _collapse(card.get("public_description")),
        _collapse(card.get("deeper_bio")),
        _collapse(card.get("town_presence")),
    ]
    signature = _collapse(card.get("signature_line"))
    if signature:
        parts.append(f"A line {_signature_intro(card.get('pronouns'))}: {signature}")
    return " ".join(p for p in parts if p)


# Subject pronoun -> (word, verb) for "A line <subject> often <verb>:". Singular
# they takes the plural verb form ("they often use"); she/he take "uses".
_SUBJECT_VERB = {
    "she": ("she", "uses"),
    "he": ("he", "uses"),
    "they": ("they", "use"),
}


def _signature_intro(pronouns: str | None) -> str:
    subject = str(pronouns or "they/them").split("/", 1)[0].strip().lower()
    word, verb = _SUBJECT_VERB.get(subject, ("they", "use"))
    return f"{word} often {verb}"


def _ts_backtick(text: str) -> str:
    # Template-literal safe: escape backslash, backtick, and ${ interpolation.
    return text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def render_descriptions(cards: dict) -> str:
    by_id = {c["id"]: c for c in cards["characters"]}
    sprites = cards["sprites"]
    plans = cards["plans"]
    lines = ["export const Descriptions = ["]
    for cid in NPC_ORDER:
        card = by_id[cid]
        identity = _ts_backtick(compose_identity(card))
        plan = _ts_backtick(_collapse(plans[cid]))
        lines.append("  {")
        lines.append(f"    name: {_ts_single(card['name'])},")
        lines.append(f"    character: '{sprites[cid]}',")
        lines.append(f"    identity: `{identity}`,")
        lines.append(f"    plan: `{plan}`,")
        lines.append("  },")
    lines.append("];")
    return "\n".join(lines)


def _ts_single(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


# Real join description in world.ts (start-of-line, not the `// description:` comment).
_WORLD_DESC_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)description: `(?:[^`\\]|\\.)*`,$"
)
_DEFAULT_NAME_RE = re.compile(r"(?m)^export const DEFAULT_NAME = '(?:[^'\\]|\\.)*';$")


def patch_juniper(cards: dict) -> list[str]:
    """Splice Juniper (human player) into constants.ts (DEFAULT_NAME) and
    world.ts (join description) so those patches are reproducible from the cards.
    Returns the list of changed file labels."""
    by_id = {c["id"]: c for c in cards["characters"]}
    juniper = by_id["juniper_feld"]
    changed: list[str] = []

    const_src = CONSTANTS_TS.read_text(encoding="utf-8")
    new_const, n = _DEFAULT_NAME_RE.subn(
        f"export const DEFAULT_NAME = {_ts_single(juniper['name'])};", const_src
    )
    if n != 1:
        raise SystemExit(f"constants.ts: expected 1 DEFAULT_NAME, found {n}")
    if new_const != const_src:
        CONSTANTS_TS.write_text(new_const, encoding="utf-8")
        changed.append("convex/constants.ts")

    world_src = WORLD_TS.read_text(encoding="utf-8")
    desc = _ts_backtick(compose_presence_blurb(juniper))
    new_world, n = _WORLD_DESC_RE.subn(
        lambda m: f"{m.group('indent')}description: `{desc}`,", world_src, count=1
    )
    if n != 1:
        raise SystemExit(f"world.ts: expected 1 join description, found {n}")
    if new_world != world_src:
        WORLD_TS.write_text(new_world, encoding="utf-8")
        changed.append("convex/world.ts")
    return changed


def main() -> int:
    cards = yaml.safe_load(CARDS.read_text(encoding="utf-8"))
    src = CHARACTERS_TS.read_text(encoding="utf-8")

    start = src.index("export const Descriptions = [")
    end_marker = "export const characters ="
    end = src.index(end_marker)
    # keep everything after the Descriptions array closes, before `characters`
    tail = src[end:]
    head = src[:start]
    new_block = render_descriptions(cards)
    new_src = f"{head}{new_block}\n\n{tail}"
    if new_src != src:
        CHARACTERS_TS.write_text(new_src, encoding="utf-8")
        print(f"updated {CHARACTERS_TS.relative_to(_SERVICE.parents[1])}")
    else:
        print("characters.ts already up to date")

    # Emit generated blurbs for players seeded outside `Descriptions`:
    #  - orion_town_card.txt   -> consumed by the embodiment bootstrap
    #  - juniper_description.txt -> reference for the world.ts join description
    by_id = {c["id"]: c for c in cards["characters"]}
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "orion_town_card.txt").write_text(
        compose_presence_blurb(by_id["orion"]) + "\n", encoding="utf-8"
    )
    (GENERATED / "juniper_description.txt").write_text(
        compose_presence_blurb(by_id["juniper_feld"]) + "\n", encoding="utf-8"
    )
    print(f"wrote {GENERATED.relative_to(_SERVICE.parents[1])}/orion_town_card.txt")

    # Juniper (human) join description + DEFAULT_NAME in the Convex source.
    juniper_changed = patch_juniper(cards)
    for label in juniper_changed:
        print(f"updated upstream/{label}")
    if not juniper_changed:
        print("world.ts/constants.ts already up to date")

    # sanity: count rendered NPC entries directly (indentation-independent)
    n = new_block.count("\n    character: '")
    print(f"Descriptions entries: {n}")
    return 0 if n == len(NPC_ORDER) else 1


if __name__ == "__main__":
    raise SystemExit(main())
