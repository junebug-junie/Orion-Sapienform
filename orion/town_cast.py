from __future__ import annotations

ORION_DISPLAY_NAME = "Orion"

TOWN_PARTICIPANT_SLUGS: dict[str, str] = {
    "Mara Vale": "mara-vale",
    "Nico Sable": "nico-sable",
    "Sofia Bell": "sofia-bell",
    "Cam Lin": "cam-lin",
    "Juniper Feld": "juniper-feld",
    "Orion": "orion",
}


def slug_for_name(name: str) -> str | None:
    key = str(name or "").strip()
    if not key:
        return None
    return TOWN_PARTICIPANT_SLUGS.get(key)


def thread_id_for(name_a: str, name_b: str) -> str | None:
    left = slug_for_name(name_a)
    right = slug_for_name(name_b)
    if left is None or right is None:
        return None
    return "--".join(sorted((left, right)))
