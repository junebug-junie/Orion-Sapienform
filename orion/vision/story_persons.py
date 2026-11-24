"""One person per linked group on the walkway story card (who-card redesign, Patch A).

docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md. The tracker often
splits one passer-by over two to four camera groups (``vision_individual`` rows).
When Juniper already said those groups are the same person (a ``same`` row in
``vision_individual_pair_verdict``, made by her "same person?" answer or by the
same-name link when she typed a name she used before), the card shows them as ONE
person and asks once. The answer is written for each camera group in it: one
already-answered ``orion_ask`` row per group, same ``story_id``, so orion-sql-writer
and the split need no change.

Pure and stdlib-only, so the Hub route (which writes the rows) and the sql-writer
tests (which apply them) share one copy of the row shape (``person_answer``).

Rules this module owns:

- **Grouping** (``group_people``): people whose ``same`` components match are one
  person, unless they carry two different names (then each name is its own person;
  a link that joins two typed names is a conflict the writer already logs).
- **Pictures, max 4** (``MAX_PICTURES``): a grouped person shows at most four
  pictures, spread over the passage, the largest box first. Each camera group is
  answered only on its own pictures that were SHOWN; a group with none shown is
  not answered (never a blind answer). So one answer writes at most four rows.
- **Idempotency** (``already_said``): a group whose latest answer on this story
  already says the same thing gets no second row (Orion's question 4).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

WHO_IS_THIS_KIND = "vision_individual"
STORY_PERSON_KIND = "vision_story_person"
SAME_PERSON_KIND = "vision_same_individual"
PAIR_SEP = "|"                      # == app/vision_individuals.PAIR_SEP (a test pins it)
MAX_PICTURES = 4                    # a story strip holds at most crop_0..crop_3
PERSON_ACTIONS = ("name", "confirm", "someone_else", "unsure")


def pair_ref(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}{PAIR_SEP}{hi}"


def link_roots(edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """individual_id -> the smallest id in its ``same`` component (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in list(parent)}


def connected(edges: Iterable[tuple[str, str]], a: str, b: str) -> bool:
    roots = link_roots(edges)
    return roots.get(a, a) == roots.get(b, b)


def _fold(name: Any) -> str:
    return " ".join(str(name or "").split()).casefold()


def _spread(items: Sequence[Any], n: int) -> list[Any]:
    """``n`` items spread over ``items`` (first and last kept), in order."""
    if len(items) <= n:
        return list(items)
    if n <= 1:
        return [items[0]]
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def _area(box: Any) -> float:
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def crop_areas(evidences: Iterable[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    """crop_id -> box area, over every evidence list given."""
    out: dict[str, float] = {}
    for ev in evidences:
        for e in ev or []:
            if isinstance(e, Mapping) and e.get("crop_id"):
                out.setdefault(str(e["crop_id"]), _area(e.get("box")))
    return out


def _person_pictures(p: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The pictures one camera group's row would show: its strip crops, else its thumbnail."""
    pics = [dict(c, individual_id=p["individual_id"]) for c in p.get("pictures") or []]
    if not pics and p.get("thumb") and p.get("thumb_crop_id") and p.get("pictured"):
        pics = [{"thumb": p["thumb"], "crop_id": p["thumb_crop_id"], "t": None, "individual_id": p["individual_id"]}]
    return pics


def group_people(people: Sequence[Mapping[str, Any]], roots: Mapping[str, str],
                 areas: Optional[Mapping[str, float]] = None,
                 links: Iterable[tuple[str, str]] = ()) -> list[dict[str, Any]]:
    """The story's ``people`` (one per camera group, first-seen order) as persons.

    ``roots``: individual_id -> ``same`` component root (``link_roots``). ``areas``: crop_id
    -> box area (the first picture is the largest). ``links``: the direct ``same`` rows
    between two of these people (``not same person?`` offers one button per row).
    """
    areas = areas or {}
    buckets: list[list[Mapping[str, Any]]] = []
    by_root: dict[str, list[list[Mapping[str, Any]]]] = {}
    for p in people:
        iid = p["individual_id"]
        root = roots.get(iid, iid)
        name = _fold(p.get("label"))
        home = None
        for b in by_root.get(root, []):
            names = {_fold(m.get("label")) for m in b} - {""}
            if not name or not names or names == {name}:
                home = b
                break
        if home is None:
            home = []
            by_root.setdefault(root, []).append(home)
            buckets.append(home)
        home.append(p)
    link_set = {tuple(sorted(e)) for e in links}
    out = []
    for members in buckets:
        ids = [m["individual_id"] for m in members]
        # Every picture of every group, deduped, in time order; then at most four, spread.
        pics: list[dict[str, Any]] = []
        seen: set[str] = set()
        for m in members:
            for c in _person_pictures(m):
                if c.get("crop_id") and c["crop_id"] not in seen:
                    seen.add(c["crop_id"])
                    pics.append(c)
        pics.sort(key=lambda c: (c.get("t") or "", ids.index(c["individual_id"])))
        shown = _spread(pics, MAX_PICTURES)
        if shown:
            big = max(range(len(shown)), key=lambda i: (areas.get(shown[i]["crop_id"], 0.0), -i))
            shown.insert(0, shown.pop(big))
        shown_ids = {c["crop_id"] for c in shown}
        groups = []
        for m in members:
            mine = [cid for cid in m.get("picture_crop_ids") or [] if cid in shown_ids]
            groups.append({"individual_id": m["individual_id"], "short_id": m.get("short_id"),
                           "role": m.get("role"), "label": m.get("label"),
                           "label_copied": bool(m.get("label_copied")), "boxes": list(m.get("boxes") or []),
                           "picture_crop_ids": mine, "asked": bool(mine), "story_action": m.get("story_action")})
        named = next((m for m in members if m.get("label") and not m.get("label_copied")), None) \
            or next((m for m in members if m.get("label")), None)
        actions = [g["story_action"] for g in groups if g["story_action"]]
        actions.sort(key=lambda a: (a.get("answered_at") or ""))
        out.append({
            "person_key": min(ids),
            "individual_ids": ids,
            "crop_ids": [c for m in members for c in m.get("crop_ids") or []],
            "role": "walker" if any(m.get("role") == "walker" for m in members) else "companion",
            "kind": members[0].get("kind") or "person",
            "label": named.get("label") if named else None,
            "label_copied": bool(named and named.get("label_copied")),
            "short_id": members[0].get("short_id"),
            "story_id": members[0].get("story_id"),
            "pictures": shown,
            "boxes": sorted((b for g in groups for b in g["boxes"]), key=lambda b: b.get("n", 0)),
            "pictured": any(g["asked"] for g in groups),
            "groups": groups,
            "asked_ids": [g["individual_id"] for g in groups if g["asked"]],
            "story_action": actions[-1] if actions else None,
            "links": [{"a": a, "b": b} for a, b in sorted(link_set) if a in ids and b in ids],
        })
    return out


def already_said(latest: Optional[Mapping[str, Any]], action: str, name: str) -> bool:
    """Would this answer repeat what this camera group's latest answer on the story says?

    A name and a yes are both a ``vision_individual`` answer: a second one with the same
    name would be a second who-is-this answer (and duplicate "same" pictures) for one
    (story, individual), so it is skipped (Orion's question 4)."""
    if not latest or not latest.get("action"):
        return False
    said, said_name = latest["action"], _fold(latest.get("name"))
    if action in ("name", "confirm") and said in ("name", "confirm"):
        return said_name == _fold(name)
    return said == action and said_name == _fold(name)


def person_answer(*, action: str, name: str, story_id: str, picture_crop_ids: Sequence[str], why: str = "",
                  person_key: Optional[str] = None, group_size: int = 1) -> dict[str, Any]:
    """The ``orion_ask`` row for one camera group: ``{kind, answer, question, detail}``.

    ``name``: the typed / confirmed name ("" for none). ``picture_crop_ids``: the pictures of
    THIS group the card showed; every picture verdict is about one of them."""
    if action not in PERSON_ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    detail: dict[str, Any] = {"story_id": story_id, "action": action, "via": "motion_story"}
    if name:
        detail["name"] = name
    if why:
        detail["why"] = why
    if person_key and group_size > 1:
        detail["person_key"] = person_key
        detail["group_size"] = int(group_size)
    if action == "name":
        return {"kind": WHO_IS_THIS_KIND, "answer": name, "question": "Who is this? (named on a walkway story)",
                "detail": detail}
    if action == "confirm":
        detail["pictures"] = [{"image_ref": f"crop:{cid}", "index": i, "verdict": "same", "crop_id": cid,
                               "name": name} for i, cid in enumerate(picture_crop_ids)]
        return {"kind": WHO_IS_THIS_KIND, "answer": name,
                "question": f"Is this {name}? Yes (said on a walkway story)", "detail": detail}
    verdict = "different" if action == "someone_else" else "unsure"
    detail["pictures"] = [{"image_ref": f"crop:{cid}", "index": i, "verdict": verdict, "crop_id": cid,
                           "name": name or None} for i, cid in enumerate(picture_crop_ids)]
    question = ("Is this someone else? (said on a walkway story)" if action == "someone_else"
                else "Not sure who this is (said on a walkway story)")
    return {"kind": STORY_PERSON_KIND, "answer": None, "question": question, "detail": detail}


def not_same_answer(*, a: str, b: str, story_id: str, person_key: str) -> dict[str, Any]:
    """Juniper's "these camera groups are not the same person" on a grouped row: the
    existing "same person?" answer (``vision_same_individual``, answer ``different``), which
    sql-writer's pair path writes over the ``same`` row (app/vision_individuals._apply_pair_answer)."""
    return {"kind": SAME_PERSON_KIND, "source_ref": pair_ref(a, b), "answer": "different",
            "question": "Are these the same person? No (said on a walkway story)",
            "detail": {"story_id": story_id, "action": "not_same", "via": "motion_story",
                       "person_key": person_key}}


__all__ = [
    "MAX_PICTURES", "PERSON_ACTIONS", "STORY_PERSON_KIND", "WHO_IS_THIS_KIND", "SAME_PERSON_KIND",
    "already_said", "connected", "crop_areas", "group_people", "link_roots", "not_same_answer",
    "pair_ref", "person_answer",
]
