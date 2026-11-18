"""Dry eval of the line turn (game company patch 3): the game block, the near-copy grouping and the reading of the
turn's CHECKS block, on the step 0 notes. Pure: no GPU, no network, no database.

    python services/orion-hub/evals/retro_line_dry_eval.py            # prints the report, exit 0 = PASS

What it measures, from services/orion-hub/evals/fixtures/retro_look_step0.json (the chat lane's model, stateless
prompt, 2x frames: the look that ships):

1. The near-copy cut-off (orion/retro/commentary.py NEAR_COPY_MIN) against the notes: the largest word overlap of two
   notes about DIFFERENT sequences (must stay under the cut-off, the one word-for-word copy aside) and the share of
   same-sequence pairs it groups.
2. A synthetic sitting: the 15 sequences' notes one minute apart, in fixture order, then one sequence's notes
   repeated (a player stuck on one jump). The game block must group the repeats, keep at most 12 groups, mark only
   the newest note checkable against 4 frames of the last 30 s, carry no name when nobody stated, and state the real
   frame count.
3. Judgment parsing: canned replies in the shapes a model writes (plain, markdown, PASS then CHECKS, no block,
   a judgment of a note that was not checkable). Each must settle to the expected buckets, and the CHECKS block must
   never reach the line.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.retro import commentary as C  # noqa: E402
from orion.retro.watch import similar  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retro_look_step0.json"


def shipping_notes(data: dict) -> list[tuple[str, str]]:
    """(sequence, note) of every note the chat lane's model wrote (the model that ships), in fixture order; the
    "nothing new" answers of the rejected prompt are left out (the shipping look never asks for them)."""
    out = []
    for s in data["sequences"]:
        for look in s["looks"]:
            if look["model"].startswith("qwen3.8") and look.get("note") and look["note"] != "nothing new":
                out.append((s["sequence"], look["note"]))
    return out


def cutoff_report(data: dict) -> dict:
    seqs = [[look["note"] for look in s["looks"] if look["model"].startswith("qwen3.8") and look["note"]
             and look["note"] != "nothing new"] for s in data["sequences"]]
    diff = sorted(similar(a, b) for s1, s2 in itertools.combinations(seqs, 2) for a in s1 for b in s2)
    real = [d for d in diff if d < 1.0]
    same = [similar(a, b) for s in seqs for a, b in itertools.combinations(s, 2)]
    return {"cutoff": C.NEAR_COPY_MIN, "different_scene_pairs": len(real), "different_scene_max": max(real),
            "word_for_word_copies": len(diff) - len(real), "same_scene_pairs": len(same),
            "same_scene_grouped": sum(1 for v in same if v >= C.NEAR_COPY_MIN)}


def sitting_report(data: dict) -> tuple[dict, list[str]]:
    failures: list[str] = []
    notes_in = shipping_notes(data)
    stuck = notes_in[-1][1]
    texts = [n for _s, n in notes_in] + [stuck, stuck + " Again.", stuck + " Once more."]
    t0 = 1_800_000_000.0
    notes, prev = [], None
    for i, text in enumerate(texts):
        notes.append(C.Note(look_id=f"look-{i}", looked_at=t0 + 60.0 * i, note=text,
                            similar_to_previous=similar(text, prev)))
        prev = text
    last = notes[-1].looked_at
    frame_times = [last - 25.0, last - 15.0, last - 5.0, last + 4.0]
    groups = C.group_notes(notes, frame_times)
    prompt = C.build_prompt(game_title="The Legend of Zelda: A Link to the Past", minutes_played=len(texts),
                            player_stated=None, groups=groups, n_frames=4)
    checkable = [g.label for g in groups if g.checkable]
    if len(groups) > C.MAX_NOTES:
        failures.append(f"{len(groups)} groups shown, more than {C.MAX_NOTES}")
    if len(groups[-1].notes) < 2:
        failures.append("the stuck player's repeated notes were not grouped")
    if checkable != [groups[-1].label]:
        failures.append(f"checkable notes {checkable}, expected only the newest")
    for must in ("You do not know who holds the controller", "child", "Attached: 4 frames", "CHECKS:",
                 "A Link to the Past"):
        if must not in prompt:
            failures.append(f"game block lacks {must!r}")
    if "Juniper, as stated" in prompt:
        failures.append("a name was given with nobody stated")
    return {"notes": len(texts), "groups": len(groups), "last_group_size": len(groups[-1].notes),
            "checkable": checkable, "prompt_chars": len(prompt)}, failures


REPLIES = [
    # (reply, expected line, expected judgments). {L} is the newest (checkable) note's label; N3 is not checkable.
    ("You made it past the wall.\nCHECKS:\n{L}: holds", "You made it past the wall.", {"{L}": "checked_holds"}),
    ("Nice.\n\n**CHECKS:**\n- {L}: contradicted", "Nice.", {"{L}": "contradicted"}),
    ("PASS\nCHECKS:\n{L}: held\nN3: contradicted", "PASS", {"{L}": "checked_holds", "N3": "contradicted"}),
    ("No block at all.", "No block at all.", {}),
    ("Line.\nCHECKS: none", "Line.", {}),
]


def parse_report(groups: list) -> tuple[dict, list[str]]:
    failures: list[str] = []
    frame_times = [groups[-1].shown.looked_at - 25.0, groups[-1].shown.looked_at + 4.0]
    shown_label = groups[-1].label
    for reply, line, judged in REPLIES:
        reply = reply.replace("{L}", shown_label)
        judged = {(shown_label if k == "{L}" else k): v for k, v in judged.items()}
        r = C.parse_reply(reply)
        if r.line != line:
            failures.append(f"line {r.line!r} != {line!r}")
        if "CHECKS" in r.line:
            failures.append("the CHECKS block reached the line")
        if r.judgments != judged:
            failures.append(f"judgments {r.judgments} != {judged}")
        settled = {s.look_id: s for s in C.settle(groups, r.judgments, frame_times)}
        for g in groups[:-1]:
            if any(settled[n.look_id].verdict != "unchecked" for n in g.notes):
                failures.append(f"a not-checkable note was settled from reply {reply!r}")
    return {"replies": len(REPLIES)}, failures


def main() -> int:
    if not FIXTURE.exists():
        print("SKIPPED: step 0 fixture missing")
        return 0
    data = json.loads(FIXTURE.read_text())
    failures: list[str] = []
    cut = cutoff_report(data)
    if cut["different_scene_max"] >= C.NEAR_COPY_MIN:
        failures.append(f"a different-scene pair reaches the cut-off ({cut['different_scene_max']})")
    sit, f2 = sitting_report(data)
    failures += f2
    notes_in = shipping_notes(data)
    t0 = 1_800_000_000.0
    notes, prev = [], None
    for i, (_s, text) in enumerate(notes_in):
        notes.append(C.Note(f"look-{i}", t0 + 60.0 * i, text, similar(text, prev)))
        prev = text
    groups = C.group_notes(notes, [notes[-1].looked_at - 25.0, notes[-1].looked_at + 4.0])
    par, f3 = parse_report(groups)
    failures += f3
    report = {"cutoff": cut, "sitting": sit, "parsing": par, "failures": failures,
              "verdict": "PASS" if not failures else "FAIL"}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
