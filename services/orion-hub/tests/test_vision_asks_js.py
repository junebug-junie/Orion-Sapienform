"""vision-asks.js pure helpers for the "Is this <name>?" card (2026-10-20), run in node."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parents[1] / "static" / "js" / "vision-asks.js"


def _node(expr: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = f"const api = require({json.dumps(str(JS))}); process.stdout.write(JSON.stringify({expr}));"
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_person_confirm_card_reads_the_name_from_the_ref() -> None:
    got = _node("""[
      api.isPersonConfirm({source_kind: "vision_person_confirm"}),
      api.isPersonConfirm({source_kind: "vision_same_individual"}),
      api.personConfirmName({source_ref: "ind-1|Juniper"}),
      api.personConfirmName({source_ref: "ind-1|Ana | Bo"}),
      api.personConfirmName({source_ref: "ind-1"}),
      api.askCardViewModel({source_kind: "vision_person_confirm", source_ref: "ind-1|Juniper"}).personName
    ]""")
    assert got == [True, False, "Juniper", "Ana | Bo", "", "Juniper"]


def test_person_confirm_sends_yes_no_or_dismisses_for_cant_tell() -> None:
    got = _node("""[
      api.personConfirmSubmission("yes"),
      api.personConfirmSubmission("no"),
      api.personConfirmSubmission("unsure"),
      api.personConfirmSubmission("")
    ]""")
    assert got == [{"action": "answer", "answer": "yes", "extra": None},
                   {"action": "answer", "answer": "no", "extra": None},
                   {"action": "dismiss", "answer": "", "extra": None},
                   None]


def test_is_this_x_with_several_pictures_sends_one_answer_per_picture() -> None:
    """2026-10-23: the pictures can be of different people, so each gets its own answer."""
    got = _node("""(function () {
      const views = [{raw: "thumb:a"}, {raw: "thumb:b"}, {raw: "thumb:c"}];
      return [
        api.personConfirmPicturesSubmission(views, {0: "different", 1: "same"}),
        api.personConfirmPicturesSubmission(views, {0: "different"}),
        api.personConfirmPicturesSubmission(views, {}),
        api.personConfirmPicturesSubmission([views[0]], {0: "same"})
      ];
    })()""")
    pics = lambda *v: [{"image_ref": f"thumb:{c}", "verdict": x} for c, x in zip("abc", v)]
    assert got == [
        {"action": "answer", "answer": "yes", "extra": {"pictures": pics("different", "same", "unsure")}},
        {"action": "answer", "answer": "no", "extra": {"pictures": pics("different", "unsure", "unsure")}},
        None,   # nothing chosen: refused, never a silent dismissal
        None,
    ]
