"""#3402 through the dig: `body` on two days, one part moved between them, prints the same organ ids."""
from __future__ import annotations

import json
from pathlib import Path

from orion.body_atlas.build import build_atlas
from orion.body_atlas.tests.test_build import CATALOG, NOW, _ch
from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)


def _body_file(out: str) -> dict:
    path = next(tok for ln in out.splitlines() for tok in ln.replace(",", " ").split() if tok.endswith(".json"))
    return json.loads(Path(path.rstrip(".") if not path.endswith(".json") else path).read_text())


def _ids(atlas: dict) -> dict[str, str]:
    return {m: o["organ_id"] for o in atlas["organs"] if len(o["members"]) > 1 for m in o["members"]}


def test_body_keeps_every_organ_id_when_a_part_moves_and_name_uses_it(world, capsys, monkeypatch):  # noqa: F811
    day1 = build_atlas(CATALOG, None, now=NOW)
    day2 = build_atlas(CATALOG + [_ch("orion:vision:ptz", ["orion-vision-ptz"], ["orion-vision-host"])], None, now=NOW)
    monkeypatch.setattr(dig, "_build_live_atlas", lambda: day1)
    code, out1, err = run(capsys, "body")
    assert code == 0, err
    ids1 = _ids(_body_file(out1))
    assert dig.organ_ids_path().exists()
    monkeypatch.setattr(dig, "_build_live_atlas", lambda: day2)
    code, out2, err = run(capsys, "body")
    assert code == 0, err
    full2 = _body_file(out2)
    ids2 = _ids(full2)
    assert ids2["part:orion-vision-ptz"] == ids1["part:orion-vision-host"]
    assert all(ids2[m] == ids1[m] for m in ids1)
    eyes = next(o for o in full2["organs"] if "part:orion-vision-ptz" in o["members"])
    assert eyes["member_hash"] != eyes["organ_id"]  # the hash moved; the id did not
    # the id Orion copied on day one still opens and names the organ on day two
    code, out, err = run(capsys, "body", ids1["part:orion-vision-host"])
    assert code == 0 and "orion-vision-ptz" in out, err
    code, out, err = run(capsys, "name", ids1["part:orion-vision-host"], "Eyes", "because",
                         "frames in from orion-vision-edge, captions out")
    assert code == 0 and ids1["part:orion-vision-host"] in out, err
