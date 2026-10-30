"""`dig name-organs`: one read-only command that lists what is left to name, with ready lines."""
from __future__ import annotations

import json
import re
from pathlib import Path

from orion.body_atlas import names
from orion.dig import __main__ as dig
from orion.dig.tests.test_body_whole import _whole_atlas
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)


def _organ(atlas, part):
    return next(o for o in atlas.organs if part in o.members)


def _file(out: str) -> dict:
    path = out.strip().splitlines()[-1].split(": ", 1)[1]
    return json.loads(Path(path).read_text())


def test_lists_unnamed_organs_largest_first_with_parts_and_a_name_line(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    eyes, cortex = _organ(atlas, "part:camera:walkway"), _organ(atlas, "part:orion-cortex-exec")
    single = next(o for o in atlas.organs if len(o.members) == 1)
    code, out, _ = run(capsys, "name-organs")
    assert code == 0
    assert "2 of 2 multi-part organs have no name" in out
    assert "takes in and gives out" in out and "cite" in out
    assert out.index(eyes.organ_id) < out.index(cortex.organ_id)  # 3 parts before 2
    for part in ("orion-vision-edge", "orion-vision-host", "camera:walkway", "orion-cortex-orch", "orion-cortex-exec"):
        assert part in out
    assert f'python -P -m orion.dig name {eyes.organ_id} "<name>" because "<reason citing a part>"' in out
    assert single.organ_id not in out and "orion-equilibrium-service" not in out
    full = _file(out)
    assert [o["organ_id"] for o in full["unnamed"]] == [eyes.organ_id, cortex.organ_id]
    assert full["unnamed"][0]["members"] == ["orion-vision-edge", "orion-vision-host", "camera:walkway"]
    assert not dig.names_path().exists()  # read-only: no name written


def test_named_organs_drop_out_and_the_name_line_it_prints_works(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    eyes, cortex = _organ(atlas, "part:camera:walkway"), _organ(atlas, "part:orion-cortex-exec")
    code, _, err = run(capsys, "name", eyes.organ_id, "Eyes", "because", "frames in from orion-vision-edge, captions out")
    assert code == 0, err
    code, out, _ = run(capsys, "name-organs")
    assert code == 0 and "1 of 2 multi-part organs have no name" in out
    assert eyes.organ_id not in out and cortex.organ_id in out


def test_all_named_says_so_and_lists_the_names(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    eyes, cortex = _organ(atlas, "part:camera:walkway"), _organ(atlas, "part:orion-cortex-exec")
    for rec in (names.check(atlas, [], eyes.organ_id, "Eyes", "frames in from orion-vision-edge", dig="t"),
                names.check(atlas, [], cortex.organ_id, "Planner", "plans go from orion-cortex-orch out", dig="t")):
        names.append(dig.names_path(), rec)
    code, out, _ = run(capsys, "name-organs")
    assert code == 0 and "all 2 multi-part organs are named" in out
    assert f'"Eyes" {eyes.organ_id}' in out and f'"Planner" {cortex.organ_id}' in out
    assert "<name>" not in out
    code, out, _ = run(capsys, "body")
    assert code == 0 and "unnamed" not in out


def test_body_points_at_name_organs_when_any_are_unnamed(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    code, out, _ = run(capsys, "body")
    assert code == 0 and "2 organs unnamed: python -P -m orion.dig name-organs" in out
    assert out.strip().splitlines()[-1].startswith("full atlas: ")  # the file path stays last
    eyes = _organ(atlas, "part:camera:walkway")
    run(capsys, "name", eyes.organ_id, "Eyes", "because", "frames in from orion-vision-edge")
    assert "1 organ unnamed: python -P -m orion.dig name-organs" in run(capsys, "body")[1]


def test_name_organs_printout_stays_capped_on_a_big_body(world, capsys, monkeypatch):  # noqa: F811
    from orion.body_atlas.model import OrganV1, PartV1

    atlas = _whole_atlas()
    parts = [PartV1(part_id=f"part:svc-{i:03d}-" + "x" * 30, name=f"svc-{i:03d}-" + "x" * 30, kind="service", declared=True,
                    status="live") for i in range(400)]
    organs = [OrganV1(organ_id=f"organ:{i:010d}", members=[p.part_id for p in parts[i * 20:(i + 1) * 20]],
                      hub=parts[i * 20].part_id, internal_weight=1.0, external_weight=0.0) for i in range(20)]
    big = atlas.model_copy(update={"parts": parts, "organs": organs})
    monkeypatch.setattr(dig, "_live_atlas", lambda: big)
    code, out, _ = run(capsys, "name-organs")
    assert code == 0 and len(out.rstrip("\n").splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS + 1
    assert "more in the file" in out and "of 20 unnamed organs" in out
    shown = int(re.search(r"showing (\d+) of 20 unnamed organs", out).group(1))
    assert out.count("python -P -m orion.dig name organ:") == shown  # every shown block is whole
    assert len(_file(out)["unnamed"]) == 20 and len(_file(out)["unnamed"][0]["members"]) == 20


def test_help_lists_name_organs(capsys):
    assert "name-organs" in run(capsys, "help")[1]


def test_rule_states_the_limits_name_checks(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _whole_atlas)
    out = run(capsys, "name-organs")[1]
    assert f"at most {names.NAME_MAX_CHARS} chars" in out
    assert f"({names.BECAUSE_MIN_CHARS}-{names.BECAUSE_MAX_CHARS} chars)" in out


def test_an_unreadable_ledger_line_is_said_in_every_branch(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    dig.names_path().parent.mkdir(parents=True, exist_ok=True)
    dig.names_path().write_text("{not json\n")
    out = run(capsys, "name-organs")[1]
    assert "1 unreadable line(s) skipped" in out
    out = run(capsys, "body")[1]
    assert "1 unreadable line(s) skipped" in out and out.strip().splitlines()[-1].startswith("full atlas: ")
    for organ in (o for o in atlas.organs if len(o.members) > 1):
        part = next(p.name for p in atlas.parts if p.part_id == organ.members[0])
        assert run(capsys, "name", organ.organ_id, "Thing", "because", f"it is fed by {part} mostly")[0] == 0
    out = run(capsys, "name-organs")[1]
    assert "all 2 multi-part organs are named" in out and "1 unreadable line(s) skipped" in out


def test_a_huge_first_part_name_still_shows_one_part_to_cite(world, capsys, monkeypatch):  # noqa: F811
    from orion.body_atlas.model import OrganV1, PartV1

    atlas = _whole_atlas()
    parts = [PartV1(part_id=f"part:{c * 400}", name=c * 400, kind="service", declared=True, status="live")
             for c in "ab"]
    organ = OrganV1(organ_id="organ:0000000001", members=[p.part_id for p in parts], hub=parts[0].part_id,
                    internal_weight=1.0, external_weight=0.0)
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas.model_copy(update={"parts": parts, "organs": [organ]}))
    out = run(capsys, "name-organs")[1]
    assert "parts: " + "a" * dig.NAME_ORGANS_MEMBERS_CHARS + " (+1 more in the file)" in out
    assert 'name organ:0000000001 "<name>"' in out
