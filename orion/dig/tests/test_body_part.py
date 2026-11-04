"""`dig body <part>`: one part's flows in and out, and the exact ids a claim on it may cite.

2026-11-04 (turn 00655f13): Orion spent 12 hand-written scripts parsing the body file to list one
part's flows before writing a capacity claim, because no dig command showed them.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from orion.body_atlas import claims
from orion.body_atlas.tests.test_capacities import _body
from orion.dig import __main__ as dig
from orion.dig.tests.test_body_whole import _whole_atlas
from orion.dig.tests.test_dig import run, world  # noqa: F401  (fixture)


def _file(out: str) -> dict:
    return json.loads(Path(out.strip().splitlines()[-1].removeprefix("full result: ")).read_text())


def _cite_line(out: str) -> str:
    return next(ln for ln in out.splitlines() if ln.startswith("cite in a claim (by full id): "))


def test_part_view_lists_flows_in_and_out_strongest_first_and_is_read_only(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    for ref in ("gateway", "part:gateway"):
        code, out, err = run(capsys, "body", ref)
        assert code == 0, err
        assert len(out.splitlines()) <= dig.MAX_LINES and len(out) <= dig.MAX_CHARS + 1
        lines = out.splitlines()
        assert lines[0].startswith("part gateway (part:gateway): service, live") and "organ organ:" in lines[0]
        assert "flows in (6), strongest first:" in out and "  <- r0 signals [llm:request] w=1" in out
        assert "flows out (4), strongest first:" in out and "  -> r0 signals [llm:reply] w=1" in out
        assert "  -> watcher signals [orion:system:health (census)] w=1" in out  # census is marked
    full = _file(out)
    assert len(full["flows_in"]) == 6 and len(full["flows_out"]) == 4
    weights = [r["weight"] for r in full["flows_in"]]
    assert weights == sorted(weights, reverse=True)
    written = sorted(p.relative_to(world / "home").as_posix() for p in (world / "home").rglob("*") if p.is_file())
    assert written == ["dig/t/body-part-gateway-1.json", "dig/t/body-part-gateway-2.json"]  # no ledger


def test_many_flows_are_capped_with_the_rest_in_the_file(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "body", "watcher")
    assert code == 0 and "flows in (12), strongest first:" in out
    assert f"  +{12 - dig.PART_FLOWS_SHOWN} more in the file" in out
    assert len(_file(out)["flows_in"]) == 12
    assert "jobs: S3* candidate (R-census-reader)" in out


def test_citable_ids_are_exactly_what_claims_accepts_and_the_first_one_writes_a_claim(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, out, _ = run(capsys, "body", "gateway")
    assert code == 0
    atlas = _body()
    reading, _, _ = dig._capacity_reading(atlas)
    census = {c.channel for c in reading.census if c.census}
    accepted = claims.evidence_ids(atlas, "part:gateway", census)
    full = _file(out)
    assert set(full["cite"]) == accepted and len(full["cite"]) == len(accepted)
    assert "orion:system:health" not in full["cite"]  # census channel: never a cite
    first = _cite_line(out).removeprefix("cite in a claim (by full id): ").split(", ")[0]
    assert first == full["cite"][0]
    ready = next(ln for ln in out.splitlines() if ln.strip().startswith("python -P -m orion.dig capacity claim "))
    argv = shlex.split(ready)[4:]
    assert argv[:4] == ["capacity", "claim", "gateway", "<job>"] and argv[4] == "because"
    assert f"seen on {first}" in argv[5]
    # the ready line, with the placeholders filled in, is a claim the ledger takes
    because = argv[5].replace("<what it does>", "it answers model calls from the requesters")
    code, out2, err = run(capsys, "capacity", "claim", "gateway", "S3", "because", because)
    assert code == 0, err
    assert out2.startswith(f"claimed gateway S3 (cites: {first})")
    assert dig.claims_path().exists()
    # and the claim now shows on the part
    _, out3, _ = run(capsys, "body", "gateway")
    assert "my claims: S3 claim unconfirmed" in out3


def test_the_ready_line_as_printed_is_refused_for_its_placeholder(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    _, out, _ = run(capsys, "body", "gateway")
    ready = next(ln for ln in out.splitlines() if ln.strip().startswith("python -P -m orion.dig capacity claim "))
    argv = shlex.split(ready)[4:]
    code, _, err = run(capsys, *argv[:3], "S3", *argv[4:])
    assert code == 1 and "placeholder" in err
    assert not dig.claims_path().exists()


def test_unknown_part_is_a_clear_error(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _body)
    code, _, err = run(capsys, "body", "part:nope")
    assert code == 1 and "no part 'nope' in the live body" in err
    code, _, err = run(capsys, "body", "nope")
    assert code == 1 and "no multi-part organ 'organ:nope'" in err and "no part is named 'nope' either" in err


def test_organ_refs_keep_their_meaning(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    organ = next(o for o in atlas.organs if "part:camera:walkway" in o.members)
    for ref in (organ.organ_id, organ.organ_id.removeprefix("organ:")):
        code, out, _ = run(capsys, "body", ref)
        assert code == 0 and out.startswith(f"organ {organ.organ_id}") and "full atlas (flows too):" in out
    code, _, err = run(capsys, "body", "organ:0123456789")
    assert code == 1 and "no multi-part organ" in err and "no part is named" not in err


def test_a_part_with_a_colon_name_and_its_host(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    code, out, err = run(capsys, "body", "camera:walkway")
    assert code == 0, err
    assert out.startswith("part camera:walkway (part:camera:walkway): sensor")
    assert "body-part-camera_walkway-1.json" in out.splitlines()[-1]
    host = next(p for p in atlas.parts if p.kind == "host")
    code, out, _ = run(capsys, "body", host.name)
    assert code == 0 and "not claimable" in out and "cite in a claim" not in out


def test_help_and_capacity_help_point_at_body_part(world, capsys):  # noqa: F811
    _, out, _ = run(capsys, "help")
    assert "body [<date>|<organ>|<part>]" in out
    _, out, _ = run(capsys, "capacity", "help")
    assert "python -P -m orion.dig body <part>" in out
