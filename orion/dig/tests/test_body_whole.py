"""`dig body` on the whole body: one organ's members, a past day's declared body, loud gaps."""
from __future__ import annotations

from orion.dig import __main__ as dig
from orion.dig.tests.test_dig import _commit, run, world  # noqa: F401  (fixture)


def _whole_atlas():
    from orion.body_atlas.build import build_atlas
    from orion.body_atlas.model import SourceReadV1
    from orion.body_atlas.sources import BodyInputs, HostIn, SensorIn
    from orion.body_atlas.tests.test_build import CATALOG, NOW

    body = BodyInputs(hosts=[HostIn(name="athena", aliases=["athena"])],
                      sensors=[SensorIn(name="walkway", readers=[("orion-vision-edge", "x")])],
                      reads=[SourceReadV1(name="live:postgres vision_events", origin="live", status="unread",
                                          detail="permission denied")])
    return build_atlas(CATALOG, None, now=NOW, body=body)


def test_body_with_an_organ_id_lists_its_members(world, capsys, monkeypatch):  # noqa: F811
    atlas = _whole_atlas()
    monkeypatch.setattr(dig, "_live_atlas", lambda: atlas)
    organ = next(o for o in atlas.organs if "part:camera:walkway" in o.members)
    for ref in (organ.organ_id, organ.organ_id.removeprefix("organ:")):
        code, out, _ = run(capsys, "body", ref)
        assert code == 0 and "camera:walkway" in out and "orion-vision-host" in out
        assert "full atlas (flows too):" in out
    code, out, _ = run(capsys, "body")
    assert code == 0 and "UNREAD 1 source(s)" in out and "bodies (machines; not in organs):" in out


def test_body_with_an_unknown_organ_is_a_short_error(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _whole_atlas)
    code, _, err = run(capsys, "body", "0123456789")
    assert code == 1 and "no multi-part organ" in err


def test_a_past_body_reads_that_days_declared_files(world, capsys):  # noqa: F811
    _commit(world / "repo", "2025-07-01T19:00:00Z", {
        "config/biometrics/node_catalog.yaml": "nodes:\n  athena:\n    role: orchestration\n    hardware: HP\n",
    }, "catalog")
    assert run(capsys, "open", "2025-07-01")[0] == 0
    code, out, _ = run(capsys, "body", "2025-07-01")
    assert code == 0 and "athena: orchestration, HP; declared_only" in out
    assert "UNREAD" not in out  # files a past checkout lacks are absent, not failures


def test_body_with_a_non_date_word_is_an_organ_lookup(world, capsys, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dig, "_live_atlas", _whole_atlas)
    code, _, err = run(capsys, "body", "eyes")
    assert code == 1 and "no multi-part organ" in err
