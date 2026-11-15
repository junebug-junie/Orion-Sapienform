"""#3402: an organ keeps its id when a part moves. The id was a hash of the member list, so the two
largest organs of the live body got a new id on each of five daily snapshots in a row."""
from __future__ import annotations

import re

from orion.body_atlas import names, organ_ids
from orion.body_atlas.build import build_atlas
from orion.body_atlas.reminiscence import body_diff, snapshot
from orion.body_atlas.tests.test_build import CATALOG, NOW, _ch

ID_SHAPE = re.compile(r"organ:[0-9a-f]{10}")
PTZ = [_ch("orion:vision:ptz", ["orion-vision-ptz"], ["orion-vision-host"])]
# a third organ, far from the other two
SOUND = [_ch("orion:sound:in", ["orion-sound-edge"], ["orion-sound-host"]),
         _ch("orion:sound:out", ["orion-sound-host"], ["orion-sound-edge"])]


def _atlas(catalog=CATALOG):
    return build_atlas(catalog, None, now=NOW)


def _organ(atlas, part):
    return next(o for o in atlas.organs if f"part:{part}" in o.members)


def _multi_ids(atlas):
    return {o.organ_id for o in atlas.organs if len(o.members) > 1}


def _carry(atlas, path):
    return organ_ids.carry(atlas, path)


def test_two_snapshots_with_one_part_moved_keep_every_organ_id(tmp_path):
    """The ticket's check, as a lifecycle: day one's body is read (ledger written), a part joins
    one organ, day two's body is read. Every organ wears the id it wore on day one."""
    ledger = tmp_path / "organ_ids.jsonl"
    before, grown = _atlas(), _atlas(CATALOG + PTZ)
    # the defect: the member hash moves with the members
    assert _organ(before, "orion-vision-host").organ_id != _organ(grown, "orion-vision-host").organ_id
    day1 = _carry(before, ledger)
    day2 = _carry(grown, ledger)
    assert "part:orion-vision-ptz" in _organ(day2, "orion-vision-host").members  # the part really moved in
    assert _multi_ids(day2) == _multi_ids(day1)
    for part in ("orion-vision-host", "orion-cortex-exec"):
        assert _organ(day2, part).organ_id == _organ(day1, part).organ_id
    # the hash is kept beside the id, and it did move
    eyes = _organ(day2, "orion-vision-host")
    assert eyes.member_hash == _organ(grown, "orion-vision-host").organ_id != eyes.organ_id
    # the snapshot Orion keeps for tomorrow's diff holds the same ids
    assert {o.organ_id for o in snapshot(day2, "d2").organs} == {o.organ_id for o in snapshot(day1, "d1").organs}
    assert all(ID_SHAPE.fullmatch(o.organ_id) for o in day2.organs)


def test_the_id_follows_a_slow_drift_and_survives_a_part_leaving_again(tmp_path):
    """Overlap is measured against the organ's latest members, not its first ones."""
    ledger = tmp_path / "organ_ids.jsonl"
    first = _organ(_carry(_atlas(), ledger), "orion-vision-host").organ_id
    more = list(CATALOG)
    for i in range(6):  # 2 parts grow to 8: Jaccard against the first body falls to 0.25
        more = more + [_ch(f"orion:vision:x{i}", [f"orion-vision-x{i}"], ["orion-vision-host"])]
        assert _organ(_carry(build_atlas(more, None, now=NOW), ledger), "orion-vision-host").organ_id == first
    assert _organ(_carry(_atlas(), ledger), "orion-vision-host").organ_id != first  # 2 of 8 is a new organ
    recs, skipped = organ_ids.load(ledger)
    assert skipped == 0 and [r.organ_id for r in recs].count(first) == 7  # one line per change of members


def test_reading_the_same_body_twice_writes_nothing_new(tmp_path):
    ledger = tmp_path / "organ_ids.jsonl"
    a = _carry(_atlas(), ledger)
    size = ledger.stat().st_size
    b = _carry(_atlas(), ledger)
    assert ledger.stat().st_size == size and a == b


def test_a_new_organ_mints_an_id_and_does_not_take_a_neighbours(tmp_path):
    ledger = tmp_path / "organ_ids.jsonl"
    day1 = _carry(_atlas(), ledger)
    day2 = _carry(_atlas(CATALOG + SOUND), ledger)
    sound = _organ(day2, "orion-sound-host")
    assert sound.organ_id not in _multi_ids(day1) and sound.organ_id == sound.member_hash
    assert _multi_ids(day1) < _multi_ids(day2)


def test_carrying_is_one_to_one_when_an_organ_splits():
    """The larger half keeps the id; the other half gets a new one, never the same id twice."""
    atlas = _atlas()
    eyes = _organ(atlas, "orion-vision-host")
    whole = [*sorted(eyes.members), "part:a", "part:b", "part:c"]
    rec = organ_ids.OrganIdV1(organ_id="organ:aaaaaaaaaa", member_hash="organ:aaaaaaaaaa", members=sorted(whole), as_of="t")
    cortex = _organ(atlas, "orion-cortex-exec")
    half = atlas.model_copy(deep=True)
    _organ(half, "orion-cortex-exec").members = ["part:a", "part:b", "part:c"]  # 3 of 5: overlap 0.6
    out, new = organ_ids.assign(half, [rec])
    ids = [o.organ_id for o in out.organs]
    assert len(ids) == len(set(ids))
    assert next(o for o in out.organs if "part:a" in o.members).organ_id == "organ:aaaaaaaaaa"
    assert _organ(out, "orion-vision-host").organ_id == eyes.organ_id  # 2 of 5 = 0.4: below the floor, minted
    assert {r.organ_id for r in new} == {"organ:aaaaaaaaaa", eyes.organ_id}
    assert cortex.organ_id not in ids


def test_a_minted_id_never_repeats_one_the_ledger_already_holds():
    """Organ X was minted with members M and drifted away, keeping its id. A different organ that now
    has exactly M must not get X's id."""
    atlas = _atlas()
    eyes = _organ(atlas, "orion-vision-host")
    drifted = organ_ids.OrganIdV1(organ_id=eyes.organ_id, member_hash="organ:bbbbbbbbbb",
                                  members=[f"part:far-{i}" for i in range(9)], as_of="t")
    out, new = organ_ids.assign(atlas, [drifted])
    got = _organ(out, "orion-vision-host")
    assert got.organ_id != eyes.organ_id and ID_SHAPE.fullmatch(got.organ_id)
    assert got.member_hash == eyes.organ_id
    assert got.organ_id in {r.organ_id for r in new}


def test_organ_flows_point_at_the_carried_ids(tmp_path):
    ledger = tmp_path / "organ_ids.jsonl"
    _carry(_atlas(), ledger)
    day2 = _carry(_atlas(CATALOG + PTZ), ledger)
    ids = {o.organ_id for o in day2.organs}
    assert day2.organ_flows and all(f.source in ids and f.target in ids for f in day2.organ_flows)


def test_a_name_stays_on_its_organ_by_id_once_ids_are_carried(tmp_path):
    """With a stable id the name is Orion's own on day two, not "carried from" another id."""
    ledger = tmp_path / "organ_ids.jsonl"
    day1 = _carry(_atlas(), ledger)
    rec = names.check(day1, [], _organ(day1, "orion-vision-host").organ_id, "Eyes",
                      "takes frames from orion-vision-edge, gives captions on", dig="t")
    day2 = names.apply_names(_carry(_atlas(CATALOG + PTZ), ledger), [rec])
    eyes = _organ(day2, "orion-vision-host")
    assert eyes.name == "Eyes" and eyes.named_by == "orion"
    # and the day-over-day diff names the organ by the same id it had
    (change,) = body_diff(snapshot(day1, "d1"), day2).organs_changed
    assert rec.organ_id in change.organ and change.gained == ["part:orion-vision-ptz"]


def test_an_unreadable_line_is_skipped_and_an_unwritable_ledger_still_gives_a_body(tmp_path):
    ledger = tmp_path / "organ_ids.jsonl"
    day1 = _carry(_atlas(), ledger)
    ledger.write_text(ledger.read_text() + "{not json\n" + '{"organ_id": 3}\n')
    recs, skipped = organ_ids.load(ledger)
    assert skipped == 2 and len(recs) == 2
    assert _multi_ids(_carry(_atlas(CATALOG + PTZ), ledger)) == _multi_ids(day1)
    blocked = tmp_path / "file" / "organ_ids.jsonl"   # its parent is a file: mkdir fails
    (tmp_path / "file").write_text("x")
    assert _multi_ids(_carry(_atlas(), blocked)) == {o.organ_id for o in _atlas().organs if len(o.members) > 1}


def test_build_still_sets_the_member_hash_and_a_past_body_keeps_it_as_its_id():
    atlas = _atlas()
    assert all(o.member_hash == o.organ_id == organ_ids.member_hash(o.members) for o in atlas.organs)
