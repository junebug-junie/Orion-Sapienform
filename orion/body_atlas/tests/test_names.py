from __future__ import annotations

import pytest

from orion.body_atlas import names
from orion.body_atlas.build import build_atlas
from orion.body_atlas.tests.test_build import CATALOG, NOW


def _atlas(catalog=CATALOG):
    return build_atlas(catalog, None, now=NOW)


def _vision(atlas):
    return next(o for o in atlas.organs if "part:orion-vision-host" in o.members)


def _named(atlas, name="Eyes", because="takes frames from orion-vision-edge, gives captions on"):
    return names.check(atlas, _vision(atlas).organ_id, name, because, dig="t")


def test_a_name_must_cite_its_own_organ():
    atlas = _atlas()
    rec = _named(atlas)
    assert rec.cites == ["orion-vision-edge"] and rec.organ_id == _vision(atlas).organ_id
    with pytest.raises(names.NamingError, match="cite"):
        _named(atlas, because="it is where thinking happens, roughly speaking")
    with pytest.raises(names.NamingError, match="words"):
        _named(atlas, name="the organ that sees all the things")
    with pytest.raises(names.NamingError, match="no multi-part organ"):
        names.check(atlas, "organ:0000000000", "Eyes", "x" * 30, dig="t")


def test_the_short_id_without_prefix_is_accepted():
    atlas = _atlas()
    short = _vision(atlas).organ_id.removeprefix("organ:")
    assert names.check(atlas, short, "Eyes", "fed by orion-vision-edge frames", dig="t").name == "Eyes"


def test_a_name_carries_when_a_part_joins_but_not_onto_a_different_organ():
    before = _atlas()
    rec = _named(before)
    grown = _atlas(CATALOG + [{"name": "orion:vision:ptz", "producer_services": ["orion-vision-ptz"],
                              "consumer_services": ["orion-vision-host"]}])
    organ = _vision(grown)
    assert organ.organ_id != rec.organ_id  # the id moved with the members
    named = names.apply_names(grown, [rec])
    eyes = [o for o in named.organs if o.name == "Eyes"]
    assert len(eyes) == 1 and eyes[0].organ_id == organ.organ_id
    assert eyes[0].named_by.startswith(f"orion, carried from {rec.organ_id}")


def test_the_latest_name_for_an_organ_wins_and_names_are_one_to_one():
    atlas = _atlas()
    first, second = _named(atlas), _named(atlas, name="Sight")
    named = names.apply_names(atlas, [first, second])
    assert [o.name for o in named.organs if o.name] == ["Sight"]
    assert _vision(named).named_by == "orion"


def test_ledger_round_trip(tmp_path):
    path = tmp_path / "self" / "organ_names.jsonl"
    assert names.load(path) == []
    rec = _named(_atlas())
    names.append(path, rec)
    names.append(path, rec)
    assert names.load(path) == [rec, rec]
