"""Whole-body build: hosts stay out of organs, new kinds are normalized, names carry over."""
from __future__ import annotations

import pytest

from orion.body_atlas import names
from orion.body_atlas.__main__ import MAX_CHARS, MAX_LINES, organ_view, summarize
from orion.body_atlas.build import build_atlas, match_part
from orion.body_atlas.model import SourceReadV1
from orion.body_atlas.sources import BodyInputs, HostIn, PlacementIn, ProcessIn, SensorIn, StoreIn
from orion.body_atlas.tests.test_build import CATALOG, NOW, RECENT

USERS = [f"orion-u{i}" for i in range(8)]
WIDE = CATALOG + [{"name": f"orion:u:{u}", "producer_services": [u], "consumer_services": ["orion-cortex-exec"]}
                  for u in USERS]


def _body(**kw) -> BodyInputs:
    base = dict(
        hosts=[HostIn(name="athena", role="orchestration", hardware="HP DL380 G10", last_seen_epoch=RECENT,
                      aliases=["athena"]),
               HostIn(name="carbon-x1", role="juniper_laptop", monitored=False, aliases=["carbon-x1"],
                      cameras=["walkway"])],
        placements=[PlacementIn(service="orion-athena-vision-host", host="athena", evidence="observed_placement"),
                    PlacementIn(service="orion-orion-athena-cortex-orch", host="athena", evidence="observed_placement"),
                    PlacementIn(service="zelda-dojo-live-0", host="athena", evidence="observed_placement")],
        sensors=[SensorIn(name="walkway", readers=[("orion-vision-edge", "x"), ("vision-host", "y")],
                          events=10, last_seen_epoch=RECENT)],
        stores=[StoreIn(name="orion_substrate", store="falkordb", rows=5,
                        links=[(u, "uses", "declared_env", "") for u in USERS]),
                StoreIn(name="empty_graph", store="falkordb", rows=0)],
        processes=[ProcessIn(name="reading.turn", family="durable_workflow", runner="orion-cortex-exec",
                             declared=True)],
        reads=[SourceReadV1(name="live:postgres vision_events", origin="live", status="unread",
                            detail="permission denied for table vision_events")],
        known_gaps=["graphify: host only"],
    )
    base.update(kw)
    return BodyInputs(**base)


def _atlas(body=None, catalog=WIDE):
    return build_atlas(catalog, None, now=NOW, body=_body() if body is None else body)


def _parts(atlas):
    return {p.name: p for p in atlas.parts}


def test_match_part_rules():
    known = {"orion-cortex-exec", "orion-bus-core", "vision-retina-walkway", "orion-energy"}
    hosts = ["athena", "circe"]
    assert match_part("orion-athena-cortex-exec", known, hosts) == "orion-cortex-exec"
    assert match_part("orion-orion-athena-bus-core", known, hosts) == "orion-bus-core"
    assert match_part("orion-vision-retina-walkway", known, hosts) == "vision-retina-walkway"
    assert match_part("orion-athena-orion-energy", known, hosts) == "orion-energy"
    assert match_part("orion-cortex-exec-1", known, hosts) == "orion-cortex-exec"
    assert match_part("zelda-dojo-live-0", known, hosts) is None


def test_hosts_are_parts_but_never_organ_members():
    atlas = _atlas()
    parts = _parts(atlas)
    assert parts["host:athena"].kind == "host" and parts["host:athena"].status == "live"
    assert parts["host:athena"].facts["hardware"] == "HP DL380 G10"
    assert parts["host:carbon-x1"].status == "declared_only"
    assert parts["host:carbon-x1"].facts["telemetry"] == "none"
    members = {m for o in atlas.organs for m in o.members}
    assert not any(m.startswith("part:host:") for m in members)
    assert atlas.stats.hosts == 2 and atlas.stats.parts_in_organs == len(members)
    vision = next(o for o in atlas.organs if "part:orion-vision-host" in o.members)
    assert vision.placement["carbon-x1"] == 1 and vision.placement["athena"] >= 1  # camera on the laptop
    assert atlas.unmatched == {"observed_placement": ["zelda-dojo-live-0@athena"]}


def test_each_new_part_adds_at_most_one_unit_over_its_edges():
    atlas = _atlas()
    store = [f for f in atlas.flows if f.target == "part:falkor:orion_substrate"]
    assert len(store) == 8 and sum(f.weight for f in store) == pytest.approx(1.0, abs=1e-3)
    assert {f.relation for f in store} == {"uses"} and {tuple(f.evidence) for f in store} == {("declared_env",)}
    cam = [f for f in atlas.flows if f.source == "part:camera:walkway" and f.relation == "sees_through"]
    assert sorted(f.target for f in cam) == ["part:orion-vision-edge", "part:orion-vision-host"]
    assert sum(f.weight for f in cam) == pytest.approx(1.0, abs=1e-3)
    run = [f for f in atlas.flows if f.target == "part:run:reading.turn"]
    assert [(f.source, f.relation, f.evidence) for f in run] == [("part:orion-cortex-exec", "runs", ["declared_runner"])]


def test_a_store_with_eight_users_does_not_pull_them_into_one_organ():
    # the users only meet at the store; one unit split eight ways must not outweigh their channels
    atlas = _atlas()
    organ_of = {m: o.organ_id for o in atlas.organs for m in o.members}
    assert organ_of["part:orion-vision-edge"] != organ_of["part:orion-cortex-exec"]
    assert atlas.stats.modularity >= 0.3


def test_new_kinds_have_status_and_facts():
    parts = _parts(_atlas())
    assert parts["camera:walkway"].status == "live" and parts["camera:walkway"].facts["on_host"] == "carbon-x1"
    assert parts["falkor:empty_graph"].status == "dormant" and parts["falkor:empty_graph"].kind == "store"
    assert parts["run:reading.turn"].status == "declared_only"


def test_persisted_channel_flow_moves_to_the_table():
    catalog = CATALOG + [{"name": "orion:chat:history", "producer_services": ["orion-hub"],
                          "consumer_services": ["orion-sql-writer", "orion-cortex-exec"]}]
    store = StoreIn(name="chat_history_log", store="postgres", rows=3,
                    links=[("orion-hub", "writes", "declared_route", "orion:chat:history")])
    with_body = build_atlas(catalog, None, now=NOW, body=BodyInputs(
        stores=[store], persisted={"orion:chat:history": "orion-sql-writer"}))
    pairs = {(f.source, f.target): f for f in with_body.flows}
    assert ("part:orion-hub", "part:orion-sql-writer") not in pairs
    assert pairs[("part:orion-hub", "part:pg:chat_history_log")].relation == "writes"
    assert ("part:orion-hub", "part:orion-cortex-exec") in pairs  # other consumers untouched
    bus_only = build_atlas(catalog, None, now=NOW)
    assert ("part:orion-hub", "part:orion-sql-writer") in {(f.source, f.target) for f in bus_only.flows}


def test_summary_shows_bodies_unread_sources_and_kind_counts_within_caps():
    text = summarize(_atlas(), out="/x/atlas.json")
    assert len(text.splitlines()) <= MAX_LINES and len(text) <= MAX_CHARS
    assert "bodies (machines; not in organs):" in text
    assert "athena: orchestration, HP DL380 G10; live" in text
    assert "UNREAD 1 source(s)" in text and "permission denied" in text
    assert "1 camera" in text and "body <organ_id>" in text
    assert text.endswith("full atlas: /x/atlas.json")


def test_summary_caps_hold_with_many_hosts_and_unread_sources():
    many = _body(hosts=[HostIn(name=f"h{i}", aliases=[f"h{i}"]) for i in range(30)],
                 reads=[SourceReadV1(name=f"live:x{i}", origin="live", status="unread", detail="e" * 200)
                        for i in range(30)])
    text = summarize(_atlas(many), out="/x/atlas.json")
    assert len(text.splitlines()) <= MAX_LINES and len(text) <= MAX_CHARS
    assert text.endswith("full atlas: /x/atlas.json")


def test_organ_view_lists_every_member_by_kind():
    atlas = _atlas()
    vision = next(o for o in atlas.organs if "part:camera:walkway" in o.members)
    text = organ_view(atlas, vision.organ_id, out="/x/a.json")
    assert f"{len(vision.members)} members" in text
    for m in vision.members:
        assert m.removeprefix("part:") in text
    assert "sensor (1)" in text and "on athena" in text and text.endswith("/x/a.json")
    with pytest.raises(names.NamingError):
        organ_view(atlas, "organ:0000000000", out=None)


def test_organ_view_is_capped_for_a_huge_organ():
    users = [f"orion-x{i}" for i in range(80)]
    catalog = [{"name": f"orion:x:{u}", "producer_services": [u], "consumer_services": ["orion-hubx"]} for u in users]
    atlas = build_atlas(catalog, None, now=NOW, body=BodyInputs())
    text = organ_view(atlas, atlas.organs[0].organ_id, out="/x/a.json")
    assert len(text.splitlines()) <= MAX_LINES and len(text) <= MAX_CHARS
    assert "the rest is in the file" in text and text.endswith("/x/a.json")


def test_a_v1_name_carries_onto_the_whole_body_and_new_parts_can_be_cited():
    bus_only = build_atlas(WIDE, None, now=NOW)
    vision = next(o for o in bus_only.organs if "part:orion-vision-host" in o.members)
    rec = names.check(bus_only, [], vision.organ_id, "Eyes", "takes frames from orion-vision-edge", dig="t")
    whole = _atlas()
    grown = next(o for o in whole.organs if "part:orion-vision-host" in o.members)
    assert grown.organ_id != rec.organ_id and "part:camera:walkway" in grown.members
    named = names.apply_names(whole, [rec])
    eyes = [o for o in named.organs if o.name == "Eyes"]
    assert [o.organ_id for o in eyes] == [grown.organ_id]
    assert eyes[0].named_by.startswith(f"orion, carried from {rec.organ_id}")
    again = names.check(whole, [rec], grown.organ_id, "Sight", "looks out through camera:walkway, all day", dig="t")
    assert again.cites == ["camera:walkway"] and again.replaces == [rec.organ_id]


def test_no_body_means_the_v1_atlas_unchanged():
    v1 = build_atlas(CATALOG, None, now=NOW)
    assert v1.sources.reads == [] and v1.known_gaps == [] and v1.stats.hosts == 0
    assert all(f.relation == "signals" for f in v1.flows)
    assert "bodies" not in summarize(v1, out=None)


def test_old_names_carry_onto_the_real_whole_body():
    # review HIGH: cameras/stores/processes diluted a bus-only name's overlap (real fixture: 4 of
    # 6 organs fell to 0.14-0.32); overlap is now over the kinds the name could have held
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_eval", Path(__file__).parents[1] / "evals" / "run_organ_structure_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    catalog, observed, now, body = ev._fixture_inputs(ev.WHOLE_BODY_FIXTURE)
    v1 = build_atlas(catalog, observed, now=now)
    vision = next(o for o in v1.organs if "part:orion-vision-host" in o.members)
    rec = names.check(v1, [], vision.organ_id, "Eyes", "fed by orion-vision-host frames", dig="t")
    whole = names.apply_names(build_atlas(catalog, observed, now=now, body=body), [rec])
    eyes = [o for o in whole.organs if o.name == "Eyes"]
    assert len(eyes) == 1 and "part:orion-vision-host" in eyes[0].members
    assert any(m.startswith("part:camera:") for m in eyes[0].members)
    assert names._overlap(eyes[0].members, rec.members) == 1.0 > names._jaccard(eyes[0].members, rec.members)
    whole_raw = build_atlas(catalog, observed, now=now, body=body)
    for organ in (o for o in v1.organs if len(o.members) > 1):  # never worse than raw Jaccard
        best = max(names._overlap(w.members, organ.members) for w in whole_raw.organs)
        assert best >= max(names._jaccard(w.members, organ.members) for w in whole_raw.organs)
