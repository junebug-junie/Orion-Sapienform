"""Rung-4 role rules: each clause can fail, counts are live-only, and gap lines come from code."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from orion.body_atlas import capacities as K
from orion.body_atlas.build import build_atlas
from orion.body_atlas.model import (
    AtlasSourcesV1, AtlasStatsV1, BodyAtlasV1, FlowV1, OrganV1, PartV1,
)

HEALTH = "orion:system:health"


def _atlas(parts: dict[str, tuple[str, str]], sig: list[tuple[str, str, str]], *, organs: dict[str, list[str]],
           other: list[FlowV1] = ()) -> BodyAtlasV1:
    """parts: name -> (kind, status). sig: (source, target, channel) signals flows."""
    flows: dict[tuple[str, str], FlowV1] = {}
    for s, t, ch in sig:
        f = flows.setdefault((s, t), FlowV1(source=f"part:{s}", target=f"part:{t}", weight=1.0,
                                            evidence=["observed_publish"]))
        f.channels.append(ch)
    return BodyAtlasV1(
        as_of="2026-10-30T08:00:00+00:00",
        sources=AtlasSourcesV1(catalog_path="x", catalog_channels=0, live=True),
        parts=[PartV1(part_id=f"part:{n}", name=n, kind=k, status=st, declared=True) for n, (k, st) in parts.items()],
        flows=[*flows.values(), *other],
        organs=[OrganV1(organ_id=f"organ:{o}", members=[f"part:{m}" for m in ms], hub=f"part:{ms[0]}",
                        internal_weight=1.0, external_weight=1.0) for o, ms in organs.items()],
        organ_flows=[],
        stats=AtlasStatsV1(parts_total=len(parts), parts_in_organs=len(parts), transient_parts=0, flows=len(flows),
                           organs=len(organs), singleton_organs=0, modularity=0.4, largest_organ_share=0.5),
    )


def _body(*, workers_overlap: bool = False, worker_status: str = "live"):
    """A pool between 6 requesters (two organs) and 3 workers, and a gateway that only serves."""
    req = [f"r{i}" for i in range(6)]
    work = req[:3] if workers_overlap else ["w0", "w1", "w2"]
    parts = {n: ("service", "live") for n in [*req, "pool", "gateway", "watcher", "quiet", "w0", "w1", "w2"]}
    for w in ("w0", "w1", "w2"):
        parts[w] = ("service", worker_status)
    sig = [(r, "pool", "lease:request") for r in req]
    sig += [("pool", r, "lease:reply") for r in req[:2]]
    sig += [(w, "pool", "worker:announce") for w in work]
    sig += [(r, "gateway", "llm:request") for r in req]
    sig += [("gateway", r, "llm:reply") for r in req[:3]]
    # every live part sends health; watcher reads it and answers one part (1 of 13 < 10%)
    sig += [(n, "watcher", HEALTH) for n in parts if n != "watcher"]
    sig += [("watcher", "pool", "watch:poke")]
    organs = {"a": req[:3] + ["pool", "w0", "w1"], "b": req[3:] + ["gateway", "watcher", "w2", "quiet"]}
    return _atlas(parts, sig, organs=organs)


def test_the_pool_between_two_disjoint_crowds_is_s2_and_the_gateway_is_only_a_server():
    r = K.read_capacities(_body())
    assert K.hits(r, "R-matchmaker") == {"part:pool"}
    pool = next(x for x in r.roles if x.part_id == "part:pool")
    assert pool.counts == {"requesters": 6, "requester_organs": 2, "replies": 2, "workers": 3}
    assert {(n.part_id, n.rule) for n in r.near_misses} >= {("part:gateway", "R-server")}


def test_a_second_crowd_that_overlaps_the_first_is_not_a_coordinator():
    r = K.read_capacities(_body(workers_overlap=True))
    assert K.hits(r, "R-matchmaker") == set()
    assert "part:pool" in {n.part_id for n in r.near_misses if n.rule == "R-server"}
    assert any(g.gap_id == "gap:s2-none" for g in r.gaps)
    assert next(j for j in r.jobs if j.job == "S2").status == "NO PART FOUND"


def test_workers_that_are_not_live_do_not_count():
    r = K.read_capacities(_body(worker_status="transient"))
    assert K.hits(r, "R-matchmaker") == set()


def test_dropping_the_second_crowd_clause_makes_the_server_s2_the_mutation_the_eval_catches():
    assert K.hits(K.read_capacities(_body(), second_set=False), "R-matchmaker") == {"part:pool", "part:gateway"}


def test_census_is_the_channel_above_the_largest_drop_and_its_quiet_reader_is_an_audit_candidate():
    r = K.read_capacities(_body())
    assert {c.channel for c in r.census if c.census} == {HEALTH}
    assert K.hits(r, "R-census-reader") == {"part:watcher"}
    assert next(x for x in r.roles if x.part_id == "part:watcher").strength == "candidate"


def test_a_census_reader_that_answers_many_routes_and_is_not_an_audit():
    body = _body()
    extra = [FlowV1(source="part:watcher", target=f"part:r{i}", weight=1.0, evidence=["observed_publish"],
                    channels=["watch:poke"]) for i in range(6)]
    body.flows.extend(extra)
    assert K.hits(K.read_capacities(body), "R-census-reader") == set()


def test_a_health_heartbeat_back_is_not_a_reply():
    body = _body()
    for f in body.flows:
        if f.source == "part:pool" and f.target in ("part:r0", "part:r1"):
            f.channels = [HEALTH]
    r = K.read_capacities(body)
    assert K.hits(r, "R-matchmaker") == set()
    assert "part:pool" in {n.part_id for n in r.near_misses if n.rule == "R-collector"}


def test_boundary_in_keeps_live_cameras_and_outside_endpoints_only():
    parts = {"cam": ("sensor", "live"), "old": ("sensor", "dormant"), "router": ("service", "live"),
             "http:anthropic": ("endpoint", "live"), "router:sub": ("endpoint", "live"),
             "operator:probe": ("endpoint", "live"), "http:gone": ("endpoint", "transient")}
    sees = [FlowV1(source=f"part:{c}", target="part:router", weight=1.0, evidence=["declared_config"],
                   relation="sees_through") for c in ("cam", "old")]
    sig = [(e, "router", "x:req") for e in ("http:anthropic", "router:sub", "operator:probe", "http:gone")]
    atlas = _atlas(parts, sig, organs={"a": list(parts)}, other=sees)
    r = K.read_capacities(atlas)
    assert K.hits(r, "R-boundary-in") == {"part:cam", "part:http:anthropic"}
    assert r.live_parts == 3  # router, http:anthropic, router:sub; operator and transient ids are out


def test_silent_services_gap_is_written_by_code_from_the_atlas():
    r = K.read_capacities(_body())
    assert r.silent_services == ["part:quiet"]
    gap = next(g for g in r.gaps if g.gap_id == "gap:s3-unseen")
    assert "1 live services" in gap.text and "quiet" in gap.text
    assert {g.job for g in r.gaps} >= {"S3", "S4", "S5", "algedonic"}


def test_printout_is_small_and_names_every_job():
    text = "\n".join(K.render(K.read_capacities(_body())))
    assert len(text.splitlines()) <= 39 and len(text) <= 3900
    for job in K.JOBS:
        assert f"\n{job} " in f"\n{text}"


_SPEC = importlib.util.spec_from_file_location(
    "run_capacity_roles_eval", Path(__file__).parents[1] / "evals" / "run_capacity_roles_eval.py")
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)


def test_the_eval_passes_on_both_frozen_bodies_and_catches_the_mutation():
    fixtures = [_EVAL.HELD_OUT, _EVAL.TODAY]
    assert _EVAL.run(fixtures, verbose=False) == 0
    assert _EVAL.run(fixtures, second_set=False, verbose=False) == len(fixtures)


def test_the_eval_fails_when_the_expected_set_is_wrong():
    catalog, observed, now, body = _EVAL._structure._fixture_inputs(_EVAL.TODAY)
    reading = K.read_capacities(build_atlas(catalog, observed, now=now, body=body))
    failures = _EVAL.evaluate(reading, {"R-matchmaker": {"part:orion-llm-gateway"}})
    assert any("precision 0.00 recall 0.00" in f for f in failures)


def test_a_service_talking_only_to_a_transient_part_is_still_silent():
    body = _body()
    body.parts.append(PartV1(part_id="part:smoke", name="smoke", kind="service", status="transient", declared=False))
    body.flows.append(FlowV1(source="part:quiet", target="part:smoke", weight=1.0, evidence=["observed_publish"],
                             channels=["x:poke"]))
    assert "part:quiet" in K.read_capacities(body).silent_services


def test_a_camera_feeding_only_dead_parts_is_not_s1_and_only_a_service_name_marks_a_sub_id():
    parts = {"cam": ("sensor", "live"), "dead": ("service", "dormant"), "router": ("service", "live"),
             "http": ("store", "dormant"), "http:anthropic": ("endpoint", "live")}
    sees = [FlowV1(source="part:cam", target="part:dead", weight=1.0, evidence=["declared_config"],
                   relation="sees_through")]
    sig = [("http:anthropic", "router", "x:req")]
    r = K.read_capacities(_atlas(parts, sig, organs={"a": list(parts)}, other=sees))
    # a part named `http` that is not a service does not hide http:* (an endpoint id always has a ':')
    assert K.hits(r, "R-boundary-in") == {"part:http:anthropic"}


def test_a_transient_requester_does_not_count():
    body = _body()
    body.parts = [p if p.name != "r5" else p.model_copy(update={"status": "transient"}) for p in body.parts]
    pool = next(x for x in K.read_capacities(body).roles if x.part_id == "part:pool")
    assert pool.counts["requesters"] == 5


def test_two_census_channels_give_one_audit_row_per_reader():
    body = _body()
    for f in body.flows:
        if f.target == "part:watcher" and HEALTH in f.channels:
            f.channels.append("orion:rpc_health:snapshot")
    r = K.read_capacities(body)
    assert {c.channel for c in r.census if c.census} == {HEALTH, "orion:rpc_health:snapshot"}
    rows = [x for x in r.roles if x.rule == "R-census-reader"]
    assert [x.part_id for x in rows] == ["part:watcher"] and len(rows[0].evidence) == 2


def test_with_no_census_the_printout_says_none():
    parts = {"a": ("service", "live"), "b": ("service", "live")}
    text = "\n".join(K.render(K.read_capacities(_atlas(parts, [("a", "b", "x:y")], organs={"o": ["a", "b"]}))))
    assert "census: none" in text
