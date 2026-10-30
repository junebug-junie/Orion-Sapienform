"""Processes link to the services that do their work, from each run's own record, weighted so
plumbing (a service in every process) cannot glue the body together."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from orion.body_atlas import names
from orion.body_atlas import sources as S
from orion.body_atlas.build import build_atlas
from orion.body_atlas.tests.test_build import NOW
from orion.body_atlas.tests.test_sources import CHANNELS, FakeSql, _root, _write

_SPEC = importlib.util.spec_from_file_location(
    "run_organ_structure_eval", Path(__file__).parents[1] / "evals" / "run_organ_structure_eval.py")
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)

SERVICES = ["orion-durable-runs", "orion-gpu-pool", "orion-cortex-exec", "orion-hub", "orion-harness-governor",
            "orion-a", "orion-b", "orion-c"]
# a ring of channels so every service is a part
CATALOG = [{"name": f"orion:x:{a}", "producer_services": [a], "consumer_services": [b]}
           for a, b in zip(SERVICES, SERVICES[1:] + SERVICES[:1])]


def _gather(tmp_path, sql=None):
    return S.gather_body(_root(tmp_path), CHANNELS, live=S.LiveReaders(sql=sql or FakeSql()))


# --- extraction from the run's own record ---------------------------------------------------

def test_doers_come_from_correlation_joins_and_follow_the_turn(tmp_path):
    procs = {p.name: p for p in _gather(tmp_path).processes}
    assert procs["reverie.visual"].doers == [("cortex-exec", "observed_trace", 40)]
    # a row keyed by detail.turn_correlation_id is labeled as the turn, not the run itself
    assert procs["situation.update"].doers == [("orion-hub", "observed_turn", 9)]
    # a trace row with an empty service names nobody
    assert ("", "observed_trace", 2) not in procs["reading.turn"].doers
    # the turn record names no service: its table's declared writer did the turn
    assert ("orion-harness-governor", "declared_turn_writer", 4) in procs["reading.turn"].doers
    # seen only in traces, never in the Literal: still a part, with the declared runner
    assert not procs["reverie.visual"].declared and procs["reverie.visual"].runner is None


def test_the_join_query_follows_both_the_run_and_its_turn():
    q = S.sql_trace_doers("grammar_events", "source_service")
    assert "detail->>'turn_correlation_id'" in q and "correlation_id AS cid, 'run'" in q
    assert "count(DISTINCT k.run_id)" in q and "t.source_service AS service" in q
    assert "service" not in S.sql_trace_doers("harness_turn_trace", None).split("FROM k")[0].split("via, ")[1]


def test_child_run_ids_become_a_process_to_process_flow(tmp_path):
    procs = {p.name: p for p in _gather(tmp_path).processes}
    assert procs["dream.carry"].children == [("reverie.visual", 3)]
    body = S.BodyInputs(processes=[
        S.ProcessIn(name="dream.carry", family="durable_workflow", runner="orion-durable-runs", declared=True,
                    children=[("reverie.visual", 3), ("ghost.workflow", 1)]),
        S.ProcessIn(name="reverie.visual", family="durable_workflow", runner="orion-durable-runs", declared=True),
    ])
    atlas = build_atlas(CATALOG, None, now=NOW, body=body)
    spawn = [f for f in atlas.flows if f.relation == "spawns"]
    assert [(f.source, f.target, f.evidence) for f in spawn] == [
        ("part:run:dream.carry", "part:run:reverie.visual", ["observed_child_run"])]
    assert atlas.unmatched["observed_child_run"] == ["ghost.workflow"]  # no such process part


def test_unreadable_trace_tables_are_unread_with_the_grant_they_need(tmp_path):
    body = _gather(tmp_path, FakeSql(deny=("substrate_reduction_cursor", "cockpit_turn_sighting")))
    reads = {r.name: r for r in body.reads}
    bad = reads["live:postgres cockpit_turn_sighting (run + turn correlation ids)"]
    assert bad.status == "unread"
    assert "needs GRANT SELECT ON cockpit_turn_sighting TO orion_readonly" in bad.detail
    # the other sources still count
    assert reads["live:postgres grammar_events (run + turn correlation ids)"].status == "read"
    assert {p.name: p for p in body.processes}["reverie.visual"].doers


def test_a_turn_record_table_with_no_declared_writer_is_unread_not_guessed(tmp_path):
    body = S.gather_body(_root(tmp_path), CHANNELS[:1], live=S.LiveReaders(sql=FakeSql()))
    bad = {r.name: r for r in body.reads}["live:postgres harness_turn_trace (turn records)"]
    assert bad.status == "unread" and "no declared writer" in bad.detail
    assert not any(e == "declared_turn_writer" for p in body.processes for _, e, _ in p.doers)


def test_a_malformed_reply_is_unread_not_a_crash(tmp_path):
    class Wrong(FakeSql):
        def run(self, sql, flags):
            if "JOIN grammar_events t" in sql:
                return "nope\n1\n", 0
            return super().run(sql, flags)

    reads = {r.name: r for r in _gather(tmp_path, Wrong()).reads}
    assert reads["live:postgres grammar_events (run + turn correlation ids)"].status == "unread"


# --- reducer sources (declared) -------------------------------------------------------------

WORKER = """
    from orion.loops.constants import CURSOR_A, SOURCE_A
    from orion.loops.constants import CURSOR_B
    LOCAL = "orion-local"
    REDUCER_SPECS: tuple = (
        ReducerSpec(reducer_key="a", cursor_name=CURSOR_A, source_service=SOURCE_A),
        ReducerSpec(reducer_key="b", cursor_name=CURSOR_B, source_service=LOCAL),
        ReducerSpec(reducer_key="c", cursor_name="c_cursor", source_service="orion-c"),
    )
"""


def _reducer_root(tmp_path, worker=WORKER):
    _write(tmp_path, "services/orion-substrate-runtime/app/worker.py", worker)
    # a re-export: the constant lives one more module down
    _write(tmp_path, "orion/loops/constants.py",
           "from orion.schemas.proj import SOURCE_A\nCURSOR_A = 'a_cursor'\nCURSOR_B = 'b_cursor'\n")
    _write(tmp_path, "orion/schemas/proj.py", "SOURCE_A = 'orion-a'\n")
    return tmp_path


def test_reducer_sources_resolve_through_imports_and_re_exports(tmp_path):
    got = S.parse_reducer_sources(_reducer_root(tmp_path))
    assert got == {"a_cursor": "orion-a", "b_cursor": "orion-local", "c_cursor": "orion-c"}


def test_an_unresolvable_reducer_spec_is_a_partial_read(tmp_path):
    worker = WORKER.replace('source_service="orion-c"', "source_service=NOWHERE")
    with pytest.raises(S.PartialRead) as exc:
        S.parse_reducer_sources(_reducer_root(tmp_path, worker))
    assert "1 REDUCER_SPECS" in str(exc.value) and "c_cursor" not in exc.value.value


def test_gathered_reducers_carry_their_source_service(tmp_path):
    class Cursors(FakeSql):
        def run(self, sql, flags):
            if "substrate_reduction_cursor" in sql:
                return "cursor_name,updated_at\na_cursor,2026-10-29 20:00:00+00\nlost_cursor,\n", 0
            return super().run(sql, flags)

    root = _reducer_root(_root(tmp_path))
    body = S.gather_body(root, CHANNELS, live=S.LiveReaders(sql=Cursors(deny=())))
    procs = {p.name: p for p in body.processes}
    assert procs["a_cursor"].doers == [("orion-a", "declared_reducer_source", 1)]
    assert procs["lost_cursor"].doers == []  # no spec names it: runner only


# --- the weighting rule ---------------------------------------------------------------------

def _wf(name, doers=(), runs=10, children=()):
    return S.ProcessIn(name=name, family="durable_workflow", runner="orion-durable-runs", declared=True,
                       runs=runs, doers=list(doers), children=list(children))


def _weights(atlas, process):
    return {(f.source if f.target == process else f.target).removeprefix("part:"): f.weight
            for f in atlas.flows if process in (f.source, f.target)}


def test_a_service_in_every_process_gets_a_small_share_and_each_process_adds_one_unit():
    body = S.BodyInputs(processes=[
        _wf("p1", [("orion-gpu-pool", "observed_trace", 10), ("orion-a", "observed_trace", 10)]),
        _wf("p2", [("orion-gpu-pool", "observed_trace", 10), ("orion-b", "observed_trace", 10)]),
        _wf("p3", [("orion-gpu-pool", "observed_trace", 10), ("orion-c", "observed_trace", 10)]),
        _wf("p4", [("orion-gpu-pool", "observed_trace", 10), ("orion-hub", "observed_trace", 10)]),
    ])
    atlas = build_atlas(CATALOG, None, now=NOW, body=body)
    w = _weights(atlas, "part:run:p1")
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    # runner and GPU lease are in all four; orion-a only in p1, with the same run count
    assert w["orion-a"] > 2 * w["orion-gpu-pool"]
    assert w["orion-gpu-pool"] == pytest.approx(w["orion-durable-runs"], abs=1e-3)
    flows = {(f.source, f.target): f for f in atlas.flows}
    assert flows[("part:orion-a", "part:run:p1")].relation == "does"
    assert flows[("part:orion-durable-runs", "part:run:p1")].relation == "runs"
    assert "orion-a" in atlas.parts[[p.name for p in atlas.parts].index("run:p1")].facts["work_split"]


def test_a_runner_also_seen_in_traces_stays_one_runs_flow_with_both_evidences():
    body = S.BodyInputs(processes=[_wf("p1", [("durable-runs", "observed_trace", 3)])])
    atlas = build_atlas(CATALOG, None, now=NOW, body=body)
    f = [f for f in atlas.flows if f.target == "part:run:p1"]
    assert len(f) == 1 and f[0].relation == "runs" and f[0].evidence == ["declared_runner", "observed_trace"]
    assert f[0].weight == pytest.approx(1.0)


def test_a_process_with_only_its_runner_keeps_the_whole_unit_on_it():
    atlas = build_atlas(CATALOG, None, now=NOW, body=S.BodyInputs(processes=[_wf("p1"), _wf("p2")]))
    assert _weights(atlas, "part:run:p1") == {"orion-durable-runs": pytest.approx(1.0)}


def test_reducers_lean_toward_the_service_they_read():
    body = S.BodyInputs(processes=[
        S.ProcessIn(name=f"r{i}", family="substrate_reducer", runner="orion-durable-runs",
                    runner_evidence="assumed_runner", declared=False,
                    doers=[(svc, "declared_reducer_source", 1)])
        for i, svc in enumerate(("orion-a", "orion-b", "orion-c"))])
    atlas = build_atlas(CATALOG, None, now=NOW, body=body)
    w = _weights(atlas, "part:reducer:r0")
    assert w["orion-a"] > w["orion-durable-runs"]
    assert {f.relation for f in atlas.flows if f.target == "part:reducer:r0"} == {"feeds", "runs"}


def test_a_doer_that_is_not_a_service_part_is_unmatched_not_linked():
    body = S.BodyInputs(processes=[_wf("p1", [("zelda-dojo-live-0", "observed_trace", 5)])])
    atlas = build_atlas(CATALOG, None, now=NOW, body=body)
    assert atlas.unmatched["observed_trace"] == ["zelda-dojo-live-0"]


# --- the real fixture ---------------------------------------------------------------------

def _fixture():
    return _EVAL._fixture_inputs(_EVAL.DOERS_FIXTURE)


def test_real_fixture_moves_processes_off_the_runner_and_passes_the_gate():
    catalog, observed, now, body = _fixture()
    atlas = build_atlas(catalog, observed, now=now, body=body)
    assert _EVAL.evaluate(atlas) == []
    runner_only = body.model_copy(deep=True)
    for p in runner_only.processes:
        p.doers, p.children = [], []
    before = build_atlas(catalog, observed, now=now, body=runner_only)
    assert atlas.stats.largest_organ_share < before.stats.largest_organ_share
    organ_of = {m: o.organ_id for o in atlas.organs for m in o.members}
    # a reducer sits with the service whose events it reads, not with its runner
    assert organ_of["part:reducer:vision_organ_grammar_reducer"] == organ_of["part:orion-vision-frame-router"]
    assert organ_of["part:run:dream.carry"] == organ_of["part:run:reverie.visual"]


def test_names_still_carry_from_the_runner_only_body_onto_the_doers_body():
    catalog, observed, now, body = _fixture()
    runner_only = body.model_copy(deep=True)
    for p in runner_only.processes:
        p.doers, p.children = [], []
    before = build_atlas(catalog, observed, now=now, body=runner_only)
    vision = next(o for o in before.organs if "part:orion-vision-host" in o.members)
    big = before.organs[0]
    recs = [names.check(before, [], vision.organ_id, "Eyes", "fed by orion-vision-host frames", dig="t")]
    hub = next(before.parts[i] for i, p in enumerate(before.parts) if p.part_id == big.hub)
    recs.append(names.check(before, recs, big.organ_id, "Spine", f"its hub is {hub.name}", dig="t"))
    after = names.apply_names(build_atlas(catalog, observed, now=now, body=body), recs)
    named = {o.name: o for o in after.organs if o.name}
    assert set(named) == {"Eyes", "Spine"}
    assert "part:orion-vision-host" in named["Eyes"].members and big.hub in named["Spine"].members
    # the reducer that reads the frame router moved into the eyes
    assert "part:reducer:vision_organ_grammar_reducer" in named["Eyes"].members

