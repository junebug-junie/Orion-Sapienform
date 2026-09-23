"""The operator's read of Orion's world view.

The thing under test is a PROJECTION, so the failure mode that matters is not a
crash — it is a panel that renders a confident number nobody can trace back to
the graph. Most of these assert on distinctions the UI must not collapse:
unreachable vs never-configured vs empty, closed vs live, "no history recorded"
vs "confidence never moved", and a run that died before writing vs a quiet one.
"""

from __future__ import annotations

import pytest

from orion.curiosity.atlas import (
    ATLAS_EDGES_CYPHER,
    ATLAS_PRIORS_CYPHER,
    ATLAS_REVISIONS_CYPHER,
    ATLAS_UNUSED_CYPHER,
    AtlasView,
    prior_claims_cypher,
    read_atlas,
    read_run_ids_since,
    read_run_nodes,
    run_ids_since_cypher,
    run_nodes_cypher,
    to_payload,
    trajectory_for,
    valid_run_id,
)
from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable


class _Reader(WorldviewReader):
    """Answers by query shape, and refuses to answer two of them the same way.

    The needle collision that made a green test stop isolating anything in
    `test_curiosity_worldview.py` is guarded here at the fixture: every needle
    must match exactly one of this module's queries.
    """

    def __init__(self, *, answers=None, raises=False) -> None:
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers = answers or {}
        self.raises = raises
        self.queries: list[str] = []

    def query(self, cypher: str):
        self.queries.append(cypher)
        if self.raises:
            raise WorldviewUnavailable("ConnectionError: nope")
        hits = [rows for needle, rows in self.answers.items() if needle in cypher]
        assert len(hits) <= 1, f"needle collision on: {cypher[:80]}"
        return hits[0] if hits else []


def _prior(pid="p1", claim="a claim", conf="0.85", status="open", tested=0,
           run="r1", last_run=""):
    return {
        "prior_id": pid, "claim": claim, "confidence": conf, "status": status,
        "times_tested": tested, "formed_from": "", "last_tested_at": "",
        "run_id": run, "last_run_id": last_run, "why": "",
    }


_PRIORS = "RETURN p.prior_id AS prior_id"
_REVS = "r.from_confidence"


# --- unreachable, unconfigured and empty are three different states --------


def test_an_unreachable_graph_is_not_an_empty_world_view() -> None:
    view = read_atlas(_Reader(raises=True))
    assert view.is_unavailable
    assert view.priors == [] and view.revisions == []
    payload = to_payload(view)
    assert payload["available"] is False
    assert "ConnectionError" in payload["reason"]


def test_an_empty_graph_is_available_and_is_not_a_dead_pool() -> None:
    """Never written a prior and closed every prior must not render the same:
    only one of them is a fault."""
    view = read_atlas(_Reader())
    assert not view.is_unavailable
    assert view.live_total == 0 and view.closed_total == 0
    assert view.pool_is_dead is False


def test_every_prior_closed_reads_as_a_dead_pool() -> None:
    view = read_atlas(_Reader(answers={_PRIORS: [
        _prior(pid="a", status="refuted"),
        _prior(pid="b", status="retired_unresolvable"),
    ]}))
    assert view.live_total == 0 and view.closed_total == 2
    assert view.pool_is_dead is True


@pytest.mark.parametrize("status", ["open", "supported", "revised", "", "typo"])
def test_a_prior_orion_has_not_closed_counts_as_live(status: str) -> None:
    """Same status rule as the reader the prompt uses. If these two drift, the
    dashboard reports a pool the loop cannot actually offer."""
    view = read_atlas(_Reader(answers={_PRIORS: [_prior(status=status)]}))
    assert view.live_total == 1 and view.closed_total == 0


# --- history is recorded, or it is honestly absent -------------------------


def test_a_prior_with_no_revision_plots_one_point_and_says_so() -> None:
    """An empty trajectory must mean "not recorded", never "did not move" — the
    page says which, and it can only do that if this flag is honest."""
    view = read_atlas(_Reader(answers={_PRIORS: [_prior(conf="0.85")]}))
    payload = to_payload(view)
    assert payload["history_recorded"] is False
    traj = payload["priors"][0]["trajectory"]
    assert len(traj) == 1
    assert traj[0]["confidence"] == pytest.approx(0.85)
    assert traj[0]["recorded"] is False


def test_a_recorded_revision_becomes_a_before_and_an_after() -> None:
    view = read_atlas(_Reader(answers={
        _PRIORS: [_prior(pid="p1", conf="0.72", status="revised", tested=1)],
        _REVS: [{
            "prior_id": "p1", "run_id": "r2", "from_confidence": "0.85",
            "to_confidence": "0.72", "from_status": "open",
            "to_status": "revised", "written_at": 1787840568235,
        }],
    }))
    payload = to_payload(view)
    assert payload["history_recorded"] is True
    traj = payload["priors"][0]["trajectory"]
    assert [round(p["confidence"], 2) for p in traj] == [0.85, 0.72]
    assert payload["revisions"][0]["delta"] == pytest.approx(-0.13)


def test_the_current_value_is_not_appended_twice() -> None:
    """A revision that already lands on the current confidence must not get a
    duplicate endpoint, or every trajectory ends in a flat segment that never
    happened."""
    view = AtlasView(
        priors=read_atlas(_Reader(answers={_PRIORS: [
            _prior(pid="p1", conf="0.72")]})).priors,
        revisions=read_atlas(_Reader(answers={_REVS: [{
            "prior_id": "p1", "run_id": "r2", "from_confidence": "0.85",
            "to_confidence": "0.72", "from_status": "open",
            "to_status": "revised", "written_at": 1,
        }]})).revisions,
    )
    traj = trajectory_for(view, "p1")
    assert [round(p["confidence"], 2) for p in traj] == [0.85, 0.72]


def test_confidence_going_down_is_representable() -> None:
    """The loop's headline acceptance check. If the projection could not carry
    a negative delta the panel could never show the thing it exists to show."""
    view = read_atlas(_Reader(answers={
        _PRIORS: [_prior(pid="p1", conf="0.40")],
        _REVS: [{
            "prior_id": "p1", "run_id": "r2", "from_confidence": "0.90",
            "to_confidence": "0.40", "from_status": "supported",
            "to_status": "revised", "written_at": 5,
        }],
    }))
    assert view.revisions[0].delta == pytest.approx(-0.50)


# --- per-run graph nodes: bounded by window, keyed by run id --------------
#
# The run-by-run read used to be `assemble_runs` over every Hop/Finding/
# TurnOutcome ever written under a 5000-row LIMIT, sorted by `n` alone. It
# now feeds `orion/curiosity/run_story.py`, reads only the runs in a window,
# and SELECTS `written_at` on hops -- the field the atlas never adopted, so a
# retried run rendered its attempts interleaved 1,1,2,2 (design doc
# 2026-09-22-curiosity-tab-redesign, "Read-model changes").


def test_hops_are_read_with_their_clock() -> None:
    cypher = run_nodes_cypher("Hop", ["446ddd7165d5"])
    assert "n.written_at AS written_at" in cypher
    assert "n.n AS n" in cypher and "n.note AS note" in cypher


def test_run_ids_reach_a_query_only_through_the_parameter_prefix() -> None:
    """Values ride in FalkorDB's `CYPHER k=v` prefix, never spliced into the
    pattern, and every id passes the allow-list first -- a run id comes off
    a URL path in `/curiosity/api/run/{run_id}`."""
    cypher = run_nodes_cypher("Hop", ["446ddd7165d5", "bad id", "x' OR 1=1 //", "20260921T174910Z-10faa7"])
    assert cypher.startswith("CYPHER ids=['20260921T174910Z-10faa7','446ddd7165d5'] MATCH")
    assert "OR 1=1" not in cypher
    assert "$ids" in cypher
    assert valid_run_id("446ddd7165d5") == "446ddd7165d5"
    assert valid_run_id("x' OR 1=1") is None
    assert valid_run_id("") is None
    assert valid_run_id("a" * 65) is None


def test_prior_ids_are_json_quoted_because_orion_writes_them_freehand() -> None:
    cypher = prior_claims_cypher(["self:it's-mine", "p2"])
    assert cypher.startswith('CYPHER ids=["p2", "self:it\'s-mine"] MATCH')
    assert "p.claim AS claim" in cypher and "p.line AS line" in cypher


def test_the_window_query_is_an_int_bound_and_skips_undated_nodes() -> None:
    cypher = run_ids_since_cypher(1790000000000.9)
    assert "CYPHER since=1790000000000 " in cypher
    assert "n.written_at IS NOT NULL AND n.written_at >= $since" in cypher
    assert "RETURN DISTINCT n.run_id AS run_id" in cypher


def test_read_run_nodes_reads_all_five_kinds_and_the_touched_priors() -> None:
    reader = _Reader(answers={
        "MATCH (n:InvestigationRole)": [{"run_id": "r1", "choice": "local_crawl", "why": "w", "written_at": 1}],
        "MATCH (n:Hop)": [{"run_id": "r1", "n": 1, "note": "h", "written_at": 2}],
        "MATCH (n:Finding)": [{"run_id": "r1", "finding_id": "f", "text": "t", "evidence": "e", "written_at": 3}],
        "MATCH (n:PriorRevision)": [{"run_id": "r1", "prior_id": "p1", "from_confidence": 0.6,
                                     "to_confidence": 0.7, "from_status": "open", "to_status": "revised", "written_at": 4}],
        "MATCH (n:TurnOutcome)": [{"run_id": "r1", "continue_line": True, "continue_note": "c",
                                   "reach_out": False, "reach_out_why": "", "written_at": 5}],
        "MATCH (p:Prior) WHERE p.prior_id IN": [{"prior_id": "p1", "claim": "the claim", "line": ""}],
        "MATCH (n:LivedAnswer)": [{"run_id": "r1", "question_id": "q", "family": "lived", "text": "t",
                                   "evidence": "e", "revises": "", "written_at": 6}],
        "MATCH (n:SelfDefinition)": [],
    })
    rows = read_run_nodes(reader, ["r1", "not valid!"])
    assert [h["written_at"] for h in rows.hops] == [2]
    assert rows.self_writes == [{"run_id": "r1", "question_id": "q", "family": "lived", "text": "t",
                                 "evidence": "e", "revises": "", "written_at": 6, "kind": "lived_answer"}]
    assert rows.roles[0]["choice"] == "local_crawl"
    assert rows.priors[0]["claim"] == "the claim"
    assert all("['r1']" in q for q in reader.queries if "n.run_id IN" in q), reader.queries
    assert '["p1"]' in next(q for q in reader.queries if "p.prior_id IN" in q)


def test_read_run_nodes_with_no_valid_ids_issues_no_query() -> None:
    reader = _Reader()
    rows = read_run_nodes(reader, ["", "bad id"])
    assert rows.hops == [] and reader.queries == []


def test_read_run_ids_since_drops_ids_that_could_not_reach_a_query() -> None:
    reader = _Reader(answers={"RETURN DISTINCT n.run_id": [{"run_id": "ok1"}, {"run_id": "bad id"}, {"run_id": None}]})
    assert read_run_ids_since(reader, 5) == ["ok1"]


def test_a_dead_graph_raises_for_the_run_readers_rather_than_reading_as_empty() -> None:
    """Unlike `read_atlas`, these return rows, not a view; the caller owns the
    `available: false` payload. An empty list here would render as "no runs
    in 14 days" during an outage."""
    with pytest.raises(WorldviewUnavailable):
        read_run_nodes(_Reader(raises=True), ["r1"])
    with pytest.raises(WorldviewUnavailable):
        read_run_ids_since(_Reader(raises=True), 1)


def test_every_atlas_read_goes_out_read_only() -> None:
    """Hub connects to FalkorDB as the unrestricted `default` user. This is the
    only thing between a bug here and Orion's own memory."""
    sent: list[tuple] = []

    class _Spy:
        def execute_command(self, *args):
            sent.append(args)
            return [[], [], []]

    read_atlas(WorldviewReader(host="x", port=1, graph_name="g", client=_Spy()))
    assert sent, "no queries were issued"
    assert all(a[0] == "GRAPH.RO_QUERY" for a in sent), sent


def test_the_atlas_takes_no_caller_input_into_a_query() -> None:
    """Every constant here is static. Nothing on this page is user-supplied, and
    keeping it that way is why there is no sanitiser to get wrong."""
    for cypher in (ATLAS_PRIORS_CYPHER, ATLAS_REVISIONS_CYPHER, ATLAS_UNUSED_CYPHER, ATLAS_EDGES_CYPHER):
        assert "{" not in cypher and "%" not in cypher


def test_a_prior_with_no_id_is_dropped_rather_than_invented() -> None:
    view = read_atlas(_Reader(answers={_PRIORS: [_prior(pid=""), _prior(pid="ok")]}))
    assert [p.prior_id for p in view.priors] == ["ok"]


def test_the_payload_is_json_safe() -> None:
    import json

    view = read_atlas(_Reader(answers={
        _PRIORS: [{**_prior(pid="p1", tested=1, last_run="r2"),
                   "last_tested_at": "2026-09-21T16:02:00+00:00"}],
        _REVS: [{"prior_id": "p1", "run_id": "r2", "from_confidence": "0.9",
                 "to_confidence": "0.85", "from_status": "open",
                 "to_status": "revised", "written_at": 7}],
    }))
    payload = to_payload(view)
    blob = json.dumps(payload)
    assert '"history_recorded": true' in blob
    assert "runs" not in payload, "the run story endpoints own runs now"
    assert "growth" not in blob
    prior = payload["priors"][0]
    assert prior["last_tested_at_ms"] == _ms("2026-09-21T16:02:00+00:00")
    assert [pt["written_at"] for pt in prior["trajectory"]] == [None, 7]


def test_the_current_point_carries_the_priors_last_tested_clock() -> None:
    view = read_atlas(_Reader(answers={_PRIORS: [
        {**_prior(pid="p1", conf="0.5"), "last_tested_at": 1790005960076}]}))
    traj = to_payload(view)["priors"][0]["trajectory"]
    assert traj == [{"run_id": "r1", "confidence": 0.5, "status": "open",
                     "recorded": False, "written_at": 1790005960076}]


# --- the surface itself -----------------------------------------------------


def test_the_operator_surface_exposes_no_write_route() -> None:
    """Read-only is a design constraint, not a phase-one scope cut: Hub never
    writes to Orion's graph, and a route that could edit a belief Orion formed
    needs an auth story, an audit trail, and an argument this does not have.
    Asserted on the router rather than trusted to review."""
    # Loaded BY PATH, not by import name. The repo root has its own `scripts`
    # package, so `from scripts.curiosity_routes import ...` resolves against
    # whichever one reached sys.path first -- green alone, ModuleNotFoundError
    # once the orion-hub suite runs in the same session.
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "services" / "orion-hub"
            / "scripts" / "curiosity_routes.py")
    spec = importlib.util.spec_from_file_location("_curiosity_routes_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    router = module.router

    assert router.routes, "the router registered nothing"
    writes = [
        r for r in router.routes if not r.methods <= {"GET", "HEAD"}
    ]
    # Exactly one, and it is a CONTROL action: it asks the loop to take a turn
    # sooner than the cooldown would have. It writes no memory, no prior, no
    # finding -- Orion still authors everything the turn produces. Pinned by
    # path so a second write route cannot be added without this going red and
    # someone having to justify it.
    # Two CONTROL actions: each asks a line to take a turn sooner than its
    # cooldown would have. The park/pin self-question routes that sat here
    # were removed 2026-09-22: nothing called them and this test had been
    # red since they landed.
    #
    # A third, 2026-09-23: `/api/run/{run_id}/reply` still writes nothing to
    # Orion's graph -- it posts Juniper's own text into chat as an ordinary
    # inbound turn (the exact path every other message she sends already
    # takes), gated on a CONFIRMED `sent` Door-A outreach decision for that
    # exact run. It is not a memory/prior/finding write any more than
    # `/api/chat` is; Orion still authors everything the turn produces.
    assert sorted(r.path for r in writes) == [
        "/curiosity/api/run-now", "/curiosity/api/run/{run_id}/reply",
        "/curiosity/api/self-inquiry/run-now",
    ], [(r.path, sorted(r.methods)) for r in writes]
    assert all(r.methods == {"POST"} for r in writes)


def test_the_schedule_keys_are_imported_from_the_loop_that_writes_them() -> None:
    """A dashboard with its own copy of `orion:curiosity:count:` would render a
    confident 0 forever the day that prefix changes."""
    from pathlib import Path

    import ast

    source = (Path(__file__).resolve().parents[1] / "services" / "orion-hub"
              / "scripts" / "curiosity_routes.py").read_text()
    assert "_COOLDOWN_KEY" in source and "_DAILY_COUNT_KEY_PREFIX" in source

    # AST rather than a substring scan: the docstring above explains this very
    # rule and names the prefix, and a check that cannot tell an explanation
    # from an implementation fails on its own documentation.
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    literals = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docstrings
    ]
    offenders = [x for x in literals if "orion:curiosity:" in x]
    assert not offenders, f"a key name was retyped in code: {offenders}"


# --- the day boundary -------------------------------------------------------


def _routes():
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "services" / "orion-hub"
            / "scripts" / "curiosity_routes.py")
    spec = importlib.util.spec_from_file_location("_curiosity_routes_tz", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ms(iso: str) -> int:
    from datetime import datetime, timezone

    d = datetime.fromisoformat(iso)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def test_a_run_is_counted_against_the_zone_the_counter_keys_on() -> None:
    """The live case, 2026-08-27 20:33 MDT: the host clock already said the
    28th while Juniper's date was still the 27th and the counter key was
    `orion:curiosity:count:2026-08-27`. Counting in UTC puts this run on the
    wrong day and the budget tile disagrees with the counter it sits beside."""
    runs = [{"run_id": "r", "line": "investigate",
             "started_at": _ms("2026-08-28T02:22:00+00:00"), "finished_at": None}]
    assert _routes()._runs_on_local_date(runs, "2026-08-27", "America/Denver") == {"investigate": 1}
    assert _routes()._runs_on_local_date(runs, "2026-08-28", "UTC") == {"investigate": 1}
    assert _routes()._runs_on_local_date(runs, "2026-08-27", "UTC") == {}


def test_a_run_with_only_an_end_clock_is_dated_by_it() -> None:
    runs = [{"run_id": "r", "line": "self_inquiry", "started_at": None,
             "finished_at": _ms("2026-08-27T10:00:00+00:00")}]
    assert _routes()._runs_on_local_date(runs, "2026-08-27", "UTC") == {"self_inquiry": 1}


def test_an_undated_run_is_not_counted_on_any_day() -> None:
    runs = [{"run_id": "killed", "line": "investigate", "started_at": None, "finished_at": None}]
    assert _routes()._runs_on_local_date(runs, "2026-08-27", "America/Denver") == {}


def test_no_local_date_reads_as_unknown_not_zero() -> None:
    """None must reach the page as None. Zero would make every tile claim a
    run vanished during a Redis outage that has nothing to do with Orion."""
    runs = [{"run_id": "r", "line": "investigate",
             "started_at": _ms("2026-08-27T10:00:00+00:00"), "finished_at": None}]
    assert _routes()._runs_on_local_date(runs, None, "America/Denver") is None


def test_an_unknown_zone_falls_back_rather_than_raising() -> None:
    runs = [{"run_id": "r", "line": "investigate",
             "started_at": _ms("2026-08-27T10:00:00+00:00"), "finished_at": None}]
    assert _routes()._runs_on_local_date(runs, "2026-08-27", "Not/AZone") == {"investigate": 1}


def test_an_iso_written_at_is_parsed_rather_than_read_as_missing() -> None:
    """Run `32b42392f495` wrote an ISO string where the prompt asks for
    `timestamp()`. Reading that as missing labelled a run that HAD written a
    :TurnOutcome as "died before writing an outcome" — Juniper caught it in the
    rendered page — and then let the undated run mask a genuinely traceless
    one."""
    from orion.curiosity.atlas import _stamp_ms

    assert _stamp_ms("2026-08-26T07:47:40.432241+00:00") == _ms(
        "2026-08-26T07:47:40.432241+00:00")
    assert _stamp_ms(1787840568235) == 1787840568235
    assert _stamp_ms("1787840568235") == 1787840568235
    assert _stamp_ms(None) is None
    assert _stamp_ms("") is None
    assert _stamp_ms("not a date") is None
