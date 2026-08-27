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
    ATLAS_GROWTH_CYPHER,
    ATLAS_PRIORS_CYPHER,
    ATLAS_REVISIONS_CYPHER,
    AtlasView,
    assemble_runs,
    read_atlas,
    to_payload,
    trajectory_for,
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
_GROWTH = "WHERE n.run_id IS NOT NULL"
_OUTCOMES = "t.continue_line"
_HOPS = "h.n AS n"
_FINDINGS = "f.finding_id"


# --- unreachable, unconfigured and empty are three different states --------


def test_an_unreachable_graph_is_not_an_empty_world_view() -> None:
    view = read_atlas(_Reader(raises=True))
    assert view.is_unavailable
    assert view.priors == [] and view.runs == []
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


# --- runs are assembled from run_id attribution ----------------------------


def test_a_run_appears_even_when_it_never_wrote_an_outcome() -> None:
    """A turn killed mid-write leaves hops and no `:TurnOutcome`. Showing
    nothing for it would hide exactly the runs worth looking at — this happened
    live on 2026-08-27 when both containers were recreated mid-turn."""
    runs = assemble_runs(
        growth_rows=[{"label": "Hop", "run_id": "killed", "n": 2}],
        outcome_rows=[],
        hop_rows=[
            {"run_id": "killed", "n": 1, "note": "got this far"},
            {"run_id": "killed", "n": 2, "note": "and this far"},
        ],
        finding_rows=[],
        priors=[],
    )
    assert [r.run_id for r in runs] == ["killed"]
    assert runs[0].hops == 2
    assert [h["n"] for h in runs[0].hop_notes] == [1, 2]
    assert runs[0].added == {"Hop": 2}
    # No outcome node, so the safe defaults hold: a killed turn must not read
    # as having decided to continue or to reach out.
    assert runs[0].written_at is None
    assert runs[0].continue_line is False
    assert runs[0].reach_out is False


def test_hop_notes_are_ordered_by_their_own_number_not_by_read_order() -> None:
    """FalkorDB returns rows unordered and the notes are a narrative: hop 3
    printed above hop 1 is a scrambled account of the turn."""
    runs = assemble_runs(
        growth_rows=[], outcome_rows=[],
        hop_rows=[{"run_id": "r", "n": n, "note": f"note {n}"} for n in (3, 1, 5, 2)],
        finding_rows=[], priors=[],
    )
    assert [h["n"] for h in runs[0].hop_notes] == [1, 2, 3, 5]


def test_a_run_with_no_timestamp_sorts_last_not_first() -> None:
    """A missing `written_at` is "unknown", not "oldest". Sorting it as 0 would
    float every killed run to the bottom of a newest-first list, or worse, the
    top."""
    runs = assemble_runs(
        growth_rows=[
            {"label": "Hop", "run_id": "dated", "n": 1},
            {"label": "Hop", "run_id": "undated", "n": 1},
        ],
        outcome_rows=[{"run_id": "dated", "written_at": 1787840568235,
                       "continue_line": True, "continue_note": "n",
                       "reach_out": False, "reach_out_why": ""}],
        hop_rows=[], finding_rows=[], priors=[],
    )
    assert [r.run_id for r in runs] == ["dated", "undated"]


def test_creating_and_re_testing_a_prior_are_attributed_to_different_runs() -> None:
    """`run_id` is who made it, `last_run_id` is who last touched it. Collapsing
    them would credit run 5 with forming a prior run 3 wrote."""
    priors = read_atlas(_Reader(answers={_PRIORS: [
        _prior(pid="p1", run="run_three", last_run="run_five", tested=1)]})).priors
    runs = assemble_runs(
        growth_rows=[], outcome_rows=[], hop_rows=[], finding_rows=[],
        priors=priors,
    )
    by_id = {r.run_id: r for r in runs}
    assert by_id["run_three"].priors_created == ["p1"]
    assert by_id["run_three"].priors_touched == []
    assert by_id["run_five"].priors_touched == ["p1"]
    assert by_id["run_five"].priors_created == []


def test_a_prior_created_and_tested_by_one_run_is_not_double_counted() -> None:
    priors = read_atlas(_Reader(answers={_PRIORS: [
        _prior(pid="p1", run="r1", last_run="r1")]})).priors
    runs = assemble_runs(growth_rows=[], outcome_rows=[], hop_rows=[],
                         finding_rows=[], priors=priors)
    assert runs[0].priors_created == ["p1"]
    assert runs[0].priors_touched == []


def test_growth_is_counted_from_labels_not_from_the_typed_readers() -> None:
    """Totals come from one `MATCH (n)` over every labelled node, so a kind
    nobody has a reader for yet still shows up rather than silently vanishing
    from the graph-growth panel."""
    assert "MATCH (n)" in ATLAS_GROWTH_CYPHER
    assert "labels(n)[0]" in ATLAS_GROWTH_CYPHER
    runs = assemble_runs(
        growth_rows=[{"label": "SomethingNew", "run_id": "r1", "n": 4}],
        outcome_rows=[], hop_rows=[], finding_rows=[], priors=[],
    )
    assert runs[0].added == {"SomethingNew": 4}
    assert runs[0].total_added == 4


# --- the reads themselves ---------------------------------------------------


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
    for cypher in (ATLAS_PRIORS_CYPHER, ATLAS_REVISIONS_CYPHER, ATLAS_GROWTH_CYPHER):
        assert "{" not in cypher and "%" not in cypher


def test_a_prior_with_no_id_is_dropped_rather_than_invented() -> None:
    view = read_atlas(_Reader(answers={_PRIORS: [_prior(pid=""), _prior(pid="ok")]}))
    assert [p.prior_id for p in view.priors] == ["ok"]


def test_the_payload_is_json_safe() -> None:
    import json

    view = read_atlas(_Reader(answers={
        _PRIORS: [_prior(pid="p1", tested=1, last_run="r2")],
        _REVS: [{"prior_id": "p1", "run_id": "r2", "from_confidence": "0.9",
                 "to_confidence": "0.85", "from_status": "open",
                 "to_status": "revised", "written_at": 7}],
        _GROWTH: [{"label": "Prior", "run_id": "r1", "n": 1}],
        _OUTCOMES: [{"run_id": "r1", "continue_line": True, "continue_note": "n",
                     "reach_out": True, "reach_out_why": "w", "written_at": 9}],
        _HOPS: [{"run_id": "r1", "n": 1, "note": "h"}],
        _FINDINGS: [{"finding_id": "f1", "text": "t", "evidence": "e",
                     "run_id": "r1", "written_at": 9}],
    }))
    blob = json.dumps(to_payload(view))
    assert '"reach_out": true' in blob
    assert '"history_recorded": true' in blob


# --- the surface itself -----------------------------------------------------


def test_the_operator_surface_exposes_no_write_route() -> None:
    """Read-only is a design constraint, not a phase-one scope cut: Hub never
    writes to Orion's graph, and a route that could edit a belief Orion formed
    needs an auth story, an audit trail, and an argument this does not have.
    Asserted on the router rather than trusted to review."""
    import sys
    from pathlib import Path

    hub = Path(__file__).resolve().parents[1] / "services" / "orion-hub"
    if str(hub) not in sys.path:
        sys.path.insert(0, str(hub))
    from scripts.curiosity_routes import router

    assert router.routes, "the router registered nothing"
    for route in router.routes:
        assert route.methods <= {"GET", "HEAD"}, (route.path, route.methods)


def test_the_schedule_keys_are_imported_from_the_loop_that_writes_them() -> None:
    """A dashboard with its own copy of `orion:curiosity:count:` would render a
    confident 0 forever the day that prefix changes."""
    import sys
    from pathlib import Path

    hub = Path(__file__).resolve().parents[1] / "services" / "orion-hub"
    if str(hub) not in sys.path:
        sys.path.insert(0, str(hub))
    import ast

    source = (hub / "scripts" / "curiosity_routes.py").read_text()
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
