"""The reducer's domain map must not outlive a domain's producer.

CLAUDE.md section 0A: "When a metric is replaced by a better one, retire the old
one completely -- do not leave a partial exclusion in place." The transport
domain is the worked example of the failure this guards:

- 2026-07-26: `node:substrate.transport`'s producer write was removed.
- 2026-07-31: `_PREDICTION_ERROR_DOMAIN_NODE_IDS` was STILL listing it, so the
  brain-frame reducer read that node every tick and handed `transport` to
  `reduce_attention_self_model()`. Live value at removal: `prediction_error=0.556`,
  `observed_at=2026-07-24` -- seven days stale.

Nothing failed for those five days, which is exactly why a comment is not
sufficient here. This is the failing gate CLAUDE.md asks for instead of a
reminder.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS

WORKER = Path(__file__).resolve().parents[1] / "app" / "worker.py"
PREDICTION_ERROR = (
    Path(__file__).resolve().parents[3] / "orion" / "substrate" / "prediction_error.py"
)
REPLAY_HARNESS = (
    Path(__file__).resolve().parents[3]
    / "scripts" / "analysis" / "measure_ast_hot_reducer.py"
)


MAP_NAME = "_PREDICTION_ERROR_DOMAIN_NODE_IDS"


def _domain_map(source: str | None = None) -> dict[str, str]:
    """Parse the map out of worker.py without importing it.

    `app.worker` pulls in the whole substrate runtime (Redis, Postgres, FalkorDB
    clients) at import time, which this gate has no reason to require.

    Review finding 2026-07-31: the first version of this helper walked only
    `tree.body` for `ast.Assign`, which made the gate bypassable in two ways,
    both reproduced against the real parser:

    - A later `_PREDICTION_ERROR_DOMAIN_NODE_IDS["node:substrate.transport"] =
      "transport"` was completely invisible. Both tests passed while the runtime
      map had six entries and the retired domain was live again -- exactly the
      bug this gate exists to stop.
    - Adding a type annotation (`: dict[str, str] = {...}`) turns the node into
      an `ast.AnnAssign`, which the isinstance check skipped, so the gate
      `pytest.fail`ed blaming a missing constant for a cosmetic edit. Fail-closed
      rather than dangerous, but misleading -- and annotated is the style BOTH
      sibling maps in this repo already use.

    So: walk the whole tree, accept both assignment forms, and reject any
    post-hoc mutation of the name outright rather than trying to evaluate it.
    """
    tree = ast.parse(source if source is not None else WORKER.read_text())

    _reject_post_hoc_mutation(tree)

    found: dict[str, str] | None = None
    for node in ast.walk(tree):
        target_names: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            target_names = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            target_names = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == MAP_NAME for t in target_names):
            continue
        if node.value is None:
            continue
        try:
            found = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            raise AssertionError(
                f"{MAP_NAME} is no longer a literal dict (comprehension, call, or "
                "computed value). This gate can only check a literal. Either keep it "
                "literal or replace this gate with one that imports the real value."
            )
    if found is None:
        raise AssertionError(f"{MAP_NAME} not found as a literal assignment in worker.py")
    return found


def _reject_post_hoc_mutation(tree: ast.AST) -> None:
    """No `map[...] = ...`, `.update(...)`, or `.setdefault(...)` on the name.

    A parser that reads one literal cannot see a mutation applied afterwards, so
    the gate forbids the shape rather than pretending to evaluate it.

    Raises `AssertionError`, not `pytest.fail()`, specifically so
    `TestTheGateHasTeeth` below can assert this fires -- `pytest.fail` raises a
    `BaseException` subclass, which `pytest.raises(Exception)` does not catch,
    so a teeth-test written against it silently proves nothing.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == MAP_NAME
                ):
                    raise AssertionError(
                        f"{MAP_NAME} is mutated by subscript assignment at line "
                        f"{node.lineno}. This gate reads the literal only, so a "
                        "post-hoc entry would be invisible to it. Put every domain "
                        "in the literal."
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "setdefault", "pop", "__setitem__"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == MAP_NAME
        ):
            raise AssertionError(
                f"{MAP_NAME}.{node.func.attr}() at line {node.lineno} mutates the map "
                "after definition, which this gate cannot see. Put every domain in "
                "the literal."
            )


def test_domain_map_matches_active_inference_domains_exactly():
    """Set equality, not subset.

    Subset in one direction alone misses the real bug: a retired domain that
    still has a reader is a superset violation, and a live domain the reducer
    silently stopped reading is a subset violation. Both are the same class of
    "the map and the reducer disagree about what is real".
    """
    mapped = set(_domain_map().values())
    assert mapped == set(ACTIVE_INFERENCE_DOMAINS), (
        "worker.py's _PREDICTION_ERROR_DOMAIN_NODE_IDS and "
        "attention_self_model.ACTIVE_INFERENCE_DOMAINS disagree.\n"
        f"  only in the node map: {sorted(mapped - set(ACTIVE_INFERENCE_DOMAINS))}\n"
        f"  only in the reducer:  {sorted(set(ACTIVE_INFERENCE_DOMAINS) - mapped)}\n"
        "If a domain is being retired, remove it from BOTH in the same patch. "
        "If one is being added, wire it into both, or add it here with a comment "
        "explaining the deliberate asymmetry (see harness_closure/codebase)."
    )


def test_replay_harness_domain_map_matches_the_live_one():
    """The AST/HOT replay harness carries a SECOND copy of this map.

    `scripts/analysis/measure_ast_hot_reducer.py` is the harness
    `attention_self_model.py`'s own module docstring names as this reducer's
    validation path ("must be measured against real historical data ... before
    anything downstream consumes it"). It keeps its own
    `PREDICTION_ERROR_DOMAIN_NODES` and calls the same
    `reduce_attention_self_model()`.

    Review finding 2026-07-31: the first version of this gate parsed worker.py
    only, so that second copy kept `transport` after the live one dropped it --
    meaning the harness would average SIX values where production averages five,
    for the same tick, and silently stop reproducing live `confidence`. A
    validation harness that disagrees with the thing it validates is worse than
    no harness.
    """
    tree = ast.parse(REPLAY_HARNESS.read_text())
    found: dict[str, str] | None = None
    for node in ast.walk(tree):
        targets = (
            list(node.targets)
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(
            isinstance(t, ast.Name) and t.id == "PREDICTION_ERROR_DOMAIN_NODES"
            for t in targets
        ) and node.value is not None:
            found = ast.literal_eval(node.value)
    assert found is not None, "PREDICTION_ERROR_DOMAIN_NODES not found in the replay harness"
    # Note the inverted key/value orientation vs the live map -- domain -> node_id
    # there, node_id -> domain here. Compared as domain sets, not raw dicts.
    assert set(found) == set(ACTIVE_INFERENCE_DOMAINS), (
        "measure_ast_hot_reducer.py's PREDICTION_ERROR_DOMAIN_NODES and "
        "ACTIVE_INFERENCE_DOMAINS disagree.\n"
        f"  only in the harness: {sorted(set(found) - set(ACTIVE_INFERENCE_DOMAINS))}\n"
        f"  only in the reducer: {sorted(set(ACTIVE_INFERENCE_DOMAINS) - set(found))}\n"
        "The harness must replay the same domain set the live reducer reads, or its "
        "output is not comparable to production."
    )


def test_every_mapped_node_id_uses_the_domain_naming_convention():
    for node_id, domain in _domain_map().items():
        assert node_id == f"node:substrate.{domain}", (
            f"{node_id!r} does not follow node:substrate.<domain>; "
            "_write_prediction_error_node() upserts that fixed identity, so a "
            "mismatch here reads a node that is never written."
        )


def test_retired_transport_producer_stays_deleted():
    """`transport_prediction_error()` was deleted 2026-07-31, not just unwired.

    It was previously "kept, not deleted -- deleting it buys nothing", which is
    how a retired instrument gets wired back in. A live-importable symbol is an
    invitation; git history is the archive.
    """
    source = PREDICTION_ERROR.read_text()
    tree = ast.parse(source)
    defined = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "transport_prediction_error" not in defined, (
        "transport_prediction_error() is back in orion/substrate/prediction_error.py. "
        "It measured a 2-Redis-Stream 'world_pulse' census, not inter-service bus "
        "traffic. The successor is bus_synaptic_prediction_error()."
    )


class TestTheGateHasTeeth:
    """The bypasses code review actually demonstrated, asserted so they stay closed.

    A gate nobody has tried to defeat is a gate nobody knows the shape of. Each
    of these reproduces a real bypass found on 2026-07-31 against the first
    version of `_domain_map()`.
    """

    _LITERAL = (
        "_PREDICTION_ERROR_DOMAIN_NODE_IDS = {\n"
        '    "node:substrate.execution": "execution",\n'
        '    "node:substrate.biometrics": "biometrics",\n'
        '    "node:substrate.chat": "chat",\n'
        '    "node:substrate.route": "route",\n'
        '    "node:substrate.bus_synaptic": "bus_synaptic",\n'
        "}\n"
    )

    def test_subscript_reintroduction_is_caught(self):
        """The exact bypass review reproduced: add the retired domain on a later line."""
        source = self._LITERAL + (
            '_PREDICTION_ERROR_DOMAIN_NODE_IDS["node:substrate.transport"] = "transport"\n'
        )
        with pytest.raises(AssertionError) as excinfo:
            _domain_map(source)
        assert "subscript" in str(excinfo.value).lower()

    def test_update_call_is_caught(self):
        source = self._LITERAL + (
            '_PREDICTION_ERROR_DOMAIN_NODE_IDS.update({"node:substrate.transport": "transport"})\n'
        )
        with pytest.raises(AssertionError) as excinfo:
            _domain_map(source)
        assert "update" in str(excinfo.value)

    def test_annotated_assignment_still_parses(self):
        """Adding a type annotation must not make the gate cry 'not found'.

        Both sibling maps in this repo (measure_ast_hot_reducer.py,
        attention_organ_routes.py) are annotated, so this is a likely edit.
        """
        source = self._LITERAL.replace(
            "_PREDICTION_ERROR_DOMAIN_NODE_IDS = {",
            "_PREDICTION_ERROR_DOMAIN_NODE_IDS: dict[str, str] = {",
        )
        assert set(_domain_map(source).values()) == set(ACTIVE_INFERENCE_DOMAINS)
