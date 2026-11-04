"""Regression: hub route modules must not build a connection pool per request.

Live 2026-10-30: nine `substrate_*_routes.py` modules called `create_engine()`
inside `_engine()`, so every request built a new pool and left its connection
open until garbage collection (eval: 22-25 pools per 25 requests). The hub held
~59 Postgres connections. See `evals/run_route_engine_pool_eval.py`.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

ROUTE_MODULES = (
    "substrate_attention_routes",
    "substrate_biometrics_routes",
    "substrate_execution_dispatch_routes",
    "substrate_field_routes",
    "substrate_observability_routes",
    "substrate_policy_routes",
    "substrate_proposal_routes",
)

# Known per-call pools that predate this gate, left alone on purpose (each
# builds a pool inside a loop/tick helper; fixing them is a separate change).
# The gate exists so this list can only shrink.
KNOWN_PER_CALL_ENGINES = frozenset(
    {
        ("chat_history_rehydrate.py", "fetch_recent_rows"),
        ("curiosity_hint.py", "_fetch_fresh_candidates"),
        ("hub_presence.py", "_write_snapshot_to_postgres"),
        ("outreach_provenance.py", "_fetch_session_rows"),
    }
)


@pytest.fixture
def fake_create_engine(monkeypatch):
    import sqlalchemy

    calls: list[tuple[str, dict]] = []

    def _fake(uri, **kwargs):
        calls.append((uri, kwargs))
        return object()

    monkeypatch.setattr(sqlalchemy, "create_engine", _fake)
    monkeypatch.setenv("POSTGRES_URI", "postgresql://u:p@127.0.0.1:1/db")
    from scripts import pg_engine

    pg_engine.module_engine.cache_clear()
    yield calls
    pg_engine.module_engine.cache_clear()


@pytest.mark.parametrize("name", ROUTE_MODULES)
def test_route_engine_is_built_once_and_reused(name, fake_create_engine) -> None:
    module = importlib.import_module(f"scripts.{name}")
    first = module._engine()
    assert all(module._engine() is first for _ in range(5))
    assert len(fake_create_engine) == 1
    assert fake_create_engine[0][1] == {"pool_pre_ping": True}


def test_each_module_keeps_its_own_pool(fake_create_engine) -> None:
    field = importlib.import_module("scripts.substrate_field_routes")
    policy = importlib.import_module("scripts.substrate_policy_routes")
    assert field._engine() is not policy._engine()
    assert len(fake_create_engine) == 2


def _per_call_engine_functions(path: Path) -> set[tuple[str, str]]:
    """Functions that call create_engine() without caching the result.

    Cached means: the function is decorated with a cache, the engine is
    assigned to a name the function declares `global`, or it is stored on an
    attribute/subscript inside `if <that same target> is None:`. A bare
    `self._engine = create_engine(...)` (e.g. in `__init__`) is NOT cached: the
    object may be rebuilt per call (felt_state_reader, 2026-11-03).
    `.setdefault(key, create_engine(...))` counts as cached: the discarded
    engine never connects.
    """
    tree = ast.parse(path.read_text())
    found: set[tuple[str, str]] = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = {ast.unparse(d).split("(")[0].split(".")[-1] for d in func.decorator_list}
        if decorators & {"lru_cache", "cache"}:
            continue
        globals_ = {n for node in ast.walk(func) if isinstance(node, ast.Global) for n in node.names}
        cached_calls = set()
        guarded = {
            ast.unparse(test.left)
            for node in ast.walk(func)
            if isinstance(node, ast.If)
            for test in ast.walk(node.test)
            if isinstance(test, ast.Compare)
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        }
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                targets = [
                    t.id
                    for target in node.targets
                    for t in ([target] if isinstance(target, ast.Name) else getattr(target, "elts", []))
                    if isinstance(t, ast.Name)
                ]
                stored = any(
                    isinstance(t, (ast.Attribute, ast.Subscript)) and ast.unparse(t) in guarded
                    for t in node.targets
                )
                if stored or any(t in globals_ for t in targets):
                    cached_calls.update(id(c) for c in ast.walk(node.value) if isinstance(c, ast.Call))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault":
                cached_calls.update(id(c) for arg in node.args for c in ast.walk(arg) if isinstance(c, ast.Call))
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func).split(".")[-1] == "create_engine"
                and id(node) not in cached_calls
            ):
                found.add((path.name, func.name))
    return found


def test_no_new_per_call_engines_in_hub_scripts() -> None:
    found: set[tuple[str, str]] = set()
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        found |= _per_call_engine_functions(path)
    new = found - KNOWN_PER_CALL_ENGINES
    assert not new, (
        f"create_engine() called per request/tick in {sorted(new)}; "
        "use scripts.pg_engine.module_engine(__name__, uri) or cache it at module level"
    )
    stale = KNOWN_PER_CALL_ENGINES - found
    assert not stale, f"fixed, remove from KNOWN_PER_CALL_ENGINES: {sorted(stale)}"


# Shared `orion/` code the hub imports. Live 2026-11-02: three substrate stores
# (review_telemetry, review_queue, policy_profiles) built a pool per call; the
# hub's 30 s autonomy loop held 149 connections. felt_state_reader built one
# per hub chat turn. Known per-call sites below predate this gate; the list can
# only shrink.
ORION_DIR = Path(__file__).resolve().parents[3] / "orion"
KNOWN_ORION_PER_CALL_ENGINES = frozenset(
    {
        ("curiosity/compaction_store.py", "engine"),
        ("discussion_window/sql_fetch.py", "fetch_discussion_window"),
        # Built once per cortex-exec runtime (endogenous_runtime.py).
        ("reasoning/calibration_profiles.py", "__init__"),
        # Real per-tick leak in orion-thought (reverie.py builds one per tick);
        # not a hub process, left for a separate change.
        ("reverie/referent_loader.py", "__init__"),
        ("social/scenario_replay.py", "_memory_service"),
        ("substrate/mutation_worker.py", "_acquire_leader_lock"),
    }
)


def test_no_new_per_call_engines_in_shared_orion_code() -> None:
    found: set[tuple[str, str]] = set()
    for path in sorted(ORION_DIR.rglob("*.py")):
        rel = path.relative_to(ORION_DIR)
        if {"tests", "evals"} & set(rel.parts):
            continue
        found |= {(rel.as_posix(), func) for _, func in _per_call_engine_functions(path)}
    new = found - KNOWN_ORION_PER_CALL_ENGINES
    assert not new, f"create_engine() called per call in {sorted(new)}; cache one engine per store/module"
    stale = KNOWN_ORION_PER_CALL_ENGINES - found
    assert not stale, f"fixed, remove from KNOWN_ORION_PER_CALL_ENGINES: {sorted(stale)}"
