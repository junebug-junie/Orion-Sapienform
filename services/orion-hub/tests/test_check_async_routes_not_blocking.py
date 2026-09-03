"""Tests for the gate that keeps blocking DB calls out of async routes.

The gate itself has now shipped green over the bug it exists to catch TWICE:
once when it only looked one level deep (the recommended fix moves the
blocking into a helper), and once when it followed helpers by name only (the
observability route reaches its loaders through a loop variable). Both times
it was caught by a human, not by anything automatic. Hence these tests: the
gate's own regressions are the failure mode with the worst track record here.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / "scripts" / "check_async_routes_not_blocking.py"


def _load():
    spec = importlib.util.spec_from_file_location("_async_gate", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_async_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load()


def _scan(tmp_path: pathlib.Path, src: str):
    f = tmp_path / "routes_under_test.py"
    f.write_text(src)
    return gate.scan_file(f, src.splitlines())


HEADER = "import asyncio\nfrom fastapi import APIRouter\nrouter = APIRouter()\n\n"


def test_flags_blocking_call_directly_in_the_route(tmp_path):
    src = HEADER + '''
@router.get("/x")
async def r():
    with engine.connect() as c:
        return c.execute("select 1")
'''
    assert _scan(tmp_path, src), "inline blocking call not flagged"


def test_flags_blocking_reached_through_a_sync_helper(tmp_path):
    """The recommended fix shape. The gate's FIRST version missed this."""
    src = HEADER + '''
def _load():
    with engine.connect() as c:
        return c.execute("select 1")


@router.get("/x")
async def r():
    return _load()
'''
    assert _scan(tmp_path, src), "one-hop helper not flagged"


def test_flags_blocking_reached_through_two_helpers(tmp_path):
    src = HEADER + '''
def _inner():
    with engine.connect() as c:
        return c.execute("select 1")


def _outer():
    return _inner()


@router.get("/x")
async def r():
    return _outer()
'''
    assert _scan(tmp_path, src), "two-hop chain not flagged"


def test_flags_helper_reached_through_a_loop_variable(tmp_path):
    """substrate_observability_routes' real shape. The SECOND version missed it.

    The callee is the loop variable `loader`, so a call-name-only check records
    the string "loader" and matches no helper.
    """
    src = HEADER + '''
def _section_a(engine):
    with engine.connect() as c:
        return c.execute("select 1")


@router.get("/x")
async def r():
    out = {}
    for name, loader in (("a", _section_a),):
        out[name] = loader(None)
    return out
'''
    assert _scan(tmp_path, src), "helper reached via a loop variable not flagged"


def test_flags_awaited_async_helper_that_blocks_internally(tmp_path):
    """`await`ing an async def whose body blocks still blocks the loop."""
    src = HEADER + '''
def _load():
    with engine.connect() as c:
        return c.execute("select 1")


async def _fused():
    return _load()


@router.get("/x")
async def r():
    return await _fused()
'''
    assert _scan(tmp_path, src), "awaited async helper that blocks not flagged"


def test_flags_blocking_evaluated_as_an_argument_to_to_thread(tmp_path):
    """to_thread's ARGUMENTS evaluate in the caller, on the loop."""
    src = HEADER + '''
def _blocking_arg():
    with engine.connect() as c:
        return c.execute("select 1")


def _work(x):
    return x


@router.get("/x")
async def r():
    return await asyncio.to_thread(_work, _blocking_arg())
'''
    assert _scan(tmp_path, src), "blocking call in a to_thread argument not flagged"


# --- must NOT fire ---------------------------------------------------------


def test_clean_when_the_blocking_half_is_threaded(tmp_path):
    src = HEADER + '''
def _load():
    with engine.connect() as c:
        return c.execute("select 1")


@router.get("/x")
async def r():
    return await asyncio.to_thread(_load)
'''
    assert _scan(tmp_path, src) == [], "correct to_thread usage was flagged"


def test_clean_for_a_nested_def_handed_to_to_thread(tmp_path):
    """grammar_atlas_routes._with_session's real shape -- correct code.

    A nested `def run()` is its own scope; inlining its body into the parent
    blames the parent for blocking that is in fact threaded.
    """
    src = HEADER + '''
async def _with_session(fn):
    def run():
        with engine.connect() as c:
            return fn(c)

    return await asyncio.to_thread(run)


@router.get("/x")
async def r():
    return await _with_session(lambda s: s)
'''
    assert _scan(tmp_path, src) == [], "threaded nested def was flagged"


def test_clean_for_an_awaited_async_client(tmp_path):
    """`await bus.connect()` is an async client, not a sync engine."""
    src = HEADER + '''
@router.get("/x")
async def r():
    await bus.connect()
    return await pool.execute("select 1")
'''
    assert _scan(tmp_path, src) == [], "awaited async client call was flagged"


def test_lifecycle_hooks_are_not_routes(tmp_path):
    """Building an engine in @app.on_event('startup') is the RIGHT pattern."""
    src = '''
import asyncio
from fastapi import FastAPI
app = FastAPI()


@app.on_event("startup")
async def startup():
    with engine.connect() as c:
        return c.execute("select 1")
'''
    assert _scan(tmp_path, src) == [], "a lifecycle hook was treated as a route"


def test_noqa_works_on_a_multi_line_call(tmp_path):
    """The escape hatch must work where it is naturally written.

    Checking only the call's FIRST line makes the comment silently inert on
    every multi-line call -- and the hub has 55 of those.
    """
    src = HEADER + '''
@router.get("/x")
async def r():
    return engine.connect(
        "a",
        "b",
    )  # noqa: async-blocking
'''
    assert _scan(tmp_path, src) == [], "noqa on the closing line did not apply"


def test_the_real_repo_is_clean(tmp_path):
    """The gate passes on the tree it ships with -- and can still fail."""
    assert gate.main() == 0
