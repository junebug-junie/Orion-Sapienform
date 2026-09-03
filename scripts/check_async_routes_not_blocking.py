#!/usr/bin/env python3
"""Gate: no `async def` HTTP route may do synchronous DB work inline.

A blocking call inside an `async def` holds the whole event loop, so EVERY
request the service is serving stalls for its duration -- not just the one
that made the call. That is invisible in a per-endpoint latency check, because
the endpoint doing the blocking is usually fast; it is the unrelated static
asset five requests later that takes 30 seconds.

Found live twice on 2026-09-03 in orion-hub, hours apart:

  * /api/biometrics/preview/induction -- a 418ms synchronous SQLAlchemy query
    on a 10s poll. Static JS requests stalled 47-60s.
  * /api/self-brain/frames/tail -- the hub's most-polled endpoint (every 3s,
    163 hits in 5 minutes), building a NEW engine per request.

Both were the same mistake, written months apart by different patches, and
neither was caught by tests or by any latency check. Hence a gate.

Exit 1 on any finding. Escape hatch for a call that is genuinely non-blocking
(an in-memory fake, a client that only looks like a DB): put
`# noqa: async-blocking` on the offending line.
"""
from __future__ import annotations

import ast
import pathlib
import sys

# Attribute/function names that mean "synchronous database work".
SYNC_CALL_NAMES = {
    "create_engine": "create_engine() builds a pool and opens a connection",
    "connect": "engine.connect()/psycopg2.connect() is synchronous",
    "execute": ".execute() on a sync SQLAlchemy connection",
}
# Modules whose calls block the loop outright.
SYNC_MODULE_CALLS = {"requests": "blocking HTTP"}

SCAN_ROOTS = [pathlib.Path("services/orion-hub/scripts")]


def _awaited_calls(tree: ast.AST) -> set[int]:
    """id() of every Call that is the direct operand of an `await`.

    `await bus.connect()` is an async client, not a sync engine -- flagging it
    was this checker's own first false positive.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            out.add(id(node.value))
    return out


def _is_route(node: ast.AsyncFunctionDef) -> bool:
    for d in node.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if isinstance(f, ast.Attribute) and getattr(getattr(f, "value", None), "id", "") in {
            "router",
            "app",
        }:
            return True
    return False


def _direct_blocking_calls(
    node: ast.AST, awaited: set[int], lines: list[str]
) -> list[tuple[int, str]]:
    """Blocking calls lexically inside `node`, ignoring to_thread-shielded ones."""
    out: list[tuple[int, str]] = []

    def walk(n: ast.AST, shielded: bool) -> None:
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in {"to_thread", "run_in_executor"}:
                for c in ast.iter_child_nodes(n):
                    walk(c, True)
                return
            if not shielded and id(n) not in awaited:
                src = lines[n.lineno - 1] if n.lineno <= len(lines) else ""
                if "noqa: async-blocking" not in src:
                    if name in SYNC_CALL_NAMES:
                        out.append((n.lineno, SYNC_CALL_NAMES[name]))
                    mod = getattr(getattr(f, "value", None), "id", None)
                    if mod in SYNC_MODULE_CALLS:
                        out.append((n.lineno, f"{mod}.{name}() -- {SYNC_MODULE_CALLS[mod]}"))
        for c in ast.iter_child_nodes(n):
            walk(c, shielded)

    for child in ast.iter_child_nodes(node):
        walk(child, False)
    return out


def _called_names(node: ast.AST, awaited: set[int]) -> set[str]:
    """Plain-name calls inside `node` NOT shielded by to_thread."""
    out: set[str] = set()

    def walk(n: ast.AST, shielded: bool) -> None:
        if isinstance(n, ast.Call):
            f = n.func
            nm = getattr(f, "attr", None) or getattr(f, "id", None)
            if nm in {"to_thread", "run_in_executor"}:
                return  # everything under here runs off the loop
            if not shielded and isinstance(f, ast.Name) and id(n) not in awaited:
                out.add(f.id)
        for c in ast.iter_child_nodes(n):
            walk(c, shielded)

    for child in ast.iter_child_nodes(node):
        walk(child, False)
    return out


def scan_file(path: pathlib.Path, lines: list[str]) -> list[tuple[int, str, str]]:
    """Flag blocking work reachable from an async route without crossing to_thread.

    Not merely lexically inside the route. The blocking half is normally
    factored into a module-local `def` helper -- that IS the recommended fix
    shape -- so a checker that only looks one level deep passes the very bug it
    exists to catch. Confirmed: this file's first version scored the reverted,
    still-broken `frames_tail` as clean, because the blocking had moved one call
    away. Helper reachability is resolved to a fixed point, so a chain is caught.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    awaited = _awaited_calls(tree)

    helper_blocking: dict[str, list[tuple[int, str]]] = {}
    helper_calls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            helper_blocking[node.name] = _direct_blocking_calls(node, awaited, lines)
            helper_calls[node.name] = _called_names(node, awaited)

    blocking_helpers: dict[str, tuple[int, str]] = {
        n: hits[0] for n, hits in helper_blocking.items() if hits
    }
    changed = True
    while changed:
        changed = False
        for name, callees in helper_calls.items():
            if name in blocking_helpers:
                continue
            for callee in callees:
                if callee in blocking_helpers:
                    blocking_helpers[name] = blocking_helpers[callee]
                    changed = True
                    break

    findings: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or not _is_route(node):
            continue
        for lineno, why in _direct_blocking_calls(node, awaited, lines):
            findings.append((lineno, node.name, why))
        for callee in _called_names(node, awaited):
            if callee in blocking_helpers:
                bl, why = blocking_helpers[callee]
                findings.append(
                    (
                        node.lineno,
                        node.name,
                        f"calls {callee}() inline, which blocks at line {bl}: {why}",
                    )
                )
    return findings


def main() -> int:
    total = 0
    for root in SCAN_ROOTS:
        if not root.is_dir():
            print(f"scan root missing: {root} -- refusing to pass vacuously")
            return 1
        files = sorted(root.glob("*.py"))
        if not files:
            print(f"no python files under {root} -- discovery is broken, not the tree clean")
            return 1
        for f in files:
            lines = f.read_text().splitlines()
            for lineno, fn, why in scan_file(f, lines):
                total += 1
                print(f"{f}:{lineno}  async def {fn}(): {why}")

    if total:
        print(
            f"\n{total} blocking call(s) inside async routes.\n"
            "Move the blocking half into a plain `def` helper and await it with\n"
            "`asyncio.to_thread(...)`. See services/orion-hub/scripts/self_brain_routes.py\n"
            "for the shape. If a call only looks blocking, mark it "
            "`# noqa: async-blocking`."
        )
        return 1
    print("no blocking calls inside async routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
