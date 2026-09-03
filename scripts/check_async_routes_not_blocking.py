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


def _noqa_in_span(lines: list[str], lo: int, hi: int) -> bool:
    """`# noqa: async-blocking` anywhere in a node's line span.

    Checking only the START line silently does nothing on a multi-line call --
    and the natural place to put the comment is the closing paren. The hub has
    55 multi-line call sites and zero existing uses of this escape hatch, so
    the only documented way out of this gate had never been exercised.
    """
    for i in range(max(0, lo - 1), min(hi, len(lines))):
        if "noqa: async-blocking" in lines[i]:
            return True
    return False


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


#: Decorators that look like routes but are not serving traffic. Building an
#: engine in `@app.on_event("startup")` is the RECOMMENDED pattern -- nothing
#: is being served yet -- so flagging it would turn CI red on correct code.
_NON_ROUTE_DECORATORS = {"on_event", "middleware", "exception_handler"}


def _is_route(node: ast.AsyncFunctionDef) -> bool:
    for d in node.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if not isinstance(f, ast.Attribute):
            continue
        if getattr(getattr(f, "value", None), "id", "") not in {"router", "app"}:
            continue
        if f.attr in _NON_ROUTE_DECORATORS:
            continue
        return True
    return False


def _direct_blocking_calls(
    node: ast.AST, awaited: set[int], lines: list[str]
) -> list[tuple[int, str]]:
    """Blocking calls lexically inside `node`, ignoring to_thread-shielded ones."""
    out: list[tuple[int, str]] = []

    def walk(n: ast.AST, shielded: bool) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return  # own scope; registered as its own helper
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in {"to_thread", "run_in_executor"}:
                for c in ast.iter_child_nodes(n):
                    walk(c, True)
                return
            if not shielded and id(n) not in awaited:
                if not _noqa_in_span(lines, n.lineno, getattr(n, "end_lineno", n.lineno)):
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


def _referenced_names(node: ast.AST) -> set[str]:
    """Every bare NAME the body reaches that is not shielded by to_thread.

    References, not just direct calls, and deliberately so. Three real shapes
    defeat a call-only check, all of them live in this repo:

      * `for name, loader in ((..., _section_a), ...): sections[n] = loader(e)`
        -- the callee is a loop variable, so a call-only check records the
        string "loader" and matches nothing. This exact shape is how the
        observability route stayed blocking while the gate reported clean.
      * `functools.partial(_blocking_helper, x)` and lambdas.
      * A blocking helper passed as an ARGUMENT to to_thread: the shielded
        subtree is the call, but arguments evaluate in the caller, on the loop.

    Under to_thread the whole subtree really is off the loop, so it is skipped;
    but a name that appears anywhere else in an async route is treated as
    reached. That over-approximates -- `# noqa: async-blocking` is the way out.
    """
    out: set[str] = set()

    def walk(n: ast.AST, shielded: bool) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A nested def is its own scope, registered separately as a
            # helper. Its body is NOT the parent's code -- `def run(): ...;
            # await asyncio.to_thread(run)` is correct, and inlining run()'s
            # body here would blame the parent for blocking that is threaded.
            return
        if isinstance(n, ast.Call):
            f = n.func
            nm = getattr(f, "attr", None) or getattr(f, "id", None)
            if nm in {"to_thread", "run_in_executor"}:
                # The callable and its args run off the loop EXCEPT arguments
                # that are themselves calls -- those evaluate in the caller.
                for arg in list(n.args) + [k.value for k in n.keywords]:
                    walk(arg, not isinstance(arg, ast.Call))
                return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and not shielded:
            out.add(n.id)
        for c in ast.iter_child_nodes(n):
            walk(c, shielded)

    for child in ast.iter_child_nodes(node):
        walk(child, False)
    return out


def scan_file(path: pathlib.Path, lines: list[str]) -> list[tuple[int, str, str]]:
    """Flag blocking work reachable from an async route without crossing to_thread.

    Not merely lexically inside the route. The blocking half is normally
    factored into a module-local helper -- that IS the recommended fix shape --
    so a checker that only looks one level deep passes the very bug it exists
    to catch. This file's first version did exactly that. Its second version
    followed plain `def` helpers only, and so missed BOTH an `async def` helper
    that blocks internally (awaiting one still blocks the loop) and a helper
    reached through a loop variable. All three shapes are live in orion-hub.

    Helper reachability is resolved to a fixed point over both `def` and
    `async def` helpers.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    awaited = _awaited_calls(tree)

    # Both sync AND async helpers: `await _helper()` where _helper's body does
    # a synchronous DB call blocks the loop just as hard as calling it directly.
    helper_blocking: dict[str, list[tuple[int, str]]] = {}
    helper_refs: dict[str, set[str]] = {}
    # ast.walk reaches nested defs too, and each is registered by its own bare
    # name -- so `run` inside `_with_session` is a helper in its own right.
    # Same-name collisions within one file resolve conservatively (a false
    # positive, silenced with `# noqa: async-blocking`), never silently clean.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            helper_blocking[node.name] = _direct_blocking_calls(node, awaited, lines)
            helper_refs[node.name] = _referenced_names(node)

    blocking_helpers: dict[str, tuple[int, str]] = {
        n: hits[0] for n, hits in helper_blocking.items() if hits
    }
    changed = True
    while changed:
        changed = False
        for name, refs in helper_refs.items():
            if name in blocking_helpers:
                continue
            for ref in refs:
                if ref in blocking_helpers and ref != name:
                    blocking_helpers[name] = blocking_helpers[ref]
                    changed = True
                    break

    findings: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or not _is_route(node):
            continue
        if _noqa_in_span(lines, node.lineno, getattr(node, "end_lineno", node.lineno)):
            continue
        for lineno, why in _direct_blocking_calls(node, awaited, lines):
            findings.append((lineno, node.name, why))
        for ref in sorted(_referenced_names(node)):
            if ref in blocking_helpers and ref != node.name:
                bl, why = blocking_helpers[ref]
                findings.append(
                    (
                        node.lineno,
                        node.name,
                        f"reaches {ref}() on the loop, which blocks at line {bl}: {why}",
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
