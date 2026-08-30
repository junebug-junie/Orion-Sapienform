#!/usr/bin/env python3
"""Every service touching the substrate control surface must be pointed at one.

The substrate control surface (`orion/substrate/mutation_control_surface.py`) is a
shared mutable store: orion-hub's mutation applier WRITES
`routing.chat_reflective_lane_threshold`, and orion-cortex-orch's
`decision_router.py` READS it on every routing decision.

`RuntimeControlSurfaceStore` fails *open*: with no database configured it silently
falls back to a per-process in-memory dict that returns the compiled-in default and
resets on restart. There is no error, no degraded flag, no log line. A writer and a
reader configured differently therefore look completely healthy while operating on
two different stores, and an adopted change can never reach the decision it was
adopted to change.

Confirmed live 2026-08-30, before this gate existed:

    orion-athena-hub          source_kind=postgres  threshold=0.5
    orion-athena-cortex-orch  source_kind=memory    threshold=0.75

Same code, same surface key, two stores, no symptom.

This gate finds every service whose source imports the control surface module and
asserts its compose file configures at least one of the keys
`_resolve_postgres_url()` actually reads, so the fail-open path cannot be reached
by accident.

Exit 0 clean, 1 on violations.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICES = REPO_ROOT / "services"
CONTROL_SURFACE_MODULE = "orion.substrate.mutation_control_surface"

# Exactly the keys _resolve_postgres_url() consults, in its order. Kept as a
# literal list so a change to the resolver that this gate does not know about
# shows up as a test failure rather than a silently weaker check.
RESOLVER_KEYS = (
    "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL",
    "SUBSTRATE_POLICY_POSTGRES_URL",
    "DATABASE_URL",
)


def _module_path(module: str) -> Path | None:
    candidate = REPO_ROOT / (module.replace(".", "/") + ".py")
    if candidate.exists():
        return candidate
    package = REPO_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _imported_orion_modules(path: Path) -> set[str]:
    """Every `orion.*` module a file imports, in either spelling.

    Substring-matching the full dotted module name misses
    `from orion.substrate import mutation_control_surface`, which is a form real
    code in this repo uses -- and misses transitive reach entirely.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("orion."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or not node.module.startswith("orion"):
                continue
            found.add(node.module)
            # `from orion.substrate import mutation_control_surface` -- the
            # submodule is a name, not part of node.module.
            for alias in node.names:
                found.add(f"{node.module}.{alias.name}")
    return found


def _reaches_control_surface(path: Path, cache: dict[Path, bool], stack: set[Path]) -> bool:
    """Does this file import the control surface, directly or transitively?

    Transitive matters: orion-field-digester reaches it through
    causal_geometry_producer -> mutation_trials -> get_chat_reflective_lane_threshold,
    and a direct-import check reports that service as clean.
    """
    if path in cache:
        return cache[path]
    if path in stack:  # import cycle
        return False
    stack.add(path)
    result = False
    for module in _imported_orion_modules(path):
        if module == CONTROL_SURFACE_MODULE or module.startswith(CONTROL_SURFACE_MODULE + "."):
            result = True
            break
        target = _module_path(module)
        if target is not None and _reaches_control_surface(target, cache, stack):
            result = True
            break
    stack.discard(path)
    cache[path] = result
    return result


def services_importing_control_surface() -> dict[str, list[Path]]:
    hits: dict[str, list[Path]] = {}
    cache: dict[Path, bool] = {}
    for path in sorted(SERVICES.glob("*/**/*.py")):
        parts = set(path.relative_to(SERVICES).parts)
        if "tests" in parts or "evals" in parts or "node_modules" in parts:
            continue
        if _reaches_control_surface(path, cache, set()):
            hits.setdefault(path.relative_to(SERVICES).parts[0], []).append(path)
    return hits


def _nonempty_assignment(text: str, key: str) -> bool:
    """True only if the key resolves to something non-empty.

    Presence is not enough. `_resolve_postgres_url()` strips each value and falls
    through on empty, so a bare passthrough is 100% fail-open:

        - SUBSTRATE_CONTROL_PLANE_POSTGRES_URL=${SUBSTRATE_CONTROL_PLANE_POSTGRES_URL}
          SUBSTRATE_CONTROL_PLANE_POSTGRES_URL: ${SUBSTRATE_CONTROL_PLANE_POSTGRES_URL:-}

    orion-hub ships exactly the first form with the key empty in .env_example, and
    works only because DATABASE_URL is also set. A presence check credits hub for
    a key that is provably unset.
    """
    # Horizontal whitespace only after the separator: `\s*` matches newlines, so
    # `KEY=` followed by `OTHER_KEY=` on the next line would capture that next
    # line as this key's value and report an empty key as configured. That is
    # exactly what it did for orion-hub, whose key IS empty.
    pattern = rf"^[^\S\n]*-?[^\S\n]*{re.escape(key)}[^\S\n]*[:=][^\S\n]*([^\n]*)$"
    for match in re.finditer(pattern, text, re.MULTILINE):
        value = match.group(1).strip().strip('"').strip("'")
        if not value:
            continue
        # ${VAR} / ${VAR:-} with no default resolves to empty when VAR is unset.
        bare = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*(:-\s*)?\}", value)
        if bare:
            continue
        return True
    return False


def compose_configures_a_resolver_key(service: str) -> tuple[bool, str | None]:
    """Look at compose AND .env_example: a service using `env_file:` legitimately
    configures keys there rather than in an `environment:` block."""
    compose = SERVICES / service / "docker-compose.yml"
    if not compose.exists():
        return False, f"no docker-compose.yml at {compose.relative_to(REPO_ROOT)}"
    sources: list[tuple[str, str]] = [("docker-compose.yml", compose.read_text(encoding="utf-8", errors="ignore"))]
    env_example = SERVICES / service / ".env_example"
    if env_example.exists():
        sources.append((".env_example", env_example.read_text(encoding="utf-8", errors="ignore")))
    for key in RESOLVER_KEYS:
        for origin, text in sources:
            if _nonempty_assignment(text, key):
                return True, f"{key} ({origin})"
    return False, None


def service_declares_sqlalchemy(service: str) -> bool:
    """The Postgres path does `from sqlalchemy import create_engine` inside a
    try/except that swallows ImportError into the in-memory fallback. Config alone
    is therefore not proof the path can run: orion-cortex-orch was correctly
    pointed at Postgres and still reported source_kind=memory with
    last_error="No module named 'sqlalchemy'"."""
    requirements = SERVICES / service / "requirements.txt"
    if not requirements.exists():
        return False
    text = requirements.read_text(encoding="utf-8", errors="ignore").lower()
    return any(
        line.strip().startswith("sqlalchemy")
        for line in text.splitlines()
        if not line.strip().startswith("#")
    )


def main() -> int:
    hits = services_importing_control_surface()
    if not hits:
        print("check_control_surface_store_parity: no services import the control surface; nothing to check")
        return 0

    violations: list[str] = []
    for service in sorted(hits):
        ok, detail = compose_configures_a_resolver_key(service)
        rel = [str(p.relative_to(REPO_ROOT)) for p in sorted(hits[service])]
        if ok and not service_declares_sqlalchemy(service):
            violations.append(
                f"  FAIL {service}: configured via {detail}, but requirements.txt does not\n"
                f"       declare SQLAlchemy.\n"
                f"       touches the control surface in: {', '.join(rel)}\n"
                f"       -> the Postgres path raises ImportError and is swallowed into\n"
                f"          the in-memory fallback. Configured correctly, still broken."
            )
            continue
        if ok:
            print(f"  OK   {service}: {detail} + SQLAlchemy")
        else:
            reason = detail or (
                "compose defines none of " + ", ".join(RESOLVER_KEYS)
            )
            violations.append(
                f"  FAIL {service}: {reason}\n"
                f"       touches the control surface in: {', '.join(rel)}\n"
                f"       -> falls back to a per-process in-memory store; reads the\n"
                f"          compiled-in default forever and never sees another\n"
                f"          service's writes."
            )

    if violations:
        print("\ncheck_control_surface_store_parity FAILED\n")
        print("\n".join(violations))
        print(
            "\nFix: add one of "
            + ", ".join(RESOLVER_KEYS)
            + " to the service's docker-compose.yml (and .env_example), pointing at\n"
            "the SAME database the other control-surface services use."
        )
        return 1

    print(f"\ncheck_control_surface_store_parity: {len(hits)} service(s) checked, all configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
