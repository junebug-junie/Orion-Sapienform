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


def services_importing_control_surface() -> dict[str, list[Path]]:
    hits: dict[str, list[Path]] = {}
    for path in SERVICES.glob("*/**/*.py"):
        if "/tests/" in str(path) or "/evals/" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if CONTROL_SURFACE_MODULE in text:
            service = path.relative_to(SERVICES).parts[0]
            hits.setdefault(service, []).append(path)
    return hits


def compose_configures_a_resolver_key(service: str) -> tuple[bool, str | None]:
    compose = SERVICES / service / "docker-compose.yml"
    if not compose.exists():
        return False, f"no docker-compose.yml at {compose.relative_to(REPO_ROOT)}"
    text = compose.read_text(encoding="utf-8", errors="ignore")
    for key in RESOLVER_KEYS:
        # Match the key in an environment position (`KEY:` or `- KEY=`), not a
        # mention inside a comment.
        if re.search(rf"^\s*-?\s*{re.escape(key)}\s*[:=]", text, re.MULTILINE):
            return True, key
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
