"""orion:grammar:event's producer list must match the code that emits to it.

The channel's `producer_services` in orion/bus/channels.yaml drifted twice
before this gate existed: it listed three vision services that never emitted a
grammar event, and omitted orion-substrate-runtime, which does. A second,
unread copy (config/substrate-lattice/grammar_producer_registry.v1.yaml, deleted
2026-09-25) drifted further still. This test derives the producer set from the
code instead of from anyone's memory.

How the scan works (static, no imports of service code):
  * Walk every non-test .py file under orion/ and services/ that mentions
    GrammarProvenanceV1.
  * For each `GrammarProvenanceV1(...)` call, resolve the `source_service=`
    keyword: a string literal, or a module-level `NAME = "literal"` constant.
  * Anything else (e.g. `self.service_name`) is UNRESOLVED and must be listed
    in DYNAMIC_SOURCE_SITES with the identity it resolves to at runtime --
    a new dynamic site fails this test until someone names it.

Known limits: a producer that builds its provenance as a plain dict and
validates it (`GrammarEventV1.model_validate({...})`) instead of calling
GrammarProvenanceV1(...) is not seen; neither is one that emits a GrammarEventV1
from a language other than Python. None exist today (checked 2026-09-25 with
grep for `"source_service":` in modules that reference GrammarEventV1: only a
ledger reader and an unrelated orchestrator payload). Live cross-check used when
this gate was written: `SELECT source_service, count(*) FROM grammar_events
WHERE created_at > now() - interval '7 days' GROUP BY 1` returned exactly the
eight producers listed in channels.yaml.
"""
from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNEL = "orion:grammar:event"

# source_service values found in code that do NOT publish on orion:grammar:event.
EXCLUDED_SOURCES: dict[str, str] = {
    "orion-substrate-organs": (
        "pressure-organ candidate events: stored in OrganEmissionV1 and published on "
        "orion:grammar:accepted-pressure, not canonical ingress (0 grammar_events rows in 60 days)"
    ),
    "orion-grammar-seed": "orion/grammar/seed_demo.py deterministic demo trace for Substrate Atlas, never published",
}

# Call sites whose source_service is not a literal, mapped to the identity they
# carry at runtime.
DYNAMIC_SOURCE_SITES: dict[str, str] = {
    # GpuPoolRuntime._publish(): service_name defaults to "orion-gpu-pool"
    # (services/orion-gpu-pool/app/settings.py SERVICE_NAME).
    "services/orion-gpu-pool/app/runtime.py": "orion-gpu-pool",
}


def _is_test_path(path: Path) -> bool:
    parts = path.relative_to(REPO_ROOT).parts
    return "tests" in parts or path.name.startswith("test_") or path.name == "conftest.py"


def _scan() -> tuple[dict[str, set[str]], set[str]]:
    """Return ({source_service: {files}}, {files with unresolved source_service})."""
    found: dict[str, set[str]] = {}
    unresolved: set[str] = set()
    candidates = list((REPO_ROOT / "orion").rglob("*.py")) + list((REPO_ROOT / "services").rglob("*.py"))
    for path in candidates:
        if _is_test_path(path) or "node_modules" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "GrammarProvenanceV1" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        constants: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "GrammarProvenanceV1":
                continue
            value = next((kw.value for kw in node.keywords if kw.arg == "source_service"), None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.setdefault(value.value, set()).add(rel)
            elif isinstance(value, ast.Name) and value.id in constants:
                found.setdefault(constants[value.id], set()).add(rel)
            else:
                unresolved.add(rel)
    return found, unresolved


def _catalog_producers() -> set[str]:
    doc = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8"))
    entry = next(c for c in doc["channels"] if c.get("name") == CHANNEL)
    return set(entry.get("producer_services") or [])


def test_unresolved_source_service_sites_are_named() -> None:
    _, unresolved = _scan()
    assert unresolved == set(DYNAMIC_SOURCE_SITES), (
        "GrammarProvenanceV1(source_service=<non-literal>) sites changed. Add new ones to "
        f"DYNAMIC_SOURCE_SITES with their runtime identity; remove gone ones. unresolved={sorted(unresolved)}"
    )


def test_exclusions_are_still_real() -> None:
    found, _ = _scan()
    stale = set(EXCLUDED_SOURCES) - set(found)
    assert not stale, f"EXCLUDED_SOURCES entries no longer emitted anywhere, delete them: {sorted(stale)}"


def test_channels_yaml_producers_match_emitting_code() -> None:
    found, _ = _scan()
    emitted = (set(found) | set(DYNAMIC_SOURCE_SITES.values())) - set(EXCLUDED_SOURCES)
    catalog = _catalog_producers()
    missing = emitted - catalog
    phantom = catalog - emitted
    assert not missing and not phantom, (
        f"{CHANNEL} producer_services in orion/bus/channels.yaml is out of sync with code.\n"
        f"  emitted in code but not cataloged: {sorted(missing)} "
        f"(sites: { {s: sorted(found.get(s, [])) for s in missing} })\n"
        f"  cataloged but no emitter found: {sorted(phantom)}"
    )
