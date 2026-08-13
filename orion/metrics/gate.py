"""Phase 4: deterministic gate over the metric semantic layer.

NOT wired into CI despite the name: no .github/workflows/ job runs any
`make check-*` target, this one included. It is a local target that can be
skipped. Wiring it in is a follow-up, stated so the name does not overclaim.

Turns three of CLAUDE.md 0A's metric-quality rules from prose into a failing
check. Every check here is provable from repo state -- no naming heuristics,
no keyword lists.

What it checks
--------------
1. **Declared-consumer existence.** Every consumer a registry *claims* must
   exist on disk, at BOTH levels: the module file, and the callable named
   after the colon. Catches the deleted
   `services.orion-spark-introspector.app.inner_state:build_inner_state_features`,
   and also the likelier rot -- a function renamed inside a file that still
   exists. This is the load-bearing check.

2. **Orphan ratchet.** The count of registered metrics that feed nothing --
   no discoverable code consumer AND no surviving declared consumer -- may not
   grow. A metric that names something but feeds nothing is the definition of
   a keyword cathedral (0A).

3. **Referential integrity.** Causal parent organs must exist in
   ORGAN_REGISTRY. The sibling `upstream` URN check is a cheap invariant that
   cannot currently fail on real data (see check_registry_integrity's
   docstring) -- it is retained deliberately but is not doing real work today,
   and is not counted as load-bearing.

What it deliberately does NOT check
-----------------------------------
**Newly introduced but unregistered metrics** -- e.g. `arena_degeneracy`,
shipped 2026-08-13 and invisible to this layer. Catching that statically would
mean guessing which new string dict keys are "metric-shaped", and the only
available signal is a suffix list (`*_pressure`, `*_load`, `*_error`, ...).
That is precisely the keyword cathedral 0A bans, and it would be both
false-positive noisy and trivially evaded by a name outside the list.

Registration is enforced by review and by the phase 3 edit-time card, not by a
heuristic pretending to be a gate. Stated here so the absence reads as a
decision rather than an oversight.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from orion.metrics.consumers import scan_repo
from orion.metrics.lineage import MetricGraph, MetricNode, build_graph

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "config" / "metrics" / "orphan_baseline.json"

# Trailing prose some registry entries append after the callable, e.g.
# "...:build_chat_stance_inputs (ctx['chat_drive_state'] via drive_state_postgres)"
_TRAILING_PROSE = re.compile(r"\s*\(.*\)\s*$", re.DOTALL)


@dataclass
class GateResult:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _resolve_consumer_path(consumer: str, repo_root: Path) -> Path | None:
    """Map a declared consumer string to the path it asserts exists.

    Two live formats:
      - dotted module + callable: `services.<svc>.app.<mod>:<func>`
      - bare compose service name: `orion-hub`

    Returns None when the claim is not checkable (wildcards).
    """
    consumer = _TRAILING_PROSE.sub("", consumer).strip()
    if not consumer or consumer == "*":
        return None

    module = consumer.split(":", 1)[0]
    if "." in module:
        # services.orion-hub.app.foo -> services/orion-hub/app/foo.py, or the
        # package form services/orion-hub/app/foo/__init__.py. Returning only
        # the .py path reported a false failure for any package-shaped module.
        base = repo_root / module.replace(".", "/")
        as_module = base.with_suffix(".py")
        if as_module.exists():
            return as_module
        if (base / "__init__.py").exists():
            return base / "__init__.py"
        return as_module
    # bare service name -> services/<name>/
    return repo_root / "services" / module


def _declared_callable_missing(consumer: str, repo_root: Path) -> str | None:
    """Check the callable half of `module:func`, not just the module file.

    The module-only check caught orion-spark-introspector solely because the
    whole service directory was deleted. Renaming the function inside a file
    that still exists left the registry's claim false while the gate stayed
    green -- which is the more likely way this rots.

    Returns a failure description, or None when the claim holds (including
    when it is not checkable).
    """
    cleaned = _TRAILING_PROSE.sub("", consumer).strip()
    if ":" not in cleaned:
        return None
    func = cleaned.split(":", 1)[1].strip()
    if not func or not func.isidentifier():
        return None
    path = _resolve_consumer_path(cleaned, repo_root)
    if path is None or not path.exists():
        return None  # module-level failure is reported by the caller
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == func:
                return None
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == func:
                    return None
    return (
        f"declares {cleaned!r} but {path.relative_to(repo_root)} defines no "
        f"{func!r} (renamed or removed?)"
    )


def check_registry_integrity(graph: MetricGraph) -> list[str]:
    """Referential integrity of the resolved graph.

    Honest scope note: the `upstream` URN check below cannot currently fail on
    real data. `upstream` is populated in exactly one place
    (lineage.resolve_inner_state), with the URN of a parent node appended a few
    lines earlier in the same loop -- so it is structurally consistent by
    construction, and only the hand-built graphs in tests exercise it. It is
    kept as a cheap invariant that will start earning its keep the moment any
    resolver populates `upstream` from a separate source, but it is not doing
    real work today and should not be counted as one of the load-bearing
    checks.

    `upstream_organs` IS checkable against live data and can genuinely fail:
    it holds organ ids copied from `causal_parent_organs`, and nothing
    otherwise guarantees a parent organ still exists in ORGAN_REGISTRY.
    """
    known = set(graph.nodes)
    failures = []
    for node in graph.nodes.values():
        for parent in node.upstream:
            if parent not in known:
                failures.append(f"dangling upstream: {node.urn} -> {parent} (no such node)")

    known_organs = _known_organ_ids()
    if known_organs:
        seen: set[tuple[str, str]] = set()
        for node in graph.nodes.values():
            for organ in node.upstream_organs:
                if organ in known_organs or (node.urn, organ) in seen:
                    continue
                seen.add((node.urn, organ))
                failures.append(
                    f"{node.urn} names causal parent organ {organ!r}, "
                    "which is not in ORGAN_REGISTRY"
                )

    if not graph.nodes:
        failures.append("metric graph resolved to zero nodes -- registries did not load")
    return failures


def _known_organ_ids() -> set[str]:
    try:
        from orion.signals.registry import ORGAN_REGISTRY
    except Exception:
        return set()
    return set(ORGAN_REGISTRY)


def missing_declared_consumers(graph: MetricGraph, repo_root: Path) -> dict[str, str]:
    """Declared consumers that do not exist on disk -> where they are declared."""
    missing: dict[str, str] = {}
    for node in graph.nodes.values():
        for consumer in node.declared_consumers:
            target = _resolve_consumer_path(consumer, repo_root)
            if target is None or target.exists():
                continue
            missing.setdefault(consumer, node.registry_source)
    return missing


def check_declared_consumers(
    graph: MetricGraph, repo_root: Path, known_missing: Iterable[str] = ()
) -> list[str]:
    """Every consumer a registry claims must exist on disk.

    Pre-existing breakage is carried in the baseline's known_missing_consumers
    rather than silently waived: the gate's job is stopping NEW false claims,
    and the recorded ones stay visible and are expected to shrink. Fixing them
    means editing orion/bus/channels.yaml consumer_services, a contract change
    (CLAUDE.md 6) that belongs in its own patch.
    """
    allowed = set(known_missing)
    failures = []
    for consumer, source in sorted(missing_declared_consumers(graph, repo_root).items()):
        if consumer in allowed:
            continue
        target = _resolve_consumer_path(consumer, repo_root)
        failures.append(
            f"{source} declares a consumer that does not exist: {consumer!r} "
            f"(looked for {target.relative_to(repo_root) if target else '?'})"
        )

    # Callable-level check: the module can exist while the function it names
    # does not. Same waiver set applies.
    seen: set[str] = set()
    for node in graph.nodes.values():
        for consumer in node.declared_consumers:
            if consumer in allowed or consumer in seen:
                continue
            seen.add(consumer)
            problem = _declared_callable_missing(consumer, repo_root)
            if problem:
                failures.append(f"{node.registry_source} {problem}")
    return sorted(failures)


def orphan_nodes(graph: MetricGraph, scan, repo_root: Path) -> list[MetricNode]:
    """Registered metrics that feed nothing: no code consumer AND no surviving
    declared consumer.

    Three corrections over the first version, each measured on the live repo:

    - **Declared consumers count.** Ignoring them made the ratchet's largest
      component mostly false: 132 of 150 `bus_channel` "orphans" had non-empty
      `consumer_services` in channels.yaml. Bus channels are subscribed via
      config, and wildcard names (`orion:effect:*`) can never match a literal
      string read at all, so they were permanent orphans by construction. A
      PR adding one legitimately-consumed channel would have failed the gate
      and forced a baseline bump -- the silent waiver this design set out to
      avoid.
    - **Nodes, not tokens.** 588 nodes collapse to 391 tokens, 60 of which map
      to multiple nodes. Counting tokens meant a newly registered metric
      reusing an existing name (a second organ with `signal_kind: gpu_load`)
      moved the count by zero.
    - **Each node keeps its own surface.** Three tokens span surfaces today
      (`memory_pressure`, `repair_pressure`, `confidence`). Attributing a
      token to the first resolver's surface let a consumer of the
      field-channel `confidence` mask an orphaned inner-state `confidence`,
      and pointed any failure at the wrong registry.
    """
    orphans: list[MetricNode] = []
    consumer_cache: dict[str, bool] = {}
    for node in graph.nodes.values():
        token = node.scan_token
        if token not in consumer_cache:
            consumer_cache[token] = bool(
                scan.consumers_for(token, exclude_paths=graph.registry_sources_for(token))
            )
        if consumer_cache[token]:
            continue
        if _has_surviving_declared_consumer(node, repo_root):
            continue
        orphans.append(node)
    return orphans


def _has_surviving_declared_consumer(node: MetricNode, repo_root: Path) -> bool:
    for consumer in node.declared_consumers:
        target = _resolve_consumer_path(consumer, repo_root)
        # A wildcard consumer ("*") is unverifiable but is a real claim of
        # consumption, so it counts as declared rather than as an orphan.
        if target is None or target.exists():
            return True
    return False


def orphan_counts(graph: MetricGraph, scan, repo_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in orphan_nodes(graph, scan, repo_root):
        counts[node.surface] = counts.get(node.surface, 0) + 1
    return counts


def load_baseline(path: Path | None = None) -> dict:
    target = path or BASELINE_PATH
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def check_orphan_ratchet(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Orphan population may shrink, never grow."""
    if not baseline:
        return [
            f"no orphan baseline at {BASELINE_PATH.relative_to(REPO_ROOT)} -- "
            "run with --update-baseline once to establish it"
        ]
    failures = []
    for surface in sorted(set(current) | set(baseline)):
        now = current.get(surface, 0)
        was = baseline.get(surface, 0)
        if now > was:
            failures.append(
                f"orphan ratchet: {surface} grew {was} -> {now}. A registered metric "
                "with no consumer feeds nothing (CLAUDE.md 0A). Wire it to a real "
                "consumer, or retire the registry entry."
            )
    return failures


def write_baseline(
    counts: dict[str, int], missing: dict[str, str], path: Path | None = None
) -> Path:
    target = path or BASELINE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "_comment": (
                    "Ratchet baseline for scripts/check_metric_lineage.py --gate. "
                    "Both sections may shrink, never grow. Regenerate deliberately "
                    "with --update-baseline when a decrease is real."
                ),
                "_orphans_comment": (
                    "Registered metrics with no discoverable code consumer, per "
                    "surface. A metric that names something but feeds nothing is a "
                    "keyword cathedral (CLAUDE.md 0A)."
                ),
                "orphans_by_surface": dict(sorted(counts.items())),
                "_missing_comment": (
                    "Consumers a registry CLAIMS but which do not exist on disk. "
                    "Pre-existing debt recorded so it stays visible; fixing means "
                    "editing consumer_services in orion/bus/channels.yaml or the "
                    "cognition_consumers tuple in orion/inner_state_registry.py, a "
                    "contract change (CLAUDE.md 6) belonging in its own patch."
                ),
                "known_missing_consumers": dict(sorted(missing.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def run_gate(baseline_path: Path | None = None) -> GateResult:
    """Run the gate against this repo.

    No repo_root parameter: build_graph() unconditionally reads
    lineage.REPO_ROOT, so accepting one produced a graph built from the real
    repo checked against a foreign tree -- every consumer missing, every
    metric an orphan. Better to not advertise support that does not exist.
    """
    root = REPO_ROOT
    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys(), repo_root=root)
    baseline = load_baseline(baseline_path)

    result = GateResult()
    result.failures.extend(check_registry_integrity(graph))
    result.failures.extend(
        check_declared_consumers(
            graph, root, baseline.get("known_missing_consumers", {}).keys()
        )
    )

    current = orphan_counts(graph, scan, root)
    result.failures.extend(
        check_orphan_ratchet(current, baseline.get("orphans_by_surface", {}))
    )
    result.notes.append(f"orphans by surface: {dict(sorted(current.items()))}")
    known_missing = baseline.get("known_missing_consumers", {})
    if known_missing:
        result.notes.append(
            f"known-missing declared consumers carried as debt: {len(known_missing)}"
        )
    result.notes.append(f"{len(graph.nodes)} URNs, {scan.files_scanned} files scanned")
    if scan.unparsed:
        result.notes.append(f"unparsed files (reported, not skipped): {len(scan.unparsed)}")
    return result
