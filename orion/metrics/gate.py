"""Phase 4: deterministic gate over the metric semantic layer.

Turns three of CLAUDE.md 0A's metric-quality rules from prose into a failing
check. Every check here is provable from repo state -- no naming heuristics,
no keyword lists.

What it checks
--------------
1. **Registry integrity.** Every registry resolves, and every `upstream` URN
   points at a node that actually exists. A registry that stopped importing,
   or a parent reference that can never resolve, is rot.

2. **Declared-consumer existence.** Every consumer a registry *claims* must
   exist on disk. This is the check that catches
   `services.orion-spark-introspector.app.inner_state:build_inner_state_features`
   still being declared as the cognition consumer for `self_state.v1` after
   that service was deleted (2026-07-28).

3. **Orphan ratchet.** The count of registered metrics with no discoverable
   code consumer may not grow. A metric that names something but feeds nothing
   is the definition of a keyword cathedral (0A), so the population of them is
   allowed to shrink and never to grow.

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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from orion.metrics.consumers import scan_repo
from orion.metrics.lineage import MetricGraph, build_graph

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
        # services.orion-hub.app.foo -> services/orion-hub/app/foo.py
        return repo_root / (module.replace(".", "/") + ".py")
    # bare service name -> services/<name>/
    return repo_root / "services" / module


def check_registry_integrity(graph: MetricGraph) -> list[str]:
    known = set(graph.nodes)
    failures = []
    for node in graph.nodes.values():
        for parent in node.upstream:
            if parent not in known:
                failures.append(
                    f"dangling upstream: {node.urn} -> {parent} (no such node)"
                )
    if not graph.nodes:
        failures.append("metric graph resolved to zero nodes -- registries did not load")
    return failures


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
    return failures


def orphan_counts(graph: MetricGraph, scan) -> dict[str, int]:
    """Registered metrics with no discoverable code consumer, per surface."""
    counts: dict[str, int] = {}
    for token, nodes in graph.scan_tokens().items():
        if scan.consumers_for(token, exclude_paths=graph.registry_sources_for(token)):
            continue
        counts[nodes[0].surface] = counts.get(nodes[0].surface, 0) + 1
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


def run_gate(repo_root: Path | None = None, baseline_path: Path | None = None) -> GateResult:
    root = repo_root or REPO_ROOT
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

    current = orphan_counts(graph, scan)
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
