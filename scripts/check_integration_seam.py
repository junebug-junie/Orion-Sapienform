#!/usr/bin/env python3
"""The integration gate: every new producer of Orion experience is registered, or exempt with a reason.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A.6. CLAUDE.md
section "Integration gate" is the rule; this is its check. The registry is
``config/integration/organs.yaml``. Structural only: YAML, AST and git, no live data.

What it detects (the spec's list, nothing more):

1. every family in ``config/physiology/families.yaml`` belongs to exactly one organ, or is in
   ``exempt_families`` with a reason;
2. every ``DurableWorkflowV1`` value (``orion/schemas/durable_run.py``) belongs to exactly one organ, or is in
   ``exempt_workflows`` with a reason;
3. every module in ``orion/prime/intention_sources/`` is listed under one organ's ``intention_sources`` (the
   organ is not required to emit yet: today's only adapter belongs to ``intention``, exempt from emitting);
4. every ``TemporalSourceKind`` value belongs to exactly one organ, and every organ with no
   ``temporal_sources`` has an ``exempt_emit`` reason;
5. every organ either names a ``reads_now`` module that exists, READS the now (imports
   ``orion.temporal_self.now`` or the shared client ``orion.integration.now_client``, or uses the
   ``/temporal-self/now`` route in code) and RECORDS cites (references ``record_use`` / ``arecord_use`` or the
   ``/temporal-self/now/use`` route in code), or has an ``exempt_read`` reason. Docstrings
   and comments never count (AST, ``ast.Expr`` strings skipped). An organ may not hold both a claim and its
   exemption (``temporal_sources`` + ``exempt_emit``, ``reads_now`` + ``exempt_read``);
6. every channel ADDED to ``orion/bus/channels.yaml`` since the merge base with ``origin/main`` is named by
   one organ's ``channels`` or by ``exempt_channels`` with a reason (existing channels are not re-litigated).

Also: a registry entry naming a producer (family, workflow, adapter, kind, channel) that does not exist fails.

Not seen (stated plainly): a producer that writes a new table with no family, workflow, adapter or channel
(``scripts/report_integration_coverage.py`` shows its kind as unregistered once it lands in
``temporal_self_event``), and a new Prime concern source (CLAUDE.md names it, but concern sources are
``_read_*`` methods in services/orion-mind/app/prime_loop.py, not modules a check can enumerate).

Run from the repo root:

    python3 scripts/check_integration_seam.py
    python3 scripts/check_integration_seam.py --base origin/main   # the ref new channels are diffed against
    python3 scripts/check_integration_seam.py --require-base       # CI: no merge base is a failure, not a skip
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY = "config/integration/organs.yaml"
FAMILIES = "config/physiology/families.yaml"
CHANNELS = "orion/bus/channels.yaml"
INTENTION_SOURCES = "orion/prime/intention_sources"
NOW_MODULE = "orion.temporal_self.now"
NOW_CLIENT = "orion.integration.now_client"  # the shared HTTP client (integration patch 3)
RECORD_NAMES = {"record_use", "arecord_use"}
NOW_ROUTE = "/temporal-self/now"
USE_ROUTE = "/temporal-self/now/use"
FIX = f"Fix: register it under an organ in {REGISTRY}, or add an exemption with a reason there."


@dataclass
class Inputs:
    """Everything the check reads, gathered once, so tests can plant any mutation."""

    registry: Mapping[str, Any]
    families: list[str]
    workflows: list[str]
    intention_sources: list[str]
    temporal_kinds: list[str]
    channels_now: list[str]
    channels_base: Optional[list[str]]  # None: no merge base (the channel rule reports it cannot run)
    repo_root: Path = REPO_ROOT
    base_ref: str = "origin/main"
    base_registry: Optional[Mapping[str, Any]] = None  # for listing NEW exemptions (reviewer read)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- gather (I/O)


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _channel_names(doc: Any) -> list[str]:
    chans = (doc or {}).get("channels") or []
    return [str(c["name"]) for c in chans if isinstance(c, dict) and c.get("name")]


def gather(root: Path = REPO_ROOT, base_ref: str = "origin/main") -> Inputs:
    from typing import get_args

    from orion.schemas.durable_run import DurableWorkflowV1
    from orion.schemas.temporal_self import TemporalSourceKind

    registry = yaml.safe_load((root / REGISTRY).read_text(encoding="utf-8")) or {}
    families = sorted(((yaml.safe_load((root / FAMILIES).read_text(encoding="utf-8")) or {}).get("families") or {}))
    sources = sorted(p.stem for p in (root / INTENTION_SOURCES).glob("*.py") if p.stem != "__init__")
    channels_now = _channel_names(yaml.safe_load((root / CHANNELS).read_text(encoding="utf-8")))
    notes: list[str] = []
    channels_base: Optional[list[str]] = None
    base_registry: Optional[Mapping[str, Any]] = None
    merge_base = (_git(root, "merge-base", "HEAD", base_ref) or "").strip()
    if merge_base:
        old = _git(root, "show", f"{merge_base}:{CHANNELS}")
        channels_base = _channel_names(yaml.safe_load(old)) if old is not None else []
        old_reg = _git(root, "show", f"{merge_base}:{REGISTRY}")
        base_registry = yaml.safe_load(old_reg) if old_reg else {}
    else:
        notes.append(f"no merge base with {base_ref}: the new-channel rule (6) did not run")
    return Inputs(registry=registry, families=families, workflows=sorted(get_args(DurableWorkflowV1)),
                  intention_sources=sources, temporal_kinds=sorted(get_args(TemporalSourceKind)),
                  channels_now=channels_now, channels_base=channels_base, repo_root=root, base_ref=base_ref,
                  base_registry=base_registry, notes=notes)


# --------------------------------------------------------------------------- check (pure but for rule 5's file reads)


def _reason(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _owners(organs: Mapping[str, Any], key: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, spec in organs.items():
        for item in (spec or {}).get(key) or []:
            out.setdefault(str(item), []).append(str(name))
    return out


def _code_strings(tree: ast.AST) -> list[str]:
    """String constants used as values in code: docstrings and other bare-expression strings are skipped."""
    skip = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


def _reads_the_now(path: Path) -> tuple[bool, bool]:
    """(reads, records): reads = imports orion.temporal_self.now or orion.integration.now_client, or uses the
    /temporal-self/now route in code; records = references ``record_use`` / ``arecord_use`` or uses the
    /temporal-self/now/use route in code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False, False
    reads = records = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name in (NOW_MODULE, NOW_CLIENT) or a.name.startswith(NOW_MODULE + ".")
                                                for a in node.names):
            reads = True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if (mod in (NOW_MODULE, NOW_CLIENT)
                    or (mod == "orion.temporal_self" and any(a.name == "now" for a in node.names))
                    or (mod == "orion.integration" and any(a.name == "now_client" for a in node.names))):
                reads = True
                records = records or any(a.name in RECORD_NAMES for a in node.names)
        elif isinstance(node, ast.Name) and node.id in RECORD_NAMES:
            records = True
        elif isinstance(node, ast.Attribute) and node.attr in RECORD_NAMES:
            records = True
    for text in _code_strings(tree):
        if USE_ROUTE in text:
            records = True
        if NOW_ROUTE in text.replace(USE_ROUTE, ""):
            reads = True
    return reads, records


def _covered(kind: str, items: Iterable[str], owners: Mapping[str, list[str]], exempt: Mapping[str, Any],
             where: str) -> list[str]:
    errs = []
    for item in items:
        own = owners.get(item, [])
        if len(own) > 1:
            errs.append(f"{kind} {item!r} is claimed by {len(own)} organs ({', '.join(own)}) in {REGISTRY}; one organ each.")
        elif not own and not _reason(exempt.get(item)):
            errs.append(f"{kind} {item!r} ({where}) belongs to no organ and has no exemption. {FIX}")
        elif own and item in exempt:
            errs.append(f"{kind} {item!r} is both claimed by {own[0]} and exempt in {REGISTRY}; pick one.")
    return errs


def _unknown(kind: str, owners: Mapping[str, list[str]], exempt: Mapping[str, Any], real: Iterable[str]) -> list[str]:
    real_set = set(real)
    stale = sorted((set(owners) | set(exempt)) - real_set)
    return [f"{kind} {s!r} is named in {REGISTRY} but does not exist; remove it." for s in stale]


def check(inp: Inputs) -> list[str]:
    reg = inp.registry or {}
    organs: Mapping[str, Any] = reg.get("organs") or {}
    if not organs:
        return [f"{REGISTRY}: no organs"]
    errors: list[str] = []
    for key in ("exempt_families", "exempt_workflows", "exempt_channels"):
        for item, why in (reg.get(key) or {}).items():
            if not _reason(why):
                errors.append(f"{key}.{item} in {REGISTRY} has no reason; an exemption always says why.")

    # 1. families
    fam_owners = _owners(organs, "families")
    errors += _covered("family", inp.families, fam_owners, reg.get("exempt_families") or {}, FAMILIES)
    errors += _unknown("family", fam_owners, reg.get("exempt_families") or {}, inp.families)
    # 2. workflows
    wf_owners = _owners(organs, "workflows")
    errors += _covered("workflow", inp.workflows, wf_owners, reg.get("exempt_workflows") or {},
                       "DurableWorkflowV1 in orion/schemas/durable_run.py")
    errors += _unknown("workflow", wf_owners, reg.get("exempt_workflows") or {}, inp.workflows)
    # 3. intention source adapters (no exemption: an adapter is a source of Prime's intentions by definition)
    src_owners = _owners(organs, "intention_sources")
    errors += _covered("intention source adapter", inp.intention_sources, src_owners, {}, INTENTION_SOURCES + "/")
    errors += _unknown("intention source adapter", src_owners, {}, inp.intention_sources)
    # 4. temporal source kinds, and an emit exemption for organs that emit nothing
    kind_owners = _owners(organs, "temporal_sources")
    errors += _covered("TemporalSourceKind", inp.temporal_kinds, kind_owners, {}, "orion/schemas/temporal_self.py")
    errors += _unknown("TemporalSourceKind", kind_owners, {}, inp.temporal_kinds)
    for name, spec in organs.items():
        spec = spec or {}
        if spec.get("temporal_sources") and spec.get("exempt_emit"):
            errors.append(f"organ {name!r} both emits and is exempt from emitting; drop exempt_emit.")
        if spec.get("reads_now") and spec.get("exempt_read"):
            errors.append(f"organ {name!r} both names reads_now and is exempt from reading; drop exempt_read.")
        if not spec.get("temporal_sources") and not _reason(spec.get("exempt_emit")):
            errors.append(f"organ {name!r} emits no TemporalSourceKind and has no exempt_emit reason. Emit through "
                          f"the experience contract (a TemporalSourceKind + a reader in temporal_self_sources.py), "
                          f"or write the exemption in {REGISTRY}.")
        # 5. read-back
        reads = spec.get("reads_now")
        if reads:
            path = inp.repo_root / str(reads)
            if not path.is_file():
                errors.append(f"organ {name!r}: reads_now {reads} does not exist.")
            else:
                did_read, did_record = _reads_the_now(path)
                if not did_read:
                    errors.append(f"organ {name!r}: reads_now {reads} neither imports {NOW_MODULE} nor uses "
                                  f"{NOW_ROUTE} in code; read the now there, or set reads_now: null with an "
                                  f"exempt_read reason.")
                elif not did_record:
                    errors.append(f"organ {name!r}: reads_now {reads} reads the now but records no cites (no "
                                  f"record_use, no {USE_ROUTE} in code); CLAUDE.md asks for both.")
        elif not _reason(spec.get("exempt_read")):
            errors.append(f"organ {name!r} has no reads_now module and no exempt_read reason. Read GET {NOW_ROUTE} "
                          f"at the organ's next run and record cites, or write the exemption in {REGISTRY}.")
    # 6. channels added since the merge base
    if inp.channels_base is not None:
        added = sorted(set(inp.channels_now) - set(inp.channels_base))
        ch_owners = _owners(organs, "channels")
        errors += _covered("channel", added, ch_owners, reg.get("exempt_channels") or {},
                           f"added to {CHANNELS} since the merge base with {inp.base_ref}")
    # Stale channel entries are checked against every current channel, base or not.
    errors += _unknown("channel", _owners(organs, "channels"), reg.get("exempt_channels") or {}, inp.channels_now)
    return errors


def new_exemptions(inp: Inputs) -> list[str]:
    """Exemptions present now and absent at the merge base: what a reviewer must read."""
    def flat(reg: Mapping[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in ("exempt_families", "exempt_workflows", "exempt_channels"):
            for item, why in ((reg or {}).get(key) or {}).items():
                out[f"{key}.{item}"] = str(why)
        for name, spec in ((reg or {}).get("organs") or {}).items():
            for key in ("exempt_emit", "exempt_read"):
                if (spec or {}).get(key):
                    out[f"organs.{name}.{key}"] = str(spec[key])
        return out

    if inp.base_registry is None:
        return []
    old, now = flat(inp.base_registry), flat(inp.registry)
    return [f"{k}: {v}" for k, v in sorted(now.items()) if old.get(k) != v]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main", help="ref new channels and new exemptions are diffed against")
    ap.add_argument("--require-base", action="store_true",
                    help="fail (not skip rule 6) when there is no merge base with --base; CI sets it")
    args = ap.parse_args(argv)
    inp = gather(REPO_ROOT, args.base)
    errors = check(inp)
    if args.require_base and inp.channels_base is None:
        errors.append(f"no merge base with {args.base}: the new-channel rule (6) cannot run. Fetch it "
                      f"(git fetch origin main; CI checks out with fetch-depth: 0).")
    for n in inp.notes:
        print(f"check_integration_seam: note: {n}")
    fresh = new_exemptions(inp)
    if fresh:
        print(f"check_integration_seam: {len(fresh)} new or changed exemption(s) since the merge base -- a reviewer reads each:")
        for line in fresh:
            print(f"  {line}")
    if errors:
        print(f"check_integration_seam: FAIL ({len(errors)} problem(s))", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("CLAUDE.md 'Integration gate': fix the registry or write the exemption; do not argue with it in prose.",
              file=sys.stderr)
        return 1
    organs = (inp.registry.get("organs") or {})
    print(f"check_integration_seam: OK ({len(organs)} organs; {len(inp.families)} families, {len(inp.workflows)} "
          f"workflows, {len(inp.intention_sources)} intention adapters, {len(inp.temporal_kinds)} source kinds covered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
