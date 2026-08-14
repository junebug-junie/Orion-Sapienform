"""Definition-change detection over the metric semantic layer.

R4 of docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md: "tell me when
someone starts messing in there", without a liveness verdict attached.

WHAT A "DEFINITION" IS HERE
---------------------------
Not the YAML text. A metric's definition is the resolved projection of it --
what `orion.metrics.lineage.build_graph()` already produces from the four
registries (field channel glossary, inner-state registry, organ registry, bus
channels.yaml). Diffing at that layer rather than at the file layer is what
makes the alert readable:

- reordering a YAML list, reflowing a comment, or renaming a variable in
  `orion/signals/registry.py` produces NO delta, because none of them change
  a resolved definition;
- adding one line to a 99KB `channels.yaml` produces exactly one delta line
  naming the channel, its schema, its producers and its consumers.

The file diff answers "what bytes moved". This answers "what did an agent
change about what a number MEANS or where it GOES".

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No liveness verdict, no value sampling, no Postgres. A definition change is
neither good nor bad on its own -- it is a thing Juniper asked to be told
about. Severity here ranks how easy the change is to miss by hand, not how
wrong it is.

FIELD CLASSES
-------------
Identity fields (surface/producer/name/field) form the URN and so cannot
"change" -- a changed identity is a removal plus an addition, which is exactly
what the rename pass below reassembles.

    semantics   meaning, schema_id
    routing     declared_consumers, feeds_dimensions, upstream,
                upstream_organs, all_producers
    annotation  notes, registry_source

`registry_source` sits under annotation on purpose: moving a declaration
between files is worth reporting but is not the same event as changing what it
declares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from orion.metrics.lineage import MetricGraph, MetricNode

# Ordered so the emitted lock is stable and its diffs read top-to-bottom in the
# same shape every time.
SEMANTIC_FIELDS: tuple[str, ...] = ("meaning", "schema_id")
ROUTING_FIELDS: tuple[str, ...] = (
    "all_producers",
    "declared_consumers",
    "feeds_dimensions",
    "upstream",
    "upstream_organs",
)
ANNOTATION_FIELDS: tuple[str, ...] = ("notes", "registry_source")

DEFINITION_FIELDS: tuple[str, ...] = SEMANTIC_FIELDS + ROUTING_FIELDS + ANNOTATION_FIELDS

_FIELD_CLASS: dict[str, str] = {
    **{f: "semantics" for f in SEMANTIC_FIELDS},
    **{f: "routing" for f in ROUTING_FIELDS},
    **{f: "annotation" for f in ANNOTATION_FIELDS},
}

# How easy is this change to miss by hand? Not how bad it is.
#
# `removed` is HIGH because of the case that motivated R4: `execution_load`,
# `bus_health` and `transport_pressure` were renamed on 2026-07-24 and were
# still sitting in live node vectors with plausible-looking frozen values three
# weeks later (services/orion-field-digester/app/tensor/channels.py::
# RETIRED_NODE_CHANNELS). Nothing announced the removal.
#
# `added` is MEDIUM: a new metric is loud by nature -- it arrives with code
# that reads it -- and it has its own gate (CLAUDE.md 0A, the metric quality
# gate). It still gets reported, because "an agent added a channel" is half of
# what Juniper asked to be told.
SEVERITY: dict[str, str] = {
    "removed": "high",
    "renamed": "high",
    "semantics_changed": "high",
    "routing_changed": "high",
    "added": "medium",
    "annotation_changed": "medium",
}

HIGH_SEVERITY_KINDS: frozenset[str] = frozenset(
    k for k, v in SEVERITY.items() if v == "high"
)


def fingerprint(node: MetricNode) -> dict[str, Any]:
    """The definition half of a node, with empties omitted.

    Empties are dropped rather than serialised as null/[] so that the lock is
    ~40% smaller and, more importantly, so that a field going from absent to
    absent-but-differently-typed cannot show up as a phantom delta.
    """
    out: dict[str, Any] = {}
    for name in DEFINITION_FIELDS:
        value = getattr(node, name, None)
        if isinstance(value, tuple):
            if value:
                out[name] = sorted(value)
        elif value not in (None, ""):
            out[name] = value
    return out


def build_lock(graph: MetricGraph) -> dict[str, dict[str, Any]]:
    """urn -> definition fingerprint, sorted by URN."""
    return {urn: fingerprint(graph.nodes[urn]) for urn in sorted(graph.nodes)}


@dataclass(frozen=True)
class DefinitionChange:
    kind: str  # a key of SEVERITY
    urn: str
    surface: str
    # Populated for renamed only.
    previous_urn: str | None = None
    # field -> (before, after). Populated for *_changed only.
    fields: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        return SEVERITY[self.kind]

    def describe(self) -> str:
        if self.kind == "renamed":
            return f"renamed {self.previous_urn} -> {self.urn}"
        if self.fields:
            bits = ", ".join(
                f"{name}: {_short(before)} -> {_short(after)}"
                for name, (before, after) in sorted(self.fields.items())
            )
            return f"{self.kind} {self.urn} ({bits})"
        return f"{self.kind} {self.urn}"


def _short(value: Any, limit: int = 60) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class DefinitionDiff:
    changes: list[DefinitionChange] = field(default_factory=list)

    @property
    def high(self) -> list[DefinitionChange]:
        return [c for c in self.changes if c.severity == "high"]

    def by_kind(self) -> dict[str, list[DefinitionChange]]:
        out: dict[str, list[DefinitionChange]] = {}
        for change in self.changes:
            out.setdefault(change.kind, []).append(change)
        return out

    def __bool__(self) -> bool:
        return bool(self.changes)


def _surface_of(urn: str) -> str:
    # metric://<surface>/... -- parsed rather than carried, because the lock
    # stores only definition fields and surface is an identity field.
    try:
        return urn.split("//", 1)[1].split("/", 1)[0]
    except IndexError:
        return "unknown"


def _pair_renames(
    removed: dict[str, dict[str, Any]],
    added: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Match removals to additions whose definitions are byte-identical.

    A rename is a removal plus an addition. This pass exists ONLY to merge
    those two lines into one clearer line -- it never changes what is flagged,
    because both halves are already reported at high and medium severity
    respectively. So a missed pairing (a rename that also edited the prose, the
    common case for a real rename) degrades to "removed + added", which is
    louder, not quieter. That is why an aggressive fuzzy matcher would be the
    wrong trade here: it could only ever make the output *less* alarming, and
    the failure it guards against is under-alarming.

    Ambiguous groups (two removals and two additions sharing one definition)
    are left unpaired -- same `winner_is_unique` discipline as
    orion/field/pressure.py's provenance.

    HOW OFTEN DEFINITIONS COLLIDE, measured rather than assumed: on the live
    595-node graph there are 342 distinct fingerprints, so **296 nodes (49.7%)
    share a fingerprint with at least one other node**. Almost all of it is
    `organ_signal`, where the registry distinguishes entries only by identity
    (`organ/kind#dimension`) and the definition content collapses to
    `organ=graph_cognition class=endogenous` for 48 nodes at once.

    That number sounds fatal and is not, for two reasons. The `len(...) == 1`
    guard applies to the removals and additions *in this diff*, not to the
    whole graph, so a 48-node group only goes ambiguous if 2+ of its members
    are removed and 2+ added in the same change. And when a mispairing does
    happen, the emitted alert names both URNs at `high` either way -- `removed`
    and `renamed` carry the same severity by design. The label degrades; the
    warning does not.
    """
    by_def: dict[str, dict[str, list[str]]] = {}
    for urn, fp in removed.items():
        key = _def_key(_surface_of(urn), fp)
        by_def.setdefault(key, {"removed": [], "added": []})["removed"].append(urn)
    for urn, fp in added.items():
        key = _def_key(_surface_of(urn), fp)
        by_def.setdefault(key, {"removed": [], "added": []})["added"].append(urn)

    pairs: list[tuple[str, str]] = []
    for group in by_def.values():
        if len(group["removed"]) == 1 and len(group["added"]) == 1:
            pairs.append((group["removed"][0], group["added"][0]))
    return pairs


def _def_key(surface: str, fp: dict[str, Any]) -> str:
    import json

    return surface + "|" + json.dumps(fp, sort_keys=True)


def diff_locks(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> DefinitionDiff:
    """Classified deltas between two locks. Pure; no I/O."""
    before_urns = set(before)
    after_urns = set(after)

    removed = {u: before[u] for u in before_urns - after_urns}
    added = {u: after[u] for u in after_urns - before_urns}

    changes: list[DefinitionChange] = []

    paired_out: set[str] = set()
    paired_in: set[str] = set()
    for old_urn, new_urn in _pair_renames(removed, added):
        paired_out.add(old_urn)
        paired_in.add(new_urn)
        changes.append(
            DefinitionChange(
                kind="renamed",
                urn=new_urn,
                surface=_surface_of(new_urn),
                previous_urn=old_urn,
            )
        )

    for urn in sorted(removed):
        if urn not in paired_out:
            changes.append(
                DefinitionChange(kind="removed", urn=urn, surface=_surface_of(urn))
            )
    for urn in sorted(added):
        if urn not in paired_in:
            changes.append(
                DefinitionChange(kind="added", urn=urn, surface=_surface_of(urn))
            )

    for urn in sorted(before_urns & after_urns):
        old_fp, new_fp = before[urn], after[urn]
        if old_fp == new_fp:
            continue
        # One URN can change in more than one class at once; each class is
        # reported separately so a routing edit is never hidden behind a
        # same-commit prose edit.
        buckets: dict[str, dict[str, tuple[Any, Any]]] = {}
        for name in sorted(set(old_fp) | set(new_fp)):
            if old_fp.get(name) == new_fp.get(name):
                continue
            klass = _FIELD_CLASS.get(name, "annotation")
            buckets.setdefault(klass, {})[name] = (old_fp.get(name), new_fp.get(name))
        for klass, fields in sorted(buckets.items()):
            changes.append(
                DefinitionChange(
                    kind=f"{klass}_changed",
                    urn=urn,
                    surface=_surface_of(urn),
                    fields=fields,
                )
            )

    changes.sort(key=lambda c: (SEVERITY[c.kind] != "high", c.kind, c.urn))
    return DefinitionDiff(changes=changes)


def format_report(diff: DefinitionDiff) -> Iterable[str]:
    """Human-readable lines. Shared by the CLI's report and --update output so
    the two can never describe the same delta differently."""
    if not diff:
        yield "no definition changes"
        return
    for kind, group in sorted(
        diff.by_kind().items(), key=lambda kv: (SEVERITY[kv[0]] != "high", kv[0])
    ):
        yield f"  [{SEVERITY[kind].upper()}] {kind} ({len(group)})"
        for change in group:
            yield f"      {change.describe()}"
