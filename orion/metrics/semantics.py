"""Per-metric "why is it in this state" semantics, and the gate over them.

R6 of docs/superpowers/specs/2026-08-13-phase5-liveness-scope.md asked "can a
metric express rest?". The lineage layer recorded what a metric MEANS and
where it GOES, but not what its resting value is, whether it is supposed to
be mostly zero, or what a missing reading means. Nothing could tell "calm"
from "dead" without reading docstrings: a 2026-10-07 sweep called seven
designed signals dead for exactly that reason (spec
docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md, section B).

These fields live on the EXISTING registry entries (the field channel
glossary and the inner-state registry), never in a new registry. lineage.py
projects them onto MetricNode, the metric lock records them, and
scripts/check_definition_drift.py reports any change to them.

Fields
------
value_kind   what shape the number has (see VALUE_KINDS)
rest         the designed resting value, and what 0 means
sparsity     how often a real reading is expected (see SPARSITIES)
absent_means what happens when there is no reading at all
polarity     field channels: DERIVED from
             orion.field.pressure.HIGHER_IS_BETTER_CHANNELS, never declared.
             Inner-state fields may declare it, since no channel set covers
             them.
prompt_sites the `module:callable` sites that put this number into an Orion
             LLM prompt. Juniper's 2026-10-10 decision: the semantic fields
             are CI-enforced only for metrics that reach a prompt. Whether a
             number reaches a prompt cannot be derived from the lineage
             graph (the AST consumer scan sees string reads, not whether the
             reader renders a template), so it is declared here, populated
             from the 2026-10-07 code trace of what reaches Orion's prompts,
             and each site is checked to exist by the same callable check
             the declared-consumer gate uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# level       a magnitude on a continuous scale; percentile/band readings apply
# trigger     fires on a condition; the value is "it fired", not a size
# binary      two values in practice (e.g. 0 / 1, or 0 / one fixed level)
# count       a small integer count, possibly scaled (few distinct values)
# bucket      a few fixed tiers dressed as a number (e.g. 0.9 / 0.6 / 0.3)
# placeholder a hard-coded constant standing in for a measurement not built yet
VALUE_KINDS: frozenset[str] = frozenset(
    {"level", "trigger", "binary", "count", "bucket", "placeholder"}
)

# per_tick         a reading every producer tick; long zero runs mean calm
# event_gated      a reading only when an event happens; silence is normal
# designed_sparse  mostly exact zeros by construction (e.g. negatives clipped)
SPARSITIES: frozenset[str] = frozenset({"per_tick", "event_gated", "designed_sparse"})

POLARITIES: frozenset[str] = frozenset({"higher_is_better", "higher_is_worse"})

# The three fields the prompt gate requires. absent_means and polarity are
# recorded and drift-tracked but not required: a number can reach a prompt
# without ever being absent, and polarity is meaningless for a trigger.
REQUIRED_FOR_PROMPT: tuple[str, ...] = ("value_kind", "rest", "sparsity")

DECLARABLE_FIELDS: tuple[str, ...] = (
    "value_kind",
    "rest",
    "sparsity",
    "absent_means",
    "polarity",
)


@dataclass(frozen=True)
class MetricSemantics:
    """Semantics for one inner-state metric (signal or scalar field).

    Field-channel glossary entries carry the same keys as YAML under a
    `semantics:` mapping; this dataclass is the Python-registry form.
    """

    value_kind: str | None = None
    rest: str | None = None
    sparsity: str | None = None
    absent_means: str | None = None
    polarity: str | None = None
    prompt_sites: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "MetricSemantics":
        raw = dict(raw or {})
        unknown = set(raw) - set(DECLARABLE_FIELDS) - {"prompt_sites"}
        if unknown:
            raise ValueError(f"unknown semantics keys: {sorted(unknown)}")
        return cls(
            value_kind=raw.get("value_kind"),
            rest=raw.get("rest"),
            sparsity=raw.get("sparsity"),
            absent_means=raw.get("absent_means"),
            polarity=raw.get("polarity"),
            prompt_sites=tuple(raw.get("prompt_sites") or ()),
        )


def derived_channel_polarity(channel: str) -> str:
    """Polarity of a field channel, from the one set that already decides it.

    orion.field.pressure merges HIGHER_IS_BETTER_CHANNELS with min() and every
    other channel with max(); that merge IS the repo's polarity decision, so
    this reads it rather than keeping a second list that could drift.
    """
    from orion.field.pressure import HIGHER_IS_BETTER_CHANNELS

    return "higher_is_better" if channel in HIGHER_IS_BETTER_CHANNELS else "higher_is_worse"


def check_node_semantics(nodes: Iterable[Any]) -> list[str]:
    """Gate over resolved MetricNodes. Returns failure lines (empty = pass).

    1. Any declared value_kind / sparsity / polarity must be a known value.
    2. A node with prompt_sites must carry every REQUIRED_FOR_PROMPT field.
    3. Every prompt site must be a checkable `module:callable` claim (the
       existence half is checked by orion.metrics.gate, which already knows
       how to resolve a callable on disk).
    """
    failures: list[str] = []
    for node in nodes:
        urn = node.urn
        if node.value_kind is not None and node.value_kind not in VALUE_KINDS:
            failures.append(
                f"{urn}: value_kind {node.value_kind!r} not in {sorted(VALUE_KINDS)}"
            )
        if node.sparsity is not None and node.sparsity not in SPARSITIES:
            failures.append(f"{urn}: sparsity {node.sparsity!r} not in {sorted(SPARSITIES)}")
        if node.polarity is not None and node.polarity not in POLARITIES:
            failures.append(f"{urn}: polarity {node.polarity!r} not in {sorted(POLARITIES)}")
        if not node.prompt_sites:
            continue
        missing = [f for f in REQUIRED_FOR_PROMPT if not getattr(node, f, None)]
        if missing:
            failures.append(
                f"{urn} reaches an Orion prompt ({', '.join(node.prompt_sites)}) "
                f"but has no {'/'.join(missing)} -- add them to its registry entry "
                f"in {node.registry_source}"
            )
        for site in node.prompt_sites:
            module, _, func = site.partition(":")
            if not func or not func.strip().isidentifier() or "." not in module:
                failures.append(
                    f"{urn}: prompt site {site!r} is not a checkable "
                    "'dotted.module:callable' claim"
                )
    return failures
