"""Metric semantic layer: a read-only projection over the registries that
already exist, plus mechanical downstream-consumer discovery.

Deliberately NOT a sixth registry. Every metric this package knows about is
derived from an entry in one of:

- config/field/field_channel_glossary.v1.yaml   (field channels)
- orion/inner_state_registry.py                 (inner-state signals)
- orion/signals/registry.py                     (organ signals)
- orion/bus/channels.yaml                       (bus channels)

If a metric cannot be derived from one of those, it is unregistered, and that
is the finding -- not something to hand-author a URN for.

See docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md.
"""
