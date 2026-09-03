"""One channel name, several unrelated meanings.

The glossary was keyed by channel name alone. That is right for the 48 raw
digester channels -- `cpu_pressure` means the same thing on every node -- and
wrong for the `node:substrate.*` domain nodes, which are not physical hosts and
each carry their own reading under the SAME name.

Live on 2026-09-02, `node:substrate.bus_synaptic.prediction_error` sat at 0.021
(the share of the bus running irregular) while `node:substrate.vision
.prediction_error` sat pinned at 1.0 (how overdue the camera was). Rendered
through the single bare `prediction_error` entry, a jittery bus and a blind
camera produce the same sentence.
"""

from __future__ import annotations

from orion.field.channel_glossary import (
    FieldChannelGlossaryEntry,
    FieldChannelNodeEntry,
    load_glossary,
    resolve_channel,
)

BUS = "node:substrate.bus_synaptic"
VISION = "node:substrate.vision"


def test_bus_and_vision_do_not_share_a_meaning() -> None:
    """The check this whole patch exists for."""
    bus = resolve_channel("prediction_error", node=BUS)
    vision = resolve_channel("prediction_error", node=VISION)
    assert bus is not None and vision is not None
    assert bus.meaning != vision.meaning
    assert "bus" in bus.meaning.lower()
    assert "camera" in vision.meaning.lower()


def test_a_node_without_its_own_entry_falls_back_to_the_bare_channel() -> None:
    """cpu_pressure means the same thing everywhere; it must not need 3 entries."""
    entry = resolve_channel("cpu_pressure", node="node:athena")
    assert isinstance(entry, FieldChannelGlossaryEntry)
    assert entry.channel == "cpu_pressure"


def test_no_node_resolves_exactly_as_before() -> None:
    """Existing callers pass no node and must see no change."""
    entry = resolve_channel("prediction_error")
    assert isinstance(entry, FieldChannelGlossaryEntry)
    assert entry.channel == "prediction_error"


def test_an_undescribed_channel_returns_none_not_a_placeholder() -> None:
    """A caller must not be able to render a confident sentence about a channel
    nobody has described. CLAUDE.md 0A: no empty-shell cognition."""
    assert resolve_channel("channel_that_does_not_exist") is None
    assert resolve_channel("channel_that_does_not_exist", node=BUS) is None


def test_the_bus_entry_carries_a_trend_pointer_and_its_policy_channel() -> None:
    """The breadcrumb: a reader gets somewhere to look, not a bare number, and
    the escalation rungs are findable without hardcoding a threshold."""
    bus = resolve_channel("prediction_error", node=BUS)
    assert isinstance(bus, FieldChannelNodeEntry)
    assert "substrate_field_state" in (bus.trend_source or "")
    assert BUS in (bus.trend_source or "")
    assert bus.policy_channel == "bus_synaptic_pressure"


def test_node_entries_are_not_mixed_into_the_bare_channel_list() -> None:
    """Hub's glossary panel iterates `entries` keyed on e.channel. Same-named
    rows added there would render as duplicates, so the node-qualified ones
    live in their own list."""
    glossary = load_glossary()
    bare = [e for e in glossary["entries"] if e.channel == "prediction_error"]
    assert len(bare) == 1, "a node-qualified entry leaked into the bare list"
    assert len(glossary["node_entries"]) >= 2


def test_every_node_entry_names_a_real_substrate_node() -> None:
    """Guards against a typo'd node id that would silently never resolve."""
    for entry in load_glossary()["node_entries"]:
        assert entry.node.startswith("node:substrate."), entry.node
        assert entry.meaning.strip()


def test_multiline_yaml_meanings_are_collapsed_to_one_line() -> None:
    """These render into a chat digest; an embedded newline would break the
    bullet it is rendered into."""
    for entry in load_glossary()["node_entries"]:
        assert "\n" not in entry.meaning
        assert "\n" not in (entry.trend_source or "")
