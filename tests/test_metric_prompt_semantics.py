"""Gate tests for the prompt-semantics check (2026-10-10, closes R6).

A metric that reaches an Orion LLM prompt must say what shape its number has
(value_kind), what its resting value is (rest) and how often a real reading
is expected (sparsity). Each test builds the breakage and asserts the gate
goes red, not only the green path.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from orion.metrics.definitions import build_lock, diff_locks
from orion.metrics.gate import check_prompt_semantics
from orion.metrics.lineage import MetricGraph, MetricNode, build_graph, resolve_field_channels
from orion.metrics.semantics import (
    PROMPT_INVENTORY_URNS,
    MetricSemantics,
    check_node_semantics,
    check_prompt_inventory,
    derived_channel_polarity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SITE = "orion.curiosity.queue_contention_disclosure:format_queue_contention_progress"


def _node(**kw) -> MetricNode:
    base = dict(
        urn="metric://inner_state/svc/sig#x",
        surface="inner_state",
        producer_service="svc",
        name="x",
        registry_source="orion/fake_registry.py",
    )
    base.update(kw)
    return MetricNode(**base)


def _complete(**kw) -> MetricNode:
    return _node(
        value_kind="level",
        rest="0.0 = calm",
        sparsity="per_tick",
        prompt_sites=(REAL_SITE,),
        **kw,
    )


# ------------------------------------------------------------ required fields


def test_complete_prompt_metric_passes():
    assert check_node_semantics([_complete()]) == []


@pytest.mark.parametrize("missing", ["value_kind", "rest", "sparsity"])
def test_prompt_metric_missing_a_required_field_fails(missing):
    node = dataclasses.replace(_complete(), **{missing: None})
    failures = check_node_semantics([node])
    assert len(failures) == 1
    assert missing in failures[0]
    assert "reaches an Orion prompt" in failures[0]


def test_empty_string_counts_as_missing():
    failures = check_node_semantics([dataclasses.replace(_complete(), rest="")])
    assert failures and "rest" in failures[0]


def test_metric_not_reaching_a_prompt_may_omit_semantics():
    # Juniper 2026-10-10: enforced only for prompt-reaching metrics.
    assert check_node_semantics([_node()]) == []


@pytest.mark.parametrize("bad", ["   ", 0.0])
def test_whitespace_or_non_string_rest_counts_as_missing(bad):
    failures = check_node_semantics([dataclasses.replace(_complete(), rest=bad)])
    assert failures and "rest" in failures[0]


def test_absent_means_and_polarity_are_not_required():
    assert check_node_semantics([_complete()]) == []  # neither set


# ------------------------------------------------------------ vocabularies


@pytest.mark.parametrize(
    "field,bad",
    [("value_kind", "magnitude"), ("sparsity", "sometimes"), ("polarity", "up")],
)
def test_unknown_vocabulary_value_fails_even_off_prompt(field, bad):
    failures = check_node_semantics([_node(**{field: bad})])
    assert len(failures) == 1 and field in failures[0]


def test_unchecked_prompt_site_form_fails():
    failures = check_node_semantics([dataclasses.replace(_complete(), prompt_sites=("stance_react.j2",))])
    assert any("not a checkable" in f for f in failures)


# ------------------------------------------------------------ prompt site existence


def test_prompt_site_with_missing_callable_fails():
    node = dataclasses.replace(
        _complete(),
        prompt_sites=("orion.curiosity.queue_contention_disclosure:no_such_function",),
    )
    failures = check_prompt_semantics(MetricGraph(nodes={node.urn: node}), REPO_ROOT)
    assert any("no_such_function" in f for f in failures)


def test_prompt_site_with_missing_module_fails():
    node = dataclasses.replace(_complete(), prompt_sites=("orion.no_such_module:render",))
    failures = check_prompt_semantics(MetricGraph(nodes={node.urn: node}), REPO_ROOT)
    assert any("does not exist" in f for f in failures)


# ------------------------------------------------------------ inventory pin


def test_dropping_a_pinned_prompt_marker_fails():
    urn = sorted(PROMPT_INVENTORY_URNS)[0]
    node = _node(urn=urn)  # same URN, prompt_sites removed
    failures = check_prompt_inventory([node], frozenset({urn}))
    assert failures and "declares no prompt_sites" in failures[0]


def test_prompt_marker_on_unpinned_urn_fails():
    node = _complete(urn="metric://inner_state/svc/sig#unpinned")
    failures = check_prompt_inventory([node], frozenset())
    assert failures and "not in PROMPT_INVENTORY_URNS" in failures[0]


def test_renaming_away_a_pinned_urn_fails():
    failures = check_prompt_inventory([], frozenset({"metric://x/y/z"}))
    assert failures and "no longer resolves" in failures[0]


def test_every_pinned_urn_resolves_and_declares_a_site():
    graph = build_graph()
    assert check_prompt_inventory(graph.nodes.values()) == []


# ------------------------------------------------------------ the real repo


def test_real_repo_prompt_semantics_gate_is_green():
    graph = build_graph()
    assert check_prompt_semantics(graph, REPO_ROOT) == []
    assert sum(1 for n in graph.nodes.values() if n.reaches_prompt) >= len(PROMPT_INVENTORY_URNS)


def test_cli_prompt_semantics_gate_passes():
    proc = subprocess.run(
        [sys.executable, "scripts/check_metric_lineage.py", "--prompt-semantics"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "prompt semantics gate: PASS" in proc.stdout


@pytest.mark.parametrize(
    "node_id,kind",
    [
        ("node:substrate.harness_closure", "placeholder"),
        ("node:substrate.cabinet", "trigger"),
        ("node:substrate.route", "count"),
        ("node:substrate.execution", "level"),
    ],
)
def test_node_qualified_prediction_error_entries_carry_semantics(node_id, kind):
    nodes = {n.name: n for n in resolve_field_channels()}
    node = nodes[f"{node_id}.prediction_error"]
    assert node.value_kind == kind
    assert node.rest and node.sparsity
    assert node.producer_service == "orion-substrate-runtime"


def test_vision_organ_has_no_prediction_error_entry():
    names = {n.name for n in resolve_field_channels()}
    assert "node:substrate.vision_organ.prediction_error" not in names


# ------------------------------------------------------------ polarity is derived


def test_field_channel_polarity_derives_from_pressure_merge_sets():
    from orion.field.pressure import HIGHER_IS_BETTER_CHANNELS, PRESSURE_CHANNELS

    for channel in HIGHER_IS_BETTER_CHANNELS:
        assert derived_channel_polarity(channel) == "higher_is_better"
    for channel in PRESSURE_CHANNELS:
        assert derived_channel_polarity(channel) == "higher_is_worse"
    nodes = {n.name: n for n in resolve_field_channels()}
    assert nodes["stability"].polarity == "higher_is_better"
    assert nodes["cpu_pressure"].polarity == "higher_is_worse"


@pytest.mark.parametrize(
    "channel", ["expected_offline_suppression", "context_gathering_ratio", "cabinet_climate_activity"]
)
def test_channel_in_neither_merge_set_gets_no_polarity(channel):
    # Review finding: max() is merely the default merge; it is not a claim
    # that more is worse. expected_offline_suppression is a suppression flag.
    nodes = {n.name: n for n in resolve_field_channels()}
    assert derived_channel_polarity(channel) is None
    assert nodes[channel].polarity is None


def test_trigger_gets_no_polarity():
    assert derived_channel_polarity("prediction_error", "trigger") is None
    nodes = {n.name: n for n in resolve_field_channels()}
    assert nodes["node:substrate.cabinet.prediction_error"].polarity is None
    assert nodes["node:substrate.execution.prediction_error"].polarity == "higher_is_worse"


def test_glossary_entry_declaring_polarity_is_rejected(tmp_path):
    bad = tmp_path / "glossary.yaml"
    bad.write_text(
        "channels:\n"
        "  - channel: cpu_pressure\n"
        "    level: [node]\n"
        "    category: physical_substrate\n"
        "    meaning: x\n"
        "    semantics:\n"
        "      polarity: higher_is_better\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="derived"):
        resolve_field_channels(bad)


def test_unknown_semantics_key_is_rejected():
    with pytest.raises(ValueError, match="unknown semantics keys"):
        MetricSemantics.from_mapping({"resting": "0"})


# ------------------------------------------------------------ inner-state wiring


def test_inner_state_semantics_for_a_non_field_key_is_rejected(monkeypatch):
    import orion.inner_state_registry as reg
    from orion.metrics import lineage

    sig = reg.get("field_state.v1")
    broken = dataclasses.replace(
        sig, semantics=(("not_a_field", MetricSemantics(value_kind="level")),)
    )
    monkeypatch.setattr(reg, "REGISTRY", (broken,))
    with pytest.raises(ValueError, match="not float fields"):
        lineage.resolve_inner_state()


def test_inner_state_field_semantics_reach_the_node():
    graph = build_graph()
    node = graph.nodes[
        "metric://inner_state/orion-substrate-runtime/attention_broadcast_projection.v1"
        "#coalition_stability_score"
    ]
    assert node.value_kind == "bucket"
    assert node.prompt_sites == ("services.orion-thought.app.reverie:build_reverie_context",)


# ------------------------------------------------------------ lock + drift


def test_semantic_field_change_is_a_high_semantics_change():
    before = _complete()
    after = dataclasses.replace(before, rest="0.087 = all-NO floor")
    diff = diff_locks(
        build_lock(MetricGraph(nodes={before.urn: before})),
        build_lock(MetricGraph(nodes={after.urn: after})),
    )
    assert [c.kind for c in diff.changes] == ["semantics_changed"]
    assert diff.changes[0].severity == "high"
    assert "rest" in diff.changes[0].fields


def test_prompt_site_change_is_a_routing_change():
    before = _complete()
    after = dataclasses.replace(before, prompt_sites=())
    diff = diff_locks(
        build_lock(MetricGraph(nodes={before.urn: before})),
        build_lock(MetricGraph(nodes={after.urn: after})),
    )
    assert [c.kind for c in diff.changes] == ["routing_changed"]


def test_gate_entry_point_enforces_the_inventory_pin():
    # Mutation survivor: check_prompt_semantics must run the pin, not only
    # check_node_semantics.
    urn = sorted(PROMPT_INVENTORY_URNS)[0]
    node = _node(urn=urn)
    failures = check_prompt_semantics(MetricGraph(nodes={urn: node}), REPO_ROOT)
    assert any("declares no prompt_sites" in f for f in failures)


@pytest.mark.parametrize("field", ["value_kind", "sparsity", "absent_means", "polarity"])
def test_every_semantic_field_is_recorded_in_the_lock(field):
    # Mutation survivor: dropping a field from SEMANTIC_FIELDS made its
    # change invisible to the drift gate.
    before = _node(**{field: "level" if field != "polarity" else "higher_is_worse"})
    after = dataclasses.replace(before, **{field: "binary" if field != "polarity" else "higher_is_better"})
    diff = diff_locks(
        build_lock(MetricGraph(nodes={before.urn: before})),
        build_lock(MetricGraph(nodes={after.urn: after})),
    )
    assert [c.kind for c in diff.changes] == ["semantics_changed"]
    assert field in diff.changes[0].fields
