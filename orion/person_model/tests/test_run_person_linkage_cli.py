"""Pure helpers of the dry-run CLI (scripts/run_person_linkage.py)."""

import importlib.util
from pathlib import Path

from orion.person_model import linkage as L
from orion.person_model.extract import PERSON_LABEL, spans_to_mentions

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("run_person_linkage", ROOT / "scripts" / "run_person_linkage.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _mentions(text, names, sid="p1"):
    spans, pos = [], 0
    for n in names:
        i = text.index(n, pos)
        spans.append({"start": i, "end": i + len(n), "label": PERSON_LABEL, "score": 0.9})
        pos = i + len(n)
    return text, spans_to_mentions(spans, text=text, source_kind="chat_prompt", source_id=sid, version="v")


def test_parse_labels_handles_the_ro_query_string_form():
    assert cli.parse_labels("[SubstrateNode, Concept]") == ("SubstrateNode", "Concept")
    assert cli.parse_labels(["SubstrateNode", "Entity"]) == ("SubstrateNode", "Entity")


def test_repeated_name_in_a_list_flags_a_split_and_possessives_are_not_names():
    src = {"p1": _mentions("my boss, Rachel, my reports: Jay, Jay, Armando, and Vincent. my wife too.",
                           ["Rachel", "Jay", "Jay", "Armando", "Vincent", "my wife"])}
    clusters = cli.cluster_mentions(src)
    assert clusters["jay"]["repeat_sources"] == ["p1"]
    assert not clusters["rachel"]["repeat_sources"]
    assert "my wife" not in clusters
    assert ("armando", "jay") in cli.cooccurring_pairs(src)


def _d(anchor, cand, group, relation="same_as", kind="substrate_node", rule=None):
    cfg = L.load_config()
    return L.PersonLinkageDecisionV1(
        decision_id=L.decision_id(anchor, cand + kind + relation, cfg.version), linkage_version=cfg.version,
        anchor_id=anchor, anchor_label=anchor, candidate_id=cand + kind, candidate_label=cand, candidate_kind=kind,
        relation=relation, group=group, upper_threshold=cfg.upper, lower_threshold=cfg.lower, rule=rule)


def test_merges_of_a_split_name_are_held_and_folded_into_one_card():
    clusters = {"jay": {"node_id": "mention:jay", "repeat_sources": ["p1"]}}
    ds = [_d("mention:jay", "jay", "auto_merge"), _d("mention:jay", "Jay", "needs_your_call", "distinct", "mention_cluster")]
    held = cli.hold_merges_of_split_names(ds, clusters)
    assert all(d.group == "needs_your_call" for d in held)
    out = cli.collapse_same_question(held)
    assert len(out) == 1 and out[0].relation == "distinct" and len(out[0].evidence["also_answers"]) == 1


def test_duplicate_review_questions_collapse_keeping_the_graph_node():
    ds = [_d("ref-j", "Juniper Feld", "needs_your_call", kind="mention_cluster"), _d("ref-j", "juniper feld", "needs_your_call"),
          _d("ref-j", "juniper", "auto_merge")]
    out = cli.collapse_same_question(ds)
    review = [d for d in out if d.group == "needs_your_call"]
    assert len(out) == 2 and len(review) == 1 and review[0].candidate_kind == "substrate_node"


def test_window_snippet():
    assert cli.window("a" * 200 + " Amanda had surgery " + "b" * 200, "Amanda", 10).startswith("...")
    assert cli.window("nothing here", "Amanda") is None


def test_window_masks_email_addresses():
    w = cli.window("Juniper Feld <someone.x@gmail.com> to skye", "Juniper")
    assert "@" not in w and "[email]" in w


def test_split_hold_also_catches_graph_nodes_with_the_same_name():
    """Review finding 4: a substrate node 'Jay' cannot auto-merge while 'Jay' means two people."""
    clusters = {"jay": {"node_id": "mention:jay", "repeat_sources": ["p1"]}}
    held = cli.hold_merges_of_split_names([_d("ref-person-jay", "Jay", "auto_merge")], clusters)
    assert held[0].group == "needs_your_call" and held[0].rule == "name_used_for_two_people"


def test_cli_is_read_only_by_construction():
    """Review finding 10: no write path to the live stores exists in the CLI."""
    src = (ROOT / "scripts" / "run_person_linkage.py").read_text(encoding="utf-8")
    assert "GRAPH.QUERY" not in src and src.count("GRAPH.RO_QUERY") >= 1
    assert "readonly=True" in src
    for kw in ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "MERGE (", "SET n.", "DETACH"):
        assert kw not in src.upper().replace("SET_SESSION", ""), kw
    store = cli.ReadOnlyStore.__new__(cli.ReadOnlyStore)
    import pytest

    for name, args in (("set_prop", ("n", "p", 1)), ("add_edge", ("a", "b", "r")), ("remove_edge", ("a", "b", "r")), ("put_alias", ("n", "a", {}))):
        with pytest.raises(RuntimeError):
            getattr(store, name)(*args)


def test_claims_by_two_people_ride_on_their_pair_question():
    pair = _d("mention:skye", "Skye Laflamme", "needs_your_call", kind="mention_cluster")
    pair = pair.model_copy(update={"candidate_id": "mention:skye laflamme", "rule": "possible_match_band"})
    claim = _d("mention:skye", "skye", "needs_your_call", rule="multiple_anchor_claims")
    claim = claim.model_copy(update={"evidence": {"claimed_by_ids": ["mention:skye", "mention:skye laflamme"]}})
    out = cli.fold_claims_into_pair_questions([pair, claim])
    assert len(out) == 1 and out[0].evidence["also_answers"] == [claim.decision_id]
