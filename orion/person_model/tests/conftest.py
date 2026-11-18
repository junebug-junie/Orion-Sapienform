"""Fixture: the 11 graph nodes whose label contains the interlocutor's name
(live orion_substrate, 2026-10-14; spec "The graph splits her"), as plain data."""

import pytest

from orion.person_model.linkage import Anchor, Candidate, LinkageContext

TF = "topic_foundry_adapter"
ENT = ("SubstrateNode", "Entity")
CON = ("SubstrateNode", "Concept")

ANCHOR = Anchor("referent-b96f", "juniper", ("juniper",), "memory.referents", ("episode_memory:m1",))

# node_id -> (label, graph labels, producer, expected relation, expected group)
NODES = {
    "sub-concept-seed-juniper": ("Juniper", CON, "seed_concepts_loader", "same_as", "auto_merge"),
    "tf-ent-ca5c": ("juniper", ENT, TF, "same_as", "auto_merge"),
    "tf-ent-e500": ("juniper feld", ENT, TF, "same_as", "needs_your_call"),
    "tf-ent-6d08": ("juniper's taste", ENT, TF, "facet_of", "auto_merge"),
    "tf-ent-1026": ("juniper affect", ENT, TF, "facet_of", "auto_merge"),
    "sub-concept-seed-orion_juniper_relationship": ("Orion-Juniper relationship", CON, "seed_concepts_loader", "no_match", "auto_reject"),
    "tf-ent-edfa": ("juniper relationship summary", ENT, TF, "junk", "auto_reject"),
    "tf-ent-eb42": ("juniper_relationship_summary", ENT, TF, "junk", "auto_reject"),
    "tf-con-11": ("Chat with Juniper", CON, TF, "no_match", "auto_reject"),
    "tf-con-9": ("mind / hey / juniper", CON, TF, "no_match", "auto_reject"),
}

PERSON_NAMES = {"Juniper": True, "juniper": True, "juniper feld": True, "Orion-Juniper relationship": False,
                "juniper affect": False, "juniper's taste": False}
EMB = {"juniper": 1.0, "juniper feld": 0.62, "orion juniper relationship": 0.41}


@pytest.fixture
def candidates():
    return [Candidate(nid, lab, "substrate_node", labels, prod) for nid, (lab, labels, prod, _, _) in NODES.items()]


@pytest.fixture
def ctx():
    from orion.person_model.normalize import fold

    return LinkageContext(
        is_person_name=lambda lab: PERSON_NAMES.get(lab),
        embedding_sim=lambda a, b: EMB.get(fold(b)),
        prompt_keys=frozenset({"juniper_relationship_summary", "orion_identity_summary"}),
    )
