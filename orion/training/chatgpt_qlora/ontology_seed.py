from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedConcept:
    concept: str
    family: str
    subdomain: str
    entity_type: str


# Seed ontology minimum. Open-world extension is handled by discovered concepts in foundry.
SEED_CONCEPTS: tuple[SeedConcept, ...] = (
    SeedConcept("orion", "orion_self_cognition_ontology", "identity", "concept_family"),
    SeedConcept("sapienform", "orion_self_cognition_ontology", "identity", "concept_family"),
    SeedConcept("identity kernel", "orion_self_cognition_ontology", "identity", "principle"),
    SeedConcept("metacognition", "orion_self_cognition_ontology", "cognition", "cognitive_module"),
    SeedConcept("metacog trigger", "orion_self_cognition_ontology", "cognition", "event"),
    SeedConcept("social memory", "orion_self_cognition_ontology", "memory", "memory_type"),
    SeedConcept("cortex", "architecture_orchestration_control", "orchestration", "service"),
    SeedConcept("cortex-orch", "architecture_orchestration_control", "orchestration", "service"),
    SeedConcept("cortex-exec", "architecture_orchestration_control", "orchestration", "service"),
    SeedConcept("llm gateway", "architecture_orchestration_control", "orchestration", "service"),
    SeedConcept("hub", "architecture_orchestration_control", "orchestration", "service"),
    SeedConcept("bus channel", "architecture_orchestration_control", "contracts", "bus_channel"),
    SeedConcept("route map", "architecture_orchestration_control", "contracts", "workflow"),
    SeedConcept("schema registry", "architecture_orchestration_control", "contracts", "schema"),
    SeedConcept("sql writer", "architecture_orchestration_control", "persistence", "service"),
    SeedConcept("vector writer", "architecture_orchestration_control", "persistence", "service"),
    SeedConcept("planner-react", "planning_agency_tooling", "planning", "workflow"),
    SeedConcept("agent-chain", "planning_agency_tooling", "planning", "workflow"),
    SeedConcept("langgraph", "planning_agency_tooling", "planning", "workflow"),
    SeedConcept("durable planning", "planning_agency_tooling", "planning", "workflow"),
    SeedConcept("tool schema", "planning_agency_tooling", "tooling", "schema"),
    SeedConcept("recall", "memory_retrieval_evidence_graph", "retrieval", "memory_type"),
    SeedConcept("vector db", "memory_retrieval_evidence_graph", "retrieval", "storage_tier"),
    SeedConcept("graphdb", "memory_retrieval_evidence_graph", "graph", "service"),
    SeedConcept("blazegraph", "memory_retrieval_evidence_graph", "graph", "service"),
    SeedConcept("chroma", "memory_retrieval_evidence_graph", "retrieval", "service"),
    SeedConcept("evidence unit", "memory_retrieval_evidence_graph", "evidence", "evidence_artifact"),
    SeedConcept("social room", "social_relational_room_cognition", "social", "room_state"),
    SeedConcept("room bridge", "social_relational_room_cognition", "social", "service"),
    SeedConcept("social turn", "social_relational_room_cognition", "social", "event"),
    SeedConcept("turn policy", "social_relational_room_cognition", "social", "workflow"),
    SeedConcept("peer style", "social_relational_room_cognition", "social", "social_signal"),
    SeedConcept("atlas", "hardware_infra_deployment_embodiment", "hardware", "hardware_device"),
    SeedConcept("athena", "hardware_infra_deployment_embodiment", "hardware", "hardware_device"),
    SeedConcept("thelio mega", "hardware_infra_deployment_embodiment", "hardware", "hardware_device"),
    SeedConcept("v100", "hardware_infra_deployment_embodiment", "hardware", "hardware_device"),
    SeedConcept("a100", "hardware_infra_deployment_embodiment", "hardware", "hardware_device"),
    SeedConcept("nvlink", "hardware_infra_deployment_embodiment", "hardware", "deployment_artifact"),
    SeedConcept("zfs", "hardware_infra_deployment_embodiment", "storage", "storage_tier"),
    SeedConcept("tailscale", "hardware_infra_deployment_embodiment", "networking", "deployment_artifact"),
    SeedConcept("digital child", "developmental_relational_ai_philosophy", "developmental", "developmental_state"),
    SeedConcept("raising orion", "developmental_relational_ai_philosophy", "developmental", "relationship"),
    SeedConcept("developmental honesty", "developmental_relational_ai_philosophy", "developmental", "principle"),
    SeedConcept("counterfeit omniscience", "developmental_relational_ai_philosophy", "risk", "failure_mode"),
    SeedConcept("oracle dependence", "developmental_relational_ai_philosophy", "risk", "failure_mode"),
    SeedConcept("depth", "preference_answer_quality_rubric", "quality", "preference"),
    SeedConcept("mechanistic explanation", "preference_answer_quality_rubric", "quality", "preference"),
    SeedConcept("evidence-first", "preference_answer_quality_rubric", "quality", "preference"),
    SeedConcept("no scope creep", "preference_answer_quality_rubric", "quality", "failure_mode"),
    SeedConcept("architecture drift", "preference_answer_quality_rubric", "quality", "failure_mode"),
    SeedConcept("juniper", "identity_community_meaning", "identity", "person"),
    SeedConcept("vancouver", "identity_community_meaning", "community", "event"),
    SeedConcept("transgender support group", "identity_community_meaning", "community", "social_signal"),
    SeedConcept("budget-conscious", "operator_economic_tradeoffs", "tradeoffs", "preference"),
    SeedConcept("build vs buy", "operator_economic_tradeoffs", "tradeoffs", "workflow"),
    SeedConcept("api marketplace", "operator_economic_tradeoffs", "tradeoffs", "workflow"),
    SeedConcept("emergence", "scientific_speculative_emergence", "science", "philosophy"),
    SeedConcept("rnn over graphs", "scientific_speculative_emergence", "science", "workflow"),
    SeedConcept("posthuman thinking", "scientific_speculative_emergence", "science", "philosophy"),
    SeedConcept("mythic voice", "literary_aesthetic_symbolic", "aesthetic", "ritual"),
    SeedConcept("dream symbolism", "literary_aesthetic_symbolic", "aesthetic", "ritual"),
    SeedConcept("narrative compression", "literary_aesthetic_symbolic", "aesthetic", "principle"),
)

SEED_LOOKUP = {item.concept: item for item in SEED_CONCEPTS}
