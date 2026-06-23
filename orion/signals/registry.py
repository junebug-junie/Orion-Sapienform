"""
Causal DAG registry for organ relationships.
"""
from typing import Dict, List
from .models import OrionOrganRegistryEntry, OrganClass

# Static dict imported by adapters, the gateway, and any downstream consumer
ORGAN_REGISTRY: Dict[str, OrionOrganRegistryEntry] = {
    "biometrics": OrionOrganRegistryEntry(
        organ_id="biometrics",
        organ_class=OrganClass.exogenous,
        service="orion-biometrics",
        signal_kinds=["gpu_load", "cpu_load", "memory_pressure", "thermal_state", "network_load", "disk_io", "power_state"],
        canonical_dimensions=["level", "trend", "volatility", "confidence"],
        causal_parent_organs=[],
        bus_channels=[
            "orion:telemetry:biometrics",
            "orion:biometrics:sample",
            "orion:biometrics:summary",
            "orion:biometrics:induction",
            "orion:biometrics:cluster",
        ],
        notes=[]
    ),
    "vision": OrionOrganRegistryEntry(
        organ_id="vision",
        organ_class=OrganClass.exogenous,
        service="orion-vision-council",
        signal_kinds=["scene_state", "person_detected", "object_event", "visual_context"],
        canonical_dimensions=["level", "valence", "novelty", "confidence"],
        causal_parent_organs=[],
        bus_channels=["orion:vision:windows", "orion:vision:events", "orion:vision:edge:raw"],
        notes=[]
    ),
    "social_room_bridge": OrionOrganRegistryEntry(
        organ_id="social_room_bridge",
        organ_class=OrganClass.exogenous,
        service="orion-social-room-bridge",
        signal_kinds=["social_turn", "room_event"],
        canonical_dimensions=["level", "valence", "confidence"],
        causal_parent_organs=[],
        bus_channels=[
            "orion:bridge:social:room:intake",
            "orion:bridge:social:room:delivery",
            "orion:bridge:social:participant",
            "orion:social:repair:signal",
            "orion:social:epistemic:signal",
        ],
        notes=[]
    ),
    "power_guard": OrionOrganRegistryEntry(
        organ_id="power_guard",
        organ_class=OrganClass.exogenous,
        service="orion-power-guard",
        signal_kinds=["power_state", "ups_event"],
        canonical_dimensions=["level", "trend", "confidence"],
        causal_parent_organs=[],
        bus_channels=["orion:power:events"],
        notes=[]
    ),
    "security_watcher": OrionOrganRegistryEntry(
        organ_id="security_watcher",
        organ_class=OrganClass.exogenous,
        service="orion-security-watcher",
        signal_kinds=["security_event"],
        canonical_dimensions=["level", "confidence"],
        causal_parent_organs=[],
        bus_channels=["orion:security:visits", "orion:security:alerts", "orion:vision:guard:signal"],
        notes=[]
    ),
    "equilibrium": OrionOrganRegistryEntry(
        organ_id="equilibrium",
        organ_class=OrganClass.hybrid,
        service="orion-equilibrium-service",
        signal_kinds=["mesh_health", "service_distress", "zen_state"],
        canonical_dimensions=["level", "trend", "confidence"],
        causal_parent_organs=["biometrics"],
        bus_channels=[
            "orion:equilibrium:snapshot",
            "orion:spark:signal",
            "orion:metacognition:tick",
            "orion:cognition:trace",
            "orion:equilibrium:metacog:trigger",
            "orion:collapse:intake",
            "orion:pad:signal",
        ],
        notes=[]
    ),
    "world_pulse": OrionOrganRegistryEntry(
        organ_id="world_pulse",
        organ_class=OrganClass.hybrid,
        service="orion-world-pulse",
        signal_kinds=["situation_state", "time_context", "environmental_context"],
        canonical_dimensions=["level", "valence", "confidence"],
        causal_parent_organs=[],
        bus_channels=[
            "orion:world_pulse:run:request",
            "orion:world_pulse:run:result",
            "orion:world_pulse:digest:created",
            "orion:world_pulse:digest:published",
            "orion:world_pulse:situation:brief:upsert",
            "orion:world_context:daily_capsule",
            "orion:hub:messages:create",
            "orion:world_pulse:graph:upsert",
        ],
        notes=[
            "Definitive DAG: no organ-bus causal parents. Situation/context ingests external feeds, "
            "hub messages, and graph channels per orion-world-pulse contracts — not upstream "
            "OrionSignalV1 parents in this registry.",
            "Reflective journal: orion-actions consumes orion:world_pulse:run:result "
            "(world.pulse.run.result.v1 / WorldPulseRunResultV1) and routes journal.compose → "
            "orion:journal:write when ACTIONS_WORLD_PULSE_JOURNAL_ENABLED.",
        ],
    ),
    "social_memory": OrionOrganRegistryEntry(
        organ_id="social_memory",
        organ_class=OrganClass.hybrid,
        service="orion-social-memory",
        signal_kinds=["social_bond_state", "relationship_continuity", "social_repair_event"],
        canonical_dimensions=["level", "valence", "trend"],
        causal_parent_organs=["social_room_bridge"],
        bus_channels=[
            "orion:chat:social:stored",
            "orion:social:participant:continuity",
            "orion:social:room:continuity",
            "orion:social:relational:update",
            "orion:social:stance:snapshot",
        ],
        notes=[]
    ),
    "collapse_mirror": OrionOrganRegistryEntry(
        organ_id="collapse_mirror",
        organ_class=OrganClass.endogenous,
        service="orion-collapse-mirror",
        signal_kinds=["cognitive_collapse", "metacog_event"],
        canonical_dimensions=["level", "valence", "confidence"],
        causal_parent_organs=["biometrics", "equilibrium"],
        bus_channels=["orion:collapse:intake", "orion:collapse:triage", "orion:collapse:sql-write"],
        notes=[]
    ),
    "recall": OrionOrganRegistryEntry(
        organ_id="recall",
        organ_class=OrganClass.endogenous,
        service="orion-recall",
        signal_kinds=["recall_result", "recall_gap", "recall_quality"],
        canonical_dimensions=["level", "trend", "confidence"],
        causal_parent_organs=["autonomy", "social_memory"],
        bus_channels=[
            "orion:exec:request:RecallService",
            "orion:exec:result:RecallService",
            "orion:recall:telemetry",
        ],
        notes=[]
    ),
    "concept_induction": OrionOrganRegistryEntry(
        organ_id="concept_induction",
        organ_class=OrganClass.endogenous,
        service="orion-spark-concept-induction",
        signal_kinds=["concept_salience", "topic_formation", "concept_drift"],
        canonical_dimensions=["salience", "novelty", "confidence"],
        causal_parent_organs=["chat_stance", "recall", "vision"],
        bus_channels=[
            "orion:chat:history:log",
            "orion:chat:history:turn",
            "orion:spark:telemetry",
            "orion:cognition:trace",
            "orion:spark:concepts:profile",
            "orion:spark:concepts:delta",
        ],
        notes=[]
    ),
    "spark_introspector": OrionOrganRegistryEntry(
        organ_id="spark_introspector",
        organ_class=OrganClass.endogenous,
        service="orion-spark-introspector",
        signal_kinds=["phi_field", "tissue_state", "spark_signal"],
        canonical_dimensions=["level", "valence", "arousal", "coherence", "novelty"],
        causal_parent_organs=["biometrics", "equilibrium", "recall", "collapse_mirror", "vision"],
        bus_channels=[
            "orion:spark:introspect:candidate*",
            "orion:cognition:trace",
            "orion:spark:telemetry",
            "orion:spark:state:snapshot",
            "orion:spark:signal",
            "orion:core:events",
        ],
        notes=[]
    ),
    "graph_cognition": OrionOrganRegistryEntry(
        organ_id="graph_cognition",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-exec",
        signal_kinds=[
            "metacog_perception",
            "coherence_state",
            "goal_pressure",
            "repair_pressure",
        ],
        canonical_dimensions=[
            "coherence",
            "tension",
            "goal_pressure",
            "confidence",
            "level",
            "specificity_demand",
            "trust_rupture",
            "coherence_gap",
            "repetition_failure",
            "operational_block",
            "explicit_repair_command",
            "assistant_accountability_demand",
        ],
        # recall omitted here to keep static causal_parent_organs acyclic (recall→autonomy→graph_cognition).
        causal_parent_organs=["social_memory"],
        bus_channels=["orion:cognition:trace", "orion:metacog:trace"],
        notes=[
            "Recall-linked metacog is carried via autonomy/recall in the mesh; not duplicated as a "
            "registry parent edge to avoid a recall↔graph_cognition cycle in static DAG validation.",
            "repair_pressure is a substrate-derived appraisal emitted by orion.substrate.appraisal; "
            "see docs/plans/substrate/2026-05-23-repair-pressure-v1.md.",
        ],
    ),
    "autonomy": OrionOrganRegistryEntry(
        organ_id="autonomy",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-exec",
        signal_kinds=["autonomy_state", "drive_pressure", "tension_state"],
        canonical_dimensions=["pressure_coherence", "pressure_continuity", "pressure_relational", "pressure_autonomy", "pressure_capability", "pressure_predictive", "confidence"],
        causal_parent_organs=["graph_cognition"],
        bus_channels=[
            "orion:cortex:exec:request",
            "orion:exec:request",
            "orion:exec:result",
            "orion:cognition:trace",
        ],
        notes=[]
    ),
    "chat_stance": OrionOrganRegistryEntry(
        organ_id="chat_stance",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-exec",
        signal_kinds=["chat_stance", "turn_effect", "metacog_residue"],
        canonical_dimensions=["coherence", "valence", "confidence"],
        causal_parent_organs=["recall", "autonomy", "equilibrium", "social_memory", "spark_introspector"],
        bus_channels=[
            "orion:cortex:exec:request",
            "orion:chat:history:turn",
            "orion:chat:history:log",
            "orion:cognition:trace",
        ],
        notes=[]
    ),
    "memory_consolidation": OrionOrganRegistryEntry(
        organ_id="memory_consolidation",
        organ_class=OrganClass.endogenous,
        service="orion-memory-consolidation",
        signal_kinds=["turn_change"],
        canonical_dimensions=["novelty", "salience", "contradiction", "confidence"],
        causal_parent_organs=["hub"],
        bus_channels=["orion:memory:turn:persisted"],
        notes=["Logprob turn change appraisal substrate perturbation."],
    ),
    "journaler": OrionOrganRegistryEntry(
        organ_id="journaler",
        organ_class=OrganClass.endogenous,
        service="orion library (orion/journaler/)",
        signal_kinds=["journal_entry", "recall_event"],
        canonical_dimensions=["level", "novelty", "valence"],
        causal_parent_organs=["collapse_mirror", "chat_stance", "agent_chain"],
        bus_channels=[
            "orion:chat:history:log",
            "orion:chat:history:turn",
            "orion:collapse:sql-write",
        ],
        notes=[]
    ),
    "state_journaler": OrionOrganRegistryEntry(
        organ_id="state_journaler",
        organ_class=OrganClass.endogenous,
        service="orion-state-journaler",
        signal_kinds=["state_frame", "state_transition"],
        canonical_dimensions=["level", "coherence", "confidence"],
        causal_parent_organs=["autonomy", "equilibrium", "recall"],
        bus_channels=["orion:spark:state:snapshot", "orion:equilibrium:snapshot"],
        notes=[]
    ),
    "dream": OrionOrganRegistryEntry(
        organ_id="dream",
        organ_class=OrganClass.endogenous,
        service="orion-dream",
        signal_kinds=["dream_cycle_output", "memory_consolidation"],
        canonical_dimensions=["level", "coherence", "novelty"],
        causal_parent_organs=["recall", "social_memory", "state_journaler"],
        bus_channels=[
            "orion:dream:trigger",
            "orion:dream:buffer",
            "orion:dream:complete",
            "orion:collapse:sql-write",
        ],
        notes=[]
    ),
    "agent_chain": OrionOrganRegistryEntry(
        organ_id="agent_chain",
        organ_class=OrganClass.endogenous,
        service="orion-agent-chain",
        signal_kinds=["action_outcome", "tool_execution", "capability_event"],
        canonical_dimensions=["level", "success", "surprise"],
        causal_parent_organs=["planner", "autonomy", "chat_stance"],
        bus_channels=[
            "orion:exec:request:AgentChainService",
            "orion:exec:result:AgentChainService",
        ],
        notes=[]
    ),
    "planner": OrionOrganRegistryEntry(
        organ_id="planner",
        organ_class=OrganClass.endogenous,
        service="orion-planner-react",
        signal_kinds=["plan_state", "goal_progress"],
        canonical_dimensions=["level", "confidence", "surprise"],
        # agent_chain omitted to keep static DAG acyclic (agent_chain already lists planner as parent).
        causal_parent_organs=["autonomy"],
        bus_channels=[
            "orion:exec:request:PlannerReactService",
            "orion:exec:result:PlannerReactService",
        ],
        notes=[
            "agent_chain is not listed as a causal parent to avoid a planner↔agent_chain cycle in static "
            "DAG validation; execution outcomes still reach planner via autonomy and bus contracts."
        ],
    ),
    "topic_foundry": OrionOrganRegistryEntry(
        organ_id="topic_foundry",
        organ_class=OrganClass.endogenous,
        service="orion-topic-foundry",
        signal_kinds=["topic_state", "topic_drift"],
        canonical_dimensions=["salience", "novelty", "coherence"],
        causal_parent_organs=["concept_induction", "chat_stance"],
        bus_channels=[
            "orion:exec:request:LLMGatewayService",
            "orion:chat:history:turn",
            "orion:chat:history:log",
        ],
        notes=[]
    ),
    "cortex_exec": OrionOrganRegistryEntry(
        organ_id="cortex_exec",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-exec",
        signal_kinds=["cognition_run", "cognition_step"],
        canonical_dimensions=["success", "step_count", "latency_level", "recall_used", "confidence"],
        causal_parent_organs=["autonomy", "graph_cognition"],
        bus_channels=["orion:cognition:trace"],
        notes=["Runtime trace fusion: PlanRunner publishes CognitionTracePayload."],
    ),
    "llm_gateway": OrionOrganRegistryEntry(
        organ_id="llm_gateway",
        organ_class=OrganClass.endogenous,
        service="orion-llm-gateway",
        signal_kinds=["cognition_step", "completion"],
        canonical_dimensions=["success", "latency_level", "confidence"],
        causal_parent_organs=["chat_stance", "cortex_exec"],
        bus_channels=["orion:exec:result:LLMGatewayService"],
        notes=[],
    ),
    "mind": OrionOrganRegistryEntry(
        organ_id="mind",
        organ_class=OrganClass.endogenous,
        service="orion-mind",
        signal_kinds=["mind_handoff", "cognition_step"],
        canonical_dimensions=["success", "confidence"],
        causal_parent_organs=["chat_stance", "recall"],
        bus_channels=["orion:mind:run:complete"],
        notes=["Milestone B: mind_handoff adapter."],
    ),
    "cortex_gateway": OrionOrganRegistryEntry(
        organ_id="cortex_gateway",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-gateway",
        signal_kinds=["route_decision"],
        canonical_dimensions=["confidence", "level"],
        causal_parent_organs=[],
        bus_channels=["orion:cortex:gateway:request"],
        notes=["Milestone B: gateway route from Hub cortex:gateway:request."],
    ),
    "cortex_orch": OrionOrganRegistryEntry(
        organ_id="cortex_orch",
        organ_class=OrganClass.endogenous,
        service="orion-cortex-orch",
        signal_kinds=["plan_resolution"],
        canonical_dimensions=["confidence", "level"],
        causal_parent_organs=["cortex_gateway"],
        bus_channels=["orion:cortex:request"],
        notes=["Milestone B: orch plan from cortex.orch.request on orion:cortex:request."],
    ),
    "hub": OrionOrganRegistryEntry(
        organ_id="hub",
        organ_class=OrganClass.exogenous,
        service="orion-hub",
        signal_kinds=["chat_turn"],
        canonical_dimensions=["confidence", "level"],
        causal_parent_organs=[],
        bus_channels=["orion:chat:history:turn"],
        notes=["Milestone B: exogenous chat_turn from Hub publish_chat_turn."],
    ),
    "sql_writer": OrionOrganRegistryEntry(
        organ_id="sql_writer",
        organ_class=OrganClass.exogenous,
        service="orion-sql-writer",
        signal_kinds=["persist"],
        canonical_dimensions=["level", "confidence", "latency_level"],
        causal_parent_organs=[],
        bus_channels=["orion:collapse:stored", "orion:journal:created"],
        notes=["Milestone B: SQL persist receipts (no row contents)."],
    ),
    "rdf_writer": OrionOrganRegistryEntry(
        organ_id="rdf_writer",
        organ_class=OrganClass.exogenous,
        service="orion-rdf-writer",
        signal_kinds=["persist"],
        canonical_dimensions=["level", "confidence", "latency_level"],
        causal_parent_organs=[],
        bus_channels=["orion:rdf:*"],
        notes=["Milestone B: RDF persist presence (channel pattern)."],
    ),
    "vector_writer": OrionOrganRegistryEntry(
        organ_id="vector_writer",
        organ_class=OrganClass.exogenous,
        service="orion-vector-writer",
        signal_kinds=["persist"],
        canonical_dimensions=["level", "confidence", "latency_level"],
        causal_parent_organs=[],
        bus_channels=["orion:evidence:index:upsert"],
        notes=["Milestone B: vector index persist presence."],
    ),
}

# Registry accuracy note:
# The causal_parent_organs entries above are first-pass structural approximations derived from code inspection.
# The implementation phase must verify each organ's actual bus channel subscriptions and cross-check against the adapters
# before treating the registry as authoritative. bus_channels (which channels the gateway subscribes to per organ)
# are also to be specified per-organ during implementation; they are omitted from the table above for readability.

# The primary causal chain (explicitly modeled):
# biometrics [exogenous]
#   → equilibrium [hybrid]
#     → collapse_mirror [endogenous]
#       → journaler [endogenous]
#         → recall [endogenous]    ← also receives: autonomy, social_memory
#           → chat_stance [endogenous]   ← also receives: autonomy, equilibrium, social_memory, spark_introspector
#             → (chat turn output)
#               → recall [next turn]   ← temporal feedback loop