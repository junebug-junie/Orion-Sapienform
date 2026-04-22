from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import _load_from_jsonl, _load_from_postgres
from .ontology_seed import SEED_CONCEPTS
from .schemas import FoundryBuildManifest, FoundryConfig, SubstrateSourceConfig, ensure_dir

RELATIONSHIP_MODES = {
    "frontier_oracle",
    "orion_childraising",
    "orion_peer",
    "generic_assistant",
    "technical_operator",
    "architect_collaboration",
    "reflective_confidant",
    "speculative_philosopher",
    "consumer_decision_support",
    "developmental_governance",
    "ontology_building",
    "hardware_operator",
    "emergency_debug",
    "emotional_processing",
}

FAILURE_MODE_CUES = {
    "generic": "genericity_risk",
    "boilerplate": "genericity_risk",
    "counterfeit omniscience": "counterfeit_omniscience_risk",
    "scope creep": "scope_creep",
    "fake certainty": "fake_certainty",
    "oracle": "oracle_posture_leakage",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_seed_concepts(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    norm = _norm(text)
    for seed in SEED_CONCEPTS:
        if seed.concept in norm:
            found.append(
                {
                    "phrase": seed.concept,
                    "normalized_concept": seed.concept,
                    "entity_type": seed.entity_type,
                    "concept_family": seed.family,
                    "subdomain": seed.subdomain,
                    "extraction_method": "seed_match",
                    "is_seed_match": True,
                    "confidence": 0.95,
                }
            )
    return found


def _extract_unknown_concepts(text: str, known: set[str]) -> list[dict[str, Any]]:
    concepts: list[dict[str, Any]] = []
    # preserve local quoted terms and hyphenated technical motifs
    for token in re.findall(r'"([^"]{3,80})"|\b([A-Za-z][A-Za-z0-9_-]{4,})\b', text):
        raw = token[0] or token[1]
        concept = _norm(raw)
        if not concept or concept in known or len(concept) < 4:
            continue
        if concept in {"assistant", "prompt", "response", "hello", "there"}:
            continue
        concepts.append(
            {
                "phrase": raw,
                "normalized_concept": concept,
                "entity_type": "concept_family",
                "concept_family": "discovered_open_world",
                "subdomain": "unknown",
                "extraction_method": "open_world_phrase",
                "is_seed_match": False,
                "confidence": 0.55,
            }
        )
    # dedupe while preserving order
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in concepts:
        key = item["normalized_concept"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _relationship_mode(text: str) -> str:
    norm = _norm(text)
    if "oracle" in norm or "frontier" in norm:
        return "frontier_oracle"
    if "raise orion" in norm or "digital child" in norm or "raising orion" in norm:
        return "orion_childraising"
    if "peer" in norm or "co-architect" in norm:
        return "orion_peer"
    if any(t in norm for t in ("gpu", "rack", "nvlink", "a100", "v100", "server")):
        return "hardware_operator"
    if any(t in norm for t in ("route map", "schema", "sql writer", "bus channel", "orchestration")):
        return "architect_collaboration"
    if any(t in norm for t in ("debug", "incident", "broken", "error")):
        return "emergency_debug"
    if any(t in norm for t in ("identity", "belonging", "care", "transition")):
        return "emotional_processing"
    if any(t in norm for t in ("budget", "buy", "lease", "cost")):
        return "consumer_decision_support"
    return "technical_operator"


def _oracle_vs_orion(mode: str) -> str:
    if mode == "frontier_oracle":
        return "oracle"
    if mode in {"orion_childraising", "orion_peer", "developmental_governance", "ontology_building"}:
        return "orion"
    return "mixed"


def _developmental_fit(mode: str) -> str:
    if mode == "frontier_oracle":
        return "low"
    if mode in {"orion_childraising", "orion_peer", "developmental_governance"}:
        return "high"
    if mode in {"emergency_debug", "generic_assistant"}:
        return "medium"
    return "medium"


def _depth_expectation(text: str) -> str:
    norm = _norm(text)
    if any(t in norm for t in ("frontier", "deeply", "mechanistic", "full logic")):
        return "frontier"
    if any(t in norm for t in ("mechanistic", "architecture", "evidence")):
        return "high"
    if any(t in norm for t in ("quick", "short", "brief")):
        return "low"
    return "medium"


def _answer_styles(text: str) -> list[str]:
    styles = ["direct", "evidence_first"]
    norm = _norm(text)
    if "mechanistic" in norm:
        styles.append("mechanistic")
    if "architect" in norm or "schema" in norm or "bus" in norm:
        styles.append("architectural")
    if "reflect" in norm or "identity" in norm:
        styles.append("reflective")
    if "developmental" in norm or "raising" in norm:
        styles.append("developmental")
    return sorted(set(styles))


def _failure_modes(text: str) -> list[str]:
    norm = _norm(text)
    out: list[str] = []
    for cue, label in FAILURE_MODE_CUES.items():
        if cue in norm:
            out.append(label)
    return sorted(set(out))


def _relations(text: str, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    norm = _norm(text)
    concepts_by_phrase = [c["normalized_concept"] for c in concepts]
    relations: list[dict[str, Any]] = []
    if "depends on" in norm and len(concepts_by_phrase) >= 2:
        relations.append({"relation_type": "depends_on", "source": concepts_by_phrase[0], "target": concepts_by_phrase[1]})
    if "preferred over" in norm and len(concepts_by_phrase) >= 2:
        relations.append({"relation_type": "preferred_over", "source": concepts_by_phrase[0], "target": concepts_by_phrase[1]})
    if "contradict" in norm and len(concepts_by_phrase) >= 2:
        relations.append({"relation_type": "contradicts", "source": concepts_by_phrase[0], "target": concepts_by_phrase[1]})
    if "rewrite" in norm and concepts_by_phrase:
        relations.append({"relation_type": "rewrite_targeted_to", "source": concepts_by_phrase[0], "target": "orion_peer"})
    return relations


def _partition(annotation: dict[str, Any]) -> tuple[str, str]:
    mode = annotation["relationship_mode"]
    policy = annotation["training_policy"]
    if policy["exclude_from_training"]:
        return "discard", "explicit exclusion policy"
    if mode == "frontier_oracle" and policy["requires_rewrite_before_sft"]:
        return "sft_rewritten_oracle", "frontier_oracle requires rewrite before SFT"
    if policy["negative_example"]:
        return "rubric_negative_examples", "negative example/failure mode"
    if policy["direct_sft_allowed"]:
        return "sft_direct_orion", "orion-aligned direct SFT allowed"
    if policy["ontology_only"]:
        return "ontology_graph", "ontology-rich but unsafe direct SFT"
    if policy["retrieval_only"]:
        return "retrieval_reference", "reference-only item"
    return "developmental_policy", "default developmental policy retention"


def build_semantic_foundry(cfg: FoundryConfig, source: SubstrateSourceConfig) -> FoundryBuildManifest:
    build_dir = ensure_dir(cfg.output_dir) / "foundry" / cfg.build_name
    build_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_from_jsonl(Path(source.examples_jsonl), source.limit) if source.examples_jsonl else _load_from_postgres(source.postgres_uri or "", source.import_run_ids, source.limit)

    annotations: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    conversation_summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"example_count": 0, "dominant_modes": Counter(), "top_concepts": Counter()})
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rewrite_candidates: list[dict[str, Any]] = []
    rubric_outputs: list[dict[str, Any]] = []
    ontology_outputs: list[dict[str, Any]] = []
    retrieval_outputs: list[dict[str, Any]] = []

    relationship_counter: Counter[str] = Counter()
    ovs_counter: Counter[str] = Counter()
    devfit_counter: Counter[str] = Counter()
    concept_family_counter: Counter[str] = Counter()
    subdomain_counter: Counter[str] = Counter()
    fail_counter: Counter[str] = Counter()
    ontology_density_counter: Counter[str] = Counter()
    discovered = 0
    direct_sft_count = 0
    excluded_count = 0
    arch_heavy_count = 0
    developmental_policy_count = 0

    for row in raw_rows:
        example_id = str(row.get("example_id") or "")
        prompt = str(row.get("prompt") or "")
        response = str(row.get("response") or "")
        text = f"{prompt}\n{response}"

        mode = _relationship_mode(text)
        ovs = _oracle_vs_orion(mode)
        devfit = _developmental_fit(mode)
        depth = _depth_expectation(text)
        styles = _answer_styles(text)
        failures = _failure_modes(text)

        seed_entities = _extract_seed_concepts(text)
        known = {e["normalized_concept"] for e in seed_entities}
        unknown_entities = _extract_unknown_concepts(text, known) if cfg.preserve_unknown_concepts else []
        all_entities = seed_entities + unknown_entities
        discovered += sum(1 for e in unknown_entities if not e["is_seed_match"])

        rels = _relations(text, all_entities)

        direct_sft_allowed = mode in {"orion_childraising", "orion_peer", "technical_operator", "architect_collaboration"} and "oracle_posture_leakage" not in failures
        requires_rewrite = mode == "frontier_oracle"
        negative_example = bool(failures)
        ontology_only = mode in {"frontier_oracle", "speculative_philosopher", "ontology_building"}

        training_policy = {
            "direct_sft_allowed": direct_sft_allowed,
            "rubric_only": negative_example,
            "retrieval_only": mode in {"consumer_decision_support"},
            "ontology_only": ontology_only,
            "teacher_exemplar": depth in {"high", "frontier"},
            "negative_example": negative_example,
            "requires_rewrite_before_sft": requires_rewrite,
            "needs_relational_sanitization": mode == "frontier_oracle",
            "depth_reference_only": depth == "frontier" and not direct_sft_allowed,
            "exclude_from_training": False,
        }

        scores = {
            "orion_direct_fit_score": 0.9 if direct_sft_allowed else 0.2,
            "depth_reference_score": 0.9 if depth in {"high", "frontier"} else 0.4,
            "relational_alignment_score": 0.8 if mode in {"orion_childraising", "orion_peer"} else 0.4,
            "genericity_risk_score": 0.7 if "genericity_risk" in failures else 0.2,
            "counterfeit_omniscience_risk_score": 0.9 if "counterfeit_omniscience_risk" in failures else 0.2,
            "ontology_density_score": min(1.0, len(seed_entities) / 8.0),
            "architecture_signal_score": 0.9 if mode in {"architect_collaboration", "technical_operator"} else 0.3,
            "developmental_signal_score": 0.9 if mode in {"orion_childraising", "orion_peer"} else 0.4,
        }

        annotation = {
            "example_id": example_id,
            "conversation_id": row.get("conversation_id"),
            "turn_id": row.get("turn_id"),
            "import_run_id": row.get("import_run_id"),
            "user_message_id": row.get("user_message_id"),
            "assistant_message_id": row.get("assistant_message_id"),
            "relationship_mode": mode,
            "oracle_vs_orion": ovs,
            "developmental_fit": devfit,
            "training_policy": training_policy,
            "depth_expectation": depth,
            "answer_style_preference": styles,
            "failure_modes_present": failures,
            "scores": scores,
            "concept_domains": sorted({e["concept_family"] for e in all_entities}),
            "concept_subdomains": sorted({e["subdomain"] for e in all_entities}),
        }
        annotations.append(annotation)

        partition_name, reason = _partition(annotation)
        partition_entry = {
            "example_id": example_id,
            "conversation_id": row.get("conversation_id"),
            "import_run_id": row.get("import_run_id"),
            "prompt": prompt,
            "response": response,
            "partition_reason": reason,
            "relationship_mode": mode,
            "oracle_vs_orion": ovs,
            "direct_sft_allowed": direct_sft_allowed,
            "requires_rewrite_before_sft": requires_rewrite,
            "preserve_for_ontology_reference": ontology_only,
        }
        partitions[partition_name].append(partition_entry)

        if requires_rewrite and cfg.include_oracle_rewrite_candidates:
            rewrite_candidates.append(
                {
                    "example_id": example_id,
                    "conversation_id": row.get("conversation_id"),
                    "import_run_id": row.get("import_run_id"),
                    "original_relationship_mode": mode,
                    "target_relationship_mode": "orion_peer",
                    "rewrite_rationale": "frontier_oracle defaults to rewrite before direct SFT",
                    "risk_signals": failures or ["oracle_posture_leakage"],
                    "target_partition": "sft_rewritten_oracle",
                    "semantic_constraints": ["preserve technical facts", "remove oracle-posture leakage"],
                }
            )

        if failures:
            rubric_outputs.append(
                {
                    "example_id": example_id,
                    "conversation_id": row.get("conversation_id"),
                    "import_run_id": row.get("import_run_id"),
                    "reward_signals": styles,
                    "reject_signals": failures,
                    "depth_expectation": depth,
                    "relationship_mode": mode,
                }
            )

        ontology_outputs.append(
            {
                "example_id": example_id,
                "conversation_id": row.get("conversation_id"),
                "import_run_id": row.get("import_run_id"),
                "nodes": all_entities,
                "edges": rels,
            }
        )
        retrieval_outputs.append(
            {
                "example_id": example_id,
                "conversation_id": row.get("conversation_id"),
                "import_run_id": row.get("import_run_id"),
                "text": text,
                "relationship_mode": mode,
                "concepts": [e["normalized_concept"] for e in all_entities],
            }
        )

        entities.extend({"example_id": example_id, "conversation_id": row.get("conversation_id"), "import_run_id": row.get("import_run_id"), **e} for e in all_entities)
        relations.extend({"example_id": example_id, "conversation_id": row.get("conversation_id"), "import_run_id": row.get("import_run_id"), **r} for r in rels)

        conversation_key = str(row.get("conversation_id") or "")
        conversation_summary[conversation_key]["example_count"] += 1
        conversation_summary[conversation_key]["dominant_modes"][mode] += 1
        for e in all_entities:
            conversation_summary[conversation_key]["top_concepts"][e["normalized_concept"]] += 1

        relationship_counter[mode] += 1
        ovs_counter[ovs] += 1
        devfit_counter[devfit] += 1
        if direct_sft_allowed:
            direct_sft_count += 1
        if partition_name == "discard":
            excluded_count += 1
        if mode in {"architect_collaboration", "technical_operator", "hardware_operator"}:
            arch_heavy_count += 1
        if partition_name == "developmental_policy":
            developmental_policy_count += 1
        for f in failures:
            fail_counter[f] += 1
        for e in all_entities:
            concept_family_counter[e["concept_family"]] += 1
            subdomain_counter[e["subdomain"]] += 1
        density_bucket = "high" if scores["ontology_density_score"] >= 0.7 else "medium" if scores["ontology_density_score"] >= 0.3 else "low"
        ontology_density_counter[density_bucket] += 1

    # finalize conversation summaries
    conversation_rows: list[dict[str, Any]] = []
    for conv_id, payload in conversation_summary.items():
        dominant = payload["dominant_modes"].most_common(3)
        concepts = payload["top_concepts"].most_common(12)
        conversation_rows.append(
            {
                "conversation_id": conv_id,
                "example_count": payload["example_count"],
                "dominant_relationship_modes": [k for k, _ in dominant],
                "top_concepts": [k for k, _ in concepts],
            }
        )

    files = {
        "annotations": str(build_dir / "annotations.example.jsonl"),
        "entities": str(build_dir / "entities.example.jsonl"),
        "relations": str(build_dir / "relations.example.jsonl"),
        "conversation_summary": str(build_dir / "conversation.summary.jsonl"),
        "rewrite_candidates": str(build_dir / "rewrite.candidates.jsonl"),
        "rubric_preferences": str(build_dir / "rubric.preferences.jsonl"),
        "ontology_graph": str(build_dir / "ontology.graph.jsonl"),
        "retrieval_reference": str(build_dir / "retrieval.reference.jsonl"),
    }

    for key in (
        "sft_direct_orion",
        "sft_rewritten_oracle",
        "rubric_preferences",
        "rubric_negative_examples",
        "ontology_graph",
        "retrieval_reference",
        "teacher_exemplars",
        "developmental_policy",
        "architecture_reference",
        "hardware_reference",
        "anti_pattern_archive",
        "discard",
    ):
        files[f"partition_{key}"] = str(build_dir / f"partition.{key}.jsonl")

    def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    _write_jsonl(files["annotations"], annotations)
    _write_jsonl(files["entities"], entities)
    _write_jsonl(files["relations"], relations)
    _write_jsonl(files["conversation_summary"], conversation_rows)
    _write_jsonl(files["rewrite_candidates"], rewrite_candidates)
    _write_jsonl(files["rubric_preferences"], rubric_outputs)
    _write_jsonl(files["ontology_graph"], ontology_outputs)
    _write_jsonl(files["retrieval_reference"], retrieval_outputs)
    for key, path in files.items():
        if key.startswith("partition_"):
            partition = key.removeprefix("partition_")
            _write_jsonl(path, partitions.get(partition, []))

    manifest = FoundryBuildManifest(
        created_at=_now(),
        build_name=cfg.build_name,
        source=source,
        files=files,
        relationship_mode_distribution=dict(relationship_counter),
        oracle_vs_orion_distribution=dict(ovs_counter),
        developmental_fit_distribution=dict(devfit_counter),
        partition_counts={k.removeprefix("partition_"): len(partitions.get(k.removeprefix("partition_"), [])) for k in files if k.startswith("partition_")},
        concept_domain_frequencies=dict(concept_family_counter),
        subdomain_frequencies=dict(subdomain_counter),
        discovered_new_concept_count=discovered,
        failure_mode_frequencies=dict(fail_counter),
        rewrite_candidate_count=len(rewrite_candidates),
        teacher_exemplar_count=sum(1 for x in annotations if x["training_policy"]["teacher_exemplar"]),
        direct_sft_allowed_count=direct_sft_count,
        excluded_count=excluded_count,
        architecture_heavy_count=arch_heavy_count,
        developmental_policy_count=developmental_policy_count,
        ontology_density_distribution=dict(ontology_density_counter),
    )
    manifest_path = build_dir / "foundry_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
