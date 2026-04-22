from __future__ import annotations

import json
from pathlib import Path
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.training.chatgpt_qlora import runtime
from orion.training.chatgpt_qlora.dataset import build_sft_dataset, _load_from_postgres
from orion.training.chatgpt_qlora.eval import evaluate_training_run
from orion.training.chatgpt_qlora.foundry import build_semantic_foundry
from orion.training.chatgpt_qlora.schemas import DatasetBuildConfig, EvalConfig, FoundryConfig, SubstrateSourceConfig, TrainingConfig
from orion.training.chatgpt_qlora.trainer import run_qlora_training
import orion.training.chatgpt_qlora.trainer as trainer_mod


def _write_examples(path: Path) -> None:
    rows = [
        {
            "example_id": "ex-1",
            "import_run_id": "run-a",
            "conversation_id": "conv-1",
            "prompt": "Frontier oracle answer about architecture drift and counterfeit omniscience",
            "response": "This oracle response should be rewritten before direct SFT.",
            "turn_id": "t1",
        },
        {
            "example_id": "ex-2",
            "import_run_id": "run-a",
            "conversation_id": "conv-2",
            "prompt": "Raise Orion with developmental honesty and peer alignment in cortex-orch route map",
            "response": "Use evidence-first mechanistic architecture with Orion peer stance.",
            "turn_id": "t2",
        },
        {
            "example_id": "ex-3",
            "import_run_id": "run-b",
            "conversation_id": "conv-3",
            "prompt": "Atlas hardware operator notes for V100 rack",
            "response": "GPU topology and NVLink wiring for practical build-vs-buy.",
            "turn_id": "t3",
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_dataset_build_is_deterministic_and_preserves_lineage(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples)

    cfg = DatasetBuildConfig(
        source=SubstrateSourceConfig(examples_jsonl=str(examples)),
        output_dir=str(tmp_path / "artifacts"),
        split_seed="seed-1",
        val_ratio=0.5,
        min_prompt_chars=2,
    )
    manifest_a = build_sft_dataset(cfg)
    manifest_b = build_sft_dataset(cfg)

    assert manifest_a.included_count == manifest_b.included_count == 3
    assert manifest_a.files["train"] == manifest_b.files["train"]

    rows = [json.loads(line) for line in Path(manifest_a.files["all"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(row.get("import_run_id") for row in rows)
    assert all(row.get("conversation_id") for row in rows)


def test_foundry_frontier_oracle_does_not_enter_direct_sft(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples)
    source = SubstrateSourceConfig(examples_jsonl=str(examples))
    fcfg = FoundryConfig(output_dir=str(tmp_path / "artifacts"), build_name="foundry")
    manifest = build_semantic_foundry(fcfg, source)

    direct_rows = [json.loads(line) for line in Path(manifest.files["partition_sft_direct_orion"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    rewrite_rows = [json.loads(line) for line in Path(manifest.files["partition_sft_rewritten_oracle"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(row["relationship_mode"] != "frontier_oracle" for row in direct_rows)
    assert any(row["relationship_mode"] == "frontier_oracle" for row in rewrite_rows)


def test_foundry_orion_aligned_can_enter_direct_sft_and_unknown_concepts_survive(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples)
    source = SubstrateSourceConfig(examples_jsonl=str(examples))
    fcfg = FoundryConfig(output_dir=str(tmp_path / "artifacts"), build_name="foundry2", preserve_unknown_concepts=True)
    manifest = build_semantic_foundry(fcfg, source)

    direct_rows = [json.loads(line) for line in Path(manifest.files["partition_sft_direct_orion"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(row["relationship_mode"] in {"orion_childraising", "orion_peer", "architect_collaboration", "technical_operator"} for row in direct_rows)
    assert manifest.discovered_new_concept_count >= 1

    rewrite_rows = [json.loads(line) for line in Path(manifest.files["rewrite_candidates"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rewrite_rows
    assert rewrite_rows[0]["target_partition"] == "sft_rewritten_oracle"


def test_foundry_manifest_rollups_and_integration_to_dataset(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples)
    source = SubstrateSourceConfig(examples_jsonl=str(examples))
    fcfg = FoundryConfig(output_dir=str(tmp_path / "artifacts"), build_name="foundry3")
    manifest = build_semantic_foundry(fcfg, source)

    assert manifest.partition_counts["sft_direct_orion"] >= 1
    assert manifest.partition_counts["sft_rewritten_oracle"] >= 1
    assert manifest.relationship_mode_distribution
    assert manifest.concept_domain_frequencies

    dcfg = DatasetBuildConfig(
        source=source,
        output_dir=str(tmp_path / "artifacts"),
        foundry_build_dir=str(Path(manifest.files["annotations"]).parent),
        foundry_partition="sft_direct_orion",
        split_seed="seed-foundry",
        val_ratio=0.5,
        min_prompt_chars=2,
    )
    ds_manifest = build_sft_dataset(dcfg)
    rows = [json.loads(line) for line in Path(ds_manifest.files["all"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert all(row["example_id"] != "ex-1" for row in rows)  # frontier oracle excluded from direct SFT


def test_simulated_training_and_eval_write_manifests(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples)
    dcfg = DatasetBuildConfig(
        source=SubstrateSourceConfig(examples_jsonl=str(examples)),
        output_dir=str(tmp_path / "artifacts"),
        split_seed="seed-2",
        val_ratio=0.5,
        min_prompt_chars=2,
    )
    d_manifest = build_sft_dataset(dcfg)
    d_manifest_path = Path(dcfg.output_dir) / "dataset" / "dataset_manifest.json"

    tcfg = TrainingConfig(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        output_dir=str(tmp_path / "artifacts"),
        run_name="test-run",
    )
    t_manifest = run_qlora_training(tcfg, dataset_manifest_path=str(d_manifest_path), simulate=True)
    assert t_manifest.status == "simulated"
    assert t_manifest.run_id
    assert t_manifest.chat_template == "orion_chatml_v1"
    assert t_manifest.training_config["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"

    training_manifest_path = Path(tcfg.output_dir) / "training" / tcfg.run_name / "training_manifest.json"
    ecfg = EvalConfig(output_dir=str(tmp_path / "artifacts"), sample_count=2)
    eval_manifest, adapter_manifest = evaluate_training_run(
        ecfg,
        training_manifest_path=str(training_manifest_path),
        simulate=True,
    )

    assert eval_manifest.format_validity is True
    assert eval_manifest.provenance_integrity is True
    assert adapter_manifest.import_run_ids == d_manifest.import_run_ids
    assert adapter_manifest.run_id == t_manifest.run_id
    assert adapter_manifest.load_hints["chat_template"] == "orion_chatml_v1"


def test_real_training_mode_fails_fast_when_runtime_packages_missing(tmp_path: Path, monkeypatch) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples)
    dcfg = DatasetBuildConfig(
        source=SubstrateSourceConfig(examples_jsonl=str(examples)),
        output_dir=str(tmp_path / "artifacts"),
        split_seed="seed-3",
        val_ratio=0.5,
        min_prompt_chars=2,
    )
    build_sft_dataset(dcfg)
    d_manifest_path = Path(dcfg.output_dir) / "dataset" / "dataset_manifest.json"

    monkeypatch.setattr(
        trainer_mod,
        "ensure_runtime_packages",
        lambda packages=runtime.REAL_RUNTIME_PACKAGES: (_ for _ in ()).throw(RuntimeError("Missing runtime packages for real QLoRA run: torch")),
    )
    tcfg = TrainingConfig(base_model="meta-llama/Llama-3.1-8B-Instruct", output_dir=str(tmp_path / "artifacts"), run_name="missing-deps")
    try:
        run_qlora_training(tcfg, dataset_manifest_path=str(d_manifest_path), simulate=False)
    except RuntimeError as exc:
        assert "Missing runtime packages" in str(exc)
    else:
        raise AssertionError("Expected real mode dependency failure")


def test_load_from_postgres_uses_substrate_table(monkeypatch) -> None:
    captured = {}

    class _FakeCursor:
        def __init__(self):
            self._rows = [{"example_id": "e1"}]

        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params

        def fetchall(self):
            return self._rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeConn:
        def cursor(self, row_factory=None):
            return _FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_module = types.SimpleNamespace(
        connect=lambda uri: _FakeConn(),
        rows=types.SimpleNamespace(dict_row=object()),
    )
    monkeypatch.setitem(sys.modules, "psycopg", fake_module)
    rows = _load_from_postgres("postgresql://demo", ["run-a"], 5)
    assert rows == [{"example_id": "e1"}]
    assert "FROM chat_gpt_derived_example" in captured["query"]
    assert captured["params"] == [["run-a"], 5]
