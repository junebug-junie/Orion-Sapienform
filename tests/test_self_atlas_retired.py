"""Self Atlas retirement (2026-11), kill means kill.

Spec: docs/superpowers/specs/2026-10-30-self-atlas-replacement-proposal.md, Patch B.
The daily self-knowledge scan, the Self Atlas topic-foundry runs, the
self_atlas_cluster history writes and the self FalkorDB graph wiring are
deleted, not flagged off. The dig kit and the body atlas replace them.

(a) No retired name is referenced in code or config.
(b) The self-inquiry ledger tells Orion what retired and where to look now.
(c) The kept pieces survive: history tables and literals, the analysis sensor.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from orion.curiosity.self_inquiry import LEDGER_RETIREMENTS, LedgerRow, format_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]

RETIRED_PATTERNS = (
    r"\bself_atlas_cluster_history\b",
    r"\bpublish_self_atlas_cluster_history\b",
    r"\bself_atlas_source_unchanged_since_last_run\b",
    r"\btrigger_topic_foundry_self_\w+",
    r"\bconcept_atlas_ingest_topic_foundry_self\b",
    r"ingest-topic-foundry-self",
    r"\b_TOPIC_FOUNDRY_SELF_\w+",
    r"\bSUBSTRATE_TOPIC_FOUNDRY_SELF_\w+",
    r"\bSUBSTRATE_SEMANTIC_STORE_SELF\b",
    r"\bFALKORDB_SELF_SUBSTRATE_GRAPH\b",
    r"\bbuild_self_falkor_substrate_store_from_env\b",
    r"\b_get_self_substrate_store\b",
    r"\b_build_self_substrate_store_from_env\b",
    r"\bself_knowledge_freshness\b",
    r"\bNEWEST_SELF_KNOWLEDGE_ITEM_SQL\b",
    r"\bnewest_self_knowledge_item_at\b",
    r"orion:self_study:items:write",
    r"self_study\.items\.write",
    r"\bSelfKnowledgeItemLogV1\b",
    r"\bSelfKnowledgeItemV1\b",
    r"\bSelfSnapshotV1\b",
    r"\bSelfRepoInspectResultV1\b",
    r"\bSelfKnowledgeSectionCountsV1\b",
    r"\brun_self_repo_inspect\b",
    r"\bself_study_refresh\w*",
    r"\bSELF_STUDY_INSPECT_INTERVAL_SEC\b",
    r"orion-self-atlas-",
)
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".j2", ".js", ".html", ".sh", ".toml", ".sql", ".cfg", ".ini", ".env_example"}
ALLOWED = {
    # Names what it forbids.
    "tests/test_self_atlas_retired.py",
    # Assert the retired pieces stay gone, so they name them.
    "services/orion-cortex-exec/tests/test_orch_paths.py",
    "services/orion-sql-writer/tests/test_self_knowledge_items_retired.py",
    "services/orion-hub/tests/test_concept_atlas_routes.py",
    "services/orion-hub/tests/test_substrate_store_warm.py",
    "orion/substrate/tests/test_falkor_store_aitown_graph.py",
    # Copies orion_substrate_self to an archive graph; must name the source.
    "scripts/archive_self_atlas_graph.py",
    "tests/test_archive_self_atlas_graph.py",
}
# Dated snapshots of the live body/bus taken before the retirement record what WAS true.
ALLOWED_PREFIXES = ("orion/body_atlas/evals/fixtures/",)


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "services", "orion", "config", "scripts", "tests", "Makefile", ".github"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def test_no_retired_self_atlas_piece_is_referenced_in_code_or_config() -> None:
    combined = re.compile("|".join(RETIRED_PATTERNS))
    hits: list[str] = []
    for rel in _tracked_files():
        if rel in ALLOWED or rel.startswith(ALLOWED_PREFIXES):
            continue
        path = REPO_ROOT / rel
        suffix = ".env_example" if path.name == ".env_example" else path.suffix
        if suffix not in SCANNED_SUFFIXES and path.name not in {"Makefile", "Dockerfile"}:
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            m = combined.search(line)
            if m:
                hits.append(f"{rel}:{lineno}: {m.group(0)}")
    assert not hits, "retired Self Atlas pieces still referenced:\n" + "\n".join(hits[:50])


def test_retired_files_are_deleted() -> None:
    for rel in (
        "services/orion-cortex-exec/app/self_study.py",
        "services/orion-cortex-exec/app/self_study_refresh.py",
        "services/orion-hub/scripts/self_atlas_cluster_history.py",
        "orion/self_knowledge_freshness.py",
        "orion/schemas/self_knowledge_item_log.py",
    ):
        assert not (REPO_ROOT / rel).exists(), rel


def test_ledger_tells_orion_what_retired_and_where_to_look() -> None:
    rows = [
        LedgerRow("self_concept_history", 9000, "2026-11-05T06:45:00+00:00"),
        LedgerRow("self_knowledge_items", 61000, "2026-11-05T22:24:00+00:00"),
    ]
    text = "\n".join(format_ledger(rows))
    assert "self_atlas_cluster" in text and "Self Atlas" in text
    assert "python -P -m orion.dig body" in text
    assert "chat_stance_belief_log" in text
    assert "rows stay as history" in text
    notes = LEDGER_RETIREMENTS["self_knowledge_items"]
    assert not any("feeds Self Atlas" in n for n in notes), "the old 'still live' note must go"


def test_kept_pieces_survive() -> None:
    from orion.schemas.self_concept_history import SelfConceptHistoryProducer
    from typing import get_args

    # Old rows still validate.
    assert "self_atlas_cluster" in get_args(SelfConceptHistoryProducer)
    # The analysis sensor shares the self_study name and must not be killed by accident.
    assert (REPO_ROOT / "services/orion-cortex-exec/app/self_study_analysis.py").exists()
    assert (REPO_ROOT / "orion/cognition/verbs/skills.self_study.analyze.v1.yaml").exists()
    # The table's ORM model stays so the table and its rows stay.
    model = (REPO_ROOT / "services/orion-sql-writer/app/models/self_knowledge_item.py").read_text(encoding="utf-8")
    assert '__tablename__ = "self_knowledge_items"' in model
