"""Regression gates for the 2026-10-30 fold of self-study into curiosity.

Proposal: docs/superpowers/specs/2026-10-29-self-study-into-curiosity-proposal.md (PR #2972).

(a) "reflect on yourself" still resolves, and now routes to a curiosity self-inquiry request.
(b) The kept analysis verb is tier 0 `autonomic` and the `self_study` family is gone.
    (Its source_ref prefixes are pinned in tests/test_self_study_analysis_runner.py.)
(c) No retired verb, workflow, service, channel, env key or module is referenced anywhere in
    code or config. Kill means kill: a fallback to the retired thing fails here.
(d) The self-inquiry ledger tells Orion what was retired, so a frozen count is not read as a fault.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from orion.cognition.verb_physiology import parse_verb_physiology
from orion.cognition.workflows.registry import get_workflow_definition, resolve_user_workflow_invocation
from orion.curiosity.self_inquiry import LEDGER_RETIREMENTS, LedgerRow, format_ledger
from orion.schemas.curiosity_self_inquiry_request import (
    SELF_INQUIRY_REQUEST_CHANNEL,
    SELF_INQUIRY_REQUEST_KIND,
    CuriositySelfInquiryRequestV1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- (a) the phrase routes to self-inquiry -----------------------------------------------------


def test_reflect_on_yourself_still_resolves_to_self_review() -> None:
    for phrase in ("reflect on yourself", "Hey Orion, run a self review please", "run self study"):
        match = resolve_user_workflow_invocation(phrase)
        assert match is not None and match.workflow_id == "self_review", phrase


def test_self_review_is_a_self_inquiry_request_not_a_self_study_verb() -> None:
    workflow = get_workflow_definition("self_review")
    assert workflow is not None
    assert [step.adapter for step in workflow.steps] == [f"bus:{SELF_INQUIRY_REQUEST_CHANNEL}"]
    assert all(not step.adapter.startswith("verb:") for step in workflow.steps)


def test_self_inquiry_request_channel_and_schema_are_registered() -> None:
    from orion.schemas.registry import SCHEMA_REGISTRY, resolve

    channels = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8"))
    [entry] = [c for c in channels["channels"] if c["name"] == SELF_INQUIRY_REQUEST_CHANNEL]
    assert entry["schema_id"] == "CuriositySelfInquiryRequestV1"
    assert entry["message_kind"] == SELF_INQUIRY_REQUEST_KIND
    assert entry["producer_services"] == ["orion-cortex-orch"]
    assert entry["consumer_services"] == ["orion-hub"]
    assert SCHEMA_REGISTRY["CuriositySelfInquiryRequestV1"].kind == SELF_INQUIRY_REQUEST_KIND
    assert resolve("CuriositySelfInquiryRequestV1") is CuriositySelfInquiryRequestV1


# --- (b) the analysis verb is tier 0 autonomic ------------------------------------------------


def test_analysis_verb_is_tier0_autonomic_and_self_study_family_is_gone() -> None:
    data = yaml.safe_load(
        (REPO_ROOT / "orion" / "cognition" / "verbs" / "skills.self_study.analyze.v1.yaml").read_text(encoding="utf-8")
    )
    block = parse_verb_physiology(data)
    assert block is not None
    assert (block.tier, block.family, list(block.lanes), block.deferrable_window_s) == (0, "autonomic", [], 0)

    families = yaml.safe_load((REPO_ROOT / "config" / "physiology" / "families.yaml").read_text(encoding="utf-8"))
    assert "self_study" not in families["families"]
    contract = (REPO_ROOT / "config" / "physiology" / "contract.yaml").read_text(encoding="utf-8")
    assert not re.search(r"^\s*self_study\s*:", contract, re.MULTILINE)


# --- (c) nothing retired is referenced ---------------------------------------------------------

# Word-boundary patterns, so the kept pieces (run_self_repo_inspect, the analysis verb,
# orion-thought's load_latest_self_study_reflection reader) do not match.
RETIRED_PATTERNS = (
    r"\bself_study\.reflect\b",
    r"\bself_concept_reflect\b",
    r"\bself_concept_induce\b",
    r"\bself_retrieve\b",
    r"\bself_repo_inspect\b",
    r"\bSelfStudyRetrieve\w*",
    r"\bSelfStudyConsumer\w*",
    r"\bSelfStudyHarness\w*",
    r"\bSelfStudyEnrichment\w*",
    r"\bSelfInducedConceptV1\b",
    r"\bSelfReflectiveFindingV1\b",
    r"orion:self_study:enrichment",
    r"orion-self-study-enrichment",
    r"\bself_study_enrichment",
    r"\bSELF_STUDY_(?:REFLECT|ENRICHMENT|STRUCTURAL_MASS|GRAPH)\w*",
    r"\bself_study_(?:harness|policy)\b",
    r"\b(?:admitted_)?reflect_graph\b",
    r"\bself_study_reflect(?!ion)\w*",
    r"\brun_self_concept_\w+",
    r"\bSELF_STUDY_REFLECT_NODES\b",
    r"\bDURABLE_RUNS_REFLECT_\w+",
)
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".j2", ".js", ".html", ".sh", ".toml", ".sql", ".cfg", ".ini", ".env_example"}
# Data snapshots of the live body/bus taken before the retirement (they record what WAS true),
# and this file, which has to name what it forbids.
ALLOWED = {
    "tests/test_self_study_fold_into_curiosity.py",
    # Asserts the retired entry points are gone, so it names them.
    "services/orion-cortex-exec/tests/test_self_study_pass1.py",
    # Asserts a long-dead env key stays out of .env_example.
    "services/orion-cortex-exec/tests/test_chat_stance_no_self_study_producer.py",
    "orion/body_atlas/evals/fixtures/whole_body_2026-10-29.json",
    "orion/body_atlas/evals/fixtures/bus_graph_2026-10-23.json",
}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "services", "orion", "config", "scripts", "tests", "Makefile", ".github"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def test_no_retired_self_study_piece_is_referenced_in_code_or_config() -> None:
    combined = re.compile("|".join(RETIRED_PATTERNS))
    hits: list[str] = []
    for rel in _tracked_files():
        if rel in ALLOWED:
            continue
        path = REPO_ROOT / rel
        suffix = ".env_example" if path.name == ".env_example" else path.suffix
        if suffix not in SCANNED_SUFFIXES and path.name not in {"Makefile", "post-commit", "Dockerfile"}:
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
    assert not hits, "retired self-study pieces still referenced:\n" + "\n".join(hits[:50])


def test_retired_service_directory_and_rebuild_entries_are_gone() -> None:
    assert not (REPO_ROOT / "services" / "orion-self-study-enrichment").exists()
    for rel in ("scripts/service_rebuild_paths.yaml", "scripts/rebuild_services_from_git_diff.sh"):
        assert "self-study-enrichment" not in (REPO_ROOT / rel).read_text(encoding="utf-8"), rel


# --- (d) the ledger tells Orion what was retired ---------------------------------------------


def test_ledger_names_the_retirement_under_the_frozen_tables() -> None:
    rows = [
        LedgerRow("dreams", 17, "2026-09-06T08:27:32+00:00"),
        LedgerRow("self_concept_history", 9000, "2026-10-30T06:45:00+00:00"),
        LedgerRow("self_knowledge_items", 61000, "2026-10-29T22:24:00+00:00"),
    ]
    lines = format_ledger(rows)
    text = "\n".join(lines)
    assert "retired 2026-10-30" in text and "layer3_reflect" in text
    assert "feeds Self Atlas only" in text
    # A table with no retirement gets exactly one line, as before.
    assert sum(1 for line in lines if "dreams" in line) == 1
    # Every annotated table is one the ledger actually reads.
    from orion.curiosity.self_inquiry import LEDGER_TS_COLUMNS

    assert set(LEDGER_RETIREMENTS) <= set(LEDGER_TS_COLUMNS)
