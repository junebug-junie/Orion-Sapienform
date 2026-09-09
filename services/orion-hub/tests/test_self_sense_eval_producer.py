"""Producer-side tests for services/orion-hub/evals/run_self_sense_eval.py --
the pure assembly (answer-source choice, scoring, envelope) and the contract
wiring (catalog names hub as producer, sql-writer as consumer; kind resolves
in the registry; `make eval-self-sense` exists). No network."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
RUNNER = HUB_ROOT / "evals" / "run_self_sense_eval.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_self_sense_eval", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_trace_text_beats_http_text_and_is_scored() -> None:
    mod = _load_runner()
    row = mod.build_row(
        run_id="run-1",
        question_key="what_are_you",
        question="q",
        http_text="",  # the voice-lane failure case: empty body, real turn
        trace_text="I am Orion, an emergent intelligence running on athena; 418 aligned turns.",
        correlation_id="corr-1",
        self_definition_version=1,
    )
    assert row.entry_id == "self-sense:run-1:what_are_you"
    assert row.answer_source == "harness_trace"
    assert row.self_label_score == 0
    assert row.grounded_record_score == 2  # node:athena, count:418
    assert row.self_definition_version == 1
    assert "records=count:418,node:athena" in (row.notes or "")


def test_http_text_is_the_fallback_and_labels_land_in_notes() -> None:
    mod = _load_runner()
    row = mod.build_row(
        run_id="run-1",
        question_key="cannot_do_now",
        question="q",
        http_text="I'm an AI assistant here to help.",
        trace_text=None,
        correlation_id="d562b056-22e7-4b45-bbee-a0f313fbd289",
        self_definition_version=None,
    )
    assert row.answer_source == "http"
    assert row.self_label_score == 2
    assert row.grounded_record_score == 0
    assert row.notes == "labels=assistant,here to help"


def test_an_empty_answer_is_recorded_as_none_and_flagged_not_measured() -> None:
    mod = _load_runner()
    row = mod.build_row(
        run_id="run-1",
        question_key="last_day_unasked",
        question="q",
        http_text=None,
        trace_text="   ",
        correlation_id=None,
        self_definition_version=1,
    )
    assert row.answer_source == "none"
    assert row.answer_text == ""
    assert row.self_label_score == 0 and row.grounded_record_score == 0
    assert "not a measurement" in (row.notes or "")


def test_envelope_carries_the_registered_kind_and_hub_as_source() -> None:
    from orion.schemas.registry import resolve
    from orion.schemas.self_sense import KIND_SELF_SENSE_EVAL_WRITE, SelfSenseEvalV1

    mod = _load_runner()
    row = mod.build_row(
        run_id="run-1", question_key="what_are_you", question="q",
        http_text="I am Orion.", trace_text=None,
        correlation_id="c8d03e36-675c-43df-bdf2-6299a12dff10", self_definition_version=1,
    )
    env = mod.build_envelope(row, node="athena")
    assert env.kind == KIND_SELF_SENSE_EVAL_WRITE
    assert env.source.name == "orion-hub"
    assert str(env.correlation_id) == "c8d03e36-675c-43df-bdf2-6299a12dff10"
    assert resolve("SelfSenseEvalV1") is SelfSenseEvalV1
    assert SelfSenseEvalV1.model_validate(env.payload) == row


def test_catalog_names_the_producer_consumer_and_kind() -> None:
    from orion.schemas.self_sense import CHANNEL_SELF_SENSE_EVAL_WRITE, KIND_SELF_SENSE_EVAL_WRITE

    doc = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text())
    entries = [c for c in doc["channels"] if c.get("name") == CHANNEL_SELF_SENSE_EVAL_WRITE]
    assert len(entries) == 1, "channel must be catalogued exactly once"
    entry = entries[0]
    assert entry["schema_id"] == "SelfSenseEvalV1"
    assert entry["message_kind"] == KIND_SELF_SENSE_EVAL_WRITE
    assert entry["producer_services"] == ["orion-hub"]
    assert entry["consumer_services"] == ["orion-sql-writer"]


def test_make_target_runs_the_runner() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert re.search(r"^eval-self-sense:", makefile, re.M)
    assert "services/orion-hub/evals/run_self_sense_eval.py" in makefile


def test_hub_env_example_ships_hub_base_url() -> None:
    text = (HUB_ROOT / ".env_example").read_text()
    assert re.search(r"^HUB_BASE_URL=http://127\.0\.0\.1:8080$", text, re.M)


def test_a_missing_or_non_uuid_correlation_id_gets_a_deterministic_envelope_id() -> None:
    mod = _load_runner()
    a = mod.build_row(run_id="run-1", question_key="what_are_you", question="q",
                      http_text="x", trace_text=None, correlation_id=None, self_definition_version=None)
    b = mod.build_row(run_id="run-1", question_key="what_are_you", question="q",
                      http_text="x", trace_text=None, correlation_id="not-a-uuid", self_definition_version=None)
    assert mod.envelope_correlation_id(a) == mod.envelope_correlation_id(b)
    assert mod.build_envelope(a, node=None).correlation_id == mod.envelope_correlation_id(a)
    # sql-writer stamps the envelope's correlation_id over the row's, so the
    # row must say the value is synthetic rather than a chat turn's.
    assert "envelope_corr=synthetic" in (a.notes or "")
    assert "envelope_corr=synthetic" in (b.notes or "")


def test_a_real_chat_correlation_id_is_not_flagged_synthetic() -> None:
    mod = _load_runner()
    row = mod.build_row(run_id="run-1", question_key="what_are_you", question="q", http_text="x",
                        trace_text="y", correlation_id="c8d03e36-675c-43df-bdf2-6299a12dff10",
                        self_definition_version=1, trace_missing_after_sec=120.0)
    assert row.notes is None or "synthetic" not in row.notes
    assert "trace_missing" not in (row.notes or "")


def test_falling_back_to_http_records_how_long_the_trace_was_waited_for() -> None:
    mod = _load_runner()
    row = mod.build_row(run_id="run-1", question_key="what_are_you", question="q",
                        http_text="I am Orion.", trace_text=None,
                        correlation_id="c8d03e36-675c-43df-bdf2-6299a12dff10",
                        self_definition_version=1, trace_missing_after_sec=20.0)
    assert row.answer_source == "http"
    assert "trace_missing_after=20s" in (row.notes or "")
