from copy import deepcopy

import pytest

from orion.world_pulse_read.verify import completion_gaps


def completed():
    row = {
        "seed_id": "reading:test", "url": "https://arxiv.org/abs/2310.19279",
        "status": "done", "stage2_status": "done", "trace_id": "s1",
        "stage2_trace_id": "s2", "landing_at": "2026-09-26T00:00:00Z",
        "handoff_json": {
            "seed_ref": {"seed_id": "reading:test", "kind": "reading", "run_id": "r", "url": "https://arxiv.org/abs/2310.19279"},
            "what_i_learned": "The paper reports a result that needs replication.",
            "trace_id": "s1", "created_at": "2026-09-26T00:00:00Z",
            "read_evidence": [{"url": "https://arxiv.org/abs/2310.19279", "tool_name": "WebFetch", "content_chars": 1500}],
        },
        "stage2_result_json": {
            "seed_id": "reading:test", "summary": "The result remains source-attributed and uncertain.",
            "trace_id": "s2", "created_at": "2026-09-26T00:00:00Z",
        },
    }
    journals = {"world_pulse_read:s1": "Stage 1 learning", "world_pulse_read_stage2:s2": "Stage 2 reflection"}
    return row, journals


def test_complete_requires_artifacts_and_both_stored_journals():
    row, journals = completed()
    assert completion_gaps(row, journals) == []


@pytest.mark.parametrize("mutation,expected", [
    (lambda r: r.update(status="pending"), "stage1_not_done"),
    (lambda r: r.update(stage2_status="failed"), "stage2_not_done"),
    (lambda r: r.update(landing_at=None), "landing_not_confirmed"),
    (lambda r: r["handoff_json"].update(read_evidence=[]), "missing_source_fetch_evidence"),
    (lambda r: r["handoff_json"].update(what_i_learned=" "), "missing_or_invalid_handoff"),
    (lambda r: r["stage2_result_json"].update(summary=" "), "missing_or_invalid_stage2_result"),
    (lambda r: r["stage2_result_json"].update(trace_id="unrelated"), "stage2_provenance_mismatch"),
    (lambda r: r.update(handoff_json=None), "missing_or_invalid_handoff"),
])
def test_done_flags_alone_are_not_a_pass(mutation, expected):
    row, journals = completed()
    row = deepcopy(row)
    mutation(row)
    assert expected in completion_gaps(row, journals)


def test_empty_or_missing_journal_is_not_a_landing():
    row, journals = completed()
    journals["world_pulse_read:s1"] = " "
    del journals["world_pulse_read_stage2:s2"]
    assert set(completion_gaps(row, journals)) == {
        "missing_world_pulse_read_journal_body", "missing_world_pulse_read_stage2_journal_body",
    }
