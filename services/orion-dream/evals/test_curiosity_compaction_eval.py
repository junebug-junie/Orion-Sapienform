"""Offline chronology/provenance replay. This does not measure live LLM fidelity."""
import json
from datetime import datetime, timedelta, timezone

from orion.curiosity.compaction import batches, next_prompt, record_answer, source_material


def test_month_replay_keeps_early_contradiction_late_revision_and_open_question():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    records = [
        ("z-early", 1, "September 2: I suspected delivery failures came from gateway timeouts; I had not established a cause."),
        ("m-middle", 15, "September 16: New records show a delivery gap even when gateway requests succeed, weakening the timeout explanation."),
        ("a-late", 29, "September 30: The delivery cause remains open. Next I need to compare consumption timestamps with successful publications."),
    ]
    journals = [{"entry_id": key, "source_ref": "curiosity:" + key, "created_at": start + timedelta(days=day), "body": body}
                for key, day, body in records]
    material, sources = source_material(journals, [])
    row = {"cadence": "monthly", "window_start": start, "window_end": start + timedelta(days=30), "status": "pending",
           "state": {"phase": "topics", "sources": sources, "batches": batches(material), "partials": []}}
    prompt = next_prompt(row)
    assert prompt.index("September 2:") < prompt.index("September 16:") < prompt.index("September 30:")
    assert "Do not turn unresolved questions into answers" in prompt
    # A reviewed hypothetical topic answer; evaluate propagation, not generation quality.
    answer = {"topics": [{"topic": "Delivery reliability", "body":
        "September 2's timeout explanation was only a suspicion. September 16's successful requests with missing deliveries weakened it. On September 30 the cause remained open; compare consumption against publication next.",
        "refs": ["s0", "s1", "s2"]}]}
    row = record_answer(row, json.dumps(answer))
    overall_prompt = next_prompt(row)
    assert "only a suspicion" in overall_prompt and "remained open" in overall_prompt
    assert row["state"]["topics"][0]["sources"] == ["z-early", "m-middle", "a-late"]
    assert row["state"]["phase"] == "overall" and row["status"] == "pending"
