from orion.core.contracts.recall import RecallQueryV1
from orion.schemas.recall_pcr import RecallPhaseV1, RetrievalIntentV1


def test_recall_query_accepts_pcr_fields():
    q = RecallQueryV1(
        fragment="hello",
        profile="chat.continuity.v1",
        recall_phase="continuity",
        retrieval_intent="continuity",
        task_hints={"task_mode": "casual"},
    )
    assert q.recall_phase == "continuity"
    assert q.retrieval_intent == "continuity"
    assert RecallPhaseV1.__args__[0] == "skip"  # type: ignore[attr-defined]
    assert "semantic" in RetrievalIntentV1.__args__  # type: ignore[attr-defined]
