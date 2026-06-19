from orion.memory_graph.draft_repository import insert_pending_draft


def test_insert_pending_draft_sql_shape():
    source = open(insert_pending_draft.__code__.co_filename, encoding="utf-8").read()
    assert "INSERT INTO memory_graph_suggest_drafts" in source
    assert "ON CONFLICT (memory_window_id)" in source
    assert "RETURNING draft_id" in source
    assert "pending_review" in source
    assert "async def list_consolidation_drafts" in source
    assert "async def get_consolidation_draft" in source
    assert "async def update_consolidation_draft_status" in source
    assert "status = 'pending_review'" in source
