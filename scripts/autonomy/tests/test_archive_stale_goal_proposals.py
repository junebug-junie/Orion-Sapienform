from orion.autonomy.goal_archive import build_archive_candidates


def test_build_archive_candidates_keeps_highest_priority_per_drive_origin():
    rows = [
        {"artifact_id": "g1", "drive_origin": "autonomy", "goal_statement_base": "Same text", "priority": 0.9, "proposal_status": "proposed"},
        {"artifact_id": "g2", "drive_origin": "autonomy", "goal_statement_base": "Same text", "priority": 0.3, "proposal_status": "proposed"},
        {"artifact_id": "g3", "drive_origin": "relational", "goal_statement_base": "Other", "priority": 0.5, "proposal_status": "proposed"},
    ]
    to_archive = build_archive_candidates(rows, max_active_per_subject=3, retention_days=30)
    assert set(to_archive) == {"g2"}
