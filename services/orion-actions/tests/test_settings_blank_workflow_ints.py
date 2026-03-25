from __future__ import annotations

from app.settings import Settings


def test_blank_workflow_int_env_values_fall_back_to_defaults() -> None:
    cfg = Settings(
        ACTIONS_WORKFLOW_SCHEDULE_CLAIM_BATCH_SIZE="",
        ACTIONS_WORKFLOW_ATTENTION_OVERDUE_MIN_SECONDS="",
        ACTIONS_WORKFLOW_ATTENTION_REMINDER_COOLDOWN_SECONDS="",
    )

    assert cfg.actions_workflow_schedule_claim_batch_size == 10
    assert cfg.actions_workflow_attention_overdue_min_seconds == 3600
    assert cfg.actions_workflow_attention_reminder_cooldown_seconds == 21600
