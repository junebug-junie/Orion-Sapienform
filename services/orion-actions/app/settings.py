from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("orion-actions", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(True, alias="ORION_BUS_ENFORCE_CATALOG")

    actions_subscribe_channel: str = Field("orion:collapse:triage", alias="ACTIONS_SUBSCRIBE_CHANNEL")
    actions_subscribe_channels: str = Field(
        "orion:collapse:triage,orion:collapse:stored,orion:actions:trigger:daily_pulse.v1,orion:actions:trigger:daily_metacog.v1,orion:actions:trigger:journal.v1,orion:notify:persistence:request,orion:equilibrium:metacog:trigger",
        alias="ACTIONS_SUBSCRIBE_CHANNELS",
    )
    actions_audit_channel: str = Field("orion:actions:audit", alias="ACTIONS_AUDIT_CHANNEL")
    cortex_request_channel: str = Field("orion:cortex:request", alias="CORTEX_REQUEST_CHANNEL")
    cortex_exec_request_channel: str = Field("orion:cortex:exec:request", alias="CORTEX_EXEC_REQUEST_CHANNEL")
    actions_verb: str = Field("actions.respond_to_juniper_collapse_mirror.v1", alias="ACTIONS_VERB")

    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")

    actions_recipient_group: str = Field("juniper_primary", alias="ACTIONS_RECIPIENT_GROUP")
    actions_session_id: str = Field("collapse_mirror", alias="ACTIONS_SESSION_ID")
    actions_dedupe_ttl_seconds: int = Field(86400, alias="ACTIONS_DEDUPE_TTL_SECONDS")
    actions_notify_dedupe_window_seconds: int = Field(86400, alias="ACTIONS_NOTIFY_DEDUPE_WINDOW_SECONDS")
    actions_max_concurrency: int = Field(2, alias="ACTIONS_MAX_CONCURRENCY")

    actions_recall_profile: str = Field("collapse_mirror.v1", alias="ACTIONS_RECALL_PROFILE")
    # Journal-specific recall: same env keys in .env_example and docker-compose `environment` → _run_journal in main.py
    # (trigger_kind daily_summary+scheduler / metacog_digest / notify_summary).
    actions_journal_scheduler_recall_profile: str = Field(
        "journal.daily.grounded.v1",
        alias="ACTIONS_JOURNAL_SCHEDULER_RECALL_PROFILE",
    )
    actions_journal_metacog_recall_profile: str = Field(
        "journal.daily.grounded.v1",
        alias="ACTIONS_JOURNAL_METACOG_RECALL_PROFILE",
    )
    actions_journal_notify_recall_profile: str = Field(
        "journal.notify.grounded.v1",
        alias="ACTIONS_JOURNAL_NOTIFY_RECALL_PROFILE",
    )
    actions_llm_route: str = Field("chat", alias="ACTIONS_LLM_ROUTE")
    actions_daily_llm_route: str | None = Field(None, alias="ACTIONS_DAILY_LLM_ROUTE")
    actions_journal_llm_route: str | None = Field(None, alias="ACTIONS_JOURNAL_LLM_ROUTE")
    actions_exec_timeout_seconds: float = Field(240.0, alias="ACTIONS_EXEC_TIMEOUT_SECONDS")
    actions_daily_timezone: str = Field("America/Denver", alias="ACTIONS_DAILY_TIMEZONE")
    actions_daily_run_on_startup: bool = Field(False, alias="ACTIONS_DAILY_RUN_ON_STARTUP")
    actions_daily_run_once_date: str | None = Field(None, alias="ACTIONS_DAILY_RUN_ONCE_DATE")

    actions_daily_pulse_enabled: bool = Field(True, alias="ACTIONS_DAILY_PULSE_ENABLED")
    actions_daily_pulse_hour_local: int = Field(8, alias="ACTIONS_DAILY_PULSE_HOUR_LOCAL")
    actions_daily_pulse_minute_local: int = Field(30, alias="ACTIONS_DAILY_PULSE_MINUTE_LOCAL")

    actions_daily_metacog_enabled: bool = Field(True, alias="ACTIONS_DAILY_METACOG_ENABLED")
    actions_daily_metacog_hour_local: int = Field(20, alias="ACTIONS_DAILY_METACOG_HOUR_LOCAL")
    actions_daily_metacog_minute_local: int = Field(15, alias="ACTIONS_DAILY_METACOG_MINUTE_LOCAL")
    actions_async_messages_enabled: bool = Field(True, alias="ACTIONS_ASYNC_MESSAGES_ENABLED")
    actions_daily_async_messages_enabled: bool = Field(True, alias="ACTIONS_DAILY_ASYNC_MESSAGES_ENABLED")
    actions_daily_email_enabled: bool = Field(True, alias="ACTIONS_DAILY_EMAIL_ENABLED")
    actions_pending_attention_enabled: bool = Field(True, alias="ACTIONS_PENDING_ATTENTION_ENABLED")
    actions_preserve_generic_notify_enabled: bool = Field(True, alias="ACTIONS_PRESERVE_GENERIC_NOTIFY_ENABLED")
    actions_world_pulse_enabled: bool = Field(False, alias="ACTIONS_WORLD_PULSE_ENABLED")
    actions_world_pulse_hour_local: int = Field(6, alias="ACTIONS_WORLD_PULSE_HOUR_LOCAL")
    actions_world_pulse_minute_local: int = Field(0, alias="ACTIONS_WORLD_PULSE_MINUTE_LOCAL")
    world_pulse_base_url: str = Field("http://orion-world-pulse:8628", alias="WORLD_PULSE_BASE_URL")

    actions_skills_scheduler_enabled: bool = Field(True, alias="ACTIONS_SKILLS_SCHEDULER_ENABLED")
    actions_skills_run_on_startup: bool = Field(False, alias="ACTIONS_SKILLS_RUN_ON_STARTUP")
    actions_skills_interval_seconds: int = Field(600, alias="ACTIONS_SKILLS_INTERVAL_SECONDS")
    actions_skills_notify_enabled: bool = Field(False, alias="ACTIONS_SKILLS_NOTIFY_ENABLED")
    actions_skills_gpu_mem_threshold: float = Field(0.9, alias="ACTIONS_SKILLS_GPU_MEM_THRESHOLD")
    actions_skills_biometrics_stability_threshold: float = Field(0.3, alias="ACTIONS_SKILLS_BIOMETRICS_STABILITY_THRESHOLD")

    actions_journaling_enabled: bool = Field(True, alias="ACTIONS_JOURNALING_ENABLED")
    actions_journaling_daily_enabled: bool = Field(False, alias="ACTIONS_JOURNALING_DAILY_ENABLED")
    actions_journaling_cooldown_seconds: int = Field(21600, alias="ACTIONS_JOURNALING_COOLDOWN_SECONDS")
    actions_journaling_collapse_dense_only: bool = Field(True, alias="ACTIONS_JOURNALING_COLLAPSE_DENSE_ONLY")
    actions_scheduler_daily_journal_messages_enabled: bool = Field(True, alias="ACTIONS_SCHEDULER_DAILY_JOURNAL_MESSAGES_ENABLED")
    actions_scheduler_daily_journal_email_enabled: bool = Field(True, alias="ACTIONS_SCHEDULER_DAILY_JOURNAL_EMAIL_ENABLED")
    actions_journal_session_id: str = Field("orion_journal", alias="ACTIONS_JOURNAL_SESSION_ID")
    actions_journal_author: str = Field("orion", alias="ACTIONS_JOURNAL_AUTHOR")
    actions_journal_write_channel: str = Field("orion:journal:write", alias="ACTIONS_JOURNAL_WRITE_CHANNEL")
    actions_journal_created_channel: str = Field("orion:journal:created", alias="ACTIONS_JOURNAL_CREATED_CHANNEL")
    actions_journal_post_persist_notify_enabled: bool = Field(True, alias="ACTIONS_JOURNAL_POST_PERSIST_NOTIFY_ENABLED")

    actions_workflow_schedule_store_path: str = Field("/tmp/orion-actions/workflow_schedules.json", alias="ACTIONS_WORKFLOW_SCHEDULE_STORE_PATH")
    actions_scheduler_cursor_store_path: str = Field("", alias="ACTIONS_SCHEDULER_CURSOR_STORE_PATH")
    actions_workflow_schedule_claim_batch_size: int = Field(10, alias="ACTIONS_WORKFLOW_SCHEDULE_CLAIM_BATCH_SIZE")
    actions_workflow_attention_overdue_min_seconds: int = Field(3600, alias="ACTIONS_WORKFLOW_ATTENTION_OVERDUE_MIN_SECONDS")
    actions_workflow_attention_reminder_cooldown_seconds: int = Field(21600, alias="ACTIONS_WORKFLOW_ATTENTION_REMINDER_COOLDOWN_SECONDS")
    actions_self_experiments_enabled: bool = Field(False, alias="ACTIONS_SELF_EXPERIMENTS_ENABLED")
    actions_self_experiments_url: str = Field("http://orion-self-experiments:7172", alias="ACTIONS_SELF_EXPERIMENTS_URL")
    actions_self_experiments_timeout_seconds: float = Field(8.0, alias="ACTIONS_SELF_EXPERIMENTS_TIMEOUT_SECONDS")

    port: int = Field(7160, alias="ACTIONS_PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True

    @field_validator(
        "actions_journal_scheduler_recall_profile",
        "actions_journal_metacog_recall_profile",
        "actions_journal_notify_recall_profile",
        mode="before",
    )
    @classmethod
    def _coerce_blank_journal_recall_profiles(cls, value: object, info: ValidationInfo) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return cls.model_fields[info.field_name].default
        return value

    @field_validator(
        "actions_workflow_schedule_claim_batch_size",
        "actions_workflow_attention_overdue_min_seconds",
        "actions_workflow_attention_reminder_cooldown_seconds",
        mode="before",
    )
    @classmethod
    def _coerce_blank_workflow_ints(cls, value: object, info: ValidationInfo) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return cls.model_fields[info.field_name].default
        return value

    def subscribe_patterns(self) -> list[str]:
        raw = self.actions_subscribe_channels or self.actions_subscribe_channel
        values = [v.strip() for v in str(raw).split(",") if v.strip()]
        if self.actions_subscribe_channel and self.actions_subscribe_channel not in values:
            values.append(self.actions_subscribe_channel)
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
