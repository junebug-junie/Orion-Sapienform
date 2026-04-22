from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


CHAT_GPT_LOG_TURN_KIND = "chat.gpt.log.v1"
CHAT_GPT_MESSAGE_KIND = "chat.gpt.message.v1"
CHAT_GPT_IMPORT_RUN_KIND = "chat.gpt.import.run.v1"
CHAT_GPT_CONVERSATION_KIND = "chat.gpt.conversation.v1"
CHAT_GPT_EXAMPLE_KIND = "chat.gpt.example.v1"


class ChatGptMessageV1(BaseModel):
    """ChatGPT imported message payload for isolated SQL + vector fanout."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    message_id: str = Field(validation_alias=AliasChoices("message_id", "id"))
    session_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "conversation_id"),
    )
    role: str = "user"
    speaker: Optional[str] = None
    content: str
    timestamp: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    source_message_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    child_message_ids: list[str] = Field(default_factory=list)
    content_type: Optional[str] = None
    content_blocks: list[Dict[str, Any]] = Field(default_factory=list)
    attachments: list[Dict[str, Any]] = Field(default_factory=list)
    shared_conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatGptLogTurnV1(BaseModel):
    """Turn-level ChatGPT import row (prompt + response) for `chat_gpt_log`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: Optional[str] = Field(default=None, description="Primary identifier for the turn row")
    correlation_id: Optional[str] = Field(default=None, description="Trace/correlation identifier")
    source: str = Field(..., description="Source label (e.g. chatgpt_import)")
    prompt: str = Field(..., description="User prompt")
    response: str = Field(..., description="Assistant response")
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    spark_meta: Optional[Dict[str, Any]] = Field(default=None)


class ChatGptImportRunV1(BaseModel):
    """Import-run level provenance and counters for ChatGPT export ingestion."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    import_run_id: str
    source_artifact_path: Optional[str] = None
    source_artifact_sha256: Optional[str] = None
    source_artifact_bytes: Optional[int] = None
    source_artifact_mtime: Optional[str] = None
    importer_name: str
    importer_version: str
    import_mode: str = "incremental"
    include_branches: bool = False
    include_system: bool = False
    force_full: bool = False
    dry_run: bool = False
    state_file: Optional[str] = None
    conversation_count: int = 0
    message_count: int = 0
    turn_count: int = 0
    example_count: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatGptConversationV1(BaseModel):
    """Conversation-level ChatGPT export record preserving graph/provenance metadata."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    conversation_id: str = Field(validation_alias=AliasChoices("conversation_id", "id"))
    import_run_id: str
    session_id: Optional[str] = None
    title: Optional[str] = None
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    current_node_id: Optional[str] = None
    message_count: int = 0
    turn_count: int = 0
    branch_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatGptDerivedExampleV1(BaseModel):
    """Prompt/response pair derived from ChatGPT export for later curation/training."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    example_id: str
    import_run_id: str
    conversation_id: str
    session_id: Optional[str] = None
    user_message_id: Optional[str] = None
    assistant_message_id: Optional[str] = None
    turn_id: Optional[str] = None
    prompt: str
    response: str
    prompt_role: str = "user"
    response_role: str = "assistant"
    prompt_timestamp: Optional[str] = None
    response_timestamp: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
