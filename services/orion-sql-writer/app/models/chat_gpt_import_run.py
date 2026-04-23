from sqlalchemy import BigInteger, Boolean, Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class ChatGptImportRunSQL(Base):
    __tablename__ = "chat_gpt_import_run"

    import_run_id = Column(String, primary_key=True)
    source_artifact_path = Column(String, nullable=True)
    source_artifact_sha256 = Column(String, index=True, nullable=True)
    source_artifact_bytes = Column(BigInteger, nullable=True)
    source_artifact_mtime = Column(DateTime(timezone=True), nullable=True)
    importer_name = Column(String, nullable=False)
    importer_version = Column(String, nullable=False)
    import_mode = Column(String, nullable=False, default="incremental")
    include_branches = Column(Boolean, nullable=False, default=False)
    include_system = Column(Boolean, nullable=False, default=False)
    force_full = Column(Boolean, nullable=False, default=False)
    dry_run = Column(Boolean, nullable=False, default=False)
    state_file = Column(String, nullable=True)
    conversation_count = Column(Integer, nullable=False, default=0)
    message_count = Column(Integer, nullable=False, default=0)
    turn_count = Column(Integer, nullable=False, default=0)
    example_count = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    meta = Column("metadata", JSONB, nullable=True)
