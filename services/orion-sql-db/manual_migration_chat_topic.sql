-- Create chat_topic join table for Topic Rail
CREATE TABLE IF NOT EXISTS chat_topic (
    chat_id           VARCHAR NOT NULL
  , correlation_id    VARCHAR
  , trace_id          VARCHAR
  , session_id        VARCHAR
  , topic_id          INTEGER
  , topic_label       TEXT
  , topic_keywords    JSONB
  , topic_confidence  DOUBLE PRECISION
  , model_version     VARCHAR NOT NULL
  , created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
  , UNIQUE (chat_id, model_version)
);

CREATE INDEX IF NOT EXISTS ix_chat_topic_model_version ON chat_topic (model_version);
CREATE INDEX IF NOT EXISTS ix_chat_topic_correlation_id ON chat_topic (correlation_id);
CREATE INDEX IF NOT EXISTS ix_chat_topic_trace_id ON chat_topic (trace_id);
CREATE INDEX IF NOT EXISTS ix_chat_topic_session_id ON chat_topic (session_id);
CREATE INDEX IF NOT EXISTS ix_chat_topic_topic_id ON chat_topic (topic_id);
CREATE INDEX IF NOT EXISTS ix_chat_topic_created_at ON chat_topic (created_at);
