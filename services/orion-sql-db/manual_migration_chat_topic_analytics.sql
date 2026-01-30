-- Create chat_topic_summary table
CREATE TABLE IF NOT EXISTS chat_topic_summary (
    model_version    VARCHAR NOT NULL
  , window_start     TIMESTAMPTZ NOT NULL
  , window_end       TIMESTAMPTZ NOT NULL
  , topic_id         INTEGER NOT NULL
  , topic_label      TEXT
  , topic_keywords   JSONB
  , doc_count        INTEGER NOT NULL
  , pct_of_window    DOUBLE PRECISION NOT NULL
  , outlier_count    INTEGER
  , outlier_pct      DOUBLE PRECISION
  , created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
  , UNIQUE (model_version, window_start, window_end, topic_id)
);

CREATE INDEX IF NOT EXISTS ix_chat_topic_summary_model_version ON chat_topic_summary (model_version);
CREATE INDEX IF NOT EXISTS ix_chat_topic_summary_window_start ON chat_topic_summary (window_start);
CREATE INDEX IF NOT EXISTS ix_chat_topic_summary_window_end ON chat_topic_summary (window_end);
CREATE INDEX IF NOT EXISTS ix_chat_topic_summary_topic_id ON chat_topic_summary (topic_id);
CREATE INDEX IF NOT EXISTS ix_chat_topic_summary_created_at ON chat_topic_summary (created_at);

-- Create chat_topic_session_drift table
CREATE TABLE IF NOT EXISTS chat_topic_session_drift (
    model_version     VARCHAR NOT NULL
  , window_start      TIMESTAMPTZ NOT NULL
  , window_end        TIMESTAMPTZ NOT NULL
  , session_id        VARCHAR NOT NULL
  , turns             INTEGER NOT NULL
  , unique_topics     INTEGER NOT NULL
  , entropy           DOUBLE PRECISION NOT NULL
  , switch_rate       DOUBLE PRECISION NOT NULL
  , dominant_topic_id INTEGER
  , dominant_pct      DOUBLE PRECISION
  , created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
  , UNIQUE (model_version, window_start, window_end, session_id)
);

CREATE INDEX IF NOT EXISTS ix_chat_topic_drift_model_version ON chat_topic_session_drift (model_version);
CREATE INDEX IF NOT EXISTS ix_chat_topic_drift_window_start ON chat_topic_session_drift (window_start);
CREATE INDEX IF NOT EXISTS ix_chat_topic_drift_window_end ON chat_topic_session_drift (window_end);
CREATE INDEX IF NOT EXISTS ix_chat_topic_drift_session_id ON chat_topic_session_drift (session_id);
CREATE INDEX IF NOT EXISTS ix_chat_topic_drift_created_at ON chat_topic_session_drift (created_at);
