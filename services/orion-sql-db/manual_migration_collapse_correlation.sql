-- Add correlation_id to collapse_mirror
ALTER TABLE collapse_mirror ADD COLUMN correlation_id VARCHAR;
CREATE INDEX ix_collapse_mirror_correlation_id ON collapse_mirror (correlation_id);

-- Add correlation_id to chat_history_log
ALTER TABLE chat_history_log ADD COLUMN correlation_id VARCHAR;
CREATE INDEX ix_chat_history_log_correlation_id ON chat_history_log (correlation_id);
