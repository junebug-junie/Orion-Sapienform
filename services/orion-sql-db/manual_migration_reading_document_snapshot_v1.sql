-- Internal-document reading sources (orion/world_pulse_read/documents.py).
-- Additive; Hub also runs this at startup via ensure_seed_queue_schema.
CREATE TABLE IF NOT EXISTS reading_document_snapshot (
    sha256 text PRIMARY KEY CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    content text NOT NULL,
    content_chars integer NOT NULL CHECK (content_chars > 0),
    first_source text NOT NULL,
    captured_at timestamptz NOT NULL DEFAULT now()
);
