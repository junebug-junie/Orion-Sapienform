"""Shared Mind contract constants (no runtime / IO)."""

# Default upper bound for serialized universe snapshot (bytes); operators may raise via service settings.
MIND_SNAPSHOT_MAX_BYTES_DEFAULT: int = 512_000

# Schema ids (wire + registry)
MIND_RUN_REQUEST_SCHEMA_ID = "mind.run.request.v1"
MIND_RUN_RESULT_SCHEMA_ID = "mind.run.result.v1"
MIND_RUN_ARTIFACT_SCHEMA_ID = "mind.run.artifact.v1"
