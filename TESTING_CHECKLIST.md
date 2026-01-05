# Testing Checklist

## 1. Verify Chat History Log Integrity

**Goal:** Ensure `chat_history_log` rows are created with correct `correlation_id` and rich `spark_meta`.

**Steps:**
1.  Generate a chat turn via Hub (UI or WebSocket).
2.  Inspect `chat_history_log` table (via psql or logs if not accessible).

**SQL Query:**
```sql
SELECT id, correlation_id, spark_meta FROM chat_history_log WHERE correlation_id IS NOT NULL ORDER BY created_at DESC LIMIT 1;
```

**Expected Result:**
- `correlation_id`: Must be a valid UUID (not null).
- `spark_meta`: Must contain rich keys (e.g., `spark_event_id`, `spark_source`, `phi_after`, etc.) and NOT just `{"mode": "brain", "use_recall": ...}`.
  - Note: Initial insert might be `null` or empty, but shortly after (ms/sec), the side-effect update should populate it. Wait a second if checking immediately.

## 2. Verify Trace ID Removal

**Goal:** Ensure `trace_id` column is gone (if migration ran) or ignored.

**SQL Query:**
```sql
-- This should fail if column is dropped, or return nothing useful if ignored code-side.
SELECT * FROM chat_history_log LIMIT 1;
```
Check that `trace_id` is NOT in the columns list.

## 3. Verify Log Lines

**Service: `orion-hub`**
- Look for `Transcript: ...`
- Look for `Routing to mode: ...`

**Service: `orion-sql-writer`**
- Look for `Written chat.history -> chat_history_log`
- Ensure NO warning about `Could not back-populate chat log spark_meta`.
- Look for `Written spark.telemetry -> spark_telemetry` (this triggers the update).

## 4. Verify Payload (Code Inspection)

**Hub Payload:**
Check `orion-hub` logs or code to ensure `chat_log_payload` sent to `chat.history` does NOT contain `trace_id` and has `spark_meta: None`.

**Gateway Response:**
Ensure `orion-cortex-gateway` is returning `correlation_id` in `CortexChatResult`. (Implicitly verified if Hub gets it).
