# Orion SQL Writer

The **Orion SQL Writer** is a foundational microservice in the Orion‑Sapienform architecture. It serves as the persistence layer, subscribing to ephemeral messages on the Orion Bus (Redis) and consolidating them into long‑term storage in PostgreSQL.

It is responsible for persisting:

- Memory formation
- Dream logging
- Biometrics telemetry
- Chat history archival

---

## 🏗 Architecture

The service follows a strict **Subscribe → Validate → Persist** flow:

1. **Ingest**  
   Listens to configured Redis channels via `OrionBus`.

2. **Map**  
   Resolves the channel name to a specific database table strategy via `settings.py`.

3. **Filter**  
   Strips metadata (like `trace_id` or `source`) that is not part of the strict Pydantic data model.

4. **Validate**  
   Uses Pydantic schemas to normalize data (e.g. setting default timestamps, parsing JSON strings).

5. **Persist**  
   Uses SQLAlchemy to `INSERT` (new records) or `UPDATE` (existing records) in PostgreSQL.

---

## ⚙️ Configuration

The service is configured via environment variables (typically loaded from a `.env` file).

### Core Environment Variables

| Variable           | Description                     | Example                                                |
|--------------------|---------------------------------|--------------------------------------------------------|
| `DATABASE_URL`     | PostgreSQL connection string    | `postgresql://user:pass@localhost:5432/orion`          |
| `ORION_BUS_URL`    | Redis connection string         | `redis://localhost:6379/0`                             |
| `ORION_BUS_ENABLED`| Master switch for the bus       | `True`                                                 |
| `LOG_LEVEL`        | Logging verbosity               | `INFO`                                                 |

---

## 🔗 Channel Mapping

Mappings between Redis channels and SQL tables are defined in `app/settings.py`.

| Channel Name           | Table Name          | Pydantic Schema     | Description                                      |
|------------------------|---------------------|---------------------|--------------------------------------------------|
| `orion:dream:trigger`  | `dreams`            | `DreamInput`        | Narrative synthesis from the Brain (LLM).        |
| `chat_history_log`     | `chat_history_log`  | `ChatHistoryInput`  | Raw conversation logs.                           |
| `orion_biometrics`     | `orion_biometrics`  | `BiometricsInput`   | Hardware telemetry (GPU/CPU stats).              |
| `enrichment_channel`   | `collapse_enrichment` | `EnrichmentInput`  | Semantic metadata and tags.                      |

---

## 📦 Data Handling Strategy

### Insert vs. Update Logic

The writer implements specific logic to handle data integrity, particularly for complex models like **Dreams**:

1. **Input Filtering**  
   Incoming payloads often contain bus metadata (e.g. `trace_id`, `source`) that are not present in the database schema. The writer dynamically filters incoming dictionaries against `Model.model_fields.keys()` to prevent `ValidationError`.

2. **Normalization**  
   Incoming messages are normalized via Pydantic. For example, if `dream_date` is missing, it defaults to **today**.

3. **Persistence**
   - **INSERT**: Uses `model_dump(mode="json")`. All fields are written, applying model defaults (e.g. `created_at`).
   - **UPDATE**: Uses `model_dump(mode="json", exclude_unset=True)`. Only fields explicitly present in the message are updated; existing database values are preserved.

---

## 🚀 Running the Service

### Docker (Recommended)

```bash
docker compose up -d orion-sql-writer
```

### Local Development

**Prerequisites**

- Python 3.10+
- PostgreSQL running locally

**Install Dependencies**

```bash
pip install -r requirements.txt
```

**Run the Worker**

```bash
python main.py
```

---

## 🛠 Troubleshooting

### `KeyError: 'dream_date'`

- **Cause**: Logic attempted to access a dictionary key that was excluded by `exclude_unset=True` before the record existed.  
- **Fix**: Logic now uses the Pydantic payload object (e.g. `payload.dream_date`) for queries, ensuring defaults are respected.

### `ValidationError (Extra fields not permitted)`

- **Cause**: The bus is sending metadata (e.g. `trace_id`) that isn't in the Pydantic schema.  
- **Fix**: The worker automatically filters message keys against the schema's known fields before validation.

### No Data Appearing in DB

Check the following:

1. **Channel Mapping**  
   Is the channel mapped in `settings.py`? (Look for _"No table mapping for channel"_ in the logs.)

2. **Publisher Channel**  
   Is the Brain publishing to the correct channel? (Ensure `orion:dream:trigger` matches the listener.)

3. **Connectivity**  
   - Can the service reach PostgreSQL using `DATABASE_URL`?
   - Can it reach Redis via `ORION_BUS_URL`?

---

## 📂 Project Structure

```text
orion-sql-writer/
├── app/
│   ├── models.py       # SQLAlchemy ORM definitions
│   ├── schemas.py      # Pydantic validation models
│   ├── db.py           # Session management
│   ├── settings.py     # Channel configuration & envs
│   └── worker.py       # Main logic (consumer loop & validation)
├── main.py             # Entrypoint
├── Dockerfile
└── requirements.txt
```
