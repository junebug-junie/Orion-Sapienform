📦 Orion Vector DB

Responsibility: Provides a persistent, standalone vector database for the Orion mesh.
1. Overview

This service runs the official chromadb/chroma container image. Its sole purpose is to provide a central, persistent server for storing and querying vector embeddings. It is the "database server" component of the vector store architecture.

It is designed to be accessed by other services on the mesh, primarily the orion-vector-writer (for ingestion) and the orion-rag service (for querying).
2. Configuration

Configuration is managed via the .env file in this directory, which defines:

    PORT: The host port to expose the ChromaDB API on.

    CHROMA_DATA_PATH: The path on the host machine where the persistent database files will be stored.

3. How to Run

This service is managed like any other in the Orion mesh.

From the project root (/Orion-Sapienform):

# Start the service
docker compose --env-file .env --env-file services/orion-vector-db/.env -f services/orion-vector-db/docker-compose.yml up -d

# Check its status
docker compose -f services/orion-vector-db/docker-compose.yml ps

# View its logs
docker compose -f services/orion-vector-db/docker-compose.yml logs -f


🧠 Orion Vector Store — Architecture Blueprint

This document outlines the architecture for a dedicated, mesh-wide vector store. This component will serve as the central long-term memory and retrieval backbone for services like RAG, chat history, and semantic search.

This design moves away from embedded databases (like the one previously in orion-rag) to a centralized, service-oriented architecture, mirroring the pattern used by orion-sql-db.
1. Architectural Overview

The architecture consists of two new, distinct services:

    orion-vector-db (The Database Server): A standalone service that runs the official ChromaDB vector database in a container. Its only job is to store and retrieve vectorized data.

    orion-vector-writer (The Ingestion Service): A "headless" worker service that acts as the single, intelligent gateway for writing data to the vector database. It listens for various events on the Orion Bus, validates their schemas, transforms them into a standardized document format, generates embeddings, and upserts them into the orion-vector-db.

This separation of concerns is critical. It means that services like collapse-mirror or orion-hub don't need to know anything about embeddings or vector databases. They just publish their standard events to the bus, and the vector-writer handles the rest.
2. The "Schema Enforcer" Pattern

The orion-vector-writer will function as the schema enforcement middleware you asked about. It will be built to understand different types of incoming data.
Subscription Channels:

The writer will listen to multiple channels simultaneously, for example:

    orion:collapse:triage (for new collapse mirror events)

    orion:chat:history:log (for chat messages to be logged for long-term memory)

    orion:rag:document:add (for explicitly adding documents to the RAG store)

Schema Mapping:

Inside the vector-writer's code, there will be a mapping that links each channel to a specific Pydantic validation schema.

    When a message arrives on orion:collapse:triage, it's validated against a CollapseMirrorEvent schema. The summary field is extracted as the text to be embedded.

    When a message arrives on orion:chat:history:log, it's validated against a ChatMessage schema. The content field is extracted.

This makes the system incredibly extensible. To add a new data source, you simply:

    Add a new channel to the writer's subscription list.

    Create a new Pydantic model for that data type.

    Add a new rule to the channel-to-schema map.

3. Data Flow Example (Collapse Mirror)

    [ orion-hub ] submits a collapse mirror form.

    [ orion-collapse-mirror ] processes it and publishes the full event to orion:collapse:triage.

    [ orion-vector-writer ] (listening on orion:collapse:triage):

        Receives the message.

        Validates it against its CollapseMirrorEvent schema.

        Extracts the summary field.

        Uses a sentence-embedding model to generate a vector for the summary.

        Connects to orion-vector-db and upserts the document, vector, and metadata.

4. Next Steps

With this blueprint, we can now proceed with the implementation, which will involve:

    Creating the docker-compose.yml and .env files for a new orion-vector-db service group.

    Creating the full Python application for the orion-vector-writer service, including its Docker configuration and application code.

    Updating the orion-rag service to remove its embedded database and instead query the
