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

The architecture consists of two primary services:

    orion-vector-db (The Database Server): A standalone service that runs the official ChromaDB vector database in a container. Its only job is to store and retrieve vectorized data.

    orion-vector-writer (The Ingestion Service): A "headless" worker service that acts as the single, intelligent gateway for writing data to the vector database. It listens for pre-embedded upsert events on the Orion Bus and persists them into the orion-vector-db. Embeddings are produced upstream (for example by the chat memory service or a dedicated embedding host) and passed through unchanged.

This separation of concerns is critical. It means that services like collapse-mirror or orion-hub don't need to know anything about vector databases. They publish normalized upsert events to the bus with embeddings already attached, and the vector-writer handles the storage.
2. The "Schema Enforcer" Pattern

The orion-vector-writer functions as a schema enforcement and storage middleware. It now consumes `memory.vector.upsert.v1` envelopes that already contain embeddings and optional latent references. Upstream services are responsible for obtaining embeddings and populating the standard payload.

Subscription Channels:

The writer listens to pre-embedded upsert events, for example:

    orion:memory:vector:upsert (for all vectorizable memory docs)

Schema Mapping:

Upsert payloads are validated against a shared `VectorDocumentUpsertV1` Pydantic schema, which carries the document text, metadata, embedding, embedding model info, and optional latent references.

This makes the system extensible. To add a new data source, emit a compliant upsert envelope with the embedding already attached and the writer will persist it.

3. Data Flow Example (Collapse Mirror)

    [ orion-hub ] submits a collapse mirror form.

    [ orion-collapse-mirror ] processes it and publishes the full event to orion:collapse:triage.

    [ orion-chat-memory / embedding host ] generates an embedding for the collapse summary and publishes a `memory.vector.upsert.v1` message.

    [ orion-vector-writer ] (listening on orion:memory:vector:upsert):

        Receives the message.

        Validates it against the VectorDocumentUpsertV1 schema.

        Upserts the document, vector, and metadata into the vector store without recomputing embeddings.

4. Next Steps

With this blueprint, we can now proceed with the implementation, which will involve:

    Creating the docker-compose.yml and .env files for a new orion-vector-db service group.

    Creating the full Python application for the orion-vector-writer service, including its Docker configuration and application code.

5. Testing

Find the IP:
    docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' orion-athena-vector-db

Run within the container:

    docker exec -it orion-athena-vector-writer python -c "

    import chromadb
    from chromadb.config import Settings
    import pprint

    print('🔌 Connecting to ChromaDB...')

    # Disable telemetry to avoid the ClientStartEvent error'
    client = chromadb.HttpClient(
        host='172.18.0.22', # or replace with IP from above 
        port=8000, 
        settings=Settings(anonymized_telemetry=False)
    )

    print('\n📊 Collections Report:')
    collections = client.list_collections()
    if not collections:
        print('   (No collections found)')

    for c in collections:
        count = c.count()
        print(f'📁 Collection: {c.name} | Items: {count}')
        if count > 0:
            print('   🔎 Sample (Metadata):')
            pprint.pprint(c.peek(limit=1)['metadatas'])
        print('-' * 40)
    "
