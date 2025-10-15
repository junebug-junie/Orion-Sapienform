✍️ Orion Vector Writer

Version: 1.0.0
Responsibility: Acts as the sole ingestion point for the Orion Vector DB.
1. Overview

This service is a "headless" worker that listens to multiple channels on the Orion Bus. It is the "schema enforcer" for the vector database. Its job is to:

    Subscribe to events from various services (e.g., orion-collapse-mirror, orion-meta-writer).

    Validate that the incoming messages conform to expected Pydantic schemas.

    Transform the validated data into a standardized document format.

    Generate vector embeddings for the document text using a sentence-transformer model.

    Connect to the orion-vector-db service and upsert the document, its embedding, and its metadata.

This decoupled architecture means that producer services don't need to know anything about vector databases or embeddings; they just publish their standard events.
2. Configuration & Usage

Configuration is managed via the .env file. To run this service, use the standard command from the project root:

docker compose --env-file .env --env-file services/orion-vector-writer/.env -f services/orion-vector-writer/docker-compose.yml up -d --build

