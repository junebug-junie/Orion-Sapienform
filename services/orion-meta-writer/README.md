🏷️ Orion Meta Writer Service

Version: 0.1.0
Purpose: A decoupled, message-driven microservice for validating and enriching data streams.
1. Architectural Role

The orion-meta-writer acts as a crucial quality control and processing step in the Orion data pipeline. It is a pure "worker" service that does not connect directly to any database.

Its responsibilities are:

    Listen: It subscribes to a specific channel on the Orion Bus (Redis).

    Validate: It uses a Pydantic schema to validate that incoming messages have the correct structure and required fields. Malformed messages are discarded.

    Enrich: It "stamps" valid data with its own metadata, such as its service name, version, and the processing timestamp.

    Republish: It publishes the validated and enriched data to a new, downstream channel on the bus for other services to consume (e.g., a dedicated database writer).

This decoupled design makes the system more robust, scalable, and easier to maintain.
2. Data Flow

    Subscribes to: orion.tags (configurable via SUBSCRIBE_CHANNEL in .env)

    Publishes to: orion.tags.enriched (configurable via PUBLISH_CHANNEL in .env)

3. Configuration

This service is configured using two environment files:

    ../../.env: The root project .env file, which provides global variables like ${PROJECT} and ${NET}.

    .env: The service-specific file located in this directory, which defines the service's identity, port, and bus channels.

Key variables in services/orion-meta-writer/.env:

    SERVICE_NAME: The name of the service.

    PORT: The port the FastAPI health check endpoint will run on.

    ORION_BUS_URL: The full URL to the Redis instance.

    SUBSCRIBE_CHANNEL: The Redis channel to listen for incoming messages.

    PUBLISH_CHANNEL: The Redis channel to publish enriched messages to.

    STARTUP_DELAY: A delay (in seconds) to wait before starting the app, ensuring network services are available.

4. How to Run

All commands must be run from the project root (/Orion-Sapienform).
To Build and Start the Service

This command explicitly loads both the root and local .env files, builds the image, and starts the container in the background.

docker compose --env-file .env --env-file services/orion-meta-writer/.env -f services/orion-meta-writer/docker-compose.yml up -d --build

To View Logs

This command follows the real-time logs for the running service.

docker compose -f services/orion-meta-writer/docker-compose.yml logs -f

To Stop the Service

docker compose -f services/orion-meta-writer/docker-compose.yml down

