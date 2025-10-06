Orion RDF Writer Service
1. Overview

The orion-rdf-writer is a microservice designed to act as a bridge between the Orion message bus (Redis) and an Ontotext GraphDB repository. Its primary responsibility is to:

    Listen for JSON-formatted events on specific Redis Pub/Sub channels.

    Transform these events into RDF N-Triples format based on a predefined structure.

    Persist the generated triples into a specified GraphDB repository.

It is built to be resilient, featuring batch processing, connection retries, and robust error handling.
2. Architecture & Data Flow

The service operates on a simple, event-driven data flow:

    Subscription: On startup, the service subscribes to one or more Redis channels defined in its configuration.

    Queueing: As messages are received, they are placed into an in-memory queue.

    Batch Processing: A background thread periodically processes messages from the queue. A batch is processed when either the BATCH_SIZE is reached or a 5-second timeout occurs, ensuring low-volume messages are not indefinitely stranded.

    Transformation: Each event in the batch is passed to a builder function (build_triples) which generates RDF triples if the event contains the required keys (mentions, relatesTo).

    Persistence: The resulting N-Triples data is sent via an HTTP POST request to the configured GraphDB instance.

    Error Handling: If an event cannot be processed (due to invalid structure or a database error), a detailed error report is published to a dedicated Redis error channel and logged to a persistent file.

+----------------+      +----------------------+      +------------------+
| Redis Pub/Sub  |----->| orion-rdf-writer     |----->| GraphDB          |
| (Message Bus)  |      | (Consumer/Processor) |      | (RDF Repository) |
+----------------+      +----------------------+      +------------------+
        |                                                    ^
        | (on failure)                                       |
        v                                                    |
+----------------+                                           |
| Redis Error    | <-----------------------------------------+
| Channel & Log  |
+----------------+

3. Configuration

The service is configured entirely through environment variables, which are loaded from a .env file at the root of the service directory.

Variable
	

Default Value
	

Description

GraphDB Settings
	


	


GRAPHDB_URL
	

http://graphdb:7200
	

The base URL for the GraphDB instance.

GRAPHDB_REPO
	

collapse
	

The name of the target repository within GraphDB.

GRAPHDB_USER
	

None
	

The username for GraphDB authentication (if required).

GRAPHDB_PASS
	

None
	

The password for GraphDB authentication (if required).

Redis Bus Settings
	


	


ORION_BUS_URL
	

redis://orion-redis:6379/0
	

The connection URL for the Redis instance.

CHANNEL_EVENTS_TAGGED
	

orion:events:tagged
	

A channel the service listens to for events.

CHANNEL_RDF_ENQUEUE
	

orion:rdf:enqueue
	

The primary channel for direct RDF ingestion requests.

CHANNEL_CORE_EVENTS
	

orion:core:events
	

A channel for core system events that may contain RDF targets.

CHANNEL_RDF_CONFIRM
	

orion:rdf:confirm
	

Channel where success confirmations are published.

CHANNEL_RDF_ERROR
	

orion:rdf:error
	

Channel where processing failures are published.

Service Behavior
	


	


SERVICE_NAME
	

orion-rdf-writer
	

The identifier for this service.

LOG_LEVEL
	

INFO
	

The logging verbosity. Can be DEBUG, INFO, WARNING, ERROR.

BATCH_SIZE
	

10
	

The number of messages to accumulate before processing a batch.

RETRY_LIMIT
	

3
	

The number of times to retry pushing to GraphDB on a connection failure.

RETRY_INTERVAL
	

2
	

The number of seconds to wait between retries.
4. Running the Service

This service is designed to be run with Docker and Docker Compose.

Build the Image:
To incorporate any code changes, rebuild the service's Docker image.

docker-compose build orion-rdf-writer

Run the Service:
To start the service in detached mode:

docker-compose up -d orion-rdf-writer

View Logs:
To view the real-time logs of the running service:

docker-compose logs -f orion-rdf-writer

5. Usage & Testing

To have the service process data, you must publish a message to one of its subscribed Redis channels. The easiest way to do this is with redis-cli inside the Redis container.
Example Valid Message

A valid message must contain either a mentions or relatesTo key with a list of strings.

# Replace 'orion-bus-orion-redis-1' with your actual Redis container name
docker exec orion-bus-orion-redis-1 redis-cli PUBLISH orion:rdf:enqueue '{"id": "event-xyz-789", "mentions": ["entity-A", "entity-B"], "relatesTo": ["topic-X"]}'

Expected Outcome: You will see a ✅ RDF inserted message in the service logs, and the corresponding triples will be present in GraphDB.
Example Invalid Message

An invalid message lacks the required keys.

docker exec orion-bus-orion-redis-1 redis-cli PUBLISH orion:rdf:enqueue '{"id": "bad-event-123", "some_other_data": "value"}'

Expected Outcome: You will see a No triples generated for event warning in the service logs. An error message will be published to the orion:rdf:error channel and logged to the error file.
6. Error Handling

The service has two mechanisms for flagging failed events:

    Redis Error Channel (orion:rdf:error): A structured JSON message detailing the failure is published to this channel. This allows other services to monitor failures in real-time.

    Persistent Log File: A detailed JSON record of the error is appended to a file on the host machine, located at /mnt/storage/rdf_logs/errors.txt. This is configured via a Docker volume in the docker-compose.yml file and provides a durable record of all failures.

// Example entry in errors.txt
{
    "timestamp": "2025-10-06T00:30:00.123Z",
    "service": "orion-rdf-writer",
    "error": "No triples generated for event bad-event-123. Check event structure.",
    "failed_event": { "id": "bad-event-123", "some_other_data": "value" }
}

