🧠 Orion RAG Service

Version: 1.0.0 (Refactored)
Primary Responsibility: Provides a retrieval-augmented generation (RAG) backend, combining vector search with language model intelligence supplied by the orion-brain service.
1. Architectural Overview

The Orion RAG service acts as a specialized "long-term memory" module for the Orion mesh. It is a pure, backend worker service that does not serve a UI. Its primary function is to listen for requests on the Orion Bus, perform a vector search to find relevant context, and then delegate the final text generation to the orion-brain service.

This refactor decouples the RAG service from direct LLM access, centralizing all language model inference within the orion-brain. This makes the entire system more modular, scalable, and easier to manage.
Data Flow

    Request: An upstream service (like orion-hub or a new "dream model" service) publishes a rag:query:request event to the Orion Bus. The payload contains the user's query and a response_channel.

    Vector Search: The RAG service receives this message, extracts the query, and performs a similarity search against its internal ChromaDB vector store to find relevant document chunks.

    Delegate to Brain: The RAG service constructs a new payload containing the original query and the retrieved context. It publishes this to the orion:brain:intake channel, specifying a unique channel for the brain to send its final answer back to.

    Receive from Brain: The RAG service listens on the unique response channel for the brain's synthesized answer.

    Final Response: The RAG service takes the final answer from the brain and publishes it to the response_channel that was specified in the original request, completing the loop.

2. Folder and File Structure

To align with the established conventions of the Orion mesh, the service will be organized as follows:

/services/orion-rag/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point, listener startup
│   ├── settings.py       # Pydantic settings management
│   ├── models.py         # Pydantic schemas for bus messages
│   └── vector_store.py   # Logic for interacting with ChromaDB
├── .env                  # Service-specific environment variables
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Container build instructions
└── README.md             # This file

