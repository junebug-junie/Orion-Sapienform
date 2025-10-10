# 🧠 Orion Bus — Mesh Message Backbone

The **Orion Bus** is the shared message-broker layer that connects every service in the Orion-Sapienform mesh.
It functions like the **nervous system** of the network — propagating events, state changes, and telemetry across all nodes.

---

## 🌐 Overview

| Component        | Purpose                                          | Port   |
| ---------------- | ------------------------------------------------ | ------ |
| **Bus Core**     | Primary message broker (temporary Redis backend) | `6379` |
| **Bus Exporter** | Prometheus metrics endpoint                      | `9121` |

All services in the mesh communicate through this bus for:

* Event emission (`orion:evt:*`)
* Internal messaging (`orion:bus:*`)
* Shared state and coordination
* Real-time introspection and health signals

---

## ⚙️ Prerequisites

* Docker / Docker Compose v2
* Shared Docker network (usually created once per node):

```bash
docker network inspect app-net >/dev/null 2>&1 || docker network create app-net
```

---

## 🚀 Bring Up / Down

```bash
# Start the bus stack
make up

# Show running containers
make ps

# Tail logs
make logs

# Stop the stack
make down
```

---

## 🔍 Monitoring

Prometheus metrics are exposed at:

```
http://localhost:9121/metrics
```

To quickly view the top of the metrics stream:

```bash
make metrics
```

---

## 🧩 Interacting with the Bus

### CLI Access

```bash
make cli
```

Opens a shell into the broker for issuing commands or inspecting streams.

### Stream Inspection

Each Orion service publishes events to canonical stream keys:

| Stream              | Purpose                    |
| ------------------- | -------------------------- |
| `orion:evt:gateway` | internal mesh event stream |
| `orion:bus:out`     | outbound message bus       |

To preview messages:

```bash
make stream            # defaults to orion:evt:gateway
make stream STREAM=orion:bus:out
```

---

## 🧠 How It Fits the Mesh

* The **Bus** carries messages between higher-order services such as:

  * **Brain** (reasoning)
  * **RAG** (retrieval / context)
  * **Mirror** (memory / collapse logging)
  * **Meta** (tagging / semantic linking)
* Each node in the Orion mesh attaches to the same network and uses the same topic conventions.
* Replacing the current Redis backend with another protocol (e.g., NATS, Kafka) will not affect higher services—the Bus interface remains stable.

---

## 🛠️ Future Roadmap

* Transition from Redis Streams to a custom event-bus abstraction.
* Introduce schema validation for inter-service messages.
* Enable persistent stream snapshots for replay and debugging.
* Integrate distributed tracing hooks for Mesh Observability.

---

### TL;DR

> The Orion Bus is not just storage—it’s the **connective tissue** that gives the Mesh a collective nervous system.
