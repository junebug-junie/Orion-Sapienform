# Orion Bus Client

A lightweight Python client for the Orion Mesh Redis bus.

## Installation

From local path:

```bash
pip install ./client
```

From Git (specific tag):

```bash
pip install git+https://github.com/YOURORG/orion-bus.git@v0.1.0#subdirectory=client
```

## Usage

```python
from orionbus import OrionBus

bus = OrionBus()
bus.publish("orion:bus:out", {"type": "ping", "content": "hello mesh!"})
```

## Features

* **Publish**: Send events to Redis channels.
* **Subscribe**: Listen to channels and yield messages.
* **Auto-config**: Picks up `ORION_BUS_URL` and `ORION_BUS_ENABLED` from env.

## Environment Variables

| Variable            | Default                      | Description               |
| ------------------- | ---------------------------- | ------------------------- |
| `ORION_BUS_URL`     | `redis://orion-redis:6379/0` | Redis connection URL      |
| `ORION_BUS_ENABLED` | `true`                       | Enable/disable publishing |

## Example: Subscribing

```python
bus = OrionBus()
for message in bus.subscribe("orion:evt:gateway"):
    print("Received event:", message)
```

## License

MIT License – see [LICENSE](LICENSE) for details.
