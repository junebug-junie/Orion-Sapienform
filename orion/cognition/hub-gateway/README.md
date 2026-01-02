# Hub Gateway (Harness)

This directory contains the bus harness for testing Cortex interactions.

Because python modules cannot contain hyphens, the actual code is located in `orion/cognition/hub_gateway/`.

## Usage

You can run the harness using the following command:

```bash
python -m orion.cognition.hub_gateway.bus_harness brain "Hello world"
```

or

```bash
python -m orion.cognition.hub_gateway.bus_harness agent "Plan a trip to Mars"
```

To tap into all bus traffic:

```bash
python -m orion.cognition.hub_gateway.bus_harness tap
```
