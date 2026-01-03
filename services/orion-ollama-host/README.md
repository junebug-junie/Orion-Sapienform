# orion-ollama-host

A simple wrapper around the official Ollama container.

## Features
- Runs `ollama serve`
- On startup, automatically pulls the model defined in `OLLAMA_MODEL_ID` or via `OLLAMA_PROFILE_NAME` (looking up `model_id` in `config/llm_profiles.yaml`).
- Exposes port 11434.

## Configuration

Set `OLLAMA_MODEL_ID` to the model tag you want (e.g., `llama3`, `mistral`).
Or set `OLLAMA_PROFILE_NAME` to a profile key in `llm_profiles.yaml`.

## Usage

```bash
docker compose -f services/orion-ollama-host/docker-compose.yml up -d
```
