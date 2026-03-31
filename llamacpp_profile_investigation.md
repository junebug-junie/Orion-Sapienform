# Llama.cpp Profile Forwarding Investigation

This note captures validated findings for the `qwen3-30b-a3b-q4km-atlas-agent-64k-think` profile in `config/llm_profiles.yaml`.

## Key finding

`services/orion-llamacpp-host` parses the selected profile name correctly, but only maps a narrow fixed subset of `llamacpp` keys into `llama-server` argv (`ctx_size`, `n_gpu_layers`, `threads`, `n_parallel`, `batch_size`, plus model/host/port). Extended Qwen-oriented fields present in YAML are currently ignored by schema and launch mapping.

## Code locations

- Profile load/selection: `services/orion-llamacpp-host/app/settings.py`
- Parsed profile schema: `services/orion-llamacpp-host/app/profiles.py`
- Argv construction: `services/orion-llamacpp-host/app/main.py`
