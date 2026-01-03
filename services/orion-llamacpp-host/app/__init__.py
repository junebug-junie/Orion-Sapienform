# services/orion-llamacpp/app/__init__.py
"""
Orion llama.cpp service.

This service:
- Reads llm_profiles.yaml + env to resolve a model + GPU layout.
- Launches llama.cpp's `llama-server` with those settings.
- Exposes OpenAI-style HTTP on LLAMACPP_HOST:LLAMACPP_PORT for llm-gateway.
"""
