from __future__ import annotations

from typing import Any, Dict


def build_health_payload(service) -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_version": service.model_version,
        "model_loaded": service.model_loaded,
        "last_fit_at": service.last_fit_at,
        "last_assign_at": service.last_assign_at,
        "last_summary_at": service.last_summary_at,
        "last_drift_at": service.last_drift_at,
        "last_error": service.last_error,
    }


def create_app(service):
    from fastapi import FastAPI

    app = FastAPI(title="orion-topic-rail")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return build_health_payload(service)

    return app


def start_http_server(service, host: str, port: int) -> None:
    import uvicorn

    app = create_app(service)
    uvicorn.run(app, host=host, port=port, log_level="info")
