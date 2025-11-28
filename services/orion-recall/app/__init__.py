# app/__init__.py
from .main import app  # so "app.main:app" works for uvicorn

__all__ = ["app"]
