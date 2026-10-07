"""ASGI entry point: ``uvicorn aibridge_exec.server.asgi:app``."""
from .app import create_app

app = create_app()
